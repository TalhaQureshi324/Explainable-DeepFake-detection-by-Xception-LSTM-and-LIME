"""
Preprocessing Module for Deepfake Detection
============================================
This module handles face extraction from videos and images.
Extracts face frames from entire videos for deepfake detection training.

Usage:
    python preprocessing.py

Output Structure:
    processed_dataset/
    ├── train/
    │   ├── REAL/
    │   │   ├── video_name/
    │   │   │   ├── frame_000_face_0.jpg
    │   │   │   ├── frame_001_face_0.jpg
    │   │   │   └── ...
    │   └── FAKE/
    └── val/
        ├── REAL/
        └── FAKE/
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import random


# ============================== CONFIG ==============================

DEFAULT_CONFIG = {
    # Paths
    "base_data_dir": r"D:\Computer Vision\FYP\data",
    "output_dir": r"D:\Computer Vision\FYP\processed_dataset",
    "model_dir": r"D:\Computer Vision\FYP\models",
    
    # Face Detection (OpenCV DNN)
    "prototxt_path": r"D:\Computer Vision\FYP\models\deploy.prototxt",
    "caffemodel_path": r"D:\Computer Vision\FYP\models\res10_300x300_ssd_iter_140000.caffemodel",
    "confidence_threshold": 0.5,
    
    # Face Processing
    "image_size": (299, 299),  # Xception input size
    "margin": 0.2,  # Margin around detected face (20%)
    "remove_background": True,  # Remove background from extracted faces
    "face_mask_scale": (0.85, 0.90),  # Ellipse scale (width, height) relative to crop
    
    # Video Processing
    "frames_per_video": 32,  # Number of frames to extract per video
    "max_faces_per_frame": 1,  # Max faces to extract per frame (1 for main face)
    
    # Data Splitting (for raw video organization)
    "train_percent": 0.10,
    "val_percent": 0.05,
    
    # Sampling
    "sample_fraction": 1.0,  # Fraction of videos to process (1.0 = 100%)
    "random_seed": 42,  # Seed for reproducible sampling
}


class FaceExtractor:
    """
    Face extractor using OpenCV DNN face detector.
    Extracts and preprocesses faces from images and videos.
    """
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.face_net = self._load_face_detector()
        
    def _load_face_detector(self):
        """Load OpenCV DNN face detector."""
        prototxt = self.config.get("prototxt_path", DEFAULT_CONFIG["prototxt_path"])
        caffemodel = self.config.get("caffemodel_path", DEFAULT_CONFIG["caffemodel_path"])
        
        if not os.path.exists(prototxt):
            raise FileNotFoundError(f"Prototxt file not found: {prototxt}")
        if not os.path.exists(caffemodel):
            raise FileNotFoundError(f"CaffeModel file not found: {caffemodel}")
            
        return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame using OpenCV DNN.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Pass blob through network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Compute box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2, confidence))
        
        # Sort by confidence and return top faces
        faces.sort(key=lambda x: x[4], reverse=True)
        max_faces = self.config.get("max_faces_per_frame", 1)
        return faces[:max_faces]
    
    def remove_background(self, face_img):
        """
        Remove background from extracted face using an elliptical mask.
        Blacks out everything outside the face ellipse so only the face remains.
        
        Args:
            face_img: Extracted face image (H x W x C)
            
        Returns:
            Face image with background removed (black outside face region)
        """
        h, w = face_img.shape[:2]
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        
        # Ellipse axes based on config scale
        scale_w, scale_h = self.config.get("face_mask_scale", (0.85, 0.90))
        axes = (int(w * scale_w / 2), int(h * scale_h / 2))
        
        # Draw filled ellipse
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Feather the edge slightly for smoother transition
        kernel_size = max(3, int(min(h, w) * 0.03))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Normalize mask to 0-1
        mask = mask.astype(np.float32) / 255.0
        
        # Apply mask to each channel
        if len(face_img.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)
        
        result = (face_img.astype(np.float32) * mask).astype(np.uint8)
        return result
    
    def extract_face(self, frame, bbox, margin=0.2):
        """
        Extract face from frame with margin and optional background removal.
        
        Args:
            frame: Source image
            bbox: (x1, y1, x2, y2) bounding box
            margin: Margin percentage to add around face
            
        Returns:
            Resized face image with background removed if configured
        """
        x1, y1, x2, y2 = bbox[:4]
        h, w = frame.shape[:2]
        
        # Calculate margin
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        # Apply margin with boundary checks
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Extract face
        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
            
        # Resize to model input size
        target_size = self.config.get("image_size", (299, 299))
        face = cv2.resize(face, target_size)
        
        # Remove background if enabled
        if self.config.get("remove_background", True):
            face = self.remove_background(face)
        
        return face
    
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image and extract faces.
        
        Args:
            image_path: Path to image file
            output_dir: Directory to save extracted faces (optional)
            
        Returns:
            List of extracted face images
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return []
        
        faces_data = self.detect_faces(frame)
        extracted_faces = []
        
        for i, face_data in enumerate(faces_data):
            face = self.extract_face(frame, face_data)
            if face is not None:
                extracted_faces.append(face)
                
                # Save if output directory provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = Path(image_path).stem
                    output_path = os.path.join(output_dir, f"{base_name}_face_{i}.jpg")
                    cv2.imwrite(output_path, face)
        
        return extracted_faces
    
    def process_video(self, video_path, output_dir, max_frames=32):
        """
        Process a video and extract face frames.
        Saves ALL face frames from the video for temporal analysis.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted face frames
            max_frames: Maximum number of frames to extract
            
        Returns:
            Number of face frames extracted
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            cap.release()
            return 0
        
        # Calculate frame indices to extract (evenly spaced)
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_count = 0
        frame_idx = 0
        saved_indices = []
        
        pbar = tqdm(total=len(frame_indices), desc=f"Processing {Path(video_path).name}", leave=False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only selected frames
            if frame_idx in frame_indices:
                faces_data = self.detect_faces(frame)
                
                for face_idx, face_data in enumerate(faces_data):
                    face = self.extract_face(frame, face_data)
                    if face is not None:
                        output_path = os.path.join(
                            output_dir, 
                            f"frame_{extracted_count:04d}_face_{face_idx}.jpg"
                        )
                        cv2.imwrite(output_path, face)
                        saved_indices.append(frame_idx)
                        extracted_count += 1
                
                pbar.update(1)
            
            frame_idx += 1
            
            # Early exit if we've processed all needed frames
            if frame_idx > frame_indices[-1]:
                break
        
        pbar.close()
        cap.release()
        
        # Save metadata
        metadata = {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "extracted_faces": extracted_count,
            "frame_indices": saved_indices,
            "target_size": self.config.get("image_size", (299, 299))
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return extracted_count


class DatasetPreprocessor:
    """
    Preprocessor for entire dataset (train/val splits).
    """
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.face_extractor = FaceExtractor(config)
    
    def process_split(self, split_name="train", categories=None):
        """
        Process all videos in a data split (train/val).
        
        Args:
            split_name: 'train' or 'val'
            categories: List of categories ['REAL', 'FAKE'] or None for both
        """
        if categories is None:
            categories = ['REAL', 'FAKE']
        
        base_data_dir = self.config.get("base_data_dir", DEFAULT_CONFIG["base_data_dir"])
        output_dir = self.config.get("output_dir", DEFAULT_CONFIG["output_dir"])
        max_frames = self.config.get("frames_per_video", 32)
        
        stats = {cat: 0 for cat in categories}
        
        for category in categories:
            input_dir = os.path.join(base_data_dir, split_name, category)
            
            if not os.path.exists(input_dir):
                print(f"Input directory not found: {input_dir}")
                continue
            
            # Get all video files
            video_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            if not video_files:
                print(f"No videos found in {input_dir}")
                continue
            
            # Sample fraction of videos if specified
            sample_fraction = self.config.get("sample_fraction", 1.0)
            if sample_fraction < 1.0:
                random.seed(self.config.get("random_seed", 42))
                sample_count = max(1, int(len(video_files) * sample_fraction))
                video_files = random.sample(video_files, sample_count)
                print(f"\nProcessing {split_name}/{category}: {len(video_files)} videos (sampled from {len(os.listdir(input_dir))} total)")
            else:
                print(f"\nProcessing {split_name}/{category}: {len(video_files)} videos")
            
            for video_file in tqdm(video_files, desc=f"{split_name}/{category}"):
                video_path = os.path.join(input_dir, video_file)
                video_name = Path(video_file).stem
                
                output_video_dir = os.path.join(output_dir, split_name, category, video_name)
                
                # Skip if already processed
                if os.path.exists(output_video_dir) and os.listdir(output_video_dir):
                    continue
                
                # Process video
                extracted = self.face_extractor.process_video(
                    video_path, 
                    output_video_dir, 
                    max_frames=max_frames
                )
                
                if extracted > 0:
                    stats[category] += 1
        
        return stats
    
    def process_all(self):
        """Process both train and val splits."""
        print("=" * 60)
        print("Starting Dataset Preprocessing")
        print("=" * 60)
        
        all_stats = {}
        
        for split in ['train', 'val']:
            print(f"\n{'='*60}")
            print(f"Processing {split.upper()} split")
            print(f"{'='*60}")
            
            stats = self.process_split(split)
            all_stats[split] = stats
        
        # Print summary
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        for split, stats in all_stats.items():
            print(f"\n{split.upper()}:")
            for category, count in stats.items():
                print(f"  {category}: {count} videos processed")
        
        return all_stats


# ============================== SINGLE FILE PROCESSING ==============================

def process_single_video(video_path, output_dir=None, max_frames=32):
    """
    Process a single video file for inference.
    
    Args:
        video_path: Path to video file
        output_dir: Optional output directory
        max_frames: Maximum frames to extract
        
    Returns:
        List of face frames as numpy arrays
    """
    extractor = FaceExtractor()
    
    if output_dir is None:
        output_dir = os.path.join("temp_frames", Path(video_path).stem)
    
    # Extract faces
    count = extractor.process_video(video_path, output_dir, max_frames)
    
    if count == 0:
        return []
    
    # Load extracted frames
    frames = []
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    
    for frame_file in frame_files:
        frame_path = os.path.join(output_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    return np.array(frames)


def process_single_image(image_path):
    """
    Process a single image file for inference.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Face image as numpy array (RGB, 299x299) or None
    """
    extractor = FaceExtractor()
    
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    
    faces_data = extractor.detect_faces(frame)
    
    if not faces_data:
        return None
    
    # Get the face with highest confidence
    face = extractor.extract_face(frame, faces_data[0])
    
    if face is None:
        return None
    
    # Convert BGR to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    return face


# ============================== MAIN ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess videos for deepfake detection")
    parser.add_argument("--mode", choices=["dataset", "video", "image"], default="dataset",
                       help="Processing mode: dataset (train/val), single video, or single image")
    parser.add_argument("--input", "-i", help="Input video or image path (for single file mode)")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=32, help="Max frames to extract per video")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Fraction of videos to process (e.g., 0.02 for 2%%)")
    parser.add_argument("--remove-bg", action="store_true", default=None, help="Remove background from faces")
    parser.add_argument("--no-remove-bg", action="store_true", help="Keep background in faces")
    parser.add_argument("--config", "-c", help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override config with CLI args
    if args.output:
        config["output_dir"] = args.output
    if args.sample_fraction is not None:
        config["sample_fraction"] = args.sample_fraction
    if args.remove_bg is True:
        config["remove_background"] = True
    elif args.no_remove_bg:
        config["remove_background"] = False
    
    if args.mode == "dataset":
        # Process entire dataset
        preprocessor = DatasetPreprocessor(config)
        preprocessor.process_all()
    
    elif args.mode == "video":
        # Process single video
        if not args.input:
            print("Error: --input required for video mode")
            exit(1)
        
        output_dir = args.output or os.path.join("temp_frames", Path(args.input).stem)
        frames = process_single_video(args.input, output_dir, args.max_frames)
        print(f"Extracted {len(frames)} face frames to {output_dir}")
    
    elif args.mode == "image":
        # Process single image
        if not args.input:
            print("Error: --input required for image mode")
            exit(1)
        
        face = process_single_image(args.input)
        if face is not None:
            output_path = args.output or f"extracted_face_{Path(args.input).stem}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"Face extracted and saved to {output_path}")
        else:
            print("No face detected in image")
