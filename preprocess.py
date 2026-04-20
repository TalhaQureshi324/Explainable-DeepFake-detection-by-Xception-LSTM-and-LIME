# import cv2
# import os
# import numpy as np
# import mediapipe as mp

# # Face detection ke liye MediaPipe use kar rahe hain (Bohat fast hai)
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# def extract_faces_from_video(video_path, output_folder, num_frames=20):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Frames ke darmiyan gap takay poori video cover ho
#     interval = max(1, total_frames // num_frames)
#     count = 0
#     frame_idx = 0

#     while cap.isOpened() and count < num_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_idx % interval == 0:
#             # BGR to RGB conversion
#             results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             if results.detections:
#                 # Sirf pehla chehra (Face) uthayen
#                 detection = results.detections[0]
#                 bbox = detection.location_data.relative_bounding_box
#                 h, w, c = frame.shape
                
#                 x, y, w_b, h_b = int(bbox.xmin * w), int(bbox.ymin * h), \
#                                  int(bbox.width * w), int(bbox.height * h)
                
#                 # Face crop karein
#                 face = frame[max(0, y):y+h_b, max(0, x):x+w_b]
                
#                 if face.size > 0:
#                     face = cv2.resize(face, (299, 299))
#                     cv2.imwrite(f"{output_folder}/frame_{count}.jpg", face)
#                     count += 1
        
#         frame_idx += 1

#     cap.release()
#     print(f"Processed: {video_path} -> {count} faces extracted.")

# # Example Usage:
# # extract_faces_from_video('dataset/REAL/video1.mp4', 'processed_data/train/REAL/video1_frames')

import cv2
import os
import numpy as np
import random
from tqdm import tqdm
import mediapipe as mp

# Standard initialization
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

print("SUCCESS: MediaPipe loaded from System Python.")
# --- CONFIG ---
# Your D: drive project path
BASE_DATA_DIR = r"D:\Semester 06\ML\DeepFake\deepfake-main\data" 
OUTPUT_DIR = "processed_5%_new"
SAMPLES_PERCENT = 0.05  
TRAIN_RATIO = 0.7       
NUM_FRAMES_PER_VIDEO = 20

def extract_faces_with_padding(video_path, output_folder, num_frames=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    interval = max(1, total_frames // num_frames)
    count = 0
    frame_idx = 0

    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % interval == 0:
            # Face Detection
            results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                
                # Conversion to pixel coordinates
                x, y, w_b, h_b = int(bbox.xmin * w), int(bbox.ymin * h), \
                                 int(bbox.width * w), int(bbox.height * h)
                
                # 15% PADDING: This removes background while keeping facial edges
                pad_w = int(w_b * 0.15)
                pad_h = int(h_b * 0.15)
                
                y1, y2 = max(0, y - pad_h), min(h, y + h_b + pad_h)
                x1, x2 = max(0, x - pad_w), min(w, x + w_b + pad_w)
                
                face = frame[y1:y2, x1:x2]
                
                if face.size > 0:
                    face = cv2.resize(face, (299, 299))
                    cv2.imwrite(f"{output_folder}/frame_{count}.jpg", face)
                    count += 1
        
        frame_idx += 1

    cap.release()
    return count

def run_balanced_pipeline():
    # Process both REAL and FAKE to maintain symmetry and avoid bias
    for category in ['REAL', 'FAKE']:
        source_path = os.path.join(BASE_DATA_DIR, 'train', category) 
        if not os.path.exists(source_path):
            print(f"Path not found: {source_path}")
            continue
        
        all_videos = [f for f in os.listdir(source_path) if f.endswith('.mp4')]
        random.shuffle(all_videos) # Shuffling to avoid majority class bias
        
        num_to_process = max(1, int(len(all_videos) * SAMPLES_PERCENT))
        selected_videos = all_videos[:num_to_process]
        
        # Split calculation for 70/30
        split_idx = int(len(selected_videos) * TRAIN_RATIO)
        
        print(f"\nProcessing {category}: {len(selected_videos)} videos")

        for i, v in enumerate(tqdm(selected_videos)):
            # Distribute into train or val folders
            split_folder = 'train' if i < split_idx else 'val'
            
            v_path = os.path.join(source_path, v)
            v_out_name = v.replace('.', '_')
            v_output_path = os.path.join(OUTPUT_DIR, split_folder, category, v_out_name)
            
            extract_faces_with_padding(v_path, v_output_path, NUM_FRAMES_PER_VIDEO)

if __name__ == "__main__":
    run_balanced_pipeline()
    print(f"\nSymmetric processing finished. Data saved in {OUTPUT_DIR}")