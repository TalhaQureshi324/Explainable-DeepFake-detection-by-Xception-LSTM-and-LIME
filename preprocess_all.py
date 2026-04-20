# import cv2
# import os
# import random

# # Load Haar Cascade - standard frontal face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# BASE_DATA_DIR = os.path.expanduser("~/Work/fyp/data")
# OUTPUT_DIR = os.path.expanduser("~/Work/fyp/processed_dataset")

# def process_video(video_path, output_folder, num_frames=20):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total_frames <= 0: 
#         cap.release()
#         return
    
#     interval = max(1, total_frames // num_frames)
#     count = 0
#     frame_idx = 0

#     while cap.isOpened() and count < num_frames:
#         ret, frame = cap.read()
#         if not ret: break

#         if frame_idx % interval == 0:
#             # Haar Cascade works best on grayscale images
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Detect faces
#             # scaleFactor=1.1 (checks at different sizes), minNeighbors=5 (reduces false positives)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             if len(faces) > 0:
#                 # Pick the largest face found (usually the subject)
#                 (x, y, w_b, h_b) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                
#                 # Optional: Add 10% padding to the face crop (Better for Deepfake detection)
#                 h_img, w_img, _ = frame.shape
#                 pad = int(w_b * 0.1)
#                 y1, y2 = max(0, y - pad), min(h_img, y + h_b + pad)
#                 x1, x2 = max(0, x - pad), min(w_img, x + w_b + pad)

#                 face = frame[y1:y2, x1:x2]
                
#                 if face.size > 0:
#                     # Resize to 299x299 (Standard for Inception/Xception models often used in FYPs)
#                     face = cv2.resize(face, (299, 299))
#                     cv2.imwrite(f"{output_folder}/frame_{count}.jpg", face)
#                     count += 1
        
#         frame_idx += 1
#     cap.release()

# def run_pipeline():
#     for split in ['train', 'val']:
#         for category in ['REAL', 'FAKE']:
#             source_path = os.path.join(BASE_DATA_DIR, split, category)
#             if not os.path.exists(source_path): continue
            
#             all_videos = [f for f in os.listdir(source_path) if f.endswith('.mp4')]
            
#             # Shuffling for randomness
#             random.shuffle(all_videos) 
            
#             # Process 1% for testing/debugging
#             num_to_process = max(1, len(all_videos) // 10) 
#             videos = all_videos[:num_to_process] 
            
#             print(f"\n--- Processing 10% of {split}/{category} ({len(videos)} out of {len(all_videos)} videos) ---")
            
#             for i, v in enumerate(videos, start=1):
#                 v_path = os.path.join(source_path, v)
#                 v_out_name = v.replace('.', '_')
#                 v_output_path = os.path.join(OUTPUT_DIR, split, category, v_out_name)
                
#                 print(f"[{i}/{len(videos)}] Processing: {v}...", end="\r")
#                 process_video(v_path, v_output_path)

# if __name__ == "__main__":
#     run_pipeline()