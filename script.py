import os
import shutil
import random

# Define your current paths based on your terminal output
base_path = os.path.expanduser("~/Work/fyp/dataset")
original_dir = os.path.join(base_path, "DFD_original sequences")
# Path based on your 'ls' which showed a nested structure
manipulated_dir = os.path.join(base_path, "DFD_manipulated_sequences/DFD_manipulated_sequences")

# Define target structure
target_base = os.path.expanduser("~/Work/fyp/data")
splits = ['train', 'val']
categories = ['REAL', 'FAKE']

def create_folders():
    for split in splits:
        for category in categories:
            path = os.path.join(target_base, split, category)
            os.makedirs(path, exist_ok=True)

def move_files(source, category, split_ratio=0.8):
    files = [f for f in os.listdir(source) if f.endswith('.mp4')]
    random.shuffle(files)
    
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    for f in train_files:
        shutil.copy(os.path.join(source, f), os.path.join(target_base, 'train', category, f))
    
    for f in val_files:
        shutil.copy(os.path.join(source, f), os.path.join(target_base, 'val', category, f))
    
    print(f"Moved {len(files)} files from {category} source to train/val folders.")

if __name__ == "__main__":
    create_folders()
    # Move Real videos
    move_files(original_dir, 'REAL')
    # Move Fake videos
    move_files(manipulated_dir, 'FAKE')
    print("Dataset reorganization complete!")