import os
import shutil
import random
import argparse

# Dataset reorganization: converts the raw FaceForensics++ folder layout
# (DFD_original_sequences / DFD_manipulated_sequences) into the
# data/train|val + REAL|FAKE structure expected by preprocessing.py.
#
# Usage:
#   python script.py \
#       --original "path/to/DFD_original sequences" \
#       --manipulated "path/to/DFD_manipulated_sequences" \
#       --target path/to/data [--split 0.8] [--seed 42]


def create_folders(target_base):
    for split in ['train', 'val']:
        for category in ['REAL', 'FAKE']:
            path = os.path.join(target_base, split, category)
            os.makedirs(path, exist_ok=True)

def move_files(source, category, target_base, split_ratio=0.8, seed=42):
    files = [f for f in os.listdir(source) if f.endswith('.mp4')]
    random.seed(seed)
    random.shuffle(files)

    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for f in train_files:
        shutil.copy(os.path.join(source, f), os.path.join(target_base, 'train', category, f))

    for f in val_files:
        shutil.copy(os.path.join(source, f), os.path.join(target_base, 'val', category, f))

    print(f"Copied {len(files)} files from {category} source to train/val folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize raw FaceForensics++ videos into train/val splits")
    parser.add_argument("--original", required=True, help="Path to 'DFD_original sequences' folder")
    parser.add_argument("--manipulated", required=True, help="Path to 'DFD_manipulated_sequences' folder")
    parser.add_argument("--target", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                        help="Output data folder (default: ./data next to this script)")
    parser.add_argument("--split", type=float, default=0.8, help="Train fraction (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    args = parser.parse_args()

    create_folders(args.target)
    # Move Real videos
    move_files(args.original, 'REAL', args.target, args.split, args.seed)
    # Move Fake videos
    move_files(args.manipulated, 'FAKE', args.target, args.split, args.seed)
    print("Dataset reorganization complete!")