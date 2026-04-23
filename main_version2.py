"""
main_version2.py
================
Training script for Deepfake Detection (Version 2).

Key improvements over main.py:
- Uses ALL frames up to 32 per video (instead of fixed 10).
- Shorter sequences are padded by repeating the last frame.
- Class weight balancing without throwing away videos.
- Model input shape: (32, 299, 299, 3).

Usage:
    python main_version2.py
"""

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2
import random
from collections import Counter

NUM_FRAMES = 32  # Max frames per video (matches preprocessing.py default)
MIN_FRAMES = 5   # Minimum frames to keep a video (allows short clips)


# --- 1. Data Generator (pads shorter sequences to 32) ---
class DeepfakeGeneratorV2(tf.keras.utils.Sequence):
    def __init__(self, base_dir, split='train', batch_size=2):
        super().__init__()
        self.batch_size = batch_size
        self.data = []
        path = os.path.join(base_dir, split)

        if not os.path.exists(path):
            return

        for category in ['REAL', 'FAKE']:
            cat_path = os.path.join(path, category)
            if os.path.exists(cat_path):
                label = 0 if category == 'REAL' else 1
                folders = os.listdir(cat_path)
                for folder in folders:
                    full_folder_path = os.path.join(cat_path, folder)
                    if os.path.isdir(full_folder_path):
                        frames_count = len([f for f in os.listdir(full_folder_path) if f.endswith('.jpg')])
                        if frames_count >= MIN_FRAMES:
                            self.data.append((full_folder_path, label))

        random.shuffle(self.data)

        real_count = sum(1 for _, l in self.data if l == 0)
        fake_count = sum(1 for _, l in self.data if l == 1)
        print(f"Total {len(self.data)} valid videos in {split} (REAL={real_count}, FAKE={fake_count}).")

    def get_class_weights(self):
        if len(self.data) == 0:
            return None
        labels = [label for _, label in self.data]
        counts = Counter(labels)
        total = len(labels)
        n_classes = len(counts)
        weights = {}
        for cls, count in counts.items():
            weights[cls] = total / (n_classes * count)
        return weights

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_data = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = [], []
        for folder_path, label in batch_data:
            frames = []
            frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])[:NUM_FRAMES]
            for f in frame_files:
                img = cv2.imread(os.path.join(folder_path, f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append((img / 255.0).astype(np.float32))
                    del img  # Free memory immediately

            # Pad shorter sequences by repeating the last frame
            if len(frames) > 0:
                while len(frames) < NUM_FRAMES:
                    frames.append(frames[-1].copy())

            if len(frames) == NUM_FRAMES:
                X.append(frames)
                y.append(label)

        if len(X) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# --- 2. Model (32-frame input) ---
def build_model_v2():
    base = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base.trainable = True

    for layer in base.layers[:100]:
        layer.trainable = False

    inp = Input(shape=(NUM_FRAMES, 299, 299, 3))
    x = TimeDistributed(base)(inp)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(64)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    PROCESSED_DIR = os.path.join(os.getcwd(), "preprocessed_new")
    if os.path.exists(PROCESSED_DIR):
        model = build_model_v2()
        model.summary()

        # NOTE: batch_size reduced to 1 because 32 frames uses 3x more memory
        train_gen = DeepfakeGeneratorV2(PROCESSED_DIR, 'train', batch_size=1)
        val_gen = DeepfakeGeneratorV2(PROCESSED_DIR, 'val', batch_size=1)

        if len(train_gen.data) > 0:
            class_weight = train_gen.get_class_weights()
            print(f"Class weights: {class_weight}")

            print("\n[V2] Training shuru ho rahi hai...")
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    "deepfake_detector_model_v2.keras",
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min"
                )
            ]

            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=10,
                callbacks=callbacks,
                class_weight=class_weight
            )
            print("[V2] Model saved as deepfake_detector_model_v2.keras")
        else:
            print("Processed data folder khali hai!")
    else:
        print("Processed folder nahi mila. Path check karein.")
