"""
test_version2.py
================
Testing script for Deepfake Detection (Version 2).

Key improvements over test.py:
- Uses ALL extracted frames per video (up to 32).
- Ensemble prediction via sliding windows + full-sequence prediction.
- Final score = average of all window predictions.
- XAI (LIME) explanation on the most representative frame.

Usage:
    python test_version2.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2

from preprocessing import FaceExtractor, DEFAULT_CONFIG

# --- CONFIG ---
MODEL_PATH = os.path.join(os.getcwd(), "deepfake_detector_model_v2.keras")
TEST_FOLDER = os.path.join(os.getcwd(), "testing")
OUTPUT_XAI_FOLDER = os.path.join(os.getcwd(), "xai_outputs_v2")
TEMP_FOLDER = os.path.join(os.getcwd(), "temp_runtime_frames_v2")

NUM_FRAMES = 32      # Must match main_version2.py
WINDOW_SIZE = 10     # Size of each sliding window (proven sweet spot from V1)
STRIDE = 5           # Step size between windows (50% overlap)
DECISION_THRESHOLD = 0.75


def pad_frames(frames, target_length=NUM_FRAMES):
    """Pad a list of frames to target_length by repeating the last frame."""
    frames = list(frames)
    if len(frames) == 0:
        return None
    while len(frames) < target_length:
        frames.append(frames[-1].copy())
    return np.array(frames, dtype=np.float32)


def preprocess_video_v2(video_path):
    """
    Preprocess video EXACTLY like preprocessing.py:
      - OpenCV DNN face detection
      - 20% margin + elliptical mask background removal
      - 299x299 resize
      - Extract up to 32 evenly spaced frames
    Returns list of RGB frames (0-255 uint8) or None.
    """
    config = DEFAULT_CONFIG.copy()
    if not os.path.exists(config["prototxt_path"]):
        raise FileNotFoundError(f"Prototxt missing: {config['prototxt_path']}")
    if not os.path.exists(config["caffemodel_path"]):
        raise FileNotFoundError(f"CaffeModel missing: {config['caffemodel_path']}")

    extractor = FaceExtractor(config)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = os.path.join(TEMP_FOLDER, video_stem)
    os.makedirs(temp_dir, exist_ok=True)

    count = extractor.process_video(video_path, temp_dir, max_frames=NUM_FRAMES)
    if count == 0:
        print(f"[ERROR] No faces detected in: {os.path.basename(video_path)}")
        return None, 0

    frames = []
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])
    for frame_file in frame_files:
        frame_path = os.path.join(temp_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    return frames, count


def ensemble_predict(model, frames_list):
    """
    Predict using ensemble approach:
      1. Full-sequence prediction (all frames padded to 32)
      2. Sliding-window predictions (10-frame windows, stride=5)
    Returns (final_avg_prediction, list_of_all_predictions, labels_for_each).
    """
    if len(frames_list) == 0:
        return None, [], []

    normalized = [f.astype(np.float32) / 255.0 for f in frames_list]
    predictions = []
    labels = []

    # --- 1. Full sequence prediction ---
    full_seq = pad_frames(normalized, NUM_FRAMES)
    if full_seq is not None:
        full_pred = model.predict(np.expand_dims(full_seq, axis=0), verbose=0)[0][0]
        predictions.append(full_pred)
        labels.append("Full-32")
        print(f"    [Window: Full-32] Score = {full_pred:.4f}")

    # --- 2. Sliding window predictions ---
    n = len(normalized)
    if n >= WINDOW_SIZE:
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            window = normalized[start:start + WINDOW_SIZE]
            padded = pad_frames(window, NUM_FRAMES)
            if padded is not None:
                pred = model.predict(np.expand_dims(padded, axis=0), verbose=0)[0][0]
                predictions.append(pred)
                labels.append(f"Win-{start}-{start + WINDOW_SIZE - 1}")
                print(f"    [Window: {start}-{start + WINDOW_SIZE - 1}] Score = {pred:.4f}")
    else:
        # Video shorter than window size: just pad entire sequence once
        padded = pad_frames(normalized, NUM_FRAMES)
        if padded is not None:
            pred = model.predict(np.expand_dims(padded, axis=0), verbose=0)[0][0]
            predictions.append(pred)
            labels.append(f"Short-{n}")
            print(f"    [Window: Short-{n}] Score = {pred:.4f}")

    avg_pred = float(np.mean(predictions))
    return avg_pred, predictions, labels


def explain_prediction_v2(model, processed_video, video_name):
    """LIME explanation on the middle frame of the padded sequence."""
    explainer = lime_image.LimeImageExplainer()
    target_frame = processed_video[0][NUM_FRAMES // 2]

    def predict_wrapper(images):
        responses = []
        for img in images:
            seq = np.stack([img] * NUM_FRAMES, axis=0)
            res = model.predict(np.expand_dims(seq, axis=0), verbose=0)
            responses.append([1 - res[0][0], res[0][0]])
        return np.array(responses)

    print("    [XAI] Generating explanation heatmap... (may take 30-60s)")
    explanation = explainer.explain_instance(
        target_frame.astype('double'),
        predict_wrapper,
        top_labels=1,
        hide_color=0,
        num_samples=100
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    explanation_img = mark_boundaries(temp, mask)
    plt.figure(figsize=(8, 8))
    plt.imshow(explanation_img)
    plt.axis('off')
    plt.title(f"XAI: {video_name}")

    os.makedirs(OUTPUT_XAI_FOLDER, exist_ok=True)
    save_path = os.path.join(OUTPUT_XAI_FOLDER, f"xai_{video_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"    [XAI] Saved to: {save_path}")
    return explanation_img


def test_single_video(video_path, model):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*70}")
    print(f"[V2] Testing: {os.path.basename(video_path)}")
    print(f"{'='*70}")

    # Step 1: Preprocess
    print("[1/3] Preprocessing (DNN + BG Removal)...")
    raw_frames, extracted_count = preprocess_video_v2(video_path)
    if raw_frames is None:
        print(f"[ERROR] Preprocessing failed.")
        return
    print(f"      -> Extracted {extracted_count} face frames from video.")

    # Step 2: Ensemble prediction
    print("[2/3] Running ensemble prediction...")
    avg_pred, all_preds, all_labels = ensemble_predict(model, raw_frames)

    if avg_pred is None:
        print("[ERROR] Prediction failed.")
        return

    label = "FAKE" if avg_pred > DECISION_THRESHOLD else "REAL"
    confidence = avg_pred if label == "FAKE" else 1 - avg_pred
    std_dev = float(np.std(all_preds))

    print(f"\n{'='*50}")
    print(f" FINAL VERDICT : {label}")
    print(f" AVG SCORE     : {avg_pred:.4f}  (closer to 1 = Fake)")
    print(f" CONFIDENCE    : {confidence * 100:.2f}%")
    print(f" STD DEV       : {std_dev:.4f}  (lower = model is more certain)")
    print(f" WINDOWS USED  : {len(all_preds)}")
    print(f"{'='*50}")

    # Step 3: XAI
    print("[3/3] Generating XAI explanation...")
    full_seq = pad_frames([f.astype(np.float32) / 255.0 for f in raw_frames], NUM_FRAMES)
    explain_prediction_v2(model, np.expand_dims(full_seq, axis=0), video_name)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Train first using: python main_version2.py")
        sys.exit(1)

    print("[V2] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[V2] Model loaded.\n")

    if not os.path.exists(TEST_FOLDER):
        print(f"[ERROR] Test folder not found: {TEST_FOLDER}")
        sys.exit(1)

    video_files = [f for f in os.listdir(TEST_FOLDER)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if len(video_files) == 0:
        print(f"[WARNING] No videos found in '{TEST_FOLDER}'")
        sys.exit(0)

    print(f"[V2] Found {len(video_files)} video(s). Starting ensemble testing...\n")

    for video_name in video_files:
        test_single_video(os.path.join(TEST_FOLDER, video_name), model)

    print(f"\n[V2] Done! XAI images saved in: {OUTPUT_XAI_FOLDER}")
