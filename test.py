import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2

# Import the EXACT same FaceExtractor and config from preprocessing.py
from preprocessing import FaceExtractor, DEFAULT_CONFIG

# --- PATHS ---
MODEL_PATH = os.path.join(os.getcwd(), "deepfake_detector_model.keras")
TEST_FOLDER = os.path.join(os.getcwd(), "testing")
OUTPUT_XAI_FOLDER = os.path.join(os.getcwd(), "xai_outputs")
TEMP_FOLDER = os.path.join(os.getcwd(), "temp_runtime_frames")


def preprocess_video_runtime(video_path):
    """
    Preprocess a video EXACTLY like preprocessing.py does for training:
      - OpenCV DNN face detector (same blob params)
      - 20% margin around detected face
      - Resize to 299x299
      - Elliptical mask background removal -> black outside face
      - Extract 32 evenly spaced frames, use first 10 (matches main.py)
      - Normalize to [0, 1]
    Returns: (1, 10, 299, 299, 3) float32 array or None
    """
    config = DEFAULT_CONFIG.copy()

    # Safety check: DNN model files must exist
    if not os.path.exists(config["prototxt_path"]):
        raise FileNotFoundError(
            f"[ERROR] Prototxt not found: {config['prototxt_path']}\n"
            f"        Download OpenCV DNN face detector or update path in preprocessing.py"
        )
    if not os.path.exists(config["caffemodel_path"]):
        raise FileNotFoundError(
            f"[ERROR] CaffeModel not found: {config['caffemodel_path']}\n"
            f"        Download OpenCV DNN face detector or update path in preprocessing.py"
        )

    extractor = FaceExtractor(config)

    # Temporary folder for this video (same logic as process_single_video)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = os.path.join(TEMP_FOLDER, video_stem)
    os.makedirs(temp_dir, exist_ok=True)

    # Extract faces with SAME default as training (32 frames)
    count = extractor.process_video(video_path, temp_dir, max_frames=32)

    if count == 0:
        print(f"[ERROR] No faces detected in: {os.path.basename(video_path)}")
        return None

    # Load saved frames (BGR on disk -> RGB in memory)
    frames = []
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])

    for frame_file in frame_files:
        frame_path = os.path.join(temp_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    if len(frames) == 0:
        return None

    # main.py generator uses first 10 frames
    frames = frames[:10]

    if len(frames) < 10:
        print(f"[WARNING] Only {len(frames)}/10 valid face frames found. "
              f"Video may be too short or missing faces in early frames.")
        return None

    # Normalize to [0, 1] just like main.py generator
    frames = np.array(frames, dtype=np.float32) / 255.0
    return np.expand_dims(frames, axis=0)  # shape: (1, 10, 299, 299, 3)


def explain_prediction(model, processed_video, video_name):
    """Generates LIME heatmap for the middle frame (index 5)."""
    explainer = lime_image.LimeImageExplainer()
    target_frame = processed_video[0][5]

    def predict_wrapper(images):
        responses = []
        for img in images:
            # Replicate 2D image 10 times to feed temporal model
            seq = np.stack([img] * 10, axis=0)
            res = model.predict(np.expand_dims(seq, axis=0), verbose=0)
            responses.append([1 - res[0][0], res[0][0]])
        return np.array(responses)

    print("[XAI] Generating explanation heatmap. This may take 30-60 seconds...")

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
    plt.title(f"XAI Explanation: {video_name}")

    os.makedirs(OUTPUT_XAI_FOLDER, exist_ok=True)
    save_path = os.path.join(OUTPUT_XAI_FOLDER, f"xai_{video_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[XAI] Explanation saved to: {save_path}")
    return explanation_img


def predict_and_explain(video_path, model):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {os.path.basename(video_path)}")
    print(f"[INFO] Step 1/3: Preprocessing (DNN Face Detect + BG Removal)...")

    processed_video = preprocess_video_runtime(video_path)

    if processed_video is None:
        print(f"[ERROR] Preprocessing failed for: {os.path.basename(video_path)}")
        return

    print(f"[INFO] Step 2/3: Running model prediction...")
    prediction = model.predict(processed_video, verbose=0)[0][0]

    label = "FAKE" if prediction > 0.75 else "REAL"
    confidence = prediction if label == "FAKE" else 1 - prediction

    print(f"\n{'='*40}")
    print(f" VIDEO        : {os.path.basename(video_path)}")
    print(f" FINAL VERDICT: {label}")
    print(f" CONFIDENCE   : {confidence * 100:.2f}%")
    print(f" RAW SCORE    : {prediction:.4f}  (closer to 1 = Fake, closer to 0 = Real)")
    print(f"{'='*40}")

    print(f"[INFO] Step 3/3: Generating XAI explanation...")
    explain_prediction(model, processed_video, video_name)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[INFO] Loading AI Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.\n")

    if not os.path.exists(TEST_FOLDER):
        print(f"[ERROR] Test folder not found: {TEST_FOLDER}")
        sys.exit(1)

    video_files = [f for f in os.listdir(TEST_FOLDER)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if len(video_files) == 0:
        print(f"[WARNING] No videos found in '{TEST_FOLDER}'")
        sys.exit(0)

    print(f"[INFO] Found {len(video_files)} video(s) in '{TEST_FOLDER}'. Starting...\n")

    for video_name in video_files:
        predict_and_explain(os.path.join(TEST_FOLDER, video_name), model)

    print(f"\n[INFO] All done! XAI images saved in: {OUTPUT_XAI_FOLDER}")
