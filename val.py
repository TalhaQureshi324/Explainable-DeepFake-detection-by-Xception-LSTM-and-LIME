"""
Validation script for the V1 (10-frame) model.

Reports loss/accuracy plus precision/recall/F1 and ROC-AUC on the
validation split. Metrics are reported at BOTH the model-native 0.5
threshold and the deployment threshold (0.75) used by app.py /
test_version2.py, so reported numbers always state which threshold they
refer to.

Usage:
    python val.py [--model deepfake_detector_model.keras] [--data-dir processed_dataset]
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from main import DeepfakeGenerator  # Reuses the exact training-time generator

DEPLOYMENT_THRESHOLD = 0.75


def report_at_threshold(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, len(y_true))
    print(f"  threshold={threshold:.2f}: acc={acc:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} F1={f1:.4f} (tp={tp} fp={fp} fn={fn} tn={tn})"
          + ("   <- deployment threshold" if threshold == DEPLOYMENT_THRESHOLD else ""))


def evaluate_model(model_path, data_dir):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    val_gen = DeepfakeGenerator(data_dir, split='val', batch_size=1)
    if len(val_gen.data) == 0:
        print("Validation data not found! Check your paths.")
        return

    print(f"Evaluating on {len(val_gen.data)} samples...")
    results = model.evaluate(val_gen)
    print(f"\nValidation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy (0.5 threshold): {results[1]*100:.2f}%")

    print("\nCollecting predictions for threshold analysis...")
    scores, y_true = [], []
    for i in range(len(val_gen)):
        X, y = val_gen[i]
        s = float(model.predict(X, verbose=0)[0][0])
        scores.append(s)
        y_true.append(int(y[0]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(val_gen)}")
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    if len(np.unique(y_true)) == 2:
        print(f"ROC-AUC: {roc_auc_score(y_true, scores):.4f}")
    print("\nMetrics by threshold:")
    report_at_threshold(y_true, scores, 0.5)
    report_at_threshold(y_true, scores, DEPLOYMENT_THRESHOLD)

    print("\n--- Sample Predictions (first 5) ---")
    for i in range(min(5, len(scores))):
        label_true = "FAKE" if y_true[i] == 1 else "REAL"
        label_pred = "FAKE" if scores[i] > DEPLOYMENT_THRESHOLD else "REAL"
        confidence = scores[i] if label_pred == "FAKE" else 1 - scores[i]
        print(f"Sample {i+1}: Actual: {label_true} | Predicted: {label_pred} "
              f"({confidence*100:.2f}% confidence @ {DEPLOYMENT_THRESHOLD} threshold)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a checkpoint on the validation split")
    parser.add_argument("--model", default="deepfake_detector_model.keras")
    parser.add_argument("--data-dir", default=os.path.join(os.getcwd(), "processed_dataset"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if os.path.exists(args.model):
        evaluate_model(args.model, args.data_dir)
    else:
        print(f"Model file {args.model} not found. Did you save it?")
