"""
Frame-Based Evaluation
======================
Evaluates a trained checkpoint on the preprocessed frame folders
(processed_dataset/<split>/<REAL|FAKE>/<video>/*.jpg) using EXACTLY the
same sequence construction the training generators used:

  - mode "v1"  (matches main.py):   first 10 sorted frames, video skipped
                                    if fewer than 10 frames exist
  - mode "v2"  (matches main_version2.py): up to 32 sorted frames,
                                    last frame repeated to reach 32,
                                    video skipped if fewer than 5 frames

Run from the repo root, e.g.:
    python evaluation/evaluate_frames.py \
        --model models_store/deepfake_detector_model_v2.keras \
        --mode v2 --split val --n-real 73 --n-fake 146 \
        --data-dir "F:/Computer Vision/FYP/processed_dataset" \
        --out evaluation/results_v2_val.json
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DECISION_THRESHOLDS = [0.5, 0.75]
LABEL_OF = {0: "REAL", 1: "FAKE"}


def build_sequence(frame_files, mode):
    """Replicates the training generators' sequence construction. Returns (array or None, reason)."""
    frames = []
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append((img / 255.0).astype(np.float32))

    if mode == "v1":
        frames = frames[:10]
        if len(frames) < 10:
            return None, f"only {len(frames)}/10 frames"
        return np.stack(frames), None

    if mode == "v2":
        if len(frames) < 5:
            return None, f"only {len(frames)} frames (<5)"
        frames = frames[:32]
        while len(frames) < 32:
            frames.append(frames[-1].copy())
        return np.stack(frames), None

    raise ValueError(f"Unknown mode: {mode}")


def collect_videos(data_dir, split):
    """Returns [(folder, label)] for every video folder in the split."""
    entries = []
    for label, category in ((0, "REAL"), (1, "FAKE")):
        category_dir = Path(data_dir) / split / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {category_dir}")
        for folder in sorted(category_dir.iterdir()):
            if folder.is_dir():
                entries.append((folder, label))
    return entries


def sample_entries(entries, n_real, n_fake, seed):
    rng = random.Random(seed)
    by_class = {0: [e for e in entries if e[1] == 0], 1: [e for e in entries if e[1] == 1]}
    picked = []
    for label in (0, 1):
        pool = sorted(by_class[label])  # deterministic order before sampling
        requested = n_real if label == 0 else n_fake
        if requested and requested < len(pool):
            picked.extend(rng.sample(pool, requested))
        else:
            picked.extend(pool)
    rng.shuffle(picked)
    return picked


def predict_scores_streaming(model, entries, mode, batch_size, log_every=25):
    """Memory-safe path: build each sequence on demand and never hold the whole
    split in RAM (219 x 32-frame sequences would need ~7.5 GB)."""
    scores, labels, evaluated, skipped = [], [], [], []
    t0 = time.time()
    pending_seqs, pending_labels, pending_folders = [], [], []

    def flush():
        if not pending_seqs:
            return
        try:
            batch = np.stack(pending_seqs)
            preds = model.predict(batch, verbose=0).reshape(-1)
        except MemoryError:
            preds = np.concatenate([
                model.predict(np.expand_dims(seq, axis=0), verbose=0).reshape(-1)
                for seq in pending_seqs
            ])
        scores.extend(float(s) for s in preds)
        labels.extend(pending_labels)
        evaluated.extend(pending_folders)
        pending_seqs.clear()
        pending_labels.clear()
        pending_folders.clear()

    for i, (folder, label) in enumerate(entries, 1):
        frame_files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".jpg")
        seq, reason = build_sequence(frame_files, mode)
        if seq is None:
            skipped.append({"video": folder.name, "reason": reason})
            continue
        pending_seqs.append(seq)
        pending_labels.append(label)
        pending_folders.append(folder)
        if len(pending_seqs) >= max(1, batch_size):
            flush()
        if i % log_every == 0 or i == len(entries):
            print(f"[eval] Processed {i}/{len(entries)} videos ({time.time() - t0:.0f}s)", flush=True)
    flush()
    return scores, labels, evaluated, skipped


def metrics_at_threshold(y_true, scores, threshold):
    y_pred = (np.asarray(scores) > threshold).astype(int)
    y_true_arr = np.asarray(y_true)
    tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true_arr == 0)))
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on preprocessed frame folders")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["v1", "v2"], required=True,
                        help="v1: first 10 frames; v2: up to 32 frames last-frame-padded")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "processed_dataset"))
    parser.add_argument("--n-real", type=int, default=0, help="Sample size for REAL (0 = all)")
    parser.add_argument("--n-fake", type=int, default=0, help="Sample size for FAKE (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag", default="", help="Label recorded in the results file")
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model, compile=False)
    print(f"[eval] Loaded {args.model} (input {model.input_shape})")

    entries = collect_videos(args.data_dir, args.split)
    picked = sample_entries(entries, args.n_real, args.n_fake, args.seed)
    counts = {0: sum(1 for _, l in picked if l == 0), 1: sum(1 for _, l in picked if l == 1)}
    print(f"[eval] Split={args.split}: evaluating {len(picked)} videos (REAL={counts[0]}, FAKE={counts[1]})")

    print(f"[eval] Running streaming inference on {len(picked)} videos ...")
    scores, labels, evaluated_entries, skipped = predict_scores_streaming(
        model, picked, args.mode, args.batch_size
    )
    print(f"[eval] Inference done ({len(scores)} scored, {len(skipped)} skipped)")

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    except Exception:
        auc = None

    results = {
        "tag": args.tag or f"{Path(args.model).stem}_{args.mode}_{args.split}",
        "model": Path(args.model).name,
        "mode": args.mode,
        "split": args.split,
        "data_dir": str(args.data_dir),
        "sample": {"real": counts[0], "fake": counts[1], "seed": args.seed,
                   "n_real_requested": args.n_real, "n_fake_requested": args.n_fake},
        "skipped": skipped,
        "evaluated": len(scores),
        "roc_auc": auc,
        "score_stats": {
            "real_mean": float(np.mean([s for s, l in zip(scores, labels) if l == 0])) if counts[0] else None,
            "real_std": float(np.std([s for s, l in zip(scores, labels) if l == 0])) if counts[0] else None,
            "fake_mean": float(np.mean([s for s, l in zip(scores, labels) if l == 1])) if counts[1] else None,
            "fake_std": float(np.std([s for s, l in zip(scores, labels) if l == 1])) if counts[1] else None,
        },
        "metrics": [metrics_at_threshold(labels, scores, t) for t in DECISION_THRESHOLDS],
        "per_video": [
            {"video": folder.name, "label": LABEL_OF[label], "score": score}
            for folder, label, score in zip(evaluated_entries, labels, scores)
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out) if args.out else REPO_ROOT / "evaluation" / f"results_{results['tag']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[eval] === {results['tag']} ===")
    if auc is not None:
        print(f"[eval] ROC-AUC: {auc:.4f}")
    for m in results["metrics"]:
        cm = m["confusion_matrix"]
        print(f"[eval] thr={m['threshold']:.2f} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} "
              f"(tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']})")
    print(f"[eval] Saved: {out}")


if __name__ == "__main__":
    main()
