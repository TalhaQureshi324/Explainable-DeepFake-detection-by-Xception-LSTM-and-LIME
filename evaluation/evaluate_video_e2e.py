"""
End-to-End Baseline Check
=========================
Runs the repository's ACTUAL inference path on raw videos:
  video -> FaceExtractor (OpenCV DNN + elliptical mask) -> model

  - V1 mode: first 10 extracted frames -> single full-sequence prediction
    (matches test.py)
  - V2 mode: ensemble prediction (full 32-frame sequence + 10-frame
    sliding windows, stride 5, mean) (matches app.py / test_version2.py)

It then compares each video's score against the same video evaluated from
its preprocessed frame folder (processed_dataset path, used by
evaluate_frames.py), so we can prove the deployment preprocessing path and
the training preprocessing path produce consistent results.

Run from the repo root:
    python evaluation/evaluate_video_e2e.py \
        --videos "F:/Computer Vision/FYP/data/val" \
        --data-dir "F:/Computer Vision/FYP/processed_dataset" \
        --models-dir models_store --split val --n-real 2 --n-fake 2
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

WINDOW_SIZE = 10
STRIDE = 5


def pad_frames(frames, target_length):
    frames = list(frames)
    if len(frames) == 0:
        return None
    while len(frames) < target_length:
        frames.append(frames[-1].copy())
    return np.array(frames, dtype=np.float32)


def ensemble_predict(model, frames_list, num_frames):
    """Same logic as app.py ensemble_predict."""
    normalized = [f.astype(np.float32) / 255.0 for f in frames_list]
    predictions, labels = [], []

    full_seq = pad_frames(normalized, num_frames)
    if full_seq is not None:
        pred = float(model.predict(np.expand_dims(full_seq, axis=0), verbose=0)[0][0])
        predictions.append(pred)
        labels.append(f"Full-{num_frames}")

    n = len(normalized)
    if n >= WINDOW_SIZE:
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            window = normalized[start : start + WINDOW_SIZE]
            padded = pad_frames(window, num_frames)
            if padded is not None:
                pred = float(model.predict(np.expand_dims(padded, axis=0), verbose=0)[0][0])
                predictions.append(pred)
                labels.append(f"Win-{start}-{start + WINDOW_SIZE - 1}")
    else:
        padded = pad_frames(normalized, num_frames)
        if padded is not None:
            pred = float(model.predict(np.expand_dims(padded, axis=0), verbose=0)[0][0])
            predictions.append(pred)
            labels.append(f"Short-{n}")

    return float(np.mean(predictions)), predictions, labels


def extract_frames_from_video(video_path, extractor, num_frames, temp_dir):
    """Extract faces via the repo's FaceExtractor, return list of RGB uint8 frames."""
    from preprocessing import FaceExtractor

    video_stem = Path(video_path).stem
    out_dir = Path(temp_dir) / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    count = extractor.process_video(str(video_path), str(out_dir), max_frames=num_frames)
    if count == 0:
        return None
    frames = []
    for frame_file in sorted(f for f in out_dir.iterdir() if f.suffix == ".jpg"):
        img = cv2.imread(str(frame_file))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames if frames else None


def frames_from_processed_folder(folder, mode):
    """Load the preprocessed frame folder the same way evaluate_frames.py does."""
    frame_files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() == ".jpg")
    frames = []
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if mode == "v1":
        frames = frames[:10]
        return frames if len(frames) >= 10 else None
    if not (5 <= len(frames)):
        return None
    return frames[:32]


def full_sequence_score(model, frames, num_frames):
    normalized = [f.astype(np.float32) / 255.0 for f in frames]
    padded = pad_frames(normalized, num_frames)
    return float(model.predict(np.expand_dims(padded, axis=0), verbose=0)[0][0])


def pick_videos(data_root, processed_dir, split, n_real, n_fake, seed):
    rng = random.Random(seed)
    picked = []
    for label, category in ((0, "REAL"), (1, "FAKE")):
        video_dir = Path(data_root) / split / category
        folder_dir = Path(processed_dir) / split / category
        if not video_dir.is_dir():
            raise FileNotFoundError(video_dir)
        candidates = []
        for video_file in sorted(video_dir.glob("*.mp4")):
            stem = video_file.stem
            pf = folder_dir / stem
            if pf.is_dir() and any(pf.glob("*.jpg")):
                candidates.append((video_file, pf, label))
        requested = n_real if label == 0 else n_fake
        if requested and requested < len(candidates):
            picked.extend(rng.sample(candidates, requested))
        else:
            picked.extend(candidates)
    rng.shuffle(picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end video inference baseline check")
    parser.add_argument("--videos", required=True, help="Folder containing split/REAL|FAKE raw videos")
    parser.add_argument("--data-dir", required=True, help="processed_dataset folder for cross-check")
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models_store"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-real", type=int, default=2)
    parser.add_argument("--n-fake", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "baseline_e2e_results.json"))
    args = parser.parse_args()

    import tensorflow as tf
    from preprocessing import FaceExtractor

    models = {
        "V1": (Path(args.models_dir) / "deepfake_detector_model.keras", 10),
        "V2": (Path(args.models_dir) / "deepfake_detector_model_v2.keras", 32),
    }
    loaded = {}
    for name, (path, num_frames) in models.items():
        if path.is_file():
            loaded[name] = (tf.keras.models.load_model(str(path), compile=False), num_frames)
            print(f"[e2e] Loaded {name}: {path.name}")
        else:
            print(f"[e2e] WARNING: missing {path}")

    extractor = FaceExtractor()
    videos = pick_videos(args.videos, args.data_dir, args.split, args.n_real, args.n_fake, args.seed)
    print(f"[e2e] {len(videos)} videos selected from split '{args.split}'")

    results, all_ok = [], True
    with __import__("tempfile").TemporaryDirectory() as temp_dir:
        for video_path, processed_folder, label in videos:
            entry = {
                "video": Path(video_path).name,
                "true_label": "REAL" if label == 0 else "FAKE",
            }
            print(f"\n[e2e] === {entry['video']} ({entry['true_label']}) ===")
            for name, (model, num_frames) in loaded.items():
                mode = "v1" if name == "V1" else "v2"
                try:
                    t0 = time.time()
                    # NOTE: both models extract up to 32 evenly spaced candidate
                    # frames (exactly like test.py / test_version2.py). V1 then
                    # consumes only the FIRST 10 extracted frames.
                    frames = extract_frames_from_video(video_path, extractor, 32, temp_dir)
                    if frames is None:
                        entry[name] = {"status": "no_faces"}
                        all_ok = False
                        continue

                    if name == "V1":
                        frames_model = frames[:10]
                        if len(frames_model) < 10:
                            entry[name] = {"status": "insufficient_frames", "extracted": len(frames)}
                            continue
                        score = full_sequence_score(model, frames_model, num_frames)
                        detail = {"windows": 1, "frames_extracted": len(frames), "frames_used": 10}
                    else:
                        score, window_scores, window_labels = ensemble_predict(model, frames, num_frames)
                        detail = {"windows": len(window_scores),
                                  "window_scores": [round(s, 4) for s in window_scores],
                                  "window_labels": window_labels,
                                  "frames_extracted": len(frames), "frames_used": len(frames)}

                    pfolder_frames = frames_from_processed_folder(processed_folder, mode)
                    reference_score = None
                    if pfolder_frames is not None:
                        if mode == "v1":
                            reference_score = full_sequence_score(model, pfolder_frames, num_frames)
                        else:
                            reference_score, _, _ = ensemble_predict(model, pfolder_frames, num_frames)

                    delta = abs(score - reference_score) if reference_score is not None else None
                    verdict = "FAKE" if score > 0.75 else "REAL"
                    entry[name] = {
                        "status": "ok",
                        "e2e_score": round(score, 6),
                        "e2e_verdict@0.75": verdict,
                        "reference_score_from_processed_frames": round(reference_score, 6) if reference_score is not None else None,
                        "abs_score_delta": round(delta, 6) if delta is not None else None,
                        "seconds": round(time.time() - t0, 1),
                        **detail,
                    }
                    print(f"[e2e]   {name}: score={score:.4f} -> {verdict} "
                          f"(reference={reference_score if reference_score is not None else 'n/a'}, "
                          f"delta={delta if delta is not None else 'n/a'})")
                except Exception as exc:  # noqa: BLE001 - record and continue
                    entry[name] = {"status": "error", "error": str(exc), "trace": traceback.format_exc()[-2000:]}
                    all_ok = False
                    print(f"[e2e]   {name}: ERROR {exc}")
            results.append(entry)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "decision_threshold": 0.75,
        "videos": results,
        "all_passed": all_ok and all(
            step.get("status") == "ok" for r in results for k, step in r.items() if k in ("V1", "V2")
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[e2e] Saved: {out}")
    print(f"[e2e] all_passed={summary['all_passed']}")


if __name__ == "__main__":
    main()
