"""
Model Manifest Generator
========================
Records the exact provenance of every trained checkpoint used by the
application: file hash, size, architecture, input shape, class mapping
and the decision threshold the deployment uses.

Run from the repo root:
    python evaluation/model_manifest.py --models-dir models_store --out evaluation/model_manifest.json
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DECISION_THRESHOLD = 0.75
MODEL_SOURCE = "https://huggingface.co/spaces/MuhammadTalha121/Deep-Fake-Detection"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_model(path: Path) -> dict:
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(str(path), compile=False)
    input_shape = [None if d is None else int(d) for d in model.input_shape]
    return {
        "file": path.name,
        "sha256": sha256_of(path),
        "size_bytes": path.stat().st_size,
        "input_shape": input_shape,
        "num_frames": int(input_shape[1]) if input_shape and input_shape[1] else None,
        "frame_size": [int(input_shape[2]), int(input_shape[3])] if len(input_shape) >= 4 else None,
        "output_shape": [None if d is None else int(d) for d in model.output_shape],
        "output_activation": "sigmoid",
        "total_params": int(model.count_params()),
        "architecture": "TimeDistributed(Xception imagenet) -> TimeDistributed(GAP) -> LSTM(64) -> Dense(1, sigmoid)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the model manifest")
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models_store"))
    parser.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "model_manifest.json"))
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    entries = []
    for name in ("deepfake_detector_model_v2.keras", "deepfake_detector_model.keras"):
        path = models_dir / name
        if not path.is_file():
            print(f"[manifest] WARNING: {path} not found, skipping.")
            continue
        print(f"[manifest] Inspecting {name} ...")
        entry = describe_model(path)
        entry["role"] = "preferred (32-frame, ensemble at inference)" if "v2" in name else "fallback (10-frame)"
        entry["source"] = MODEL_SOURCE
        entries.append(entry)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_mapping": {"0": "REAL", "1": "FAKE"},
        "score_semantics": "sigmoid output = P(FAKE); higher score = more fake",
        "decision_threshold": DECISION_THRESHOLD,
        "decision_rule": "FAKE if mean score > threshold else REAL; no INCONCLUSIVE band in current deployment",
        "normalization": "frames scaled to [0, 1] (no ImageNet mean subtraction)",
        "models": entries,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] Written: {out_path}")


if __name__ == "__main__":
    main()
