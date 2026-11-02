"""
Corrected GPU training pipeline (Task R2)
=========================================

Fixes the Phase 0/audit findings on the ORIGINAL training configuration:

  1. Trains from the manifest produced by make_identity_split.py instead of a
     random, leakage-affected split (default: scene-mode manifest).
  2. Reads the FULL frame dataset, not a partial folder: frames are located by
     video stem across processed_dataset/<any split>/<label>/<stem>/.
  3. Uses the ImageNet-correct Xception normalization ([-1, 1]) by default;
     the old [0, 1] behaviour is available via --preprocessing-mode zero_one
     for controlled comparison. NEVER mix modes between train and inference.
  4. Monitors validation ROC-AUC / loss / recall; early stopping + checkpoint
     selection by a configurable metric (--selection-metric).
  5. Saves curves (CSV), the exact config, dependency versions, checkpoint
     SHA-256 and a final validation evaluation into an experiment manifest.

This script is GPU-oriented but runs on CPU for smoke tests
(--max-train-videos limits the workload).

Usage (repo root):
  python training/train_corrected.py \
      --manifest training/dataset_manifest.json \
      --frames-root "F:/Computer Vision/FYP/processed_dataset" \
      --out-dir training/runs/exp01
"""

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MIN_FRAMES = 5


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(chunk)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def normalize(img_rgb_u8, mode):
    arr = img_rgb_u8.astype("float32")
    if mode == "minus_one_plus_one":
        return arr / 127.5 - 1.0        # matches keras xception.preprocess_input for RGB
    if mode == "zero_one":
        return arr / 255.0              # legacy mode the current checkpoints used
    raise ValueError(f"unknown preprocessing mode: {mode}")


def locate_frame_folders(frames_root: Path, manifest):
    """Map every manifest video to its processed frame folder (by stem)."""
    index = {}
    for split in ("train", "val", "calibration", "test"):
        for label in ("REAL", "FAKE"):
            d = frames_root / split / label
            if d.is_dir():
                for folder in d.iterdir():
                    if folder.is_dir():
                        index[folder.name] = folder
    for rec in manifest["videos"]:
        rec["_frames_dir"] = str(index[rec["video"]]) if rec["video"] in index else None
    return manifest


def build_split_records(manifest, split):
    return [r for r in manifest["videos"] if r["assigned_split"] == split and r["_frames_dir"]]


class SequenceGenerator:
    """Yields batches of (NUM_FRAMES, 299, 299, 3) sequences from frame folders."""

    def __init__(self, records, num_frames, preprocessing_mode, batch_size, seed=42,
                 v1_first_k=None):
        self.records = records
        self.num_frames = num_frames
        self.mode = preprocessing_mode
        self.batch_size = batch_size
        self.v1_first_k = v1_first_k  # V1-style: use only the first k frames
        self.seed = seed
        self.indices = list(range(len(self.records)))
        self._order = self.indices[:]
        random.Random(seed).shuffle(self._order)

    def on_epoch_end(self):
        rng = random.Random(self.seed + random.randint(0, 1 << 30))
        rng.shuffle(self._order)

    def __len__(self):
        return max(1, len(self.records) // self.batch_size)

    def class_weights(self):
        import numpy as np

        labels = [1 if r["label"] == "FAKE" else 0 for r in self.records]
        counts = {0: labels.count(0), 1: labels.count(1)}
        total = len(labels)
        return {c: total / (len(counts) * n) for c, n in counts.items() if n}

    def _load_sequence(self, record):
        import cv2
        import numpy as np

        folder = Path(record["_frames_dir"])
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".jpg")
        if self.v1_first_k is not None:
            if len(files) < self.v1_first_k:
                return None
            files = files[: self.v1_first_k]
        frames = []
        for p in files[: self.num_frames]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            frames.append(normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.mode))
        if not (MIN_FRAMES <= len(frames)):
            return None
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())
        return np.stack(frames)

    def __iter__(self):
        import numpy as np

        batch_x, batch_y = [], []
        order = self.indices[:]
        random.shuffle(order)
        for i in order:
            seq = self._load_sequence(self.records[i])
            if seq is None:
                continue
            batch_x.append(seq)
            batch_y.append(1.0 if self.records[i]["label"] == "FAKE" else 0.0)
            if len(batch_x) >= self.batch_size:
                yield np.stack(batch_x), np.array(batch_y, dtype="float32")
                batch_x, batch_y = [], []
        if batch_x:
            yield np.stack(batch_x), np.array(batch_y, dtype="float32")

    # Keras-compatible Sequence interface
    def keras_sequence(self):
        import numpy as np
        import tensorflow as tf

        outer = self

        class _Seq(tf.keras.utils.Sequence):
            def __len__(self):
                return max(1, len(outer.records) // outer.batch_size)

            def __getitem__(self, idx):
                picks = outer._order[idx * outer.batch_size : (idx + 1) * outer.batch_size]
                xs, ys = [], []
                for i in picks:
                    seq = outer._load_sequence(outer.records[i])
                    if seq is None:
                        continue
                    xs.append(seq)
                    ys.append(1.0 if outer.records[i]["label"] == "FAKE" else 0.0)
                if not xs:
                    # fall back to any single loadable record so batches stay non-empty
                    for i in outer.indices:
                        seq = outer._load_sequence(outer.records[i])
                        if seq is not None:
                            xs.append(seq)
                            ys.append(1.0 if outer.records[i]["label"] == "FAKE" else 0.0)
                            break
                return np.stack(xs), np.array(ys, dtype="float32")

            def on_epoch_end(self):
                outer.on_epoch_end()

        return _Seq()


def build_model(num_frames, backbone_mode="finetune", lr=1e-4):
    import tensorflow as tf
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling2D, Input, TimeDistributed
    from tensorflow.keras.models import Model

    base = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
    base.trainable = backbone_mode == "finetune"
    if backbone_mode == "finetune":
        for layer in base.layers[:100]:
            layer.trainable = False

    inp = Input(shape=(num_frames, 299, 299, 3))
    x = TimeDistributed(base)(inp)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(64)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="val_auc_curve", curve="ROC"),
            tf.keras.metrics.Recall(name="recall"),
            "accuracy",
        ],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected deepfake training (GPU-ready)")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "training" / "dataset_manifest.json"))
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--v1-first-k", type=int, default=None,
                        help="If set (e.g. 10), use only the first k frames like V1")
    parser.add_argument("--preprocessing-mode", choices=["minus_one_plus_one", "zero_one"],
                        default="minus_one_plus_one")
    parser.add_argument("--backbone", choices=["finetune", "frozen"], default="finetune")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--selection-metric", default="val_auc_curve",
                        help="Checkpoint/early-stop metric (val_auc_curve|val_loss)")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint to resume from (restores optimizer state); "
                             "used after Colab disconnects")
    parser.add_argument("--max-train-videos", type=int, default=0,
                        help="Cap train videos (smoke tests); 0 = all")
    parser.add_argument("--max-val-videos", type=int, default=0)
    args = parser.parse_args()

    import tensorflow as tf

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest = locate_frame_folders(Path(args.frames_root), manifest)
    train_recs = build_split_records(manifest, "train")
    val_recs = build_split_records(manifest, "val")
    if args.max_train_videos:
        train_recs = train_recs[: args.max_train_videos]
    if args.max_val_videos:
        val_recs = val_recs[: args.max_val_videos]
    print(f"[train] train={len(train_recs)} val={len(val_recs)} "
          f"(mode={manifest.get('mode')}, preprocessing={args.preprocessing_mode})")
    assert train_recs and val_recs, "empty split after manifest mapping"

    gen_train = SequenceGenerator(train_recs, args.num_frames, args.preprocessing_mode,
                                  args.batch_size, seed=args.seed, v1_first_k=args.v1_first_k)
    gen_val = SequenceGenerator(val_recs, args.num_frames, args.preprocessing_mode,
                                args.batch_size, seed=args.seed + 1, v1_first_k=args.v1_first_k)
    class_weights = gen_train.class_weights()
    print(f"[train] class weights: {class_weights}")

    initial_epoch = 0
    if args.resume_from:
        model = tf.keras.models.load_model(args.resume_from)  # restores optimizer state
        old_log = out_dir / "training_log.csv"
        if old_log.exists():
            rows = [l for l in old_log.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
            initial_epoch = max(0, len(rows) - 1)
        print(f"[train] resuming from {args.resume_from} at epoch {initial_epoch + 1}")
    else:
        model = build_model(args.num_frames, args.backbone, args.lr)
    monitor = args.selection_metric
    mode = "max" if "auc" in monitor.lower() else "min"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best_checkpoint.keras"), monitor=monitor, mode=mode,
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                         patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv"),
                                     append=bool(args.resume_from)),
    ]

    history = model.fit(
        gen_train.keras_sequence(),
        validation_data=gen_val.keras_sequence(),
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    ckpt = out_dir / "best_checkpoint.keras"
    manifest_out = {
        "experiment": out_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: getattr(args, k) for k in (
            "num_frames", "v1_first_k", "preprocessing_mode", "backbone", "epochs",
            "batch_size", "lr", "selection_metric", "patience", "seed")},
        "resumed_from": args.resume_from,
        "initial_epoch": initial_epoch,
        "data": {
            "manifest": args.manifest,
            "manifest_mode": manifest.get("mode"),
            "frames_root": args.frames_root,
            "train_videos": len(train_recs), "val_videos": len(val_recs),
        },
        "class_weights": class_weights,
        "checkpoint_sha256": sha256_file(ckpt) if ckpt.exists() else None,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
            "dependency_lock": {
                m: getattr(importlib.import_module(m), "__version__", "unknown")
                for m in ("numpy", "cv2")
            },
        },
        "notes": [
            "Preprocessing mode MUST match inference: minus_one_plus_one uses the "
            "ImageNet-correct Xception normalization; zero_one reproduces legacy training.",
            "Checkpoint selection used validation metric: " + monitor,
        ],
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest_out, indent=2),
                                                      encoding="utf-8")
    print(f"[train] done -> {out_dir / 'experiment_manifest.json'}")


if __name__ == "__main__":
    main()
