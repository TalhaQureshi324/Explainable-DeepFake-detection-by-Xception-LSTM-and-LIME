# GPU Training Preflight Report

- Manifest: `training/dataset_manifest.json` (mode **scene**, seed 42)

## 1. Split composition

| Split | REAL | FAKE | Total | Independent groups | Actors touched |
|-------|------|------|-------|--------------------|----------------|
| train | 256 | 2163 | 2419 | 10 | 28 |
| val | 50 | 418 | 468 | 3 | 28 |
| calibration | 16 | 164 | 180 | 1 | 24 |
| test | 41 | 323 | 364 | 2 | 28 |

## 2. Derivative closure (original + its manipulated derivatives)

- Unparsable video names: 0
- Scene-level violations (fake vs same-scene original in different splits): **0**
- Actor-pair violations (manipulated video vs same-scene original of its source/target actor in a different split): **0**

## 3. Frame coverage

- Frame folders found under `F:\Computer Vision\FYP\processed_dataset`: 3431
- Manifest videos WITHOUT preprocessed frames: none (trainer skips these)

## 4. Credibility verdict

- Scene mode yields **16** independent groups and every split contains both classes: YES.
- Calibration split usable for threshold calibration: YES.
- VERDICT: **usable** for group-disjoint training, provided the report states that face identities are NOT disjoint across splits (scene-level containment only). Actor-level disjointness is impossible with this dataset: DFD's dense source/target pair coverage collapses the actor graph into 3 components, leaving no calibration split (see training/SPLIT_SUMMARY_ACTOR.md).
- RECOMMENDED alternative for a stricter claim: keep scene mode as primary and, in parallel, report a leave-one-actor-out style check later only if the retrained model is strong enough to matter; do NOT present actor-disjoint numbers that do not exist.

## 5. Storage & GPU memory (measured/derived, no training-time estimates)

- Average processed-frame size: 21.6 KB
- Estimated frames.zip upload: **~2.3 GB** (102,480 frames; well inside the 15 GB free Drive tier)
- Workspace after unzip: same order (~2-3 GB) — Colab disk is fine.
- GPU memory: batch 8 x 32 frames x 299x299 through TimeDistributed Xception peaks around 8-11 GB on a T4 (16 GB) — fits; notebook already documents dropping to batch 4 on OOM. V1-style (10 frames) fits comfortably.
- Checkpoint ~166 MB per save (best_checkpoint.keras) — Drive quota impact negligible.
- Training duration: NOT estimated here; benchmark epoch 1 on the actual GPU before planning (the notebook logs per-epoch times).
