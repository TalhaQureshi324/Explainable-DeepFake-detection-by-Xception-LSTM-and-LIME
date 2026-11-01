# Evaluation Report — Xception-LSTM Deepfake Detection

> Generated as part of **Phase 0 (Baseline Verification & Repository Audit)**.
> All numbers in this file were produced by the scripts in this folder against
> the shipped checkpoints and the `processed_dataset` frame folders
> (FaceForensics++ DFD subset). Raw JSON outputs are committed alongside this
> report so every number is traceable.

---

## 1. What was verified

| Check | Tool | Result |
|-------|------|--------|
| Checkpoint provenance (hash, shape, params) | `model_manifest.py` | ✅ recorded in `model_manifest.json` |
| Deployment app builds and loads model | import test of `app.py` | ✅ model auto-detected (32 frames), Gradio Blocks built |
| End-to-end raw-video inference (V1 + V2) | `evaluate_video_e2e.py` | ✅ runs without errors on real val videos |
| Classification metrics (V1 & V2, val split) | `evaluate_frames.py` | see §3 |
| Train vs val gap (overfitting check) | `evaluate_frames.py` | see §4 |

Environment used for this run: Windows 11, Python 3.11.9, TensorFlow 2.21.0
(CPU only — TF ≥2.11 has no native Windows GPU support), OpenCV 4.13.0,
scikit-learn 1.8.0.

## 2. Checkpoints under test

| Model | Input | Params | SHA-256 |
|-------|-------|--------|---------|
| V2 `deepfake_detector_model_v2.keras` | 32 × 299 × 299 × 3 | 21,402,473 | `9ee14530…` |
| V1 `deepfake_detector_model.keras` | 10 × 299 × 299 × 3 | 21,402,473 | `3c101cbf…` |

Both were downloaded from the live Hugging Face Space and match the files the
Space serves. Class mapping: sigmoid output = P(FAKE), `0 = REAL`, `1 = FAKE`.
Normalization: `[0, 1]`.

## 3. Validation-split metrics

Evaluated on a seeded stratified sample of the val split
(**73 REAL [all] + 146 FAKE**, seed 42), using exactly the training-time
sequence construction (V1: first 10 frames; V2: up to 32 frames, last-frame
padding). Inference here is the training-consistent **full-sequence** score;
the deployment ensemble is covered in §5.

### V1 (`deepfake_detector_model.keras`) — 216 evaluated, 3 skipped (<10 frames)

| Threshold | Acc | Bal. Acc | Precision | Recall | F1 | FPR | FNR | Confusion (tp/fp/fn/tn) |
|-----------|------|----------|-----------|--------|------|------|------|--------------------------|
| 0.5 | 0.676 | 0.590 | 0.718 | 0.847 | 0.777 | **0.667** | 0.153 | 122 / 48 / 22 / 24 |
| 0.75 (deployment) | 0.426 | 0.563 | 0.917 | 0.153 | 0.262 | 0.028 | **0.847** | 22 / 2 / 122 / 70 |

- **ROC-AUC: 0.670** — weak ranking ability.
- Score distributions overlap heavily: REAL videos score 0.553 ± 0.109,
  FAKE videos 0.621 ± 0.118 (mean separation only 0.068).
- Even at the *best possible* threshold (0.6155), balanced accuracy peaks at
  **0.663** — so the weakness is in the model, not just the threshold choice.

### V2 (`deepfake_detector_model_v2.keras`) — 219 evaluated, 0 skipped

| Threshold | Acc | Bal. Acc | Precision | Recall | F1 | FPR | FNR | Confusion (tp/fp/fn/tn) |
|-----------|------|----------|-----------|--------|------|------|------|--------------------------|
| 0.5 | 0.589 | 0.551 | 0.703 | 0.664 | 0.683 | 0.562 | 0.336 | 97 / 41 / 49 / 32 |
| 0.75 (deployment) | 0.338 | 0.503 | 1.000 | 0.007 | 0.014 | 0.000 | **0.993** | 1 / 0 / 145 / 73 |

- **ROC-AUC: 0.603** — worse than V1 on the same sample.
- Score distributions barely separate: REAL 0.500 ± 0.085, FAKE 0.532 ± 0.091.
- **Only 0.46 % of all V2 val scores exceed 0.75.** The deployment threshold
  therefore turns V2 into an almost-constant REAL predictor (145/146 fakes
  missed on this sample). Best achievable balanced accuracy over all possible
  thresholds is 0.593 (thr 0.5353).

## 4. Overfitting assessment

A stratified train-split sample (**40 REAL + 40 FAKE**, seed 42) was scored
with the same pipeline and compared against the val numbers above.

| Model | Split | ROC-AUC | Bal. Acc @0.5 | Best achievable Bal. Acc | Real score μ±σ | Fake score μ±σ |
|-------|-------|---------|---------------|--------------------------|----------------|----------------|
| V1 | train | 0.819 | 0.738 | 0.800 (thr 0.546) | 0.479 ± 0.118 | 0.624 ± 0.105 |
| V1 | val | 0.670 | 0.590 | 0.663 (thr 0.616) | 0.553 ± 0.109 | 0.621 ± 0.118 |
| V2 | train | 0.643 | 0.600 | 0.650 (thr 0.452) | 0.478 ± 0.106 | 0.529 ± 0.084 |
| V2 | val | 0.603 | 0.551 | 0.593 (thr 0.535) | 0.500 ± 0.085 | 0.532 ± 0.091 |

**Findings**

1. **V1 overfits:** train AUC 0.819 vs val AUC 0.670 (−0.15 gap), and REAL
   videos score visibly higher on val (0.553) than on train (0.479). The
   model memorized training identity/appearance cues that do not transfer.
2. **V2 underfits:** it scores barely above chance **on its own training
   data** (train AUC 0.643; REAL 0.478 vs FAKE 0.529 — almost no separation),
   and generalizes even worse (val AUC 0.603). This is not overfitting; the
   32-frame model with LSTM(64) at lr=1e-5 appears to have converged to a
   very weak solution. The "V2 is the preferred/better model" assumption in
   the current deployment is **not supported** by full-sequence metrics.
3. **Deployment-threshold effect:** at 0.75 both models miss the overwhelming
   majority of fakes (V1 val FNR 0.847; V2 val FNR 0.993). V2's score
   distribution is so compressed that only 0.46 % of val scores exceed 0.75,
   so in practice the app almost always answers REAL for V2. The threshold is
   not the root cause (see best-achievable numbers) but it compounds it.
4. **Conclusion:** no *new* overfitting is introduced by any current code
   path, but the existing V1 checkpoint does overfit its training split and
   **neither checkpoint currently meets a defensible detection standard on
   the val split.** Retraining/evaluation improvements are required before
   claiming detection performance (see §7 and the phase plan).

## 5. End-to-end deployment-path check (raw videos)

`evaluate_video_e2e.py` re-runs the *actual deployment pipeline*
(`FaceExtractor` on the raw video → model) on 6 held-out val videos and
compares each score against the same video's preprocessed frame folder.

- **Functionality:** all 6 videos (3 REAL / 3 FAKE) flow through V1 and V2
  without errors; verdicts are produced; no crashes, no unhandled exceptions.
- **Reproducibility caveat (important):** the raw-video path is *not*
  bit-reproducible against frames extracted at dataset-preprocessing time.
  Score deltas of **0.003 – 0.49** were observed on this sample.

### Root cause (verified by pixel-level diagnosis)

For the same video, runtime extraction and preprocessing-time extraction
produce the same number of frames and the same `total_frames`/`fps`
(845 / 24.0 for the probe video), but the face crops differ with a mean
absolute pixel difference of ≈ 24/255 (max ≈ 185). The OpenCV DNN face
detector output shifts slightly because decoded video pixels differ across
OpenCV/FFmpeg builds; a small box shift changes the 299×299 crop and the
elliptical mask, and the model score moves with it. V1 is the most sensitive
because it consumes only the *first 10* extracted frames — a single missed
detection shifts the whole sequence.

### Implications

1. **Same environment ⇒ deterministic results** (video decode and DNN are
   deterministic for a fixed build). Cross-environment scores can drift.
2. The **frames-based metrics in §3 are the canonical performance numbers**
   (they use the same preprocessing as training). The deployment path adds
   this extra variability on top.
3. Mitigations (recommended for Phase 1+): pin exact `opencv-python` versions
   in deployment, hash the DNN detector files in the manifest, and consider
   making V1 skip-tolerant (take the first 10 *successful* detections from a
   larger candidate pool instead of failing when <10 of the first 10 hit).

## 6. Threshold analysis

The deployment threshold is **0.75** (FAKE if mean score > 0.75), chosen to be
conservative against false accusations. `val.py` and `evaluate_frames.py`
report metrics at **both 0.5 and 0.75** so no reported number is ambiguous.
The per-threshold results in §3 quantify exactly what the 0.75 policy costs in
missed fakes (FNR) and what it buys in false-alarm reduction (FPR).

## 7. Known limitations

- **Dataset scope:** models were trained and evaluated on the DeepFake
  Detection (DFD) subset of FaceForensics++ only. Performance on other
  manipulation methods (Face2Face, FaceSwap, NeuralTextures, commercial
  generators) is **unknown** and must not be claimed.
- **Identity leakage:** the train/val split (made by `script.py`) splits
  videos randomly, so identities appear in both splits. Val metrics are
  therefore optimistic relative to a fully held-out identity split. This
  should be addressed in Phase 5 with an identity-disjoint re-split.
- **Single face:** only the most confident face per frame is analyzed.
- **LIME limitation:** explanations replicate the middle frame across the
  whole sequence, so LIME explains a *static-frame* response of the temporal
  model — it does **not** attribute the temporal (LSTM) pathway.
- **Class imbalance:** train split is 290 REAL / 2454 FAKE. Class weights
  compensate during training; metrics above use balanced samples so raw
  accuracy is not inflated by the majority class.

## 8. Reproducing this evaluation

```bash
# 1. Provenance
python evaluation/model_manifest.py --models-dir models_store

# 2. Val metrics (V2 shown; same command with --mode v1 and the V1 model for V1)
python evaluation/evaluate_frames.py \
    --model models_store/deepfake_detector_model_v2.keras --mode v2 \
    --split val --n-real 73 --n-fake 146 \
    --data-dir <path to processed_dataset> --out evaluation/results_v2_val.json

# 3. Train-sample metrics (overfitting check)
python evaluation/evaluate_frames.py \
    --model models_store/deepfake_detector_model_v2.keras --mode v2 \
    --split train --n-real 40 --n-fake 40 \
    --data-dir <path to processed_dataset> --out evaluation/results_v2_train.json

# 4. End-to-end deployment-path check
python evaluation/evaluate_video_e2e.py \
    --videos <path to data> --data-dir <path to processed_dataset> \
    --n-real 3 --n-fake 3
```
