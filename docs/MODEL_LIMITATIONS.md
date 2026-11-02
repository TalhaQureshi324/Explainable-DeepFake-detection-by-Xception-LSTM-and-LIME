# Model Limitations — read before trusting any output

**Status: research prototype.** The current checkpoints have documented,
measured weaknesses. The application displays genuine model output, but that
output must not be presented — by the UI, reports, or documentation — as
forensic evidence.

## Measured performance (validation split, Phase 0 + threshold study)

| Model / path | ROC-AUC | Notes |
|--------------|---------|-------|
| V1 full-sequence (n=216) | 0.670 | overfits: train AUC 0.819 |
| V2 full-sequence (n=219) | 0.603 | underfits: train AUC 0.643 |
| V2 deployed ensemble (n=24 sample) | 0.819 | better than full-sequence; small sample — needs confirmation |

At the historical 0.75 threshold V1 missed 84.7% of fakes and V2 missed 99.3%
on the full-sequence eval. The 0.60 development threshold on the ensemble path
recovers recall (0.92) at the cost of false alarms (FPR 0.50) on the same small
sample. Full numbers: `evaluation/EVALUATION.md`, `evaluation/THRESHOLD_ANALYSIS.md`.

## Why the models are weak (evidence-ranked)

1. **Both checkpoints were trained on 67 videos** (54 train / 13 val) — the
   `preprocessed_new` folder they consumed is a tiny subset of the full
   dataset. This is the dominant cause. (See `training/TRAINING_NOTES.md`.)
2. Xception received `[0,1]` inputs against its ImageNet `[-1,1]`
   expectation; measured on pretrained features (50% top-1 flips).
3. Split leakage: identities/original-manipulated pairs crossed splits, so
   measured val metrics are optimistic.
4. Optimization settings (lr=1e-5, batch 1, 10 epochs, checkpointing on a
   13-video val_loss).
5. Face-detector variability across OpenCV builds shifts scores (±0.49
   observed) — reproducibility limit, not fixable by retraining alone.

## Standing limitations of any release using these checkpoints

- Scores are **uncalibrated sigmoid outputs**; do not call them probabilities
  or "confidence percentages".
- Trained only on the DFD subset of FaceForensics++; unknown performance on
  other manipulation methods or generators.
- Single-face analysis only; multi-person media is not fully covered.
- LIME explanations describe a replicated static frame; they do not explain
  the temporal pathway and do not prove which pixels were manipulated.
- Image analysis reuses the video model (replicated-frame compatibility mode);
  it is not a validated image classifier.
- The corrected retraining pipeline (`training/`) exists precisely to replace
  these checkpoints; until a retrained model passes the LOCKED test split,
  keep a research/prototype disclaimer visible in the UI.
