# Leakage-Controlled Split Summary

- Source data root: `F:/Computer Vision/FYP/data`
- Grouping mode: **actor**
- Videos: 3431 (REAL=363, FAKE=3068)
- Source groups: 3
- Seed: 42 (deterministic for the same input)

| Split | REAL | FAKE | Total | Share |
|-------|------|------|-------|-------|
| train | 243 | 2604 | 2847 | 83.0% |
| val | 63 | 271 | 334 | 9.7% |
| calibration | 0 | 0 | 0 | 0.0% |
| test | 57 | 193 | 250 | 7.3% |

## Leakage properties (actor mode)

- Prevented: face-identity and scene-identity leakage (connected components over
  the actor graph; a manipulated video and both of its identities stay together).
- Remaining limitation: dense source/target pair coverage in DFD collapses the
  graph into very few components, so splits are coarse and per-split diversity
  is limited.

- The test split is LOCKED: use it only for final reporting, never for threshold
  selection or hyperparameter tuning (calibration split is for threshold calibration).
