# Leakage-Controlled Split Summary

- Source data root: `F:/Computer Vision/FYP/data`
- Grouping mode: **scene**
- Videos: 3431 (REAL=363, FAKE=3068)
- Source groups: 16
- Seed: 42 (deterministic for the same input)

| Split | REAL | FAKE | Total | Share |
|-------|------|------|-------|-------|
| train | 256 | 2163 | 2419 | 70.5% |
| val | 50 | 418 | 468 | 13.6% |
| calibration | 16 | 164 | 180 | 5.2% |
| test | 41 | 323 | 364 | 10.6% |

## Leakage properties (scene mode)

- Prevented: original/manipulated pairs across splits — every video sharing a
  scene name (the original plus ALL manipulated derivatives of it) is assigned
  to one split atomically, regardless of which actor convention the dataset uses.
- Remaining limitation: face identities are NOT disjoint across splits (the same
  actor appears in multiple scenes); run `--mode actor` for strict
  identity-disjointness and accept far fewer, larger groups.

- The test split is LOCKED: use it only for final reporting, never for threshold
  selection or hyperparameter tuning (calibration split is for threshold calibration).
