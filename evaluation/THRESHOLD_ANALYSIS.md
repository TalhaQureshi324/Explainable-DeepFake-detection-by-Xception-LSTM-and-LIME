# Threshold Analysis (existing checkpoints)

Metrics computed from the per-video scores recorded during Phase 0 evaluation
(`results_*.json`) — no new inference required. Thresholds 0.50/0.60/0.75 shown
explicitly; `best` = threshold maximizing balanced accuracy on THIS sample.
Scores are sigmoid outputs, NOT calibrated probabilities (no calibration
evidence exists), so treat thresholds as operating points, not probabilities.

### V1 full-sequence (val, n=216)

| Threshold | Acc | Bal. Acc | Precision | Recall | Specificity | F1 | FPR | FNR | tp/fp/fn/tn |
|-----------|------|----------|-----------|--------|-------------|------|------|------|-------------|
| 0.50 | 0.676 | 0.590 | 0.718 | 0.847 | 0.333 | 0.777 | 0.667 | 0.153 | 122/48/22/24 |
| 0.60 | 0.634 | 0.642 | 0.788 | 0.618 | 0.667 | 0.693 | 0.333 | 0.382 | 89/24/55/48 |
| 0.75 | 0.426 | 0.562 | 0.917 | 0.153 | 0.972 | 0.262 | 0.028 | 0.847 | 22/2/122/70 |
| **best** | — | **0.663** | — | — | — | — | — | — | thr=0.6155 |

### V2 full-sequence (val, n=219)

| Threshold | Acc | Bal. Acc | Precision | Recall | Specificity | F1 | FPR | FNR | tp/fp/fn/tn |
|-----------|------|----------|-----------|--------|-------------|------|------|------|-------------|
| 0.50 | 0.589 | 0.551 | 0.703 | 0.664 | 0.438 | 0.683 | 0.562 | 0.336 | 97/41/49/32 |
| 0.60 | 0.461 | 0.568 | 0.818 | 0.247 | 0.890 | 0.379 | 0.110 | 0.753 | 36/8/110/65 |
| 0.75 | 0.338 | 0.503 | 1.000 | 0.007 | 1.000 | 0.014 | 0.000 | 0.993 | 1/0/145/73 |
| **best** | — | **0.592** | — | — | — | — | — | — | thr=0.5353 |

### V1 full-sequence (train sample, n=80)

| Threshold | Acc | Bal. Acc | Precision | Recall | Specificity | F1 | FPR | FNR | tp/fp/fn/tn |
|-----------|------|----------|-----------|--------|-------------|------|------|------|-------------|
| 0.50 | 0.738 | 0.738 | 0.694 | 0.850 | 0.625 | 0.764 | 0.375 | 0.150 | 34/15/6/25 |
| 0.60 | 0.738 | 0.738 | 0.828 | 0.600 | 0.875 | 0.696 | 0.125 | 0.400 | 24/5/16/35 |
| 0.75 | 0.550 | 0.550 | 0.750 | 0.150 | 0.950 | 0.250 | 0.050 | 0.850 | 6/2/34/38 |
| **best** | — | **0.800** | — | — | — | — | — | — | thr=0.5458 |

### V2 full-sequence (train sample, n=80)

| Threshold | Acc | Bal. Acc | Precision | Recall | Specificity | F1 | FPR | FNR | tp/fp/fn/tn |
|-----------|------|----------|-----------|--------|-------------|------|------|------|-------------|
| 0.50 | 0.600 | 0.600 | 0.605 | 0.575 | 0.625 | 0.590 | 0.375 | 0.425 | 23/15/17/25 |
| 0.60 | 0.562 | 0.562 | 0.667 | 0.250 | 0.875 | 0.364 | 0.125 | 0.750 | 10/5/30/35 |
| 0.75 | 0.500 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0/0/40/40 |
| **best** | — | **0.650** | — | — | — | — | — | — | thr=0.4523 |

### V2 DEPLOYED ENSEMBLE (val sample, n=24)

| Threshold | Acc | Bal. Acc | Precision | Recall | Specificity | F1 | FPR | FNR | tp/fp/fn/tn |
|-----------|------|----------|-----------|--------|-------------|------|------|------|-------------|
| 0.50 | 0.667 | 0.667 | 0.600 | 1.000 | 0.333 | 0.750 | 0.667 | 0.000 | 12/8/0/4 |
| 0.60 | 0.708 | 0.708 | 0.647 | 0.917 | 0.500 | 0.759 | 0.500 | 0.083 | 11/6/1/6 |
| 0.75 | 0.667 | 0.667 | 1.000 | 0.333 | 1.000 | 0.500 | 0.000 | 0.667 | 4/0/8/12 |
| **best** | — | **0.792** | — | — | — | — | — | — | thr=0.6270 |

## Interpretation guidance

- 0.60 is an EXPERIMENTAL development setting (see docs/PHASE_PROGRESS.md);
  it is NOT claimed to be statistically optimal.
- The `best` row is an upper bound measured on the SAME sample used to pick
  it — it will look optimistic on new data and must not be reported as the
  model's performance.
- V1's best threshold sits near 0.62, V2's near 0.53 on the val sample; this
  is why model-specific threshold configuration is supported in the backend.
- Changing thresholds changes the trade-off between missed fakes (FNR) and
  false accusations (FPR); it cannot fix a weak model (see EVALUATION.md §4).