# Training Investigation & Corrected Pipeline (Task R3)

Why the original checkpoints perform poorly, what evidence exists for each
cause, and the corrected retraining pipeline that addresses them. Written as
part of the Phase 1 parallel workstream — every claim links to a committed
measurement.

---

## 1. Root causes, ranked by evidence strength

### Cause 1 — Both checkpoints were trained on 67 videos, not the dataset (smoking gun)

`main.py` / `main_version2.py` train from `preprocessed_new/`, which on the
training machine contained **54 training videos (49 FAKE / 5 REAL)** and
**13 validation videos (12 FAKE / 1 REAL)** — a strict subset of the full
`processed_dataset/` (2,744 train / 687 val), verified by name matching.

Consequences:

- The models never saw the full dataset. Everything else is secondary to this.
- `ModelCheckpoint(monitor="val_loss")` selected "best" weights against
  **13 videos** — statistically meaningless.
- The actual training class ratio was ~10:1 FAKE:REAL (class weights
  up-weighted just 5 REAL videos → noisy gradients).

**Verification:** run `python -c` folder-count comparison (documented in
`docs/PHASE_PROGRESS.md`); `preprocessed_new` ⊂ `processed_dataset`, labels
consistent.

### Cause 2 — Xception input normalization mismatch

Training used `[0, 1]` scaling while ImageNet-pretrained Xception expects
`[-1, 1]` (`keras.applications.xception.preprocess_input`). Measured on 10
training face crops with the pretrained network:

- Top-1 ImageNet prediction agreement between the two scalings: **50%**
- Penultimate-feature cosine similarity: 0.97 (direction largely preserved)

Because the first 100 Xception layers were frozen, the network could not adapt
them to `[0, 1]` inputs. This is a real, measured transfer-learning handicap —
but **secondary** to Cause 1. The corrected script defaults to
`[-1, 1]` and offers `--preprocessing-mode zero_one` for controlled comparison.
The deployed checkpoints must KEEP `[0, 1]` at inference; never mix modes.

### Cause 3 — Optimization configuration too weak for V2

32 frames/video, batch size 1, lr=1e-5, 10 epochs, LSTM(64). Combined with
Cause 1 (54 videos), the 21.4M-parameter model was effectively undertrained —
consistent with V2's measured underfitting (train AUC 0.643).

### Cause 4 — Split leakage in the original dataset organization

`script.py` split randomly per video: the same actor — and original/manipulated
pairs — could span train and val. V1's train→val AUC drop (0.819 → 0.670) is
consistent with memorizing identities. Fixed by
`training/make_identity_split.py` (scene- or actor-connected groups).

### Cause 5 — Preprocessing reproducibility

Same video, different OpenCV builds → face-detector box shifts → measured score
deltas up to 0.49 (Phase 0, `evaluation/EVALUATION.md` §5). Not a training bug;
a deployment risk. Mitigations: pinned opencv versions, hashed detector files,
and re-extracting frames in the SAME environment as inference when retraining.

---

## 2. The corrected pipeline

| Piece | File | What changed |
|-------|------|--------------|
| Splitting | `training/make_identity_split.py` | Scene- or actor-disjoint groups; train/val/calibration/**locked test**; auditable manifest |
| Training | `training/train_corrected.py` | Manifest-driven full-data training; correct Xception normalization default; AUC-based checkpointing; early stopping; class weights; experiment manifest (config, hashes, versions, curves) |
| GPU run | `training/COLAB_TRAINING.ipynb` | Colab/Kaggle-ready: GPU check → repo clone → Drive mount → sanity checks → train → curves → evaluate on val + calibration → threshold calibration → export |
| Evaluation | `evaluation/evaluate_frames.py`, `threshold_analysis.py` | Same metrics for new checkpoints as Phase 0, at explicit thresholds + best-achievable |

Both scripts were smoke-tested locally on CPU (1 epoch, 6 videos) — outputs:
`training/runs/smoke_test/` (gitignored).

## 3. Experiment matrix (run on GPU in this order)

| # | Experiment | Command sketch | Question answered |
|---|------------|----------------|-------------------|
| 0 | Sanity | `train_corrected.py --epochs 1 --max-train-videos 20` on Colab | pipeline runs on GPU end-to-end |
| 1 | Normalization | `--preprocessing-mode minus_one_plus_one` vs `zero_one`, otherwise identical | how much the Xception scaling mismatch cost |
| 2 | Backbone | `--backbone frozen` vs `finetune` | whether full fine-tuning + higher LR is needed |
| 3 | Temporal depth | `--num-frames 32` vs `--num-frames 10 --v1-first-k 10` | whether the LSTM/temporal path helps at all |
| 4 | Final candidate | best of #1–3, full epochs, early stopping | the model to calibrate + test |
| 5 | Locked test | `evaluate_frames.py --split test` **once** on the final candidate | the only reportable generalization number |

Protocol rules:

- Threshold selection happens **only** on the calibration split
  (`evaluation/threshold_analysis.py`), never on test.
- A new checkpoint replaces V1/V2 in the registry only after #5 beats the
  Phase 0 baselines (V1 val AUC 0.670 / V2 val AUC 0.603, ensemble AUC 0.819
  on the 24-video sample).
- Report calibration status explicitly: sigmoid scores are NOT probabilities
  until calibration is measured (e.g., reliability curves on the calibration split).
