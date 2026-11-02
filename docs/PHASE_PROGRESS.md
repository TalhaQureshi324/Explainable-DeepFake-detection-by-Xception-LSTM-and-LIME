# Phase Progress

| Phase | Scope | Status | Evidence |
|-------|-------|--------|----------|
| Phase 0 | Baseline audit, model provenance, evaluation, honest metrics | ✅ DONE (merged to main) | `evaluation/EVALUATION.md`, `evaluation/model_manifest.json`, commits `735468a..8496d82` |
| ML diagnostics | Training-folder truth, normalization, thresholds, ensemble-vs-full-seq | ✅ DONE (Phase 1 branch) | See "Diagnostic findings" below; `evaluation/THRESHOLD_ANALYSIS.md` |
| Phase 1 | FastAPI backend, model registry, threshold config, tests | ✅ DONE (Phase 1 branch) | 33/33 tests pass; real-media smoke: API score matches direct reference (Δ 3.6e-07) |
| Retraining prep (R1–R4) | Leakage-controlled split, corrected trainer, GPU notebook, investigation | ✅ PREPARED + PREFLIGHT-VERIFIED (GPU run pending — user runs `training/COLAB_TRAINING.ipynb`) | `training/make_identity_split.py`, `training/train_corrected.py` (CPU smoke-tested), `training/COLAB_TRAINING.ipynb`, `training/TRAINING_NOTES.md` |
| Phase 2 | Next.js frontend | ✅ DONE (branch `phase-2-frontend`, merged to main) | `frontend/`; e2e with real image+video through the proxy: scores identical to direct API calls; 34 backend tests; screenshots in `docs/screenshots/`; evidence in `docs/E2E_EVIDENCE_PHASE2.md` |
| Phase 3 | Interactive report + PDF | not started | report data structure implemented; report-preview UI already renders the full document (`frontend/app/analysis/[id]/report`) |
| Phase 4 | Supabase history | not started | needs Supabase project (user) |
| Phase 5 | Deployment hardening | not started | needs Vercel + hosting decision (user) |

## Diagnostic findings (between Phase 0 and Phase 1)

1. **Both shipped checkpoints were trained on only 67 videos** — `preprocessed_new`
   (54 train / 13 val) is a strict subset of `processed_dataset` (2,744 / 687).
   This is the dominant cause of weak performance; labels are consistent, crops
   are clean.
2. **Xception normalization mismatch confirmed** (`[0,1]` vs expected `[-1,1]`):
   50% top-1 flips on pretrained features (10-sample probe); real but secondary.
3. **Threshold study**: V1 best operating point ≈ 0.62, V2 full-sequence ≈ 0.53,
   V2 deployed ensemble ≈ 0.63. 0.60 is a defensible dev setting for the
   ensemble path (bal-acc 0.708, recall 0.92, FPR 0.50 on the 24-video sample);
   NOT a final scientific conclusion.
4. **Deployed ensemble beats full-sequence scoring**: ensemble AUC 0.819 vs
   full-sequence 0.681 on the paired 24-video val sample (mean paired |Δ| 0.137).
   The Phase 0 full-sequence numbers underestimated the deployed path.
5. **API adds no numerical drift**: in-process API result equals a direct call
   of the same inference functions (Δ 3.6e-07 = the API's 6-decimal rounding).

## Threshold policy

- `DEEPFAKE_THRESHOLD` env var (default **0.60**, version label `dev-v1`),
  per-model overrides `DEEPFAKE_THRESHOLD_V1/_V2`.
- Historical 0.75 preserved in `evaluation/` results; never silently substituted.
- Recalibrate on the new **calibration split** after retraining; the test split
  stays locked for final reporting.

## Commits (Phase 1 branch `phase-1-backend`)

See `git log phase-1-backend` — logical commits for: diagnostics+threshold tool,
backend package, tests, training pipeline, docs. Merge to `main` after review.

## Blockers / user actions

- GPU training: create `frames.zip` (processed_dataset) + upload with
  `training/dataset_manifest.json` to Drive, run `training/COLAB_TRAINING.ipynb`.
- Vercel/Supabase accounts only needed from Phase 2 deployment / Phase 4 onward.
