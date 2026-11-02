# Architecture

Status: Phase 1 (inference backend refactor) — implemented and verified locally.

## System overview

```
┌──────────────────────────┐        ┌──────────────────────────────────────────┐
│  Next.js frontend        │  HTTP  │  FastAPI inference backend (this repo)   │
│  (Phase 2, on Vercel)    │───────▶│  backend/app                              │
└──────────────────────────┘        │    api/routes      endpoints             │
                                    │    services/       model registry,       │
        ┌───────────────────┐      │                    face extraction,      │
        │ Supabase (Ph. 4)  │      │                    inference, LIME,      │
        │ history/storage   │◀─────│                    report data           │
        └───────────────────┘      │    workers/        bounded job runner    │
                                    │    schemas/        versioned JSON APIs   │
                                    └───────────────┬──────────────────────────┘
                                                    │ loads once at startup
                                            ┌───────▼────────┐
                                            │ *.keras        │  V1 / V2 / future
                                            │ checkpoints    │  checkpoints (registry)
                                            └────────────────┘
```

Design rules that came out of Phase 0 / the master plan:

1. **Model interchangeability.** The registry (`services/model_registry.py`)
   is the only code that knows checkpoint filenames. Registering a future V3 is
   one `register()` call — routes, schemas, reports and the frontend stay
   unchanged. Model version + checkpoint SHA-256 are echoed in every response.
2. **Centralized threshold.** One `Settings` object owns the decision
   threshold (env `DEEPFAKE_THRESHOLD`, dev default 0.60) with per-model
   overrides (`DEEPFAKE_THRESHOLD_V2`, …). The active threshold + its version
   label are part of every prediction and report. Historical 0.75 remains
   recorded in evaluation results only.
3. **One preprocessing implementation.** The backend imports the repo's
   `preprocessing.FaceExtractor` — the exact validated OpenCV DNN + elliptical
   mask pipeline used by training data generation and the Gradio app.
4. **Presentation-free API.** Endpoints return structured JSON; no Markdown.
   The Gradio `app.py` stays functional until the new stack fully replaces it.
5. **Persistence is an implementation detail.** Jobs live in a thread-safe
   in-memory store (`core/storage.py`). Phase 4 swaps in Supabase behind the
   same interface without touching services or routes.
6. **Explainability is failure-isolated.** A LIME error can never invalidate a
   base prediction; it surfaces as a warning.

## Analysis job lifecycle

```
POST /v1/analyses  ──▶  QUEUED ─▶ PREPROCESSING ─▶ INFERENCING ─▶ EXPLAINING
                        (upload stored under jobs/<id>/, SHA-256 hashed)      │
                                                                  GENERATING_REPORT
                                                                        │
                                                              COMPLETED ─┴─▶ FAILED
```

- States are only those actually implemented (`core/config.JOB_STATES`).
- Concurrency is bounded (`DEEPFAKE_MAX_WORKERS`, default 1). Uploads beyond
  capacity are rejected with HTTP 503 rather than silently queued forever.
- Temporary media/artifacts are deleted when a job finishes
  (`DEEPFAKE_KEEP_JOB_DIRS=1` to debug).

## Scoring paths (preserve deployed behaviour)

| Model | Sequence construction | Scoring | Warning surface |
|-------|----------------------|---------|-----------------|
| v1 | first 10 of ≤32 extracted candidates | single full-sequence score | rejected if <10 usable frames |
| v2 | all extracted frames (≤32, last-frame padded) | full-sequence + 10-frame windows (stride 5), averaged | padding warning if short |
| image | single face replicated across frame axis | `replicated_frame_sequence` | explicitly labelled a video-model adaptation |

## Deployment options

The backend is a plain FastAPI/uvicorn app with a single GPU/CPU process
contract, so it can run on:

- **Hugging Face Spaces** (current Gradio app stays as-is for now), or
- a containerized Python service (Phase 5 adds the Dockerfile/health wiring),
- any VM with Python 3.11 + the checkpoints on disk.

Vercel never runs TensorFlow; the frontend only talks HTTP to this API.
