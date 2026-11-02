# Inference API reference (v1)

Base URL (local dev): `http://127.0.0.1:8000`

```
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: `/docs` (Swagger, generated from the schemas).

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `DEEPFAKE_MODELS_DIR` | `<repo>/models_store` | folder with `.keras` checkpoints |
| `DEEPFAKE_MODEL_VERSION` | `v2` | preferred version (fallback `v1`) |
| `DEEPFAKE_THRESHOLD` | `0.60` | default decision threshold (dev setting; historical baseline 0.75) |
| `DEEPFAKE_THRESHOLD_V1` / `_V2` | — | model-specific threshold overrides |
| `DEEPFAKE_THRESHOLD_VERSION` | `dev-v1` | label recorded with every prediction |
| `DEEPFAKE_MAX_UPLOAD_MB` | `200` | upload size cap |
| `DEEPFAKE_MAX_WORKERS` | `1` | concurrent analysis jobs |
| `DEEPFAKE_ENABLE_XAI` | `false` | default LIME setting for requests |
| `DEEPFAKE_JOBS_DIR` | `backend/.jobs` | temp job workspace |
| `DEEPFAKE_KEEP_JOB_DIRS` | unset | keep temp dirs for debugging |

## Endpoints

### `GET /health` → `{"status": "ok", "version": "v1"}`

### `GET /ready`
`{"ready": true, "model_version": "v2"}` — false + detail when no checkpoint is loadable.

### `GET /v1/models`
Lists every registered checkpoint with input shape, SHA-256, availability and
the decision threshold that applies to it.

### `POST /v1/analyses` (multipart, returns **202**)
Fields:
- `file` (required): video (`.mp4 .avi .mov .mkv .webm`) or image (`.jpg .jpeg .png .bmp .webp`)
- `model_version` (optional): request a specific registered checkpoint
- `explain` (optional): request LIME for this analysis

Errors: `400` bad/empty file, `413` too large, `415` unsupported type, `503` at capacity.

```json
{"analysis_id": "…uuid…", "status": "QUEUED", "poll_url": "/v1/analyses/<id>"}
```

### `GET /v1/analyses/{id}`
Job status + full result (schema: `backend/app/schemas/analysis.py`):

```json
{
  "analysis_id": "…", "status": "COMPLETED",
  "media": {"type": "video", "filename": "clip.mp4", "sha256": "…", "fps": 24.0,
            "duration_seconds": 35.2, "frames_extracted": 32, "frames_analyzed": 32},
  "prediction": {
    "label": "FAKE", "fake_score": 0.610271,
    "threshold": 0.6, "threshold_version": "dev-v1",
    "method": "sliding_window_ensemble",
    "score_is_calibrated": false
  },
  "model": {"name": "Xception-LSTM", "version": "v2",
            "checkpoint_sha256": "…", "input_frames": 32},
  "processing": {"duration_seconds": 41.8, "warnings": []},
  "explainability": {"available": false, "method": null, "artifact_id": null, "limitation": null},
  "windows": [{"label": "Full-32", "score": 0.609, "frame_start": 0, "frame_end": 843,
               "start_seconds": 0.0, "end_seconds": 35.13}],
  "report_available": true
}
```

Rules encoded in the contract:

- `fake_score` is the raw sigmoid output (rounded to 6 decimals) — **not** a
  calibrated probability (`score_is_calibrated: false` until evidence exists).
- `threshold` is the ACTIVE threshold used for `label`; historical 0.75 is not
  silently substituted.
- Images report `method: "replicated_frame_sequence"` plus an explicit warning
  that this is a video-model adaptation, not a validated image classifier.
- Window timestamps are computed from the fps + source frame indices captured
  during face extraction — never estimated after the fact.
- Failure surfaces as `status: "FAILED"` with `processing.error_code` in
  {`NO_FACE_DETECTED`, `INSUFFICIENT_FRAMES`, `MEDIA_UNREADABLE`,
  `MODEL_UNAVAILABLE`, `SERVER_BUSY`, `INTERNAL_ERROR`}; internal tracebacks
  are never returned.

### `GET /v1/analyses/{id}/report`
Structured report document (see `docs/REPORT_SCHEMA.md`). `409` until the
analysis is `COMPLETED`.

### `GET /v1/artifacts/{artifact_id}`
Serves analysis artifacts as PNG (processed-face preview, LIME heatmap).
Artifact IDs are strict `[A-Za-z0-9_]{4,64}` tokens generated server-side; the
path cannot traverse out of the artifacts directory. `404` for unknown/expired
artifacts. Completed analyses expose the URLs in the additive
`artifacts.processed_face` / `artifacts.lime` response fields. Artifacts
persist in `DEEPFAKE_ARTIFACTS_DIR` (default `backend/.artifacts/`) after the
job's temp directory is cleaned up; retention policy arrives with Phase 4.

> Added in Phase 2 (smallest change needed for the Results/Report screens to
> display real images); documented here per the master plan's backend-extension
> rule.

## Frontend integration contract (Next.js, Phase 2)

1. `POST /v1/analyses` with the media as multipart; store `analysis_id`.
2. Poll `GET /v1/analyses/{id}` every ~2 s; drive the progress UI ONLY from
   `status` (QUEUED/PREPROCESSING/INFERENCING/EXPLAINING/GENERATING_REPORT/
   COMPLETED/FAILED). Never invent percentages.
3. On `COMPLETED`: render `prediction.label`, `prediction.fake_score`,
   `prediction.threshold` + `threshold_version`, `model.version`,
   `processing.warnings`, `windows` (timeline chart uses `start_seconds`/`end_seconds`),
   and `explainability` if available.
4. On `FAILED`: map `processing.error_code` to user-facing copy; offer retry.
5. Treat all fields as additive-compatible: new optional fields may appear at
   any time; existing fields keep their types.
6. Phase 1 has NO persistence: analysis results are lost on backend restart —
   the frontend must keep the `analysis_id` locally until Phase 4 (Supabase).
