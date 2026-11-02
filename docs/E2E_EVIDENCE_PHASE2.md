# Phase 2 End-to-End Verification Evidence

Date: 2026-11-02 · Branch: `phase-2-frontend` · All commands run locally against
the REAL FastAPI backend (V2 checkpoint) through the Next.js production server
and its same-origin proxy.

## Environment

| Piece | Value |
|-------|-------|
| Backend | `uvicorn backend.app.main:app --port 8200` (port 8000 blocked by another process on this machine) |
| Frontend | `next start -p 3000` with `INFERENCE_API_URL=http://127.0.0.1:8200` baked at build time |
| Media | real validation data: `e2e_test_media/test_video.mp4` (FAKE val video, 0.2 MB, 6 frames) and `test_face.jpg` (frame 0 of REAL val video `06__talking_against_wall.mp4`) |

## Results (through http://localhost:3000/api/inference/*)

| Check | Video | Image |
|-------|-------|-------|
| POST /v1/analyses | 202, analysis_id returned | 202, analysis_id returned |
| Final status | COMPLETED | COMPLETED |
| Prediction | REAL · **0.325892** | FAKE · **0.78528** |
| Threshold / version | 0.60 / dev-v1 | 0.60 / dev-v1 |
| Method | sliding_window_ensemble (2 windows) | replicated_frame_sequence |
| Short-frame warning | present ("Only 5 face frames…") | compat-mode warning present |
| Face artifact | HTTP 200 · 60,866 B · valid PNG | HTTP 200 · 77,706 B · valid PNG |
| Report endpoint | 200 · assessment REAL · 6 limitations | 200 · assessment FAKE · 6 limitations |

**Score consistency:** the video score 0.325892 is bit-identical to the same
video's score from the Phase 1 API smoke test (`backend/scripts/smoke_api.py`) —
the full frontend→proxy→API→model stack adds zero numerical drift.

## Bug found and fixed during verification

Headless-browser screenshots of `/analysis/[id]` caught a client-side crash:
the pages used the Next 15 `params: Promise` convention under Next 14, where
`params` is a plain object. Fixed in both dynamic pages; rebuild + re-screenshot
confirmed.

## Screenshots (docs/screenshots/)

| File | Shows |
|------|-------|
| `home_light.png` | Homepage, light theme: hero, prototype badge, disclaimer, architecture cards |
| `home_dark.png` | Homepage, dark theme |
| `home_urdu_rtl.png` | Homepage in Urdu with full RTL layout + Nastaliq typography |
| `analyze_light.png` | Upload page: dropzone, formats, size limit |
| `results_video.png` | Results for the real video: verdict card, score vs threshold, uncalibrated notice, model/checkpoint info, warning, window timeline chart, processed face, empty-state LIME, media metadata |
| `report_preview.png` | Full report preview: overview, prediction summary, timeline, face, technical, interpretation, warnings + limitations, disabled PDF button |

## Automated checks

- `python -m pytest backend/tests` → **34 passed** (includes artifact-route
  validation/traversal tests and the additive `artifacts` response field).
- `npm run build` → production build passes (typed routes, ESLint clean).
