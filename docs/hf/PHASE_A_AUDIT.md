# Phase A — Live Hugging Face Space audit (2026-11-04)

Space: `MuhammadTalha121/Deep-Fake-Detection` (public, Gradio, python 3.13)
Client: official `gradio_client` (python). Raw evidence: `space_audit.json`.

## Real smoke tests

| Test | Result | Duration |
|------|--------|----------|
| Connect | Space awake | 5.3 s |
| Image (test_face.jpg, no LIME) | FAKE · avg score **0.7853** · conf 78.53% · 2 windows | 62.5 s |
| Video (5-frame clip, no LIME) | REAL · avg score 0.4652 · 2 windows · "5 face frames extracted" | 16.8 s |
| Image with LIME | same FAKE score + heatmap file | **495.8 s (~8 min!)** |

**Score consistency across backends:** the same image scored 0.7853/0.78528 via
the FastAPI stack (Phases 1-4) and 0.7853 via the live Space — one model, two
entry points.

## Exact response shape of `/predict_deepfake`

Inputs: `input_file` (FileData), `enable_xai` (bool), `lang_display`
("English" | "اردو"). Returns a 3-tuple:

1. Markdown string: verdict emoji header (🟥 FAKE / 🟩 REAL) + table
   (Confidence %, Average Score, Std Deviation, Ensemble Windows count,
   Input Type).
2. Processed-face image file.
3. LIME heatmap file — or `None` when `enable_xai=False`.

## Metadata the Space does NOT provide (verified missing)

- Decision threshold (the Space app uses **0.75** internally — preserved as
  `hf-space-original` in our adapter; never reported as the 0.60 dev threshold)
- Model version / checkpoint identifier (Space prefers v2, falls back to v1 —
  not reported per request)
- SHA-256, fps, duration, resolution, frame indices/timestamps
- Per-window scores (only a window COUNT is present)
- Analysis IDs, queue position, persistent artifacts

## Frontend features that depended on FastAPI (Phase 5 disposition)

| Feature | Disposition |
|---------|-------------|
| Upload → prediction | ✅ re-implemented via `@gradio/client` (browser → Space direct) |
| Structured results/report | 🟡 basic mode: adapted Markdown-derived fields; full structured data arrives with the `/analyze_structured` Space extension |
| LIME display | ✅ Space returns the heatmap image |
| PDF report | 🟡 Phase C/D: Space-side ReportLab endpoint (Option A chosen: reuses proven ReportLab layout, no Vercel function limits) |
| Auth + history + dashboard | ⏸ hidden in HF guest mode until Supabase integration |
| Window timeline chart | ⏸ hidden in basic mode (no per-window data) — returns with the structured endpoint (window labels/scores ARE computable Space-side) |
| Real frame timestamps | ⏸ not computed by the Space pipeline — not invented |
