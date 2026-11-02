# Phase 3 End-to-End Verification Evidence

Date: 2026-11-02 · Branch: `phase-3-reporting` · Real media through the full
stack: Next.js proxy → FastAPI → V2 checkpoint → report document → PDF.

## Workflow verified

```
Upload → FastAPI inference → Results → Interactive Report → PDF Download
```

| Check | VIDEO (with LIME) | IMAGE (no LIME requested) |
|-------|-------------------|---------------------------|
| Analysis | `dcbe720a…` COMPLETED | `3495ed1b…` COMPLETED |
| Prediction | REAL · **0.325892** | FAKE · **0.78528** |
| Threshold / version | 0.6 / dev-v1 | 0.6 / dev-v1 |
| Method | sliding_window_ensemble | replicated_frame_sequence |
| LIME artifact | generated + served | correctly absent, report still renders |
| PDF | 200 · 3 pages · 133 KB · 3 images (chart+face+LIME) | 200 · 2 pages · ~150 KB · 1 image (face) |
| Headers | `application/pdf` · `attachment; filename="deepfake_report_dcbe720a.pdf"` | same pattern |
| Score text in PDF | present | present |
| Threshold + version in PDF | present | present |
| Checkpoint SHA-256 in PDF | present | present |
| Uncalibrated framing + limitations | every page footer + §9 | same |

Score determinism: 0.325892 for the same video across four independent runs
spanning Phase 1 smoke test → Phase 2 e2e → Phase 3 e2e (twice).

## Defects found by verification and fixed

1. **LIME files invisible to serving/PDF** — `generate_lime_heatmap` appended a
   second `artifacts/` segment, landing PNGs outside the artifacts directory.
   The PDF showed "not available" for a analysis that HAD LIME. Fixed; file
   relocated; verified embedded in the PDF (3 images).
2. **Face preview vs LIME frame mismatch** — the preview showed the middle raw
   frame while LIME explained the middle of the *padded* sequence (different
   frames visible in the first PDF render). Preview now uses exactly the LIME
   target frame (verified: identical frame in §5 Visual evidence).
3. **Orphaned PDF heading** — "4. Video prediction timeline" could strand at a
   page bottom with its chart on the next page. Fixed with `KeepTogether`.
4. (Carried from Phase 2 verification) Next 14 `params` handling.

## Automated tests

`python -m pytest backend/tests` → **42 passed** (8 new PDF tests: valid output +
filename, value consistency against the stored record, embedded artifact
images, cache identity across repeat downloads, 409/404 states, graceful
rendering without artifacts, artifacts field in the report document).
`npm run build` → production build green.

## Review artifacts (committed)

- `docs/samples/phase3_report_video.pdf` — 3-page video report with timeline
  chart, processed face, and LIME heatmap
- `docs/samples/phase3_report_image.pdf` — 2-page image report (replicated-frame
  mode, LIME section degrades gracefully)
- `docs/screenshots/pdf_video_p1.png` / `pdf_video_p2.png` — rendered PDF pages
- `docs/screenshots/pdf_image_p1.png` / `pdf_image_p2.png`
- `docs/screenshots/report_page_phase3.png` — interactive report (light theme)
- Phase 2 screenshots remain for the rest of the UI

## Known limitations (documented, intentional)

- PDFs are cached under `backend/.artifacts/` and analysis records stay
  in-memory until Phase 4 (Supabase): a backend restart invalidates both.
  Uploaded media is still deleted the moment an analysis finishes.
- Timeline charts cluster to a point for ultra-short videos (all windows fall
  in the same fraction of a second) — the data is real; the axis just has
  nothing to spread across.
- PDF typography uses built-in Helvetica; Urdu text is not rendered inside the
  PDF (the web report carries the bilingual/RTL experience; the PDF is English
  for typographic reliability). This is stated here rather than silently
  attempted.
