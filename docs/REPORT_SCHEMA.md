# Report data schema (Phase 3 basis)

`GET /v1/analyses/{id}/report` returns this structure (built by
`backend/app/services/report_data.py`). The web report AND the future PDF must
render from this one structure so numbers can never diverge. All fields are
populated only with genuine measured values; missing data is `null`.

```jsonc
{
  "analysis_id": "uuid",
  "generated_from_status": "COMPLETED",

  "overview": {
    "created_at": "ISO-8601",
    "media_type": "video | image",
    "model_version": "v2",
    "assessment": "REAL | FAKE",
    "assessment_basis": "fake_score 0.61 vs decision threshold 0.6 (dev-v1)"
  },

  "media": {                       // only keys with real values are included
    "filename": "clip.mp4", "size_bytes": 10485760, "sha256": "…",
    "width": 1280, "height": 720, "fps": 24.0,
    "duration_seconds": 35.2,
    "frames_extracted": 32,        // face crops extracted
    "frames_analyzed": 32,         // actually fed to the model
    "valid_face_detections": 32
  },

  "prediction": {
    "fake_score": 0.610271,        // raw sigmoid output, NOT calibrated
    "threshold": 0.6,
    "threshold_version": "dev-v1",
    "label": "FAKE",
    "method": "sliding_window_ensemble | full_sequence | replicated_frame_sequence",
    "score_is_calibrated": false,
    "window_variation": {"min": 0.34, "max": 0.61, "stdev": 0.09}   // when windows exist
  },

  "timeline": {
    "note": "Window scores are placed on the source timeline using fps and the
             source frame index recorded during face extraction.",
    "windows": [
      {"label": "Win-0-9", "score": 0.58,
       "frame_start": 0, "frame_end": 216,             // ORIGINAL video frame indices
       "start_seconds": 0.0, "end_seconds": 9.0}       // ORIGINAL timeline seconds
    ],
    "threshold": 0.6                  // chart reference line
  },

  "explainability": {
    "available": true, "method": "LIME", "artifact_id": "xai_hex",
    "limitation": "LIME highlights regions of a single replicated frame … "
                  "does NOT prove manipulation and does not explain the LSTM pathway."
  },

  "artifacts": {                    // relative backend URLs (added in Phase 3)
    "processed_face": "/v1/artifacts/face_<id>",
    "lime": "/v1/artifacts/xai_<id> | null"
  },

  "technical": {
    "model": {"name": "Xception-LSTM", "version": "v2", "checkpoint_sha256": "…",
              "input_frames": 32, "input_size": 299},
    "prediction_method": "sliding_window_ensemble",
    "processing_duration_seconds": 41.8,
    "warnings": ["Only 12 face frames extracted (model expects 32); …"]
  },

  "interpretation": "Plain-language paraphrase of the stored numbers; template "
                    "forbids forensic claims ('definitely fake' is unrepresentable).",

  "limitations": [ /* fixed list: uncalibrated scores, DFD-only training, leakage, … */ ]
}
```

Rules for Phase 3 rendering (web + PDF):

1. Every number shown must come from this document — no recomputation in the UI.
2. The threshold line and the `score_is_calibrated: false` state must be visible.
3. The LIME limitation text must accompany any heatmap.
4. The `limitations` list must appear in full in every rendered report/PDF.
5. Missing charts/artifacts degrade gracefully (omit section + note), never
   silently, and never invalidate the base prediction.
6. The server-rendered PDF (`report.pdf`) renders THIS document via
   `backend/app/services/pdf_report.py` — one data source for web and PDF.
