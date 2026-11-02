"""
Report data builder.

Constructs the structured report document from a COMPLETED analysis record.
Both the future web report (Phase 3) and the PDF export must render from this
single structure so numbers can never diverge between views.

The interpretation template is deliberately conservative: it paraphrases the
stored numbers and never produces forensic claims.
"""

from typing import Optional

LIMITATIONS = [
    "Scores are sigmoid outputs from an uncalibrated model; they are not proven probabilities.",
    "Models were trained only on the DeepFake Detection (DFD) subset of FaceForensics++; "
    "performance on other manipulation methods is unknown.",
    "The train/val split has identity leakage, so measured metrics are optimistic.",
    "Only the most confident face per frame is analyzed; multi-face videos are not fully covered.",
    "LIME explains a replicated static frame; it does not attribute the temporal pathway.",
    "Face-detection variability across OpenCV builds can change scores for the same video.",
]


def build_interpretation(label: str, fake_score: float, threshold: float,
                         method: str, warnings: list) -> str:
    margin = fake_score - threshold
    if label == "FAKE":
        stance = ("The model scored this media above the configured decision threshold, "
                  "which maps to its FAKE class.")
        margin_txt = f"The score exceeds the threshold by {margin:.3f}."
    else:
        stance = ("The model scored this media below the configured decision threshold, "
                  "which maps to its REAL class.")
        margin_txt = f"The score falls short of the threshold by {abs(margin):.3f}."
    return (
        f"{stance} {margin_txt} "
        f"Scoring method: {method}. "
        "Because the model is uncalibrated and was evaluated as weak on held-out data "
        "(see evaluation/EVALUATION.md), this assessment must be treated as a "
        "research-grade indication and reviewed by a human before any consequential use."
        + (" Quality warnings were raised during processing." if warnings else "")
    )


def build_report(analysis: dict) -> dict:
    """analysis: the internal record from the job store (completed only)."""
    media = analysis.get("media") or {}
    prediction = analysis.get("prediction") or {}
    model = analysis.get("model") or {}
    processing = analysis.get("processing") or {}
    windows = analysis.get("windows") or []

    window_scores = [w.get("score") for w in windows if isinstance(w.get("score"), (int, float))]
    variation = None
    if window_scores:
        import statistics
        variation = {
            "min": min(window_scores),
            "max": max(window_scores),
            "stdev": statistics.pstdev(window_scores) if len(window_scores) > 1 else 0.0,
        }

    return {
        "analysis_id": analysis.get("analysis_id"),
        "generated_from_status": analysis.get("status"),
        "overview": {
            "created_at": analysis.get("created_at"),
            "media_type": media.get("type"),
            "model_version": model.get("version"),
            "assessment": prediction.get("label"),
            "assessment_basis": (
                f"fake_score {prediction.get('fake_score')} vs decision threshold "
                f"{prediction.get('threshold')} ({prediction.get('threshold_version')})"
            ),
        },
        "media": {k: media.get(k) for k in (
            "filename", "size_bytes", "sha256", "width", "height", "fps",
            "duration_seconds", "frames_extracted", "frames_analyzed",
            "valid_face_detections") if media.get(k) is not None},
        "prediction": {
            "fake_score": prediction.get("fake_score"),
            "threshold": prediction.get("threshold"),
            "threshold_version": prediction.get("threshold_version"),
            "label": prediction.get("label"),
            "method": prediction.get("method"),
            "score_is_calibrated": prediction.get("score_is_calibrated", False),
            "window_variation": variation,
        },
        "timeline": {
            "note": "Window scores are placed on the source timeline using fps and the "
                    "source frame index recorded during face extraction.",
            "windows": windows,
            "threshold": prediction.get("threshold"),
        },
        "explainability": analysis.get("explainability") or {},
        "artifacts": analysis.get("artifacts") or {},
        "technical": {
            "model": model,
            "frames_analyzed": media.get("frames_analyzed"),
            "prediction_method": prediction.get("method"),
            "processing_duration_seconds": processing.get("duration_seconds"),
            "warnings": processing.get("warnings") or [],
        },
        "interpretation": build_interpretation(
            prediction.get("label", ""),
            prediction.get("fake_score", 0.0),
            prediction.get("threshold", 0.0),
            prediction.get("method", ""),
            processing.get("warnings") or [],
        ),
        "limitations": LIMITATIONS,
    }
