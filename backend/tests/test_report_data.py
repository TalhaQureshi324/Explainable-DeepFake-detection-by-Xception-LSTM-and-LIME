"""Report data builder: conservative interpretation, complete structure."""

from app.services.report_data import LIMITATIONS, build_report


def _completed_analysis():
    return {
        "analysis_id": "abc-123",
        "status": "COMPLETED",
        "created_at": "2026-01-01T00:00:00+00:00",
        "media": {
            "type": "video", "filename": "clip.mp4", "sha256": "a" * 64,
            "width": 1280, "height": 720, "fps": 25.0, "duration_seconds": 40.0,
            "frames_extracted": 32, "frames_analyzed": 32, "valid_face_detections": 32,
        },
        "prediction": {
            "label": "REAL", "fake_score": 0.41, "threshold": 0.60,
            "threshold_version": "dev-v1", "method": "sliding_window_ensemble",
            "score_is_calibrated": False,
        },
        "model": {"version": "v2", "name": "Xception-LSTM"},
        "processing": {"duration_seconds": 12.5, "warnings": ["Only 12 face frames..."]},
        "explainability": {"available": False},
        "windows": [
            {"label": "Full-32", "score": 0.41},
            {"label": "Win-0-9", "score": 0.52},
            {"label": "Win-5-14", "score": 0.34},
        ],
    }


def test_report_contains_all_core_sections():
    report = build_report(_completed_analysis())
    for key in ("overview", "media", "prediction", "timeline", "technical",
                "interpretation", "limitations"):
        assert key in report
    assert report["overview"]["assessment"] == "REAL"
    assert report["prediction"]["threshold_version"] == "dev-v1"


def test_window_variation_is_computed():
    report = build_report(_completed_analysis())
    var = report["prediction"]["window_variation"]
    assert abs(var["min"] - 0.34) < 1e-9
    assert abs(var["max"] - 0.52) < 1e-9
    assert var["stdev"] > 0


def test_interpretation_never_makes_forensic_claims():
    text = build_report(_completed_analysis())["interpretation"].lower()
    assert "definitely" not in text
    assert "proof" in text or "research-grade" in text


def test_limitations_carried_into_report():
    report = build_report(_completed_analysis())
    assert len(report["limitations"]) == len(LIMITATIONS) >= 5


def test_missing_optional_fields_do_not_crash():
    minimal = {"analysis_id": "x", "status": "COMPLETED"}
    report = build_report(minimal)
    assert report["analysis_id"] == "x"
    assert report["prediction"]["fake_score"] is None
