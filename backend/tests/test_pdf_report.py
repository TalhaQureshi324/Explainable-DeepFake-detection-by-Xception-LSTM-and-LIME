"""PDF report endpoint tests (no TensorFlow: records crafted via the job store)."""

import json
import re
import sys
from io import BytesIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.storage import JobStore  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.report_data import build_report  # noqa: E402


def _completed_record(analysis_id="11111111-2222-3333-4444-555555555555"):
    """Mirror the real worker's record shape for a completed video analysis."""
    return {
        "analysis_id": analysis_id,
        "status": "COMPLETED",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:01:00+00:00",
        "media": {
            "type": "video", "filename": "clip.mp4", "size_bytes": 10485760,
            "sha256": "a" * 64, "width": 1280, "height": 720, "fps": 25.0,
            "duration_seconds": 40.0, "frames_extracted": 32, "frames_analyzed": 32,
            "valid_face_detections": 32,
        },
        "prediction": {
            "label": "FAKE", "fake_score": 0.610271, "threshold": 0.6,
            "threshold_version": "dev-v1", "method": "sliding_window_ensemble",
            "score_is_calibrated": False,
        },
        "model": {"name": "Xception-LSTM", "version": "v2", "description": "d",
                  "checkpoint_sha256": "b" * 64, "input_frames": 32, "input_size": 299},
        "processing": {"duration_seconds": 41.8, "warnings": ["Only 5 face frames..."]},
        "explainability": {"available": True, "method": "LIME",
                           "artifact_id": "xai_deadbeef", "limitation": "lim"},
        "artifacts": {"processed_face": "/v1/artifacts/face_deadbeef",
                      "lime": "/v1/artifacts/xai_deadbeef"},
        "windows": [
            {"label": "Full-32", "score": 0.61, "frame_start": 0, "frame_end": 843,
             "start_seconds": 0.0, "end_seconds": 33.72},
            {"label": "Win-0-9", "score": 0.58, "frame_start": 0, "frame_end": 216,
             "start_seconds": 0.0, "end_seconds": 8.64},
        ],
        "report_available": True,
    }


@pytest.fixture()
def pdf_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings

    monkeypatch.setenv("DEEPFAKE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    store = JobStore()
    registry = __import__("tests.conftest", fromlist=["FakeRegistry"]).FakeRegistry()
    worker = __import__("tests.conftest", fromlist=["FakeWorker"]).FakeWorker(store)
    app = create_app(store=store, registry=registry, worker=worker)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (64, 64), (30, 120, 200)).save(artifacts / "face_deadbeef.png")
    Image.new("RGB", (64, 64), (200, 30, 30)).save(artifacts / "xai_deadbeef.png")

    store.create(_completed_record()["analysis_id"], _completed_record())
    yield TestClient(app), store, artifacts
    get_settings.cache_clear()


def _pdf_pages(data: bytes) -> int:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def test_pdf_endpoint_serves_valid_pdf(pdf_client):
    client, _store, _artifacts = pdf_client
    r = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert "attachment" in r.headers["content-disposition"]
    fname = re.search(r'filename="([^"]+)"', r.headers["content-disposition"]).group(1)
    assert fname == "deepfake_report_11111111.pdf"
    data = r.content
    assert data[:4] == b"%PDF"
    assert _pdf_pages(data) >= 1


def test_pdf_contains_consistent_values(pdf_client):
    """The PDF must carry the exact stored prediction values — no recompute."""
    client, store, _artifacts = pdf_client
    r = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    import pymupdf

    with pymupdf.open(stream=r.content, filetype="pdf") as doc:
        text = " ".join(p.get_text() for p in doc)
    rec = store.get("11111111-2222-3333-4444-555555555555")
    expected = build_report(dict(rec))
    assert "0.610271" in text
    assert "FAKE" in text
    assert str(expected["prediction"]["threshold"]) in text
    assert "dev-v1" in text
    assert "b" * 24 in text[: len(text)]  # checkpoint hash prefix present
    assert "uncalibrated" in text.lower()
    assert expected["limitations"][0][:30] in text


def test_pdf_embeds_artifact_images(pdf_client):
    client, _store, artifacts = pdf_client
    r = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    import pymupdf

    with pymupdf.open(stream=r.content, filetype="pdf") as doc:
        images = [img for page in doc for img in page.get_images(full=True)]
    # face + LIME artifacts must appear (matplotlib chart adds more on later reads)
    assert len(images) >= 2
    assert (artifacts / "face_deadbeef.png").is_file()


def test_pdf_is_cached_not_regenerated(pdf_client, monkeypatch):
    client, _store, artifacts = pdf_client
    r1 = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    cache = artifacts / "report_11111111-2222-3333-4444-555555555555.pdf"
    assert cache.is_file()
    r2 = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    assert r1.content == r2.content


def test_pdf_states_are_rejected(pdf_client):
    client, store, _artifacts = pdf_client
    store.update("11111111-2222-3333-4444-555555555555", status="INFERENCING")
    r = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    assert r.status_code == 409

    store.update("11111111-2222-3333-4444-555555555555", status="FAILED")
    r = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    assert r.status_code == 409


def test_pdf_missing_analysis_denied(pdf_client):
    client, _store, _artifacts = pdf_client
    # unauthenticated: denied without existence disclosure
    assert client.get("/v1/analyses/nope/report.pdf").status_code == 401


def test_report_document_includes_artifacts(pdf_client):
    client, _store, _artifacts = pdf_client
    r = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report")
    body = r.json()
    assert body["artifacts"]["processed_face"] == "/v1/artifacts/face_deadbeef"
    assert body["artifacts"]["lime"] == "/v1/artifacts/xai_deadbeef"


def test_pdf_without_artifacts_still_renders(pdf_client, tmp_path):
    """Missing LIME/face images must not break generation (graceful degradation)."""
    import os

    client, store, artifacts = pdf_client
    (artifacts / "face_deadbeef.png").unlink()
    (artifacts / "xai_deadbeef.png").unlink()
    r = client.get("/v1/analyses/11111111-2222-3333-4444-555555555555/report.pdf")
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"
