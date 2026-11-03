"""Shared fixtures: build the app with fakes so tests never load TensorFlow."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.config import Settings, get_settings  # noqa: E402
from app.services.model_registry import ModelNotAvailableError  # noqa: E402
from app.core.storage import JobStore  # noqa: E402
from app.main import create_app  # noqa: E402


class FakeRegistry:
    """Registry stand-in with the same surface the routes use."""

    class _Entry:
        def __init__(self, version):
            self.version = version
            self.input_frames = 32 if version == "v2" else 10
            self.input_size = 299
            self.checkpoint_sha256 = "fake" * 8
            self.description = "fake"
            self.file_path = Path(f"fake_{version}.keras")

    def __init__(self, versions=("v1", "v2"), available=True):
        self._versions = list(versions)
        self.available = available

    def resolve_version(self, requested=None):
        if not self.available:
            raise Exception("no models installed")
        for candidate in (requested, "v2", "v1"):
            if candidate in self._versions:
                return candidate
        raise Exception("no usable model")

    def get_entry(self, version):
        if version not in self._versions:
            raise ModelNotAvailableError(f"unknown model {version}")
        return self._Entry(version)

    def threshold_for_version(self, version):
        return {"v1": 0.60, "v2": 0.60}[version]

    def describe_all(self):
        return {
            "preferred_version": self._versions[-1],
            "models": [
                {"version": v, "description": "fake", "available": True, "file": f"fake_{v}.keras",
                 "checkpoint_sha256": "fake" * 8, "input_frames": 32 if v == "v2" else 10,
                 "input_size": 299, "decision_threshold": 0.60, "loaded": False}
                for v in self._versions
            ],
        }


class FakeWorker:
    """Completes jobs synchronously with a fixed, schema-conformant result."""

    def __init__(self, store: JobStore, label: str = "FAKE", score: float = 0.68):
        self.store = store
        self.label = label
        self.score = score
        self.submitted = []

    def queue_capacity(self):
        return 1

    def submit(self, media_path, media_type, original_filename, user=None,
               requested_model=None, request_xai=None):
        import uuid

        analysis_id = str(uuid.uuid4())
        self.submitted.append((analysis_id, media_type, original_filename))
        self.store.create(analysis_id, {
            "media": {
                "type": media_type, "filename": original_filename,
                "size_bytes": 1234, "sha256": "a" * 64,
                "width": 64, "height": 64, "fps": 25.0,
                "duration_seconds": 2.0, "frames_extracted": 32, "frames_analyzed": 32,
                "valid_face_detections": 32,
            } if media_type == "video" else {
                "type": media_type, "filename": original_filename,
                "size_bytes": 1234, "sha256": "a" * 64,
                "frames_extracted": 1, "frames_analyzed": 1, "valid_face_detections": 1,
            },
            "prediction": {
                "label": self.label, "fake_score": self.score,
                "threshold": 0.60, "threshold_version": "dev-v1",
                "method": "sliding_window_ensemble" if media_type == "video" else "replicated_frame_sequence",
                "score_is_calibrated": False,
            },
            "model": {"name": "Xception-LSTM", "version": "v2", "description": "fake",
                      "checkpoint_sha256": "fake" * 8, "input_frames": 32, "input_size": 299},
            "processing": {"duration_seconds": 0.01, "warnings": (
                ["Image analysis reuses the VIDEO model by replicating one face crop "
                 "across the frame axis; it is not a separately validated image classifier."]
                if media_type == "image" else [])},
            "explainability": {"available": False, "method": None, "artifact_id": None,
                               "limitation": None},
            "windows": [
                {"label": "Full-32", "score": self.score, "frame_start": 0, "frame_end": 31,
                 "start_seconds": 0.0, "end_seconds": 2.0},
                {"label": "Win-0-9", "score": min(1.0, self.score + 0.02), "frame_start": 0,
                 "frame_end": 9, "start_seconds": 0.0, "end_seconds": 0.36},
            ] if media_type == "video" else [],
            "report_available": True,
            "status": "COMPLETED",
        })
        return analysis_id


@pytest.fixture()
def fake_app():
    store = JobStore()
    registry = FakeRegistry()
    worker = FakeWorker(store)
    get_settings.cache_clear()
    app = create_app(store=store, registry=registry, worker=worker)
    app.state.worker = worker
    return app


@pytest.fixture()
def client(fake_app):
    from fastapi.testclient import TestClient

    return TestClient(fake_app)
