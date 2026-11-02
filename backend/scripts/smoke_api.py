"""
Real-media API smoke test.

Runs the FastAPI app IN-PROCESS with the real model registry, pushes one real
validation video and one real validation image through POST /v1/analyses, and
compares the returned fake_score against an independently computed reference
(the same functions, called directly) — they must match exactly, proving the
API layer adds no numerical drift. Optionally cross-checks against the Phase 0
evaluation JSON if available.

Usage (repo root):
    "C:/Program Files/Python311/python.exe" backend/scripts/smoke_api.py \
        --videos-dir "F:/Computer Vision/FYP/data/val" \
        --models-dir models_store
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from fastapi.testclient import TestClient  # noqa: E402


def wait_for(client, analysis_id, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = client.get(f"/v1/analyses/{analysis_id}")
        status = r.json()["status"]
        if status in ("COMPLETED", "FAILED"):
            return r.json()
        time.sleep(3)
    raise TimeoutError(f"analysis {analysis_id} did not finish in {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models_store"))
    args = parser.parse_args()

    os.environ["DEEPFAKE_MODELS_DIR"] = args.models_dir
    os.environ["DEEPFAKE_JOBS_DIR"] = str(Path(tempfile.mkdtemp(prefix="df_api_smoke_")))

    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    registry = app.state.registry
    version = registry.resolve_version()
    print(f"[smoke] preferred model: {version} | threshold: "
          f"{registry.threshold_for_version(version)}")

    from fastapi.testclient import TestClient  # noqa: F811

    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    print(f"[smoke] /ready -> {ready}")

    models = client.get("/v1/models").json()
    print(f"[smoke] /v1/models -> {[m['version'] for m in models['models']]}")

    # ---- pick a small real video + a real image from the raw val folder ------
    val_dir = Path(args.videos_dir)
    videos = sorted(p for p in val_dir.rglob("*.mp4"))
    videos.sort(key=lambda p: p.stat().st_size)  # smallest first for speed
    video = videos[0]
    print(f"[smoke] video: {video.name} ({video.stat().st_size/1e6:.1f} MB)")

    with open(video, "rb") as f:
        r = client.post("/v1/analyses", files={"file": (video.name, f.read(), "video/mp4")})
    assert r.status_code == 202, r.text
    video_result = wait_for(client, r.json()["analysis_id"])
    assert video_result["status"] == "COMPLETED", video_result["processing"]
    pred = video_result["prediction"]
    print(f"[smoke] API video result: label={pred['label']} score={pred['fake_score']} "
          f"method={pred['method']} threshold={pred['threshold']} "
          f"windows={len(video_result['windows'])}")
    print(f"[smoke] media info: fps={video_result['media']['fps']} "
          f"duration={video_result['media']['duration_seconds']} "
          f"frames_extracted={video_result['media']['frames_extracted']} "
          f"sha256={str(video_result['media']['sha256'])[:12]}...")

    # ---- direct reference with the same code path --------------------------
    import shutil
    from app.services import face_extraction, inference as inference_service

    tmp = Path(tempfile.mkdtemp(prefix="df_ref_"))
    faces = face_extraction.extract_faces_from_video(str(video), tmp, max_frames=32)
    model = registry.load(version)
    num_frames = registry.get_entry(version).input_frames
    reference = inference_service.predict_video(model, version, num_frames, faces)
    delta = abs(reference.fake_score - pred["fake_score"])
    print(f"[smoke] direct-reference score={reference.fake_score:.6f} "
          f"delta_vs_api={delta:.2e}")
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- report endpoint ----------------------------------------------------
    report = client.get(f"/v1/analyses/{r.json()['analysis_id']}/report")
    assert report.status_code == 200
    rep = report.json()
    print(f"[smoke] report: assessment={rep['overview']['assessment']} "
          f"windows_in_timeline={len(rep['timeline']['windows'])} "
          f"limitations={len(rep['limitations'])}")

    ok = delta <= 1e-6  # API deliberately rounds scores to 6 decimals
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'} (api score matches direct reference: {ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
