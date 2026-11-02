"""API contract tests (fakes injected; no TensorFlow, no real checkpoints)."""

from io import BytesIO


def _png_bytes(size=(4, 4)):
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "v1"}


def test_models_listing(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["preferred_version"] in ("v1", "v2")
    versions = {m["version"] for m in body["models"]}
    assert versions == {"v1", "v2"}
    assert all(m["decision_threshold"] == 0.60 for m in body["models"])


def test_video_analysis_round_trip(client):
    r = client.post(
        "/v1/analyses",
        files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
    )
    assert r.status_code == 202
    body = r.json()
    analysis_id = body["analysis_id"]
    assert body["status"] == "QUEUED"
    assert body["poll_url"] == f"/v1/analyses/{analysis_id}"

    r2 = client.get(f"/v1/analyses/{analysis_id}")
    assert r2.status_code == 200
    result = r2.json()
    assert result["status"] == "COMPLETED"
    assert result["prediction"]["threshold"] == 0.60
    assert result["prediction"]["threshold_version"] == "dev-v1"
    assert result["prediction"]["score_is_calibrated"] is False
    assert result["model"]["version"] == "v2"
    assert result["media"]["type"] == "video"
    assert len(result["windows"]) == 2
    assert result["report_available"] is True


def test_image_analysis_flags_compat_mode(client):
    r = client.post(
        "/v1/analyses",
        files={"file": ("photo.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 202
    result = client.get(f"/v1/analyses/{r.json()['analysis_id']}").json()
    assert result["media"]["type"] == "image"
    assert result["prediction"]["method"] == "replicated_frame_sequence"
    assert any("not a separately validated image classifier" in w
               for w in result["processing"]["warnings"])


def test_report_endpoint_matches_analysis(client):
    r = client.post("/v1/analyses", files={"file": ("clip.mp4", b"x", "video/mp4")})
    analysis_id = r.json()["analysis_id"]
    report = client.get(f"/v1/analyses/{analysis_id}/report").json()
    assert report["analysis_id"] == analysis_id
    assert report["overview"]["assessment"] == "FAKE"
    assert "threshold" in report["prediction"]
    assert len(report["limitations"]) >= 5
    assert report["timeline"]["threshold"] == 0.60
    interp = report["interpretation"]
    assert "research-grade" in interp and "definitely" not in interp


def test_unsupported_extension_rejected(client):
    r = client.post("/v1/analyses", files={"file": ("movie.exe", b"x", "application/octet-stream")})
    assert r.status_code == 415


def test_empty_upload_rejected(client):
    r = client.post("/v1/analyses", files={"file": ("clip.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_upload_size_limit(client, fake_app):
    settings = fake_app.state.settings
    settings.max_upload_mb = 0  # force the limit with a tiny value
    r = client.post("/v1/analyses", files={"file": ("clip.mp4", b"x" * (2 * 1024 * 1024), "video/mp4")})
    assert r.status_code == 413


def test_unknown_model_version_rejected(client):
    r = client.post(
        "/v1/analyses",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={"model_version": "v9"},
    )
    assert r.status_code == 400


def test_analysis_404(client):
    assert client.get("/v1/analyses/does-not-exist").status_code == 404


def test_report_404_and_409(client):
    assert client.get("/v1/analyses/does-not-exist/report").status_code == 404


def test_no_internal_path_leakage(client):
    r = client.post("/v1/analyses", files={"file": ("clip.mp4", b"x", "video/mp4")})
    analysis_id = r.json()["analysis_id"]
    body = client.get(f"/v1/analyses/{analysis_id}").content.decode()
    assert "_job_dir" not in body
    assert "_stored_path" not in body
    assert "upload_" not in body
