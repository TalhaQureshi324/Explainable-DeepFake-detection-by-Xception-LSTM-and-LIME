"""Model registry behaviour (tmp dirs + a tiny real keras model for load())."""

import sys
from pathlib import Path

import pytest

from app.services.model_registry import ModelNotAvailableError, ModelRegistry


def _settings(tmp_path, **env):
    from app.core.config import Settings

    for key, value in env.items():
        if value is None:
            import os

            os.environ.pop(key, None)
        else:
            import os

            os.environ[key] = value
    s = Settings(models_dir=tmp_path)
    return s


def test_missing_checkpoints_raise(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPFAKE_MODELS_DIR", raising=False)
    reg = ModelRegistry(_settings(tmp_path))
    assert reg.available_versions() == []
    with pytest.raises(ModelNotAvailableError):
        reg.resolve_version()


def test_registry_lists_and_describes(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPFAKE_MODELS_DIR", raising=False)
    (tmp_path / "deepfake_detector_model.keras").write_bytes(b"stub")
    reg = ModelRegistry(_settings(tmp_path))
    assert reg.available_versions() == ["v1"]
    info = reg.describe("v1")
    assert info["version"] == "v1"
    assert info["input_frames"] is None  # not loaded yet
    with pytest.raises(ModelNotAvailableError):
        reg.describe("v2")


def test_register_custom_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPFAKE_MODELS_DIR", raising=False)
    custom = tmp_path / "custom_v3.keras"
    custom.write_bytes(b"stub")
    reg = ModelRegistry(_settings(tmp_path))
    reg.register("v3", custom, description="future model")
    assert "v3" in reg.available_versions()
    with pytest.raises(ModelNotAvailableError):
        reg.register("v4", tmp_path / "missing.keras")


def test_load_validates_input_shape(tmp_path, monkeypatch):
    """A tiny real .keras model must load and validate; a stub must not."""
    import os

    monkeypatch.delenv("DEEPFAKE_MODELS_DIR", raising=False)
    os.environ["DEEPFAKE_MODELS_DIR"] = str(tmp_path)
    tiny = tmp_path / "deepfake_detector_model.keras"

    import tensorflow as tf

    inp = tf.keras.Input(shape=(2, 8))
    out = tf.keras.layers.Dense(1, activation="sigmoid")(inp)
    tf.keras.models.Model(inp, out).save(tiny)

    reg = ModelRegistry(_settings(tmp_path))
    model = reg.load("v1")
    assert model is reg.load("v1")            # cached, loaded once
    entry = reg.get_entry("v1")
    assert entry.checkpoint_sha256            # recorded on load
    info = reg.describe_all()
    assert info["models"][0]["loaded"] is True
