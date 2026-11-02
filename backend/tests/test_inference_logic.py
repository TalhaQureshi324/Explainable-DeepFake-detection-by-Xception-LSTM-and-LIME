"""Inference service behaviour with a stubbed model (no TensorFlow)."""

import numpy as np
import pytest

from app.services.inference import (
    ENSEMBLE, FULL_SEQUENCE, REPLICATED_FRAME, InferenceInputError,
    predict_image_compat, predict_video,
)


class FakeModel:
    """predict() returns a deterministic function of the input mean."""

    def predict(self, x, verbose=0):
        score = float(np.clip(x.mean() * 2.0, 0.0, 1.0))
        return np.array([[score]])


def _faces(n_frames=32, fps=25.0):
    rng = np.random.default_rng(0)
    return {
        "frames": [rng.integers(0, 255, (299, 299, 3), dtype=np.uint8) for _ in range(n_frames)],
        "fps": fps,
        "frame_indices": list(range(0, n_frames * 3, 3)),
        "duration_seconds": n_frames * 3 / fps,
        "frames_extracted": n_frames,
    }


def test_v1_rejects_insufficient_frames():
    faces = _faces(n_frames=7)
    with pytest.raises(InferenceInputError) as exc:
        predict_video(FakeModel(), "v1", 10, faces)
    assert exc.value.code == "INSUFFICIENT_FRAMES"


def test_v1_full_sequence_scoring():
    result = predict_video(FakeModel(), "v1", 10, _faces())
    assert result.method == FULL_SEQUENCE
    assert len(result.windows) == 1
    assert 0.0 <= result.fake_score <= 1.0


def test_v2_ensemble_mean_matches_windows():
    result = predict_video(FakeModel(), "v2", 32, _faces(n_frames=32))
    assert result.method == ENSEMBLE
    # full + windows over 32 frames stride 5: starts 0,5,10,15,20 -> 6 predictions
    assert len(result.windows) == 6
    expected = float(np.mean([w.score for w in result.windows]))
    assert abs(expected - result.fake_score) < 1e-6


def test_v2_short_video_gets_padded_with_warning():
    result = predict_video(FakeModel(), "v2", 32, _faces(n_frames=8))
    assert result.method == ENSEMBLE
    assert any("padded" in w for w in result.warnings)


def test_window_timestamps_use_real_fps_and_indices():
    faces = _faces(n_frames=32, fps=25.0)
    result = predict_video(FakeModel(), "v2", 32, faces)
    win = result.windows[1]  # Win-0-9
    # frame index 9 in extraction order maps to source frame 27 -> 27/25 s
    assert win.end_seconds == pytest.approx(27 / 25.0)


def test_image_compat_is_labelled_and_warns():
    face = np.full((299, 299, 3), 128, dtype=np.uint8)
    result = predict_image_compat(FakeModel(), 32, face)
    assert result.method == REPLICATED_FRAME
    assert result.frames_used == 1
    assert any("not a separately validated image classifier" in w for w in result.warnings)


def test_score_bounds_and_label_boundary():
    face = np.full((299, 299, 3), 255, dtype=np.uint8)   # mean*2 clips to 1.0
    result = predict_image_compat(FakeModel(), 32, face)
    assert result.fake_score == 1.0
