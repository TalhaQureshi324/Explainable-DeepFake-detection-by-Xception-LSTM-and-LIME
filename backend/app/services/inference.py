"""
Inference service
=================
Sequence construction and scoring, preserving the deployed behaviour of
test.py (V1) and app.py (V2) exactly:

  V1  - first 10 of the extracted candidate frames -> one full-sequence score
        (videos with <10 usable frames are rejected, as in training)
  V2  - all extracted frames (up to 32, last-frame padded) -> full-sequence
        score + 10-frame sliding windows (stride 5), averaged (app.py)

Images run through a clearly-labelled compatibility mode: the single face is
replicated across the model's frame axis. This is NOT a validated image
classifier and responses say so explicitly.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.core.config import WINDOW_SIZE, STRIDE

logger = logging.getLogger("deepfake.inference")

FULL_SEQUENCE = "full_sequence"
ENSEMBLE = "sliding_window_ensemble"
REPLICATED_FRAME = "replicated_frame_sequence"


class InferenceInputError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class WindowScore:
    label: str
    score: float
    frame_start: Optional[int] = None      # index within the extracted sequence
    frame_end: Optional[int] = None
    start_seconds: Optional[float] = None  # mapped onto the source timeline
    end_seconds: Optional[float] = None


@dataclass
class PredictionResult:
    fake_score: float
    method: str
    windows: List[WindowScore] = field(default_factory=list)
    frames_used: int = 0
    warnings: List[str] = field(default_factory=list)


def _pad(frames: List[np.ndarray], target: int) -> np.ndarray:
    frames = list(frames)
    while len(frames) < target:
        frames.append(frames[-1].copy())
    return np.stack(frames).astype(np.float32) / 255.0


def _predict(model, seq: np.ndarray) -> float:
    return float(model.predict(np.expand_dims(seq, axis=0), verbose=0)[0][0])


def _timestamp(frame_index: Optional[int], fps: Optional[float]):
    if frame_index is None or not fps:
        return None
    return float(frame_index) / float(fps)


def predict_video(model, model_version: str, num_frames: int, faces: dict) -> PredictionResult:
    frames = faces["frames"]
    fps = faces.get("fps")
    indices = faces.get("frame_indices") or []

    if model_version == "v1":
        # test.py behaviour: extract up to 32 candidates, consume the first 10.
        frames_model = frames[:num_frames]
        if len(frames_model) < num_frames:
            raise InferenceInputError(
                "INSUFFICIENT_FRAMES",
                f"Only {len(frames_model)}/{num_frames} usable face frames; "
                "the 10-frame model requires at least 10.",
            )
        score = _predict(model, _pad(frames_model, num_frames))
        window = WindowScore(
            label=f"Full-{num_frames}", score=score,
            frame_start=indices[0] if indices else None,
            frame_end=indices[num_frames - 1] if len(indices) >= num_frames else None,
            start_seconds=_timestamp(indices[0] if indices else None, fps),
            end_seconds=_timestamp(indices[num_frames - 1] if len(indices) >= num_frames else None, fps),
        )
        return PredictionResult(
            fake_score=score, method=FULL_SEQUENCE, windows=[window],
            frames_used=num_frames,
            warnings=["V1 uses a single full-sequence score (no ensemble)."],
        )

    # V2 ensemble (app.py behaviour)
    normalized = [f.astype(np.float32) / 255.0 for f in frames]
    windows: List[WindowScore] = []

    full_seq = _pad(normalized, num_frames)
    full_score = _predict(model, full_seq)
    windows.append(WindowScore(
        label=f"Full-{num_frames}", score=full_score,
        frame_start=indices[0] if indices else None,
        frame_end=indices[-1] if indices else None,
        start_seconds=_timestamp(indices[0] if indices else None, fps),
        end_seconds=_timestamp(indices[-1] if indices else None, fps),
    ))

    n = len(normalized)
    if n >= WINDOW_SIZE:
        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            seg = normalized[start : start + WINDOW_SIZE]
            score = _predict(model, _pad(seg, num_frames))
            w_start = indices[start] if start < len(indices) else None
            w_end = indices[min(start + WINDOW_SIZE - 1, len(indices) - 1)] if indices else None
            windows.append(WindowScore(
                label=f"Win-{start}-{start + WINDOW_SIZE - 1}", score=score,
                frame_start=w_start, frame_end=w_end,
                start_seconds=_timestamp(w_start, fps),
                end_seconds=_timestamp(w_end, fps),
            ))
    else:
        score = _predict(model, _pad(normalized, num_frames))
        windows.append(WindowScore(
            label=f"Short-{n}", score=score,
            frame_start=indices[0] if indices else None,
            frame_end=indices[-1] if indices else None,
            start_seconds=_timestamp(indices[0] if indices else None, fps),
            end_seconds=_timestamp(indices[-1] if indices else None, fps),
        ))

    fake_score = float(np.mean([w.score for w in windows]))
    warnings = []
    if len(frames) < num_frames:
        warnings.append(
            f"Only {len(frames)} face frames extracted (model expects {num_frames}); "
            "sequence was padded by repeating the last frame, which can weaken reliability."
        )
    return PredictionResult(
        fake_score=fake_score, method=ENSEMBLE, windows=windows,
        frames_used=len(frames), warnings=warnings,
    )


def predict_image_compat(model, num_frames: int, face_rgb: np.ndarray) -> PredictionResult:
    """Compatibility mode: replicate one face crop across the frame axis."""
    seq = np.stack([face_rgb.astype(np.float32) / 255.0] * num_frames)
    score = _predict(model, seq)
    return PredictionResult(
        fake_score=score, method=REPLICATED_FRAME, windows=[],
        frames_used=1,
        warnings=[
            "Image analysis reuses the VIDEO model by replicating one face crop "
            "across the frame axis; it is not a separately validated image classifier.",
        ],
    )
