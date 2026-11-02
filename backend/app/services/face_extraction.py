"""
Face extraction & preprocessing service.

Wraps the repository's validated `preprocessing.FaceExtractor` so the API uses
EXACTLY the same OpenCV DNN detection + elliptical masking pipeline as the
deployed Gradio app and the training data generation. No new computer-vision
logic is introduced here.

Frame/timestamp provenance: FaceExtractor.process_video writes metadata.json
with the original video's fps and the source frame index of every extracted
face, which we read back so window scores can be placed on the real video
timeline instead of being estimated afterwards.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("deepfake.face_extraction")

# Ensure the repository root is importable regardless of how the backend runs.
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing import FaceExtractor  # noqa: E402  (repo-root module)


class FaceExtractionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def get_extractor() -> FaceExtractor:
    return FaceExtractor()


def extract_faces_from_video(
    video_path: str,
    work_dir: Path,
    max_frames: int = 32,
    extractor: Optional[FaceExtractor] = None,
) -> dict:
    """
    Extract up to `max_frames` evenly spaced face crops from a video.

    Returns dict with:
      frames       list of RGB uint8 arrays (extraction order)
      fps          source video fps (or None if unknown)
      frame_indices source frame index of each extracted face
      total_frames total frames reported by the container
      duration_seconds source duration if fps known
      frames_extracted  number of face crops
    """
    extractor = extractor or get_extractor()
    out_dir = Path(work_dir) / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    count = extractor.process_video(str(video_path), str(out_dir), max_frames=max_frames)
    if count == 0:
        raise FaceExtractionError(
            "NO_FACE_DETECTED", "No faces were detected in the provided video."
        )

    frames = []
    for frame_file in sorted(f for f in out_dir.iterdir() if f.suffix == ".jpg"):
        img = cv2.imread(str(frame_file))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    meta = {"fps": None, "frame_indices": [], "total_frames": None}
    meta_path = out_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Could not parse %s; timestamps will be unavailable.", meta_path)

    fps = meta.get("fps")
    duration = None
    try:
        if fps and meta.get("total_frames"):
            duration = float(meta["total_frames"]) / float(fps)
    except (TypeError, ZeroDivisionError):
        duration = None

    return {
        "frames": frames,
        "fps": fps,
        "frame_indices": meta.get("frame_indices") or [],
        "total_frames": meta.get("total_frames"),
        "duration_seconds": duration,
        "frames_extracted": len(frames),
    }


def extract_face_from_image(image_path: str, extractor: Optional[FaceExtractor] = None) -> np.ndarray:
    """Single highest-confidence face crop (RGB uint8, 299x299)."""
    extractor = extractor or get_extractor()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FaceExtractionError("MEDIA_UNREADABLE", "The image could not be read.")
    faces = extractor.detect_faces(frame)
    if not faces:
        raise FaceExtractionError(
            "NO_FACE_DETECTED", "No face was detected in the provided image."
        )
    face = extractor.extract_face(frame, faces[0])
    if face is None:
        raise FaceExtractionError("NO_FACE_DETECTED", "Face crop failed for the provided image.")
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def probe_media(path: str) -> dict:
    """Read container metadata (resolution/duration/fps) without full decode."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FaceExtractionError("MEDIA_UNREADABLE", "The video could not be opened.")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps and frames else None
        return {"width": width, "height": height, "fps": fps or None,
                "total_frames": frames or None, "duration_seconds": duration}
    finally:
        cap.release()
