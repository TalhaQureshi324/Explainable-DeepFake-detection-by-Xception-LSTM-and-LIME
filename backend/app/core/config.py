"""
Central configuration.

Everything is environment-driven; no machine-specific paths are hardcoded.
Defaults resolve relative to the repository root so the backend runs from a
fresh clone once checkpoints are placed / env vars point at them.

Key environment variables:
  DEEPFAKE_MODELS_DIR        directory containing *.keras checkpoints
  DEEPFAKE_MODEL_VERSION     preferred model version key (default: v2, fallback v1)
  DEEPFAKE_THRESHOLD         default decision threshold (default: 0.60, dev setting)
  DEEPFAKE_THRESHOLD_V1      optional model-specific threshold override
  DEEPFAKE_THRESHOLD_V2      optional model-specific threshold override
  DEEPFAKE_THRESHOLD_VERSION label recorded with every prediction (default: dev-v1)
  DEEPFAKE_MAX_UPLOAD_MB     max accepted upload size (default: 200)
  DEEPFAKE_JOBS_DIR          temp job workspace (default: <repo>/backend/.jobs)
  DEEPFAKE_ENABLE_XAI        default for LIME explainability (default: false)
  DEEPFAKE_MAX_WORKERS       concurrent analysis jobs (default: 1)

Historical note: the deployed Gradio app used 0.75. 0.60 is the experimental
development threshold agreed for Phase 1 (docs/PHASE_PROGRESS.md); evaluation
results in evaluation/ keep 0.75 as the historical baseline.
"""

import os
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

WINDOW_SIZE = 10
STRIDE = 5

# Job states actually implemented by the backend worker.
JOB_STATES = [
    "QUEUED",
    "PREPROCESSING",
    "INFERENCING",
    "EXPLAINING",
    "GENERATING_REPORT",
    "COMPLETED",
    "FAILED",
]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("DEEPFAKE_MODELS_DIR", REPO_ROOT / "models_store")))
    preferred_model_version: str = os.environ.get("DEEPFAKE_MODEL_VERSION", "v2")
    fallback_model_version: str = os.environ.get("DEEPFAKE_FALLBACK_MODEL_VERSION", "v1")

    # Central decision threshold. 0.60 is the experimental development default;
    # model-specific overrides win when provided.
    default_threshold: float = field(default_factory=lambda: _env_float("DEEPFAKE_THRESHOLD", 0.60))
    threshold_version: str = os.environ.get("DEEPFAKE_THRESHOLD_VERSION", "dev-v1")
    model_threshold_overrides: dict = field(default_factory=lambda: {
        "v1": os.environ.get("DEEPFAKE_THRESHOLD_V1"),
        "v2": os.environ.get("DEEPFAKE_THRESHOLD_V2"),
    })

    max_upload_mb: int = field(default_factory=lambda: _env_int("DEEPFAKE_MAX_UPLOAD_MB", 200))
    jobs_dir: Path = field(default_factory=lambda: Path(os.environ.get("DEEPFAKE_JOBS_DIR", REPO_ROOT / "backend" / ".jobs")))
    enable_xai_default: bool = field(default_factory=lambda: _env_bool("DEEPFAKE_ENABLE_XAI", False))
    max_workers: int = field(default_factory=lambda: _env_int("DEEPFAKE_MAX_WORKERS", 1))

    # Replicated-frame image inference is a compatibility mode (video model on
    # a single frame), never presented as a validated image classifier.
    image_inference_mode: str = "replicated_frame_sequence"

    def threshold_for(self, model_version: str) -> float:
        override = self.model_threshold_overrides.get(model_version)
        if override:
            try:
                return float(override)
            except (TypeError, ValueError):
                pass
        return self.default_threshold


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
