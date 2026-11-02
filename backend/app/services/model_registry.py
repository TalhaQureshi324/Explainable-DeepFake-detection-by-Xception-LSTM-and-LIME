"""
Model Registry
==============
Central, lazily-loaded checkpoint management.

- Registers V1/V2 checkpoints by version key from a configurable directory.
- Records SHA-256 and input-shape validation for every checkpoint.
- Loads each checkpoint ONCE and caches it in memory.
- Carries model-specific decision thresholds from central Settings.
- Raises ModelNotAvailableError with a friendly message when missing.

Adding a future checkpoint (e.g. V3) means dropping the .keras file in the
models directory and adding one registry entry — no frontend changes.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("deepfake.model_registry")


class ModelNotAvailableError(RuntimeError):
    """Raised when a requested checkpoint is missing or invalid."""


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelEntry:
    version: str                      # "v1" | "v2" | future keys
    file_path: Path
    description: str
    input_frames: Optional[int] = None        # validated at load time
    input_size: Optional[int] = None
    checkpoint_sha256: Optional[str] = None
    _model: object = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ModelRegistry:
    def __init__(self, settings):
        self.settings = settings
        self._entries = {}
        self._lock = threading.Lock()
        self._register_defaults()

    # ---- registration -------------------------------------------------

    def _register_defaults(self) -> None:
        filename_by_version = {
            "v1": "deepfake_detector_model.keras",
            "v2": "deepfake_detector_model_v2.keras",
        }
        descriptions = {
            "v1": "Xception-LSTM, 10-frame full-sequence scoring",
            "v2": "Xception-LSTM, 32-frame; deployed with sliding-window ensemble",
        }
        for version, filename in filename_by_version.items():
            path = Path(self.settings.models_dir) / filename
            if path.is_file():
                self._entries[version] = ModelEntry(
                    version=version, file_path=path, description=descriptions[version]
                )
            else:
                logger.warning("Checkpoint for %s not found at %s", version, path)

    def register(self, version: str, file_path, description: str = "") -> None:
        """Register an additional checkpoint without frontend changes."""
        path = Path(file_path)
        if not path.is_file():
            raise ModelNotAvailableError(f"Checkpoint not found: {path}")
        with self._lock:
            self._entries[version] = ModelEntry(
                version=version, file_path=path, description=description or f"checkpoint {path.name}"
            )

    # ---- introspection -------------------------------------------------

    def available_versions(self):
        return sorted(self._entries.keys())

    def get_entry(self, version: str) -> ModelEntry:
        entry = self._entries.get(version)
        if entry is None:
            raise ModelNotAvailableError(
                f"Model version '{version}' is not available. "
                f"Installed: {self.available_versions() or 'none'}."
            )
        return entry

    def resolve_version(self, requested: Optional[str] = None) -> str:
        """Pick the requested version, else the preferred, else the fallback."""
        for candidate in (requested, self.settings.preferred_model_version, self.settings.fallback_model_version):
            if candidate and candidate in self._entries:
                return candidate
        raise ModelNotAvailableError(
            "No usable checkpoint found. Install a .keras checkpoint in "
            f"{self.settings.models_dir} or set DEEPFAKE_MODELS_DIR."
        )

    def threshold_for_version(self, version: str) -> float:
        return self.settings.threshold_for(version)

    # ---- loading --------------------------------------------------------

    def load(self, version: str):
        """Return the cached keras model for `version`, loading it once."""
        entry = self.get_entry(version)
        if entry._model is not None:
            return entry._model
        with entry._lock:
            if entry._model is None:
                import tensorflow as tf  # deferred: keeps unit tests TF-free

                logger.info("Loading checkpoint %s (%s)...", entry.file_path.name, version)
                model = tf.keras.models.load_model(str(entry.file_path), compile=False)
                shape = model.input_shape
                try:
                    entry.input_frames = int(shape[1])
                    entry.input_size = int(shape[2])
                except (TypeError, ValueError, IndexError):
                    raise ModelNotAvailableError(
                        f"Checkpoint {entry.file_path.name} has unexpected input shape {shape}; "
                        "expected (frames, 299, 299, 3)."
                    )
                if entry.checkpoint_sha256 is None:
                    entry.checkpoint_sha256 = sha256_of(entry.file_path)
                entry._model = model
                logger.info(
                    "Model %s ready: frames=%d size=%d sha256=%s...",
                    version, entry.input_frames, entry.input_size, entry.checkpoint_sha256[:12],
                )
        return entry._model

    def describe(self, version: str) -> dict:
        entry = self.get_entry(version)
        return {
            "version": entry.version,
            "description": entry.description,
            "available": True,
            "file": entry.file_path.name,
            "checkpoint_sha256": entry.checkpoint_sha256,
            "input_frames": entry.input_frames,
            "input_size": entry.input_size,
            "decision_threshold": self.threshold_for_version(entry.version),
            "loaded": entry._model is not None,
        }

    def describe_all(self) -> dict:
        return {
            "preferred_version": self.resolve_version() if self._entries else None,
            "models": [self.describe(v) for v in self.available_versions()],
        }
