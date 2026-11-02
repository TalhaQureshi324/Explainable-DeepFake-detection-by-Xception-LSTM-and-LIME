"""
FastAPI application factory.

Run locally from the repo root:
    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

The router is split out (api/routes/analyses.py) so tests can build the app
with fakes injected instead of loading TensorFlow.
"""

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from .api.routes.analyses import create_router  # noqa: E402
from .core.config import Settings, get_settings  # noqa: E402
from .core.logging import configure_logging  # noqa: E402
from .core.storage import JobStore  # noqa: E402
from .services import model_registry as registry_module  # noqa: E402
from .workers.analysis_worker import AnalysisWorker  # noqa: E402

logger = logging.getLogger("deepfake")


def create_app(settings: Settings | None = None,
               store: JobStore | None = None,
               registry: registry_module.ModelRegistry | None = None,
               worker: AnalysisWorker | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    store = store or JobStore()
    registry = registry or registry_module.ModelRegistry(settings)
    worker = worker or AnalysisWorker(settings, store, registry)

    app = FastAPI(
        title="Deepfake Detection Inference API",
        version="1.0.0",
        description="Phase 1 inference backend. Structured analyses over the existing "
                    "Xception-LSTM checkpoints with a configurable decision threshold "
                    "(development default 0.60; historical deployment baseline 0.75).",
    )
    app.state.settings = settings
    app.state.store = store
    app.state.registry = registry
    app.state.worker = worker
    app.include_router(create_router(store, registry, worker))
    return app


app = create_app()
