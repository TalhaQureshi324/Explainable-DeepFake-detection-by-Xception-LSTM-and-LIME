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
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from .api.routes.analyses import create_router  # noqa: E402
from .api.routes.auth_history import create_router as create_auth_history_router  # noqa: E402
from .core.config import Settings, get_settings  # noqa: E402
from .core.logging import configure_logging  # noqa: E402
from .core.storage import JobStore  # noqa: E402
from .persistence import get_repository as _get_repository  # noqa: E402
import app.persistence as persistence  # noqa: E402
from .services import model_registry as registry_module  # noqa: E402
from .workers.analysis_worker import AnalysisWorker  # noqa: E402

logger = logging.getLogger("deepfake")


def create_app(settings: Settings | None = None,
               store: JobStore | None = None,
               registry: registry_module.ModelRegistry | None = None,
               worker: AnalysisWorker | None = None,
               repository=None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    store = store or JobStore()
    registry = registry or registry_module.ModelRegistry(settings)
    repository = repository or persistence.get_repository(settings)
    worker = worker or AnalysisWorker(settings, store, registry, repository=repository)

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
    app.state.repository = repository
    app.include_router(create_router(store, registry, worker, settings, repository))
    app.include_router(create_auth_history_router(repository, settings))

    # When the backend is hosted separately from the frontend (recommended for
    # the TensorFlow workload), the browser may talk to it directly for large
    # uploads. origins come from CORS_ALLOW_ORIGINS (comma-separated); no CORS
    # is configured unless the variable is set, keeping the proxied local
    # setup unchanged.
    cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    return app


app = create_app()
