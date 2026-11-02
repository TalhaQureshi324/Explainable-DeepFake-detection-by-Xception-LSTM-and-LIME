"""API v1 routes: health, readiness, model listing, analyses."""

import logging
import os
import re
from pathlib import Path

from fastapi import Response

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import (
    SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS, Settings, get_settings,
)
from app.core.storage import JobStore
from app.schemas.analysis import (
    AnalysisCreateAccepted, AnalysisResponse, HealthResponse, ModelsResponse, ReadyResponse,
)
from app.services import model_registry as registry_module
from app.services.report_data import build_report
from app.workers.analysis_worker import AnalysisWorker

logger = logging.getLogger("deepfake.api")

MAX_EXTENSION = 8  # filename extension sanity bound


def _public_record(job: dict) -> dict:
    """Strip worker-internal keys (leading underscore) before responding."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _record_to_response(job: dict) -> AnalysisResponse:
    return AnalysisResponse(**_public_record(job))


def create_router(store: JobStore, registry: registry_module.ModelRegistry,
                  worker: AnalysisWorker) -> APIRouter:
    router = APIRouter()

    # ---- health / readiness -------------------------------------------------

    @router.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", version="v1")

    @router.get("/ready", response_model=ReadyResponse)
    def ready(settings: Settings = Depends(get_settings)):
        try:
            version = registry.resolve_version()
            return ReadyResponse(ready=True, model_version=version)
        except registry_module.ModelNotAvailableError as exc:
            return ReadyResponse(ready=False, detail=str(exc))

    # ---- models ---------------------------------------------------------

    @router.get("/v1/models", response_model=ModelsResponse)
    def list_models():
        return ModelsResponse(**registry.describe_all())

    # ---- analyses ---------------------------------------------------------

    @router.post("/v1/analyses", response_model=AnalysisCreateAccepted, status_code=202)
    async def create_analysis(
        file: UploadFile = File(...),
        model_version: str = Form(default=None),
        explain: bool = Form(default=None),
        settings: Settings = Depends(get_settings),
    ):
        if worker.queue_capacity() <= 0:
            raise HTTPException(status_code=503, detail="Server is at capacity; retry shortly.")

        filename = Path(file.filename or "").name
        ext = Path(filename).suffix.lower()
        if len(filename) > 255 or not ext or len(ext) > MAX_EXTENSION:
            raise HTTPException(status_code=400, detail="Invalid or missing filename extension.")

        if ext in SUPPORTED_VIDEO_EXTENSIONS:
            media_type = "video"
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            media_type = "image"
        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Supported: video "
                       f"{SUPPORTED_VIDEO_EXTENSIONS}, image {SUPPORTED_IMAGE_EXTENSIONS}.",
            )

        max_bytes = settings.max_upload_mb * 1024 * 1024
        size = 0
        tmp_path = Path(settings.jobs_dir) / f"incoming_{os.urandom(6).hex()}_{filename}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(tmp_path, "wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                        )
                    out.write(chunk)
        except HTTPException:
            tmp_path.unlink(missing_ok=True)
            raise
        if size == 0:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Empty upload rejected.")

        if model_version is not None:
            try:
                registry.get_entry(model_version)
            except registry_module.ModelNotAvailableError as exc:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=str(exc))

        analysis_id = worker.submit(tmp_path, media_type, filename,
                                    requested_model=model_version, request_xai=explain)
        return AnalysisCreateAccepted(
            analysis_id=analysis_id, status="QUEUED",
            poll_url=f"/v1/analyses/{analysis_id}",
        )

    @router.get("/v1/analyses/{analysis_id}", response_model=AnalysisResponse)
    def get_analysis(analysis_id: str):
        job = store.get(analysis_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return _record_to_response(job)

    @router.get("/v1/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str):
        """Serve analysis artifacts (processed-face preview / LIME PNG).

        Artifact IDs are strict hex-prefixed tokens generated server-side, so
        the path can never traverse outside the artifacts directory."""
        if not re.fullmatch(r"[A-Za-z0-9_]{4,64}", artifact_id):
            raise HTTPException(status_code=400, detail="Malformed artifact id.")
        settings = get_settings()
        candidate = (Path(settings.artifacts_dir) / f"{artifact_id}.png").resolve()
        if candidate.parent != Path(settings.artifacts_dir).resolve() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found (or expired).")
        return Response(content=candidate.read_bytes(), media_type="image/png")

    @router.get("/v1/analyses/{analysis_id}/report")
    def get_report(analysis_id: str):
        job = store.get(analysis_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        if job.get("status") == "FAILED":
            raise HTTPException(status_code=409, detail="Analysis failed; no report is available.")
        if job.get("status") != "COMPLETED":
            raise HTTPException(status_code=409, detail="Analysis is not completed yet.")
        return build_report(_public_record(job))

    return router
