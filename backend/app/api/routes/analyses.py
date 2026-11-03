"""API v1 routes: health, readiness, model listing, analyses."""

import logging
import os
import re
from pathlib import Path


from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, Response
from app.core.deps import get_current_user

from app.core.config import (
    SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS, Settings, get_settings,
)
from app.core.storage import JobStore
from app.schemas.analysis import (
    AnalysisCreateAccepted, AnalysisResponse, HealthResponse, ModelsResponse, ReadyResponse,
)
from app.services import model_registry as registry_module
from app.services.pdf_report import build_pdf, pdf_filename
from app.services.report_data import build_report
from app.workers.analysis_worker import AnalysisWorker

logger = logging.getLogger("deepfake.api")

MAX_EXTENSION = 8  # filename extension sanity bound


def _public_record(job: dict) -> dict:
    """Strip worker-internal keys (leading underscore) before responding."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _strip_internal(record: dict) -> dict:
    """Repository records -> API-shaped response dict."""
    record = {k: v for k, v in record.items() if not k.startswith("_")}
    record.setdefault("processing", {"warnings": []})
    record.setdefault("explainability", {"available": False})
    record.setdefault("artifacts", {"processed_face": None, "lime": None})
    record.setdefault("windows", [])
    return record


def _record_to_response(job: dict) -> AnalysisResponse:
    return AnalysisResponse(**_public_record(job))


def create_router(store: JobStore, registry: registry_module.ModelRegistry,
                  worker: AnalysisWorker, settings: Settings = None,
                  repository=None) -> APIRouter:
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
        user: dict | None = Depends(get_current_user),
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

        analysis_id = worker.submit(tmp_path, media_type, filename, user=user,
                                    requested_model=model_version, request_xai=explain)
        return AnalysisCreateAccepted(
            analysis_id=analysis_id, status="QUEUED",
            poll_url=f"/v1/analyses/{analysis_id}",
        )

    @router.get("/v1/analyses/{analysis_id}", response_model=AnalysisResponse)
    def get_analysis(analysis_id: str,
                     user: dict | None = Depends(get_current_user)):
        job = store.get(analysis_id)
        if job is not None:
            # hot cache: guests and the owner during processing; a persisted
            # analysis is only readable by its owner
            if job.get("user_id") and (not user or user["id"] != job["user_id"]):
                raise HTTPException(status_code=404, detail="Analysis not found.")
            return _record_to_response(job)
        if repository is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        if user is None:
            raise HTTPException(status_code=401,
                                detail="Sign in to access saved analyses.")
        record = repository.get_analysis(analysis_id, user["id"])
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return AnalysisResponse(**_strip_internal(record))

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

    def _source_record(analysis_id: str, user):
        """Hot cache first, then the repository (owner-only)."""
        job = store.get(analysis_id)
        if job is not None:
            if job.get("user_id") and (not user or user["id"] != job["user_id"]):
                raise HTTPException(status_code=404, detail="Analysis not found.")
            return _public_record(job)
        if user is None:
            raise HTTPException(status_code=401,
                                detail="Sign in to access saved analyses.")
        record = repository.get_analysis(analysis_id, user["id"]) if repository else None
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return _strip_internal(record)

    @router.get("/v1/analyses/{analysis_id}/report")
    def get_report(analysis_id: str,
                   user: dict | None = Depends(get_current_user)):
        job = _source_record(analysis_id, user)
        if job.get("status") == "FAILED":
            raise HTTPException(status_code=409, detail="Analysis failed; no report is available.")
        if job.get("status") != "COMPLETED":
            raise HTTPException(status_code=409, detail="Analysis is not completed yet.")
        return build_report(job)

    @router.get("/v1/analyses/{analysis_id}/report.pdf")
    def get_report_pdf(analysis_id: str,
                       user: dict | None = Depends(get_current_user)):
        """Server-side PDF rendering of the SAME structured report document the
        JSON endpoint returns. Never re-runs the model."""
        job = _source_record(analysis_id, user)
        if job.get("status") == "FAILED":
            raise HTTPException(status_code=409, detail="Analysis failed; no report is available.")
        if job.get("status") != "COMPLETED":
            raise HTTPException(status_code=409, detail="Analysis is not completed yet.")
        settings = get_settings()
        artifacts_dir = Path(settings.artifacts_dir).resolve()
        cache = artifacts_dir / f"report_{analysis_id}.pdf"
        try:
            if cache.is_file():
                data = cache.read_bytes()
            else:
                data = build_pdf(build_report(_public_record(job)), artifacts_dir)
                cache.write_bytes(data)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001 - never leak internals
            logger.exception("PDF generation failed for %s", analysis_id)
            raise HTTPException(status_code=500, detail="PDF generation failed; please retry.")
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{pdf_filename(analysis_id)}"'},
        )

    return router
