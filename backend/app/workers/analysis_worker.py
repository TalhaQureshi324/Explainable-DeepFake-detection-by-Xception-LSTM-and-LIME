"""
Analysis worker
===============
Runs analyses on a bounded thread pool with the job-state machine:

  QUEUED -> PREPROCESSING -> INFERENCING -> EXPLAINING -> GENERATING_REPORT
         -> COMPLETED | FAILED

Design notes:
- Only states actually implemented here are emitted.
- Media hashing (SHA-256) happens during PREPROCESSING.
- Explainability failure is isolated: the base prediction still completes.
- Temporary files live under the job's own directory and are deleted when the
  job finishes (unless DEEPFAKE_KEEP_JOB_DIRS=1 for debugging).
- Concurrency is bounded by the pool size (Settings.max_workers); the upload
  route rejects additional work with 503 when the queue is saturated.
"""

import hashlib
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from app.core.config import Settings
from app.core.storage import JobStore
from app.services import face_extraction
from app.services import inference as inference_service
from app.services import model_registry as registry_module
from app.services import report_data
from app.services.explainability import XAI_LIMITATION, XAI_METHOD, generate_lime_heatmap, lime_available
from app.services.face_extraction import FaceExtractionError

logger = logging.getLogger("deepfake.worker")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class AnalysisWorker:
    def __init__(self, settings: Settings, store: JobStore, registry: registry_module.ModelRegistry):
        self.settings = settings
        self.store = store
        self.registry = registry
        self._pool = ThreadPoolExecutor(max_workers=max(1, settings.max_workers),
                                        thread_name_prefix="analysis")
        self._semaphore = threading.BoundedSemaphore(max(1, settings.max_workers))

    def submit(self, media_path: Path, media_type: str, original_filename: str,
               requested_model: Optional[str] = None, request_xai: Optional[bool] = None) -> str:
        analysis_id = str(uuid.uuid4())
        job_dir = Path(self.settings.jobs_dir) / analysis_id
        job_dir.mkdir(parents=True, exist_ok=True)
        stored_path = job_dir / ("upload_" + original_filename)
        shutil.move(str(media_path), stored_path)

        self.store.create(analysis_id, {
            "media": {
                "type": media_type,
                "filename": original_filename,
                "size_bytes": stored_path.stat().st_size,
            },
            "requested_model": requested_model,
            "request_xai": self.settings.enable_xai_default if request_xai is None else request_xai,
            "_job_dir": str(job_dir),
            "_stored_path": str(stored_path),
        })
        self._pool.submit(self._run, analysis_id)
        return analysis_id

    def queue_capacity(self) -> int:
        return max(0, self.settings.max_workers - self.store.count_by_status(("QUEUED",)))

    # ------------------------------------------------------------------

    def _fail(self, analysis_id: str, code: str, message: str) -> None:
        logger.info("Analysis %s FAILED [%s]: %s", analysis_id, code, message)
        self.store.update(analysis_id, status="FAILED")
        self.store.update(analysis_id, processing={
            "error": message, "error_code": code,
            "warnings": (self.store.get(analysis_id) or {}).get("processing", {}).get("warnings", []),
        })
        self._cleanup(analysis_id)

    def _cleanup(self, analysis_id: str) -> None:
        job = self.store.get(analysis_id) or {}
        job_dir = job.get("_job_dir")
        if job_dir and os.environ.get("DEEPFAKE_KEEP_JOB_DIRS") != "1":
            shutil.rmtree(job_dir, ignore_errors=True)

    def _run(self, analysis_id: str) -> None:
        job = self.store.get(analysis_id)
        if job is None:
            return
        if not self._semaphore.acquire(blocking=False):
            self._fail(analysis_id, "SERVER_BUSY", "The server is busy; retry later.")
            return
        started = time.time()
        try:
            self._execute(analysis_id, job)
        except FaceExtractionError as exc:
            self._fail(analysis_id, exc.code, str(exc))
        except inference_service.InferenceInputError as exc:
            self._fail(analysis_id, exc.code, str(exc))
        except registry_module.ModelNotAvailableError as exc:
            self._fail(analysis_id, "MODEL_UNAVAILABLE", str(exc))
        except Exception as exc:  # noqa: BLE001 - final backstop
            logger.exception("Analysis %s crashed", analysis_id)
            self._fail(analysis_id, "INTERNAL_ERROR", "Internal error while processing the analysis.")
        finally:
            self._semaphore.release()

    def _execute(self, analysis_id: str, job: dict) -> None:
        started = time.time()
        media_type = job["media"]["type"]
        stored_path = Path(job["_stored_path"])
        job_dir = Path(job["_job_dir"])

        # ---- PREPROCESSING ------------------------------------------
        self.store.update(analysis_id, status="PREPROCESSING")
        version = self.registry.resolve_version(job.get("requested_model"))
        entry = self.registry.get_entry(version)
        if entry.input_frames is None:
            self.registry.load(version)  # validates shape + fills entry metadata

        if media_type == "video":
            probe = face_extraction.probe_media(str(stored_path))
            num_frames = entry.input_frames
            faces = face_extraction.extract_faces_from_video(
                str(stored_path), job_dir, max_frames=max(num_frames, 32)
            )
            self.store.update(analysis_id, media={
                **job["media"],
                "sha256": sha256_file(stored_path),
                "width": probe.get("width"), "height": probe.get("height"),
                "fps": faces.get("fps") or probe.get("fps"),
                "duration_seconds": faces.get("duration_seconds") or probe.get("duration_seconds"),
                "frames_extracted": faces["frames_extracted"],
                "valid_face_detections": faces["frames_extracted"],
            })
        else:
            face = face_extraction.extract_face_from_image(str(stored_path))
            faces = {"frames": [face], "fps": None, "frame_indices": [],
                     "duration_seconds": None, "frames_extracted": 1}
            self.store.update(analysis_id, media={
                **job["media"],
                "sha256": sha256_file(stored_path),
                "frames_extracted": 1,
                "valid_face_detections": 1,
            })

        # ---- INFERENCING ---------------------------------------------
        self.store.update(analysis_id, status="INFERENCING")
        model = self.registry.load(version)
        num_frames = entry.input_frames

        if media_type == "video":
            result = inference_service.predict_video(model, version, num_frames, faces)
        else:
            result = inference_service.predict_image_compat(model, num_frames, faces["frames"][0])

        threshold = self.registry.threshold_for_version(version)
        label = "FAKE" if result.fake_score > threshold else "REAL"
        self.store.update(analysis_id,
                          prediction={
                              "label": label,
                              "fake_score": round(result.fake_score, 6),
                              "threshold": threshold,
                              "threshold_version": self.settings.threshold_version,
                              "method": result.method,
                              "score_is_calibrated": False,
                          },
                          model={
                              "name": "Xception-LSTM",
                              "version": version,
                              "description": entry.description,
                              "checkpoint_sha256": entry.checkpoint_sha256,
                              "input_frames": entry.input_frames,
                              "input_size": entry.input_size,
                          },
                          windows=[{
                              "label": w.label, "score": round(w.score, 6),
                              "frame_start": w.frame_start, "frame_end": w.frame_end,
                              "start_seconds": w.start_seconds, "end_seconds": w.end_seconds,
                          } for w in result.windows])
        for warning in result.warnings:
            self.store.push_warning(analysis_id, warning)

        # ---- EXPLAINING (optional, failure-isolated) ------------------
        request_xai = bool(job.get("request_xai"))
        explainability = {"available": False, "method": None, "artifact_id": None,
                          "limitation": XAI_LIMITATION if request_xai else None}
        if request_xai:
            self.store.update(analysis_id, status="EXPLAINING")
            if not lime_available():
                self.store.push_warning(
                    analysis_id, "Explainability requested but lime/scikit-image are not installed server-side."
                )
            else:
                try:
                    artifact = generate_lime_heatmap(
                        model, num_frames, faces["frames"], job_dir,
                    )
                    explainability = {
                        "available": True, "method": XAI_METHOD,
                        "artifact_id": artifact["artifact_id"],
                        "limitation": XAI_LIMITATION,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.warning("XAI failed for %s: %s", analysis_id, exc)
                    self.store.push_warning(
                        analysis_id,
                        "Explainability generation failed; the base prediction is unaffected."
                    )
        self.store.update(analysis_id, explainability=explainability)

        # ---- REPORT ----------------------------------------------------
        self.store.update(analysis_id, status="GENERATING_REPORT")
        self.store.update(analysis_id,
                          processing={
                              "duration_seconds": round(time.time() - started, 3),
                              "warnings": self.store.get(analysis_id)["processing"].get("warnings", []),
                          },
                          report_available=True,
                          status="COMPLETED")
        logger.info("Analysis %s COMPLETED (%s=%.4f, thr=%.2f)",
                    analysis_id, version, result.fake_score, threshold)
        self._cleanup(analysis_id)
