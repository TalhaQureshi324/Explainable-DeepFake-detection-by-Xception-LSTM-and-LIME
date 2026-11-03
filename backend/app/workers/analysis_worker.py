"""
Analysis worker (Phase 4): durable job processing on top of the repository.

Two execution paths behind one service:

- **Authenticated analyses (persistent)**: the record is written to the
  repository BEFORE inference and a job is enqueued in the durable queue.
  Dispatcher threads claim jobs atomically (lease + SKIP LOCKED in Supabase
  mode), renew the lease while running, update the database as processing
  progresses, and classify failures: recoverable failures re-queue up to
  max_attempts, permanent failures (no face, unreadable media, insufficient
  frames) mark the job DEAD. Stalled jobs (worker died mid-run) are reclaimed
  on startup and by a periodic sweep. Uploaded media lives under
  jobs/uploads/<id>/ until SUCCESS (retention policy: delete after success).
- **Guest analyses (ephemeral)**: exactly the Phase 1-3 in-memory behaviour.
  Guests get real inference but nothing is persisted (no history, no DB rows).

In both cases the TensorFlow inference itself is unchanged. Only states
actually implemented here are emitted; explainability failure is isolated from
the base prediction.
"""

import hashlib
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from app.core.config import Settings
from app.core.storage import JobStore
from app.services import face_extraction
from app.services import inference as inference_service
from app.services import model_registry as registry_module
from app.services.explainability import (
    XAI_LIMITATION, XAI_METHOD, generate_lime_heatmap, lime_available,
)
from app.services.face_extraction import FaceExtractionError
from app.services.inference import InferenceInputError

logger = logging.getLogger("deepfake.worker")

CLAIM_LEASE_SECONDS = 300
RENEW_EVERY = 60
SWEEP_EVERY = 3600
UPLOAD_GRACE_SECONDS = 24 * 3600

PERMANENT_ERROR_CODES = {"NO_FACE_DETECTED", "INSUFFICIENT_FRAMES", "MEDIA_UNREADABLE"}


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
    def __init__(self, settings: Settings, store: JobStore,
                 registry: registry_module.ModelRegistry, repository=None):
        self.settings = settings
        self.store = store
        self.registry = registry
        self.repository = repository
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self._pool = ThreadPoolExecutor(max_workers=max(1, settings.max_workers),
                                        thread_name_prefix="analysis")
        self._semaphore = threading.BoundedSemaphore(max(1, settings.max_workers))
        self._stop = threading.Event()

        if repository is not None:
            reclaimed = repository.reclaim_stalled_jobs()
            if reclaimed:
                logger.info("Reclaimed %d stalled job(s) on startup", reclaimed)
            for _ in range(max(1, settings.max_workers)):
                threading.Thread(target=self._dispatcher_loop, daemon=True,
                                 name="job-dispatcher").start()
            threading.Thread(target=self._sweep_loop, daemon=True,
                             name="retention-sweeper").start()

    # ------------------------------------------------------------------ submit

    def submit(self, media_path: Path, media_type: str, original_filename: str,
               user: Optional[dict] = None, requested_model: Optional[str] = None,
               request_xai: Optional[bool] = None) -> str:
        analysis_id = str(uuid.uuid4())
        if user:
            # ---- persistent path: durable record + queued job, THEN inference
            uploads_dir = Path(self.settings.jobs_dir) / "uploads" / analysis_id
            uploads_dir.mkdir(parents=True, exist_ok=True)
            stored_path = uploads_dir / ("upload_" + original_filename)
            shutil.move(str(media_path), stored_path)
            record = {
                "analysis_id": analysis_id,
                "user_id": user["id"],
                "status": "QUEUED",
                "media": {"type": media_type, "filename": original_filename,
                          "size_bytes": stored_path.stat().st_size},
                "requested_model": requested_model,
                "request_xai": self.settings.enable_xai_default if request_xai is None else request_xai,
                "created_at": self._now(),
                "updated_at": self._now(),
            }
            self.repository.create_analysis(record)
            self.repository.enqueue_job(analysis_id)
            self.store.create(analysis_id, record)  # hot cache for status polling
            return analysis_id

        # ---- guest path: ephemeral, in-memory (Phase 1-3 behaviour)
        job_dir = Path(self.settings.jobs_dir) / analysis_id
        job_dir.mkdir(parents=True, exist_ok=True)
        stored_path = job_dir / ("upload_" + original_filename)
        shutil.move(str(media_path), stored_path)
        self.store.create(analysis_id, {
            "media": {"type": media_type, "filename": original_filename,
                      "size_bytes": stored_path.stat().st_size},
            "requested_model": requested_model,
            "request_xai": self.settings.enable_xai_default if request_xai is None else request_xai,
            "_job_dir": str(job_dir),
            "_stored_path": str(stored_path),
        })
        self._pool.submit(self._run_guest, analysis_id)
        return analysis_id

    def queue_capacity(self) -> int:
        return max(0, self.settings.max_workers - self.store.count_by_status(("QUEUED",)))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _strip(record: dict) -> dict:
        return {k: v for k, v in (record or {}).items() if not k.startswith("_")}

    # ------------------------------------------------------- persistent path

    def _dispatcher_loop(self) -> None:
        last_reclaim = 0.0
        while not self._stop.is_set():
            try:
                job = self.repository.claim_job(self._worker_id, CLAIM_LEASE_SECONDS)
                if not job:
                    # periodic stall recovery: a worker that died mid-run leaves a
                    # leased job behind; once the lease expires it must re-queue
                    if time.time() - last_reclaim > 60:
                        reclaimed = self.repository.reclaim_stalled_jobs()
                        if reclaimed:
                            logger.info("Reclaimed %d stalled job(s)", reclaimed)
                        last_reclaim = time.time()
                    self._stop.wait(2.0)
                    continue
                self._pool.submit(self._run_persistent, job)
            except Exception:  # noqa: BLE001
                logger.exception("Dispatcher loop error")
                self._stop.wait(2.0)

    def _run_persistent(self, job: dict) -> None:
        analysis_id = job["analysis_id"]
        stop_renew = threading.Event()

        def _renew():
            while not stop_renew.wait(RENEW_EVERY):
                try:
                    self.repository.renew_lease(job["job_id"], self._worker_id,
                                                CLAIM_LEASE_SECONDS)
                except Exception:  # noqa: BLE001
                    logger.warning("Lease renewal failed for %s", analysis_id)

        renewer = threading.Thread(target=_renew, daemon=True)
        renewer.start()
        try:
            self._run_with_guards(job)
        finally:
            stop_renew.set()

    def _run_with_guards(self, job: dict):
        analysis_id = job["analysis_id"]
        if not self._semaphore.acquire(blocking=False):
            # worker busy: give the lease back; the job re-queues after a delay
            self.repository.finish_job(job["job_id"], "QUEUED", "recoverable", "worker busy")
            return
        try:
            self._execute(analysis_id, self._load_job(analysis_id), persistent=True,
                          job_ref=job)
        except FaceExtractionError as exc:
            self._fail(analysis_id, exc.code, str(exc), True, job)
        except InferenceInputError as exc:
            self._fail(analysis_id, exc.code, str(exc), True, job)
        except registry_module.ModelNotAvailableError as exc:
            self._fail(analysis_id, "MODEL_UNAVAILABLE", str(exc), True, job)
        except Exception:  # noqa: BLE001 - final backstop
            logger.exception("Analysis %s crashed", analysis_id)
            self._fail(analysis_id, "INTERNAL_ERROR",
                       "Internal error while processing the analysis.", True, job)
        finally:
            self._semaphore.release()

    def _load_job(self, analysis_id: str) -> dict:
        """Rebuild the working context from the store cache or the repository."""
        record = self.store.get(analysis_id)
        if record is None:
            record = self.repository.get_analysis_system(analysis_id)
            if record is None:
                raise RuntimeError(f"Analysis {analysis_id} vanished from the repository")
            # re-register in the hot store: status polling and _execute's store
            # updates depend on it after a restart
            self.store.create(analysis_id, self._strip(record))
            record = self.store.get(analysis_id)
        filename = (record.get("media") or {}).get("filename", "upload")
        uploads_dir = Path(self.settings.jobs_dir) / "uploads" / analysis_id
        record["_stored_path"] = str(uploads_dir / ("upload_" + filename))
        record["_job_dir"] = str(Path(self.settings.jobs_dir) / analysis_id)
        record["_persistent_media"] = True
        return record

    def _sync(self, analysis_id: str):
        """Push the current in-memory record to the repository."""
        record = self.store.get(analysis_id)
        if record:
            try:
                self.repository.update_analysis(analysis_id, self._strip(record))
            except Exception:  # noqa: BLE001
                logger.exception("Repository sync failed for %s", analysis_id)

    def _fail(self, analysis_id: str, code: str, message: str, persistent: bool,
              job_ref: Optional[dict] = None):
        logger.info("Analysis %s FAILED [%s]: %s", analysis_id, code, message)
        self.store.update(analysis_id, status="FAILED")
        current = self.store.get(analysis_id) or {}
        self.store.update(analysis_id, processing={
            "error": message, "error_code": code,
            "warnings": (current.get("processing") or {}).get("warnings", []),
        })
        if persistent and job_ref:
            self._sync(analysis_id)
            attempts = job_ref.get("attempts", 1)
            if code in PERMANENT_ERROR_CODES or attempts >= job_ref.get("max_attempts", 3):
                self.repository.finish_job(job_ref["job_id"], "DEAD", "permanent",
                                           f"{code}: {message}")
            else:
                # recoverable: analysis goes back to QUEUED for the retry
                self.store.update(analysis_id, status="QUEUED")
                self._sync(analysis_id)
                self.repository.finish_job(job_ref["job_id"], "QUEUED", "recoverable",
                                           f"{code}: {message}")
        else:
            self._cleanup_guest(analysis_id)

    def _cleanup_guest(self, analysis_id: str):
        job = self.store.get(analysis_id) or {}
        job_dir = job.get("_job_dir")
        if job_dir and os.environ.get("DEEPFAKE_KEEP_JOB_DIRS") != "1":
            shutil.rmtree(job_dir, ignore_errors=True)

    # ------------------------------------------------------------- guest path

    def _run_guest(self, analysis_id: str) -> None:
        job = self.store.get(analysis_id)
        if job is None:
            return
        if not self._semaphore.acquire(blocking=False):
            self._fail(analysis_id, "SERVER_BUSY", "The server is busy; retry later.", False)
            return
        try:
            self._execute(analysis_id, job, persistent=False)
        except FaceExtractionError as exc:
            self._fail(analysis_id, exc.code, str(exc), False)
        except InferenceInputError as exc:
            self._fail(analysis_id, exc.code, str(exc), False)
        except registry_module.ModelNotAvailableError as exc:
            self._fail(analysis_id, "MODEL_UNAVAILABLE", str(exc), False)
        except Exception:  # noqa: BLE001 - final backstop
            logger.exception("Analysis %s crashed", analysis_id)
            self._fail(analysis_id, "INTERNAL_ERROR",
                       "Internal error while processing the analysis.", False)
        finally:
            self._semaphore.release()

    # ------------------------------------------------------------ core logic

    def _execute(self, analysis_id: str, job: dict, persistent: bool,
                 job_ref: Optional[dict] = None) -> None:
        started = time.time()
        media_type = job["media"]["type"]
        stored_path = Path(job["_stored_path"])
        job_dir = Path(job["_job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)

        # ---- PREPROCESSING ----
        self.store.update(analysis_id, status="PREPROCESSING")
        if persistent:
            self._sync(analysis_id)

        version = self.registry.resolve_version(job.get("requested_model"))
        entry = self.registry.get_entry(version)
        if entry.input_frames is None:
            self.registry.load(version)

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
        if persistent:
            self._sync(analysis_id)

        # ---- INFERENCING ----
        self.store.update(analysis_id, status="INFERENCING")
        if persistent:
            self._sync(analysis_id)
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
        if persistent:
            self._sync(analysis_id)

        # ---- face preview artifact (aligned with the LIME target frame) ----
        artifacts_dir = Path(self.settings.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        preview_id = f"face_{analysis_id.replace('-', '')[:20]}"
        try:
            padded = list(faces["frames"])
            while len(padded) < num_frames:
                padded.append(padded[-1].copy())
            middle = padded[num_frames // 2]
            Image.fromarray(middle).save(artifacts_dir / f"{preview_id}.png")
        except Exception as exc:  # noqa: BLE001
            preview_id = None
            logger.warning("Face preview artifact failed for %s: %s", analysis_id, exc)

        # ---- EXPLAINING (optional, failure-isolated) ----
        request_xai = bool(job.get("request_xai"))
        explainability = {"available": False, "method": None, "artifact_id": None,
                          "limitation": XAI_LIMITATION if request_xai else None}
        if request_xai:
            self.store.update(analysis_id, status="EXPLAINING")
            if persistent:
                self._sync(analysis_id)
            if not lime_available():
                self.store.push_warning(
                    analysis_id,
                    "Explainability requested but lime/scikit-image are not installed server-side.")
            else:
                try:
                    artifact = generate_lime_heatmap(model, num_frames, faces["frames"],
                                                     artifacts_dir)
                    explainability = {
                        "available": True, "method": XAI_METHOD,
                        "artifact_id": artifact["artifact_id"],
                        "limitation": XAI_LIMITATION,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.warning("XAI failed for %s: %s", analysis_id, exc)
                    self.store.push_warning(
                        analysis_id,
                        "Explainability generation failed; the base prediction is unaffected.")

        artifacts_block = {
            "processed_face": f"/v1/artifacts/{preview_id}" if preview_id else None,
            "lime": (f"/v1/artifacts/{explainability['artifact_id']}"
                     if explainability.get("artifact_id") else None),
        }

        # ---- REPORT ----
        self.store.update(analysis_id, status="GENERATING_REPORT")
        if persistent:
            self._sync(analysis_id)
        self.store.update(analysis_id,
                          explainability=explainability,
                          artifacts=artifacts_block,
                          processing={
                              "duration_seconds": round(time.time() - started, 3),
                              "warnings": (self.store.get(analysis_id)["processing"]
                                           .get("warnings", [])),
                          },
                          report_available=True,
                          status="COMPLETED",
                          completed_at=self._now())
        if persistent:
            self._sync(analysis_id)
            if job_ref:
                self.repository.finish_job(job_ref["job_id"], "COMPLETED")
            # retention policy: raw media is deleted after successful processing
            shutil.rmtree(Path(self.settings.jobs_dir) / "uploads" / analysis_id,
                          ignore_errors=True)
        logger.info("Analysis %s COMPLETED (%s=%.4f, thr=%.2f)",
                    analysis_id, version, result.fake_score, threshold)
        if not persistent:
            self._cleanup_guest(analysis_id)

    # ------------------------------------------------------------- retention

    def _sweep_loop(self) -> None:
        while not self._stop.wait(SWEEP_EVERY):
            try:
                self.sweep_now()
            except Exception:  # noqa: BLE001
                logger.exception("Retention sweep failed")

    def sweep_now(self) -> dict:
        """Failed/abandoned upload cleanup + configurable retention (4C)."""
        removed = {"upload_dirs": 0, "expired_analyses": 0}
        uploads_root = Path(self.settings.jobs_dir) / "uploads"
        if uploads_root.is_dir():
            now_ts = time.time()
            for entry in uploads_root.iterdir():
                if not entry.is_dir():
                    continue
                analysis_id = entry.name
                rec = self.store.get(analysis_id)
                if rec and rec.get("status") in ("QUEUED", "PREPROCESSING", "INFERENCING",
                                                 "EXPLAINING", "GENERATING_REPORT"):
                    continue  # active job keeps its media
                if now_ts - entry.stat().st_mtime > UPLOAD_GRACE_SECONDS:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed["upload_dirs"] += 1
        if self.settings.retention_days and self.repository is not None:
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(days=self.settings.retention_days)).isoformat()
            for analysis_id, paths in self.repository.purge_expired(cutoff):
                removed["expired_analyses"] += 1
                for rel in paths:
                    name = rel.rsplit("/", 1)[-1]
                    (Path(self.settings.artifacts_dir) / f"{name}.png").unlink(missing_ok=True)
                shutil.rmtree(Path(self.settings.jobs_dir) / "uploads" / analysis_id,
                              ignore_errors=True)
                (Path(self.settings.artifacts_dir) / f"report_{analysis_id}.pdf").unlink(missing_ok=True)
        return removed

    def shutdown(self) -> None:
        self._stop.set()

