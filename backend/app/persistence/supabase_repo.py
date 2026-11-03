"""
Supabase repository — PostgREST + Supabase Storage + Supabase Auth
(Phase 4A/4B/4C). Activated when SUPABASE_URL and SUPABASE_SERVICE_KEY are set.
Service-role key is used ONLY server-side (worker bookkeeping bypasses RLS);
user-facing ownership is enforced by explicit user_id filters here and by the
RLS policies in supabase/migrations/0002_rls.sql.
"""

import logging
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger("deepfake.persistence.supabase")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DuplicateEmail(Exception):
    pass


class SupabaseRepository:
    def __init__(self, url: str, service_key: str, anon_key: str | None = None):
        self.url = url.rstrip("/")
        self.service_key = service_key
        self.anon_key = anon_key or service_key
        self._client = httpx.Client(timeout=30)
        self._uid_cache: dict[str, str] = {}

    def _headers(self, key: str | None = None, json_body: bool = True):
        key = key or self.service_key
        h = {"apikey": key, "Authorization": f"Bearer {key}"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    # ---- auth (Supabase GoTrue) ---------------------------------------------

    def register_user(self, email: str, password: str) -> dict:
        r = self._client.post(
            f"{self.url}/auth/v1/signup",
            headers=self._headers(self.anon_key),
            json={"email": email, "password": password},
        )
        if r.status_code == 422 and "already" in r.text.lower():
            raise DuplicateEmail("An account with this email already exists.")
        if r.status_code >= 400:
            detail = (r.json() or {}).get("msg", "Registration failed.")
            raise RuntimeError(detail)
        user = (r.json() or {}).get("user") or {}
        return {"id": user.get("id"), "email": user.get("email", email)}

    def authenticate_user(self, email: str, password: str) -> dict | None:
        r = self._client.post(
            f"{self.url}/auth/v1/token?grant_type=password",
            headers=self._headers(self.anon_key),
            json={"email": email, "password": password},
        )
        if r.status_code >= 400:
            return None
        body = r.json()
        return {"id": (body.get("user") or {}).get("id"), "email": email,
                "access_token": body.get("access_token"), "refresh_token": body.get("refresh_token")}

    def get_user_by_token(self, access_token: str) -> dict | None:
        r = self._client.get(
            f"{self.url}/auth/v1/user",
            headers=self._headers(access_token),
        )
        if r.status_code >= 400:
            return None
        u = r.json()
        return {"id": u.get("id"), "email": u.get("email")}

    # local-mode interface stubs (never called in supabase mode)
    def session_secret(self) -> str:  # pragma: no cover
        return ""

    def sign_token(self, user_id: str, ttl_days: int = 7) -> str:  # pragma: no cover
        return ""

    def verify_token(self, token: str) -> dict | None:
        return self.get_user_by_token(token)

    # ---- analyses -----------------------------------------------------------

    @staticmethod
    def _analysis_row(record: dict) -> dict:
        media = record.get("media") or {}
        pred = record.get("prediction") or {}
        model = record.get("model") or {}
        proc = record.get("processing") or {}
        return {
            "id": record["analysis_id"],
            "user_id": record.get("user_id"),
            "media_type": media.get("type", "video"),
            "filename": media.get("filename", "upload"),
            "size_bytes": media.get("size_bytes"), "sha256": media.get("sha256"),
            "width": media.get("width"), "height": media.get("height"),
            "fps": media.get("fps"), "duration_seconds": media.get("duration_seconds"),
            "frames_extracted": media.get("frames_extracted"),
            "frames_analyzed": media.get("frames_analyzed"),
            "valid_face_detections": media.get("valid_face_detections"),
            "status": record.get("status", "QUEUED"),
            "model_name": model.get("name"), "model_version": model.get("version"),
            "checkpoint_sha256": model.get("checkpoint_sha256"),
            "input_frames": model.get("input_frames"), "input_size": model.get("input_size"),
            "label": pred.get("label"), "fake_score": pred.get("fake_score"),
            "threshold": pred.get("threshold"), "threshold_version": pred.get("threshold_version"),
            "prediction_method": pred.get("method"),
            "score_is_calibrated": bool(pred.get("score_is_calibrated")),
            "processing_duration_seconds": proc.get("duration_seconds"),
            "error_code": proc.get("error_code"), "error": proc.get("error"),
            "warnings": proc.get("warnings", []),
            "explainability": record.get("explainability") or {},
            "result_json": record,
            "completed_at": _now() if record.get("status") == "COMPLETED" else None,
        }

    def create_analysis(self, record: dict) -> None:
        row = self._analysis_row(record)
        row["created_at"] = row["updated_at"] = _now()
        r = self._client.post(
            f"{self.url}/rest/v1/analyses",
            headers={**self._headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
            json=row,
        )
        if r.status_code >= 400:
            logger.error("create_analysis failed: %s %s", r.status_code, r.text[:300])
            r.raise_for_status()
        self._replace_windows(record["analysis_id"], record.get("windows") or [])

    def update_analysis(self, analysis_id: str, record: dict) -> None:
        row = self._analysis_row(record)
        row["updated_at"] = _now()
        r = self._client.patch(
            f"{self.url}/rest/v1/analyses?id=eq.{analysis_id}",
            headers={**self._headers(), "Prefer": "return=minimal"},
            json=row,
        )
        if r.status_code >= 400:
            logger.error("update_analysis failed: %s %s", r.status_code, r.text[:300])
        if record.get("windows"):
            self._replace_windows(analysis_id, record["windows"])

    def _replace_windows(self, analysis_id: str, windows: list) -> None:
        self._client.delete(
            f"{self.url}/rest/v1/analysis_windows?analysis_id=eq.{analysis_id}",
            headers=self._headers(),
        )
        rows = [{
            "analysis_id": analysis_id, "window_index": i, "label": w.get("label", f"w{i}"),
            "score": w.get("score", 0.0), "frame_start": w.get("frame_start"),
            "frame_end": w.get("frame_end"), "start_seconds": w.get("start_seconds"),
            "end_seconds": w.get("end_seconds"),
        } for i, w in enumerate(windows)]
        if rows:
            self._client.post(
                f"{self.url}/rest/v1/analysis_windows",
                headers={**self._headers(), "Prefer": "return=minimal"},
                json=rows,
            )

    def get_analysis(self, analysis_id: str, user_id: str | None) -> dict | None:
        # ownership enforced by the user_id filter: other users' rows match nothing
        params = {"id": f"eq.{analysis_id}", "select": "*"}
        if user_id:
            params["user_id"] = f"eq.{user_id}"
        else:
            params["user_id"] = "is.null"
        r = self._client.get(f"{self.url}/rest/v1/analyses", headers=self._headers(), params=params)
        if r.status_code >= 400 or not r.json():
            return None
        row = r.json()[0]
        record = row.get("result_json") or {}
        record.update({
            "analysis_id": row["id"], "user_id": row["user_id"], "status": row["status"],
            "media": record.get("media") or {
                "type": row["media_type"], "filename": row["filename"],
                "size_bytes": row["size_bytes"], "sha256": row["sha256"],
                "width": row["width"], "height": row["height"], "fps": row["fps"],
                "duration_seconds": row["duration_seconds"],
                "frames_extracted": row["frames_extracted"],
                "frames_analyzed": row["frames_analyzed"],
                "valid_face_detections": row["valid_face_detections"],
            },
            "prediction": record.get("prediction") or {
                "label": row["label"], "fake_score": row["fake_score"],
                "threshold": row["threshold"], "threshold_version": row["threshold_version"],
                "method": row["prediction_method"], "score_is_calibrated": False,
            },
            "model": record.get("model") or {
                "name": row["model_name"], "version": row["model_version"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "input_frames": row["input_frames"], "input_size": row["input_size"],
            },
            "processing": record.get("processing") or {
                "duration_seconds": row["processing_duration_seconds"],
                "warnings": row.get("warnings") or [],
                "error": row["error"], "error_code": row["error_code"],
            },
            "explainability": row.get("explainability") or {},
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "report_available": row["status"] == "COMPLETED",
        })
        return record

    def list_analyses(self, user_id: str, label=None, media_type=None, status=None,
                      date_from=None, date_to=None, search=None, cursor=None, limit=20):
        params = {"user_id": f"eq.{user_id}", "select":
                  "id,status,media_type,filename,label,fake_score,threshold,model_version,"
                  "created_at,error_code",
                  "order": "created_at.desc,id.desc", "limit": str(limit + 1)}
        if label:
            params["label"] = f"eq.{label}"
        if media_type:
            params["media_type"] = f"eq.{media_type}"
        if status:
            params["status"] = f"eq.{status}"
        if date_from:
            params["created_at"] = f"gte.{date_from}"
        if date_to:
            params["created_at"] = f"lte.{date_to}"
        if search:
            params["or"] = f"(filename.ilike.*{search}*,id.ilike.*{search}*)"
        if cursor:
            try:
                import base64
                created, last_id = base64.urlsafe_b64decode(
                    cursor + "=" * (-len(cursor) % 4)).decode().split("|", 1)
                params["or"] = (
                    f"(and(created_at.lt.{created},id.lt.{last_id}),created_at.lt.{created})"
                )
            except Exception:
                pass
        r = self._client.get(f"{self.url}/rest/v1/analyses", headers=self._headers(), params=params)
        if r.status_code >= 400:
            return [], None
        rows = r.json()
        items = rows[:limit]
        next_cursor = None
        if len(rows) > limit and items:
            import base64

            last = items[-1]
            next_cursor = base64.urlsafe_b64encode(
                f"{last['created_at']}|{last['id']}".encode()).decode().rstrip("=")
        return items, next_cursor

    def get_analysis_system(self, analysis_id: str) -> dict | None:
        r = self._client.get(
            f"{self.url}/rest/v1/analyses",
            headers=self._headers(),
            params={"id": f"eq.{analysis_id}", "select": "*"},
        )
        rows = r.json() if r.status_code < 400 else []
        if not rows:
            return None
        row = rows[0]
        return self.get_analysis(analysis_id, row.get("user_id"))

    def purge_expired(self, cutoff: str) -> list[tuple[str, list[str]]]:
        r = self._client.get(
            f"{self.url}/rest/v1/analyses",
            headers=self._headers(),
            params={"created_at": f"lt.{cutoff}", "select": "id"},
        )
        out = []
        for row in r.json() if r.status_code < 400 else []:
            aid = row["id"]
            ar = self._client.get(
                f"{self.url}/rest/v1/analysis_artifacts",
                headers=self._headers(),
                params={"analysis_id": f"eq.{aid}", "select": "storage_path"},
            )
            paths = [x["storage_path"] for x in ar.json()] if ar.status_code < 400 else []
            self._client.delete(
                f"{self.url}/rest/v1/analyses?id=eq.{aid}",
                headers={**self._headers(), "Prefer": "return=minimal"},
            )
            out.append((aid, paths))
        return out

    def stats(self, user_id: str) -> dict:
        stats = {}
        for key, filt in (("total", None), ("completed", "eq.COMPLETED"),
                          ("failed", "eq.FAILED")):
            params = {"user_id": f"eq.{user_id}", "select": "id"}
            if filt:
                params["status"] = filt
            r = self._client.get(
                f"{self.url}/rest/v1/analyses",
                headers={**self._headers(), "Prefer": "count=exact", "Range-Unit": "items",
                         "Range": "0-0"},
                params=params,
            )
            stats[key] = self._content_range_count(r)
        for key, col, val in (("images", "media_type", "image"), ("videos", "media_type", "video"),
                              ("fakes", "label", "FAKE"), ("reals", "label", "REAL")):
            r = self._client.get(
                f"{self.url}/rest/v1/analyses",
                headers={**self._headers(), "Prefer": "count=exact", "Range-Unit": "items",
                         "Range": "0-0"},
                params={"user_id": f"eq.{user_id}", col: f"eq.{val}", "select": "id"},
            )
            stats[key] = self._content_range_count(r)
        r = self._client.get(
            f"{self.url}/rest/v1/analyses",
            headers=self._headers(),
            params={"user_id": f"eq.{user_id}", "select":
                    "id,status,media_type,filename,label,fake_score,created_at",
                    "order": "created_at.desc", "limit": "5"},
        )
        stats["recent"] = r.json() if r.status_code < 400 else []
        return stats

    @staticmethod
    def _content_range_count(r) -> int:
        cr = r.headers.get("content-range", "")
        if "/" in cr:
            tail = cr.rsplit("/", 1)[1]
            return int(tail) if tail.isdigit() else 0
        return 0

    def delete_analysis(self, analysis_id: str, user_id: str) -> list[str]:
        r = self._client.get(
            f"{self.url}/rest/v1/analysis_artifacts",
            headers=self._headers(),
            params={"analysis_id": f"eq.{analysis_id}", "select": "storage_path"},
        )
        paths = [row["storage_path"] for row in r.json()] if r.status_code < 400 else []
        r = self._client.delete(
            f"{self.url}/rest/v1/analyses?id=eq.{analysis_id}&user_id=eq.{user_id}",
            headers={**self._headers(), "Prefer": "return=minimal"},
        )
        return paths if r.status_code < 400 and not r.text.strip("[null]") else []

    def add_artifact(self, analysis_id: str, kind: str, bucket: str, path: str) -> None:
        self._client.post(
            f"{self.url}/rest/v1/analysis_artifacts",
            headers={**self._headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
            json={"analysis_id": analysis_id, "kind": kind, "bucket": bucket, "storage_path": path},
        )

    # ---- durable job queue ---------------------------------------------------

    def enqueue_job(self, analysis_id: str) -> None:
        self._client.post(
            f"{self.url}/rest/v1/analysis_jobs",
            headers={**self._headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
            json={"analysis_id": analysis_id},
        )

    def claim_job(self, worker_id: str, lease_seconds: int = 300) -> dict | None:
        r = self._client.post(
            f"{self.url}/rest/v1/rpc/claim_job",
            headers=self._headers(),
            json={"p_worker_id": worker_id, "p_lease_seconds": lease_seconds},
        )
        if r.status_code >= 400 or not r.json():
            return None
        row = r.json()[0] if isinstance(r.json(), list) else r.json()
        return {"job_id": row["job_id"], "analysis_id": row["analysis_id"],
                "attempts": row["attempts"], "max_attempts": row["max_attempts"]}

    def renew_lease(self, job_id: str, worker_id: str, lease_seconds: int = 300) -> bool:
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        r = self._client.patch(
            f"{self.url}/rest/v1/analysis_jobs?id=eq.{job_id}&worker_id=eq.{worker_id}&state=eq.RUNNING",
            headers={**self._headers(), "Prefer": "return=minimal"},
            json={"lease_until": lease_until, "updated_at": _now()},
        )
        return r.status_code < 400

    def finish_job(self, job_id: str, state: str, error_kind=None, last_error=None,
                   retry_delay_seconds=30) -> None:
        body = {"state": state, "error_kind": error_kind, "last_error": last_error,
                "worker_id": None, "lease_until": None, "updated_at": _now()}
        if state == "QUEUED":
            body["run_after"] = (datetime.now(timezone.utc)
                                 + timedelta(seconds=retry_delay_seconds)).isoformat()
        self._client.patch(
            f"{self.url}/rest/v1/analysis_jobs?id=eq.{job_id}",
            headers={**self._headers(), "Prefer": "return=minimal"},
            json=body,
        )

    def reclaim_stalled_jobs(self, grace_seconds: int = 60) -> int:
        r = self._client.post(
            f"{self.url}/rest/v1/rpc/reclaim_stalled_jobs", headers=self._headers(), json={}
        )
        return r.json() if r.status_code < 400 and r.text.strip().isdigit() else 0

    def job_for(self, analysis_id: str) -> dict | None:
        r = self._client.get(
            f"{self.url}/rest/v1/analysis_jobs",
            headers=self._headers(),
            params={"analysis_id": f"eq.{analysis_id}", "select": "*"},
        )
        rows = r.json() if r.status_code < 400 else []
        return rows[0] if rows else None
