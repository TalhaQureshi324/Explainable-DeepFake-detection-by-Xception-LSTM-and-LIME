"""
SQLite repository — local persistence backend (Phase 4A).

Mirrors supabase/migrations/0001_init.sql: analyses, analysis_windows,
analysis_artifacts, model_versions, analysis_jobs (+ users for local auth mode).
WAL journaling; a single connection guarded by a lock (FastAPI threadpool-safe
for this workload).
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA = """
create table if not exists users (
  id text primary key,
  email text not null unique,
  password_hash text not null,
  created_at text not null
);
create table if not exists analyses (
  id text primary key,
  user_id text references users(id) on delete cascade,
  media_type text not null,
  filename text not null,
  size_bytes integer, sha256 text,
  width integer, height integer,
  fps real, duration_seconds real,
  frames_extracted integer, frames_analyzed integer, valid_face_detections integer,
  status text not null default 'QUEUED',
  model_name text, model_version text, checkpoint_sha256 text,
  input_frames integer, input_size integer,
  label text, fake_score real,
  threshold real, threshold_version text,
  prediction_method text, score_is_calibrated integer default 0,
  processing_duration_seconds real,
  error_code text, error text,
  warnings_json text not null default '[]',
  explainability_json text not null default '{}',
  result_json text,
  created_at text not null, updated_at text not null, completed_at text,
  retained_until text
);
create index if not exists analyses_user_created_idx on analyses (user_id, created_at desc);
create index if not exists analyses_status_idx on analyses (status);
create table if not exists analysis_windows (
  id text primary key,
  analysis_id text not null references analyses(id) on delete cascade,
  window_index integer not null,
  label text not null, score real not null,
  frame_start integer, frame_end integer,
  start_seconds real, end_seconds real,
  created_at text not null,
  unique (analysis_id, window_index)
);
create table if not exists analysis_artifacts (
  id text primary key,
  analysis_id text not null references analyses(id) on delete cascade,
  kind text not null,
  storage_path text not null,
  bucket text not null,
  created_at text not null,
  unique (analysis_id, kind, storage_path)
);
create table if not exists model_versions (
  id text primary key,
  version text not null unique, name text not null,
  checkpoint_sha256 text not null,
  input_frames integer not null, input_size integer not null,
  preprocessing_version text not null, notes text, created_at text not null
);
create table if not exists analysis_jobs (
  id text primary key,
  analysis_id text not null unique references analyses(id) on delete cascade,
  state text not null default 'QUEUED',
  attempts integer not null default 0,
  max_attempts integer not null default 3,
  lease_until text, worker_id text,
  error_kind text, last_error text,
  run_after text not null,
  created_at text not null, updated_at text not null
);
create index if not exists analysis_jobs_claim_idx on analysis_jobs (state, run_after);
"""

_INDEXED_FIELDS = (
    "user_id", "media_type", "filename", "size_bytes", "sha256", "width", "height",
    "fps", "duration_seconds", "frames_extracted", "frames_analyzed",
    "valid_face_detections", "status", "model_name", "model_version",
    "checkpoint_sha256", "input_frames", "input_size", "label", "fake_score",
    "threshold", "threshold_version", "prediction_method", "score_is_calibrated",
    "processing_duration_seconds", "error_code", "error",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DuplicateEmail(Exception):
    pass


class SQLiteRepository:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "app.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("pragma journal_mode=wal")
        self._conn.execute("pragma foreign_keys=on")
        self._lock = threading.RLock()  # reentrant: update_analysis nests window/artifact writes
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        self._secret_file = self.data_dir / "session_secret"

    # ---- auth (local mode) -------------------------------------------------

    def _hash_password(self, password: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()

    def register_user(self, email: str, password: str) -> dict:
        uid = str(__import__("uuid").uuid4())
        salt = secrets.token_hex(16)
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "insert into users (id, email, password_hash, created_at) values (?,?,?,?)",
                    (uid, email.strip().lower(), f"{salt}${self._hash_password(password, bytes.fromhex(salt))}", _now()),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateEmail("An account with this email already exists.") from exc
        return {"id": uid, "email": email.strip().lower()}

    def authenticate_user(self, email: str, password: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "select * from users where email = ?", (email.strip().lower(),)
            ).fetchone()
        if not row:
            return None
        salt, digest = row["password_hash"].split("$", 1)
        calc = self._hash_password(password, bytes.fromhex(salt))
        if not hmac.compare_digest(calc, digest):
            return None
        return {"id": row["id"], "email": row["email"]}

    def get_user(self, user_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "select id, email from users where id = ?", (user_id,)
            ).fetchone()
        return {"id": row["id"], "email": row["email"]} if row else None

    def session_secret(self) -> str:
        from . import generate_secret

        return generate_secret(self.data_dir)

    def sign_token(self, user_id: str, ttl_days: int = 7) -> str:
        expiry = str(int((datetime.now(timezone.utc) + timedelta(days=ttl_days)).timestamp()))
        payload = f"{user_id}.{expiry}"
        sig = hmac.new(self.session_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def verify_token(self, token: str) -> dict | None:
        try:
            uid, expiry, sig = token.split(".")
            payload = f"{uid}.{expiry}"
            expected = hmac.new(self.session_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected) or int(expiry) < datetime.now(timezone.utc).timestamp():
                return None
            return self.get_user(uid)
        except (ValueError, TypeError):
            return None

    # ---- analyses -----------------------------------------------------------

    def create_analysis(self, record: dict) -> None:
        media = record.get("media") or {}
        pred = record.get("prediction") or {}
        model = record.get("model") or {}
        proc = record.get("processing") or {}
        with self._lock, self._conn:
            self._conn.execute(
                """insert into analyses (id, user_id, media_type, filename, size_bytes, sha256,
                    width, height, fps, duration_seconds, frames_extracted, frames_analyzed,
                    valid_face_detections, status, model_name, model_version, checkpoint_sha256,
                    input_frames, input_size, label, fake_score, threshold, threshold_version,
                    prediction_method, score_is_calibrated, processing_duration_seconds,
                    error_code, error, warnings_json, explainability_json, result_json,
                    created_at, updated_at)
                 values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["analysis_id"], record.get("user_id"), media.get("type", "video"),
                    media.get("filename", "upload"), media.get("size_bytes"), media.get("sha256"),
                    media.get("width"), media.get("height"), media.get("fps"),
                    media.get("duration_seconds"), media.get("frames_extracted"),
                    media.get("frames_analyzed"), media.get("valid_face_detections"),
                    record.get("status", "QUEUED"), model.get("name"), model.get("version"),
                    model.get("checkpoint_sha256"), model.get("input_frames"),
                    model.get("input_size"), pred.get("label"), pred.get("fake_score"),
                    pred.get("threshold"), pred.get("threshold_version"), pred.get("method"),
                    1 if pred.get("score_is_calibrated") else 0,
                    proc.get("duration_seconds"), proc.get("error_code"), proc.get("error"),
                    json.dumps(proc.get("warnings", [])),
                    json.dumps(record.get("explainability") or {}),
                    json.dumps(record), _now(), _now(),
                ),
            )
            self._replace_windows(record["analysis_id"], record.get("windows") or [])
            for kind, rel in (record.get("artifacts") or {}).items():
                if rel:
                    self.add_artifact(record["analysis_id"], kind, "analysis-artifacts", rel)

    def update_analysis(self, analysis_id: str, record: dict) -> None:
        """Persist the FULL current record (columns + result_json snapshot)."""
        media = record.get("media") or {}
        pred = record.get("prediction") or {}
        model = record.get("model") or {}
        proc = record.get("processing") or {}
        completed_at = _now() if record.get("status") == "COMPLETED" else None
        with self._lock, self._conn:
            self._conn.execute(
                """update analyses set status=?, model_name=?, model_version=?,
                    checkpoint_sha256=?, input_frames=?, input_size=?, label=?, fake_score=?,
                    threshold=?, threshold_version=?, prediction_method=?, score_is_calibrated=?,
                    processing_duration_seconds=?, error_code=?, error=?, warnings_json=?,
                    explainability_json=?, result_json=?, updated_at=?,
                    completed_at=coalesce(?, completed_at),
                    width=coalesce(?, width), height=coalesce(?, height),
                    fps=coalesce(?, fps), duration_seconds=coalesce(?, duration_seconds),
                    frames_extracted=coalesce(?, frames_extracted),
                    frames_analyzed=coalesce(?, frames_analyzed),
                    valid_face_detections=coalesce(?, valid_face_detections),
                    size_bytes=coalesce(?, size_bytes), sha256=coalesce(?, sha256)
                 where id=?""",
                (
                    record.get("status"), model.get("name"), model.get("version"),
                    model.get("checkpoint_sha256"), model.get("input_frames"),
                    model.get("input_size"), pred.get("label"), pred.get("fake_score"),
                    pred.get("threshold"), pred.get("threshold_version"), pred.get("method"),
                    1 if pred.get("score_is_calibrated") else 0,
                    proc.get("duration_seconds"), proc.get("error_code"), proc.get("error"),
                    json.dumps(proc.get("warnings", [])),
                    json.dumps(record.get("explainability") or {}),
                    json.dumps(record), _now(), completed_at,
                    media.get("width"), media.get("height"), media.get("fps"),
                    media.get("duration_seconds"), media.get("frames_extracted"),
                    media.get("frames_analyzed"), media.get("valid_face_detections"),
                    media.get("size_bytes"), media.get("sha256"), analysis_id,
                ),
            )
            if record.get("windows"):
                self._replace_windows(analysis_id, record["windows"])
            for kind, rel in (record.get("artifacts") or {}).items():
                if rel:
                    self.add_artifact(analysis_id, kind, "analysis-artifacts", rel)

    def _replace_windows(self, analysis_id: str, windows: list) -> None:
        self._conn.execute("delete from analysis_windows where analysis_id = ?", (analysis_id,))
        for i, w in enumerate(windows):
            self._conn.execute(
                """insert into analysis_windows (id, analysis_id, window_index, label, score,
                    frame_start, frame_end, start_seconds, end_seconds, created_at)
                   values (?,?,?,?,?,?,?,?,?,?)""",
                (str(__import__("uuid").uuid4()), analysis_id, i, w.get("label", f"w{i}"),
                 w.get("score", 0.0), w.get("frame_start"), w.get("frame_end"),
                 w.get("start_seconds"), w.get("end_seconds"), _now()),
            )

    def get_analysis(self, analysis_id: str, user_id: str | None) -> dict | None:
        with self._lock:
            if user_id is None:
                row = self._conn.execute(
                    "select * from analyses where id = ? and user_id is null", (analysis_id,)
                ).fetchone()
            else:
                # ownership enforced here: another user's analysis is invisible
                row = self._conn.execute(
                    "select * from analyses where id = ? and user_id = ?", (analysis_id, user_id)
                ).fetchone()
        if not row:
            return None
        record = json.loads(row["result_json"]) if row["result_json"] else {}
        record.update({
            "analysis_id": row["id"],
            "user_id": row["user_id"],
            "status": row["status"],
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
                "warnings": json.loads(row["warnings_json"]),
                "error": row["error"], "error_code": row["error_code"],
            },
            "explainability": json.loads(row["explainability_json"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "report_available": row["status"] == "COMPLETED",
        })
        return record

    def get_analysis_system(self, analysis_id: str) -> dict | None:
        """Internal read WITHOUT ownership filter (service/worker use only)."""
        with self._lock:
            row = self._conn.execute(
                "select user_id from analyses where id = ?", (analysis_id,)
            ).fetchone()
        if not row:
            return None
        return self.get_analysis(analysis_id, row["user_id"])

    def purge_expired(self, cutoff: str) -> list[tuple[str, list[str]]]:
        """Retention sweep: delete analyses older than cutoff; returns (id, paths)."""
        with self._lock:
            rows = self._conn.execute(
                "select id from analyses where created_at < ?", (cutoff,)
            ).fetchall()
        out = []
        for r in rows:
            out.append((r["id"], self.delete_analysis(r["id"],
                        self._owner_of(r["id"]))))
        return out

    def _owner_of(self, analysis_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "select user_id from analyses where id = ?", (analysis_id,)
            ).fetchone()
        return row["user_id"] if row else ""

    def list_analyses(self, user_id: str, label=None, media_type=None, status=None,
                      date_from=None, date_to=None, search=None, cursor=None, limit=20):
        where = ["user_id = ?"]
        params: list = [user_id]
        if label:
            where.append("label = ?")
            params.append(label)
        if media_type:
            where.append("media_type = ?")
            params.append(media_type)
        if status:
            where.append("status = ?")
            params.append(status)
        if date_from:
            where.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            where.append("created_at <= ?")
            params.append(date_to)
        if search:
            where.append("(filename like ? or id like ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        if cursor:
            try:
                created, last_id = base64_url_decode(cursor).split("|", 1)
                where.append("(created_at < ? or (created_at = ? and id < ?))")
                params.extend([created, created, last_id])
            except Exception:
                pass  # malformed cursor: ignore and page from the start
        q = (f"select id, status, media_type, filename, label, fake_score, threshold, "
             f"model_version, created_at, error_code from analyses "
             f"where {' and '.join(where)} order by created_at desc, id desc limit ?")
        params.append(limit + 1)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        items = [dict(r) for r in rows[:limit]]
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = base64_url_encode(f"{last['created_at']}|{last['id']}")
        return items, next_cursor

    def stats(self, user_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                """select count(*) total,
                      sum(case when status='COMPLETED' then 1 else 0 end) completed,
                      sum(case when media_type='image' then 1 else 0 end) images,
                      sum(case when media_type='video' then 1 else 0 end) videos,
                      sum(case when label='FAKE' then 1 else 0 end) fakes,
                      sum(case when label='REAL' then 1 else 0 end) reals,
                      sum(case when status='FAILED' then 1 else 0 end) failed
                   from analyses where user_id = ?""",
                (user_id,),
            ).fetchone()
            recent = self._conn.execute(
                """select id, status, media_type, filename, label, fake_score, created_at
                   from analyses where user_id = ? order by created_at desc limit 5""",
                (user_id,),
            ).fetchall()
        return {**dict(row), "recent": [dict(r) for r in recent]}

    def delete_analysis(self, analysis_id: str, user_id: str) -> list[str]:
        """Delete rows (windows/artifacts cascade) for the owner; returns artifact paths."""
        with self._lock, self._conn:
            rows = self._conn.execute(
                "select storage_path from analysis_artifacts where analysis_id = ?",
                (analysis_id,),
            ).fetchall()
            cur = self._conn.execute(
                "delete from analyses where id = ? and user_id = ?", (analysis_id, user_id)
            )
            deleted = cur.rowcount > 0
        return [r["storage_path"] for r in rows] if deleted else []

    def add_artifact(self, analysis_id: str, kind: str, bucket: str, path: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """insert or ignore into analysis_artifacts
                   (id, analysis_id, kind, storage_path, bucket, created_at)
                   values (?,?,?,?,?,?)""",
                (str(__import__("uuid").uuid4()), analysis_id, kind, bucket, path, _now()),
            )

    # ---- durable job queue --------------------------------------------------

    def enqueue_job(self, analysis_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """insert or ignore into analysis_jobs
                   (id, analysis_id, state, run_after, created_at, updated_at)
                   values (?,?,?,?,?,?)""",
                (str(__import__("uuid").uuid4()), analysis_id, "QUEUED", _now(), _now(), _now()),
            )

    def claim_job(self, worker_id: str, lease_seconds: int = 300) -> dict | None:
        now = _now()
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._conn:
            # dead-letter exhausted jobs so they never claim again
            self._conn.execute(
                "update analysis_jobs set state='DEAD', updated_at=? "
                "where state='QUEUED' and attempts >= max_attempts", (now,))
            row = self._conn.execute(
                """select j.id, j.analysis_id, j.attempts, j.max_attempts from analysis_jobs j
                   where j.state='QUEUED' and j.run_after <= ?
                   order by j.run_after limit 1""",
                (now,),
            ).fetchone()
            if not row:
                return None
            cur = self._conn.execute(
                """update analysis_jobs set state='RUNNING', worker_id=?, lease_until=?,
                    attempts=attempts+1, updated_at=?
                 where id=? and state='QUEUED'""",
                (worker_id, lease_until, _now(), row["id"]),
            )
            if cur.rowcount != 1:
                return None  # another worker claimed it between select and update
            return {"job_id": row["id"], "analysis_id": row["analysis_id"],
                    "attempts": row["attempts"] + 1, "max_attempts": row["max_attempts"]}

    def renew_lease(self, job_id: str, worker_id: str, lease_seconds: int = 300) -> bool:
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """update analysis_jobs set lease_until=?, updated_at=?
                 where id=? and worker_id=? and state='RUNNING'""",
                (lease_until, _now(), job_id, worker_id),
            )
        return cur.rowcount == 1

    def finish_job(self, job_id: str, state: str, error_kind=None, last_error=None,
                   retry_delay_seconds=30) -> None:
        run_after = (datetime.now(timezone.utc) + timedelta(seconds=retry_delay_seconds)).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """update analysis_jobs set state=?, error_kind=?, last_error=?,
                    worker_id=null, lease_until=null,
                    run_after=case when ?='QUEUED' then ? else run_after end,
                    updated_at=? where id=?""",
                (state, error_kind, last_error, state, run_after, _now(), job_id),
            )

    def reclaim_stalled_jobs(self, grace_seconds: int = 60) -> int:
        """Jobs whose worker died mid-run (lease expired) go back to the queue."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=grace_seconds)).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """update analysis_jobs set state='QUEUED', worker_id=null, lease_until=null,
                    updated_at=? where state='RUNNING' and (lease_until is null or lease_until < ?)""",
                (_now(), cutoff),
            )
        return cur.rowcount

    def job_for(self, analysis_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "select * from analysis_jobs where analysis_id = ?", (analysis_id,)
            ).fetchone()
        return dict(row) if row else None


# ---- cursor helpers ----------------------------------------------------------

def base64_url_encode(s: str) -> str:
    import base64

    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def base64_url_decode(s: str) -> str:
    import base64

    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode()
