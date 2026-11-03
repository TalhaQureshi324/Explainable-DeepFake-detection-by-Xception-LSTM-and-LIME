"""
In-memory analysis job store (Phase 1).

Thread-safe dict keyed by analysis_id. This is intentionally the ONLY place
that knows about persistence: Phase 4 swaps it for a Supabase-backed store
without touching services, workers or routes. Jobs are lost on restart by
design in Phase 1 (documented in docs/ARCHITECTURE.md).
"""

import threading
from datetime import datetime, timezone
from typing import Optional


class JobStore:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, analysis_id: str, record: dict) -> dict:
        record = dict(record)
        record.setdefault("analysis_id", analysis_id)
        record.setdefault("status", "QUEUED")
        record.setdefault("created_at", self._now())
        record.setdefault("updated_at", record["created_at"])
        record.setdefault("processing", {"warnings": []})
        record.setdefault("windows", [])
        record.setdefault("explainability", {"available": False})
        with self._lock:
            self._jobs[analysis_id] = record
        return record

    def get(self, analysis_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(analysis_id)

    def update(self, analysis_id: str, **fields) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(analysis_id)
            if job is None:
                return None
            job.update(fields)
            job["updated_at"] = self._now()
            return job

    def remove(self, analysis_id: str) -> None:
        with self._lock:
            self._jobs.pop(analysis_id, None)

    def count_by_status(self, statuses) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.get("status") in statuses)

    def push_warning(self, analysis_id: str, warning: str) -> None:
        with self._lock:
            job = self._jobs.get(analysis_id)
            if job is not None:
                proc = job.setdefault("processing", {"warnings": []})
                proc.setdefault("warnings", []).append(warning)
                job["updated_at"] = self._now()
