"""
Persistence layer (Phase 4A).

Two interchangeable backends behind one interface:

- SQLiteRepository: zero-dependency local backend (stdlib sqlite3, WAL mode).
  The schema mirrors supabase/migrations/0001_init.sql exactly, so the whole
  application — durable jobs, history, deletion — runs and is testable without
  any external credentials.
- SupabaseRepository: PostgREST + Supabase Storage via the service-role key,
  talking to the schema created by the SQL migrations (RLS in 0002_rls.sql).

Selection: SUPABASE_URL + SUPABASE_SERVICE_KEY set -> Supabase; otherwise local.
"""

import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger("deepfake.persistence")


def get_repository(settings):
    """Pick the repository backend from configuration."""
    if settings.supabase_url and settings.supabase_service_key:
        from .supabase_repo import SupabaseRepository

        logger.info("Persistence: Supabase (%s)", settings.supabase_url)
        return SupabaseRepository(settings.supabase_url, settings.supabase_service_key)
    from .sqlite_repo import SQLiteRepository

    data_dir = Path(settings.jobs_dir).parent / ".data"
    logger.info("Persistence: local SQLite (%s)", data_dir)
    return SQLiteRepository(data_dir)


def generate_secret(data_dir: Path) -> str:
    """Persistent random secret for local session-token signing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    secret_file = data_dir / "session_secret"
    if secret_file.is_file():
        return secret_file.read_text(encoding="utf-8").strip()
    secret = secrets.token_hex(32)
    secret_file.write_text(secret, encoding="utf-8")
    os.chmod(secret_file, 0o600)
    return secret
