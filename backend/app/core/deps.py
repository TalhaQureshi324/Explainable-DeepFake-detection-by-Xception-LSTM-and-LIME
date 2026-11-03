"""Shared FastAPI dependencies: settings, repository, current-user resolution."""

import logging
from typing import Optional

from fastapi import Depends, Request

from app.core.config import Settings, get_settings

logger = logging.getLogger("deepfake.deps")


def get_repository(request: Request):
    return request.app.state.repository


def _bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def get_current_user(request: Request,
                     settings: Settings = Depends(get_settings)) -> Optional[dict]:
    """Resolve the caller from the Bearer token or the df_session cookie.

    NEVER trusts a user id from the request body/query: the id always comes
    from a signed token verified against the auth backend (local HMAC session
    or Supabase GoTrue). Returns None for guests.
    """
    repository = request.app.state.repository
    token = _bearer_token(request) or request.cookies.get("df_session")
    if not token:
        return None
    try:
        return repository.verify_token(token)
    except Exception:  # noqa: BLE001 - auth backend failures degrade to guest
        logger.exception("Token verification failed")
        return None


def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency for endpoints that MUST be authenticated."""
    if not user:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Authentication required.")
    return user
