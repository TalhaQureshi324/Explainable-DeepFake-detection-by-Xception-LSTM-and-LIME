"""Auth (4B) and analysis history (4E) + deletion (4F) routes.

Auth supports two backends behind the same contract:
- local mode (no Supabase configured): pbkdf2-hashed users in the SQLite
  repository, HMAC-signed session tokens
- supabase mode: GoTrue signup/login; the backend verifies access tokens via
  /auth/v1/user and maps them to the analyses.user_id owner

Sessions are httpOnly cookies (SameSite=Lax) so the Next.js same-origin proxy
keeps everything first-party; Bearer tokens are accepted for API clients.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr

from app.core.config import Settings, get_settings
from app.core.deps import get_current_user, get_repository, require_user

logger = logging.getLogger("deepfake.auth")

COOKIE_NAME = "df_session"
COOKIE_MAX_AGE = 7 * 24 * 3600


class Credentials(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str | None = None


def _set_session_cookie(response: Response, token: str):
    response.set_cookie(
        COOKIE_NAME, token, max_age=COOKIE_MAX_AGE, httponly=True,
        samesite="lax", secure=False,  # secure=True when served over HTTPS
        path="/",
    )


def _login_response(repository, email: str, password: str, response: Response):
    user = repository.authenticate_user(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = repository.sign_token(user["id"])
    _set_session_cookie(response, token)
    return {"user": {"id": user["id"], "email": user.get("email")}, "access_token": token}


def create_router(repository, settings: Settings) -> APIRouter:
    router = APIRouter()

    # ---- auth ------------------------------------------------------------

    @router.post("/v1/auth/register", status_code=201)
    def register(body: Credentials, response: Response):
        if len(body.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
        try:
            user = repository.register_user(body.email, body.password)
        except Exception as exc:
            name = type(exc).__name__
            if name == "DuplicateEmail":
                raise HTTPException(status_code=409, detail=str(exc))
            logger.exception("Registration failed")
            raise HTTPException(status_code=400, detail=str(exc))
        token = repository.sign_token(user["id"])
        _set_session_cookie(response, token)
        return {"user": user, "access_token": token}

    @router.post("/v1/auth/login")
    def login(body: Credentials, response: Response):
        return _login_response(repository, body.email, body.password, response)

    @router.post("/v1/auth/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"ok": True}

    @router.get("/v1/auth/me")
    def me(user: dict | None = Depends(get_current_user)):
        if not user:
            return {"user": None}
        return {"user": {"id": user.get("id"), "email": user.get("email")}}

    # ---- history + dashboard ------------------------------------------------

    @router.get("/v1/history")
    def history(
        user: dict = Depends(require_user),
        cursor: str | None = None,
        label: str | None = None,
        media_type: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        search: str | None = None,
        limit: int = 12,
    ):
        items, next_cursor = repository.list_analyses(
            user["id"], label=label, media_type=media_type, status=status,
            date_from=date_from, date_to=date_to, search=search,
            cursor=cursor, limit=min(max(limit, 1), 50),
        )
        return {"items": items, "next_cursor": next_cursor}

    @router.get("/v1/history/stats")
    def history_stats(user: dict = Depends(require_user)):
        return repository.stats(user["id"])

    # ---- deletion (owner-verified; never via client-supplied owner id) -------

    from fastapi import Request

    @router.delete("/v1/analyses/{analysis_id}")
    def delete_analysis(analysis_id: str, request: Request,
                        user: dict = Depends(require_user),
                        settings: Settings = Depends(get_settings)):
        paths = repository.delete_analysis(analysis_id, user["id"])
        if not paths and repository.get_analysis(analysis_id, user["id"]) is None:
            # either not owned or already gone -> do not leak existence
            raise HTTPException(status_code=404, detail="Analysis not found.")
        # remove local caches / cached PDF / uploaded media leftovers
        import shutil

        artifacts_dir = settings.artifacts_dir
        for stale in artifacts_dir.glob(f"*{analysis_id.replace('-', '')[:20]}*"):
            stale.unlink(missing_ok=True)
        for stale in artifacts_dir.glob(f"report_{analysis_id}.pdf"):
            stale.unlink(missing_ok=True)
        jobs_dir = settings.jobs_dir / "uploads" / analysis_id
        shutil.rmtree(jobs_dir, ignore_errors=True)
        request.app.state.store.remove(analysis_id)
        return {"deleted": True}

    return router
