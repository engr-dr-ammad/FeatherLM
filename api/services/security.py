"""Authentication dependencies.

* Inference endpoints: require `Authorization: Bearer <FEATHERLM_API_KEY>` when that
  variable is set (open otherwise, for local use).
* Training/admin endpoints: require FEATHERLM_ADMIN_KEY when set. When it is NOT set,
  they are only accepted from loopback clients (127.0.0.1 / ::1), so a server bound to a
  public interface never exposes unauthenticated training controls. Requests carrying
  proxy headers (X-Forwarded-For etc.) are treated as non-local. They can also be
  switched off entirely with FEATHERLM_ENABLE_TRAINING_API=false.
"""
from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request, status

LOOPBACK = {"127.0.0.1", "::1"}


def _token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "")


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_api_key(request: Request) -> None:
    s = request.app.state.settings
    keys = [k for k in (s.api_key, s.admin_key) if k]
    if not s.api_key:
        return
    tok = _token(request)
    if not any(_eq(tok, k) for k in keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key",
                            headers={"WWW-Authenticate": "Bearer"})


def require_admin(request: Request) -> None:
    s = request.app.state.settings
    if not s.enable_training_api:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "training/admin API is disabled on this server")
    if s.admin_key:
        if not _eq(_token(request), s.admin_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "admin key required",
                                headers={"WWW-Authenticate": "Bearer"})
        return
    host = request.client.host if request.client else ""
    proxied = any(h in request.headers for h in ("x-forwarded-for", "x-real-ip", "forwarded"))
    if host not in LOOPBACK or proxied:  # a reverse proxy makes every client look local
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "admin endpoints need FEATHERLM_ADMIN_KEY when accessed from non-local clients")


ApiKey = Depends(require_api_key)
Admin = Depends(require_admin)
