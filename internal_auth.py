from __future__ import annotations

from fastapi import Header, HTTPException, status

from .db import DOCS_AUTH_ENABLED, INTERNAL_TOKEN


def require_internal_token(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> None:
    if not INTERNAL_TOKEN or x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def require_docs_access(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> None:
    if not DOCS_AUTH_ENABLED:
        return
    if not INTERNAL_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "config_error",
                "message": "Docs auth is enabled but INTERNAL_TOKEN is not configured",
                "details": {"setting": "INTERNAL_TOKEN"},
            },
        )
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
