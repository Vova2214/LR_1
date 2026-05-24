from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from http import HTTPStatus
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from .routes import events, internal, links, objects, semantic_tools, sessions, turns
from .application import turn_application_service, session_lifecycle_service
from .db import run_with_new_session
from . import schemas
from . import models
from .background_workers import (
    allow_new_workers,
    begin_shutdown,
    start_background_worker,
    wait_for_background_workers,
)
from .crud_embeddings_ops import MEMORY_EVENT_OBJECT_TYPE, MEMORY_FACT_OBJECT_TYPE
from .crud_consequences import CONSEQUENCE_WINDOW_OBJECT_TYPE
from .outbox_runtime import (
    OUTBOX_PRIMARY_EVENT_TYPES,
    OUTBOX_REFRESH_EVENT_TYPES,
    run_outbox_poller_loop,
    run_retention_janitor_loop,
)
from .workers.registry import startup_worker_specs
from .db import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    EMBEDDING_DIM,
    ALLOW_DEBUG_PATCH,
    ENABLE_DEBUG_ROUTER,
    OPENCODE_API_KEY,
    OPENCODE_BASE_URL,
    OPENCODE_CHAT_TIMEOUT_SECONDS,
    OPENROUTER_CHAT_MODEL,
    OPENROUTER_EMBED_MODEL,
    USE_EMBEDDINGS,
    get_db,
)
from .internal_auth import require_docs_access
from .observability import (
    configure_structured_logging,
    get_trace_id,
    render_prometheus,
    reset_trace_id,
    set_trace_id,
    trace_extra,
)

logger = logging.getLogger(__name__)
configure_structured_logging()
_STARTED_AT = time.monotonic()
_PLAYGROUND_PATH = Path(__file__).resolve().parent / "static" / "mini_chat_playground.html"


@asynccontextmanager
async def _app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    allow_new_workers()
    for spec in startup_worker_specs():
        start_background_worker(
            target=spec.target,
            kwargs=dict(spec.kwargs),
            name=spec.name,
        )
    try:
        yield
    finally:
        begin_shutdown()
        still_running = wait_for_background_workers(timeout_seconds=8.0)
        if still_running > 0:
            logger.warning(
                "Background workers still running during shutdown: count=%s",
                still_running,
            )


app = FastAPI(
    title="Text RPG Graph API",
    version="0.1.0",
    lifespan=_app_lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

def _status_message(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, Exception):
        return f"{type(obj).__name__}: {str(obj)}"
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    detail: Any,
    details: Any,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    trace_id = get_trace_id()
    payload = {
        "error": {
            "code": code,
            "message": message,
            "status": status_code,
            "trace_id": trace_id,
            "details": _sanitize_for_json(details),
        },
        "detail": _sanitize_for_json(detail),
    }
    merged_headers: dict[str, str] = dict(headers or {})
    if trace_id:
        merged_headers.setdefault("X-Trace-Id", trace_id)
    return JSONResponse(status_code=status_code, content=payload, headers=merged_headers)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    code = f"http_{exc.status_code}"
    message = _status_message(exc.status_code)
    details: Any = {}
    if isinstance(detail, dict):
        detail_code = detail.get("code")
        detail_message = detail.get("message")
        if isinstance(detail_code, str) and detail_code.strip():
            code = detail_code.strip()
        if isinstance(detail_message, str) and detail_message.strip():
            message = detail_message.strip()
        if "details" in detail:
            details = detail["details"]
        elif "detail" in detail:
            details = detail["detail"]
        else:
            details = detail
    elif isinstance(detail, str):
        if detail.strip():
            message = detail.strip()
    elif isinstance(detail, list):
        details = detail
    elif detail is not None:
        details = detail
    return _error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
        detail=detail,
        details=details,
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = exc.errors()
    return _error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed",
        detail=detail,
        details=detail,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    sanitized = {"code": "internal_error", "message": str(type(exc).__name__) + ": " + str(exc)[:150]}
    return _error_response(
        status_code=500,
        code="internal_error",
        message="Internal server error",
        detail=sanitized,
        details={},
    )


@app.get("/openapi.json", include_in_schema=False)
def openapi_json(_: None = Depends(require_docs_access)) -> dict[str, Any]:
    return app.openapi()


@app.get("/docs", include_in_schema=False)
def swagger_docs(_: None = Depends(require_docs_access)) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
    )


@app.get("/redoc", include_in_schema=False)
def redoc_docs(_: None = Depends(require_docs_access)) -> HTMLResponse:
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - ReDoc",
    )


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/playground", status_code=307)


@app.get("/playground", include_in_schema=False)
def mini_chat_playground() -> HTMLResponse:
    return HTMLResponse(content=_PLAYGROUND_PATH.read_text(encoding="utf-8"))


app.include_router(sessions.router)
app.include_router(objects.router)
app.include_router(links.router)
app.include_router(events.router)
app.include_router(turns.router)
app.include_router(semantic_tools.router)
app.include_router(internal.router)

if ENABLE_DEBUG_ROUTER:
    from .routes import debug

    app.include_router(debug.router)


@app.middleware("http")
async def add_trace_id(request: Request, call_next):  # type: ignore[override]
    trace_id = uuid.uuid4().hex[:8]
    token = set_trace_id(trace_id)
    started = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "request_failed",
                extra=trace_extra(
                    {
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    }
                ),
            )
            response = await unhandled_exception_handler(request, exc)
        else:
            logger.info(
                "request_completed",
                extra=trace_extra(
                    {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    }
                ),
            )
        response.headers["X-Trace-Id"] = trace_id
        return response
    finally:
        reset_trace_id(token)


def _check_db(db: Session) -> dict[str, object]:
    try:
        value = db.execute(text("SELECT 1")).scalar_one()
        return {"ok": bool(value == 1)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _skipped_provider_check(*, reason: str, message: str) -> dict[str, object]:
    return {
        "ok": True,
        "required": False,
        "skipped": True,
        "reason": reason,
        "message": message,
    }


def _check_provider_config(*, provider: str, api_key: str, required: bool = True) -> dict[str, object]:
    if not required:
        return _skipped_provider_check(
            reason="not_required",
            message=f"{provider} is not required for the current runtime configuration",
        )
    if api_key:
        return {"ok": True, "required": True}
    return {
        "ok": False,
        "required": True,
        "error": "api_key_missing",
        "reason": "api_key_missing",
        "message": f"{provider} API key is not configured",
    }


def _check_opencode_config() -> dict[str, object]:
    return _check_provider_config(provider="OpenCode", api_key=OPENCODE_API_KEY, required=True)

    chat_models = _configured_openrouter_chat_models()
    missing_fields = [
        f"{service}_model"
        for service, model_name in chat_models.items()
        if not model_name
    ]
    if require_embeddings and not str(OPENROUTER_EMBED_MODEL or "").strip():
        missing_fields.append("embed_model")
    if missing_fields:
        return {
            "ok": False,
            "required": True,
            "error": "model_missing",
            "reason": "model_missing",
            "message": "OpenRouter model configuration is incomplete",
            "missing_fields": missing_fields,
        }
    return {
        "ok": True,
        "required": True,
        "chat_models": chat_models,
        "embed_model": str(OPENROUTER_EMBED_MODEL or "").strip() if require_embeddings else None,
    }


def _check_opencode_reachable() -> dict[str, object]:
    return _check_opencode_config()
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "required": True, "error": f"httpx_unavailable:{exc}"}
    try:
        response = httpx.get(
            f"{OPENROUTER_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=2.0,
        )
        if not (200 <= int(response.status_code) < 300):
            return {"ok": False, "required": True, "status_code": response.status_code}
        payload = response.json()
        raw_models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(raw_models, list):
            return {
                "ok": False,
                "required": True,
                "status_code": response.status_code,
                "reason": "invalid_response",
                "message": "OpenRouter models endpoint returned invalid JSON",
            }
        available_model_ids = {
            str(item.get("id")).strip()
            for item in raw_models
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        required_model_ids = set(_configured_openrouter_chat_models().values())
        missing_models = sorted(model_id for model_id in required_model_ids if model_id not in available_model_ids)
        if missing_models:
            return {
                "ok": False,
                "required": True,
                "status_code": response.status_code,
                "reason": "model_not_found",
                "message": "One or more configured OpenRouter models are unavailable",
                "missing_models": missing_models,
            }
        return {
            "ok": True,
            "required": True,
            "status_code": response.status_code,
            "chat_models": _configured_openrouter_chat_models(),
            "embed_model": str(OPENROUTER_EMBED_MODEL or "").strip() if require_embeddings else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "required": True, "error": str(exc)}


def _check_required_provider_config() -> dict[str, dict[str, object]]:
    return {
        "openrouter_api": _check_openrouter_reachable(require_embeddings=USE_EMBEDDINGS),
    }


def _required_checks_ok(checks: Mapping[str, object]) -> bool:
    for raw_check in checks.values():
        if not isinstance(raw_check, Mapping):
            return False
        if raw_check.get("required", True) and not bool(raw_check.get("ok")):
            return False
    return True


def _uptime_seconds() -> float:
    return round(max(time.monotonic() - _STARTED_AT, 0.0), 3)


@app.get("/livez")
def livez() -> dict[str, object]:
    return {"status": "alive", "uptime_seconds": _uptime_seconds()}


@app.get("/readyz")
def readyz(db: Session = Depends(get_db)) -> JSONResponse:
    db_check = _check_db(db)
    provider_checks = _check_required_provider_config()
    checks: dict[str, object] = {
        "database": db_check,
        **provider_checks,
    }
    is_ready = bool(db_check.get("ok")) and _required_checks_ok(provider_checks)
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
            "uptime_seconds": _uptime_seconds(),
        },
    )


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, object]:
    db_check = _check_db(db)
    opencode_check = _check_opencode_reachable()
    return {
        "status": "ok",
        "opencode_api": opencode_check,
    }
    provider_checks = {
        "opencode_api": opencode_check,
    }
    all_ok = bool(db_check.get("ok")) and _required_checks_ok(provider_checks)
    return {"status": "ok" if all_ok else "degraded", "checks": checks}


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> Response:
    active_sessions = db.execute(select(func.count()).select_from(models.SessionModel)).scalar_one() or 0
    memory_facts_total = db.execute(
        select(func.count())
        .select_from(models.ObjectModel)
        .where(models.ObjectModel.type == MEMORY_FACT_OBJECT_TYPE)
    ).scalar_one() or 0
    memory_events_total = db.execute(
        select(func.count())
        .select_from(models.ObjectModel)
        .where(models.ObjectModel.type == MEMORY_EVENT_OBJECT_TYPE)
    ).scalar_one() or 0
    pending_consequences = db.execute(
        select(func.count())
        .select_from(models.ObjectModel)
        .where(
            models.ObjectModel.type == CONSEQUENCE_WINDOW_OBJECT_TYPE,
            models.ObjectModel.data["status"].astext.in_(("open", "shown", "suppressed")),
        )
    ).scalar_one() or 0

    payload = render_prometheus(
        {
            "rpg_active_sessions": float(active_sessions),
            "rpg_memory_facts_total": float(memory_facts_total),
            "rpg_memory_events_total": float(memory_events_total),
            "rpg_pending_consequences": float(pending_consequences),
        }
    )
    return Response(
        content=payload,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# =============================================================================
# Playground API - максимально простой эндпоинт для фронтенда
# =============================================================================

class PlaygroundChatIn(BaseModel):
    session_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=4000)


class PlaygroundChatOut(BaseModel):
    session_id: uuid.UUID
    narration: str


@app.post("/playground/chat", response_model=PlaygroundChatOut)
def playground_chat(payload: PlaygroundChatIn) -> PlaygroundChatOut:
    """
    Самый простой способ отправить реплику игроком.
    Создаёт сессию при необходимости и возвращает текст рассказчика.
    """
    if payload.session_id is None:
        new_session = run_with_new_session(
            session_lifecycle_service.create_session,
            schemas.SessionCreateIn(world_name="Playground")
        )
        session_id = new_session.session_id
    else:
        session_id = payload.session_id

    turn_row = run_with_new_session(
        turn_application_service.run_turn,
        session_id,
        schemas.TurnIn(user_input=payload.message),
        allow_debug_patch=ALLOW_DEBUG_PATCH,
    )

    narration = getattr(turn_row, "ai_text", None) or "(Рассказчик молчит...)"

    return PlaygroundChatOut(
        session_id=session_id,
        narration=narration
    )
