from __future__ import annotations

import contextvars
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from . import models
from .db import SessionLocal
from .llm_pricing import estimate_cost_cents, pricing_revision
from .observability import get_trace_id

logger = logging.getLogger(__name__)

_UNSET = object()
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_session_id", default=None)
_turn_index_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("llm_turn_index", default=None)
_request_type_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_request_type", default=None)
_origin_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_origin_trace_id", default=None)
_telemetry_disabled_until: float = 0.0


def _coerce_uuid(raw: Any) -> uuid.UUID | None:
    if isinstance(raw, uuid.UUID):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return uuid.UUID(text)
        except ValueError:
            return None
    return None


def _coerce_int(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


@contextmanager
def telemetry_context(
    *,
    session_id: str | uuid.UUID | None | object = _UNSET,
    turn_index: int | None | object = _UNSET,
    request_type: str | None | object = _UNSET,
    origin_trace_id: str | None | object = _UNSET,
) -> Iterator[None]:
    tokens: list[tuple[str, contextvars.Token[Any]]] = []
    try:
        if session_id is not _UNSET:
            value = str(session_id).strip() if session_id is not None else None
            tokens.append(("session", _session_id_var.set(value or None)))
        if turn_index is not _UNSET:
            tokens.append(("turn", _turn_index_var.set(_coerce_int(turn_index))))
        if request_type is not _UNSET:
            value = str(request_type).strip() if request_type is not None else None
            tokens.append(("request", _request_type_var.set(value or None)))
        if origin_trace_id is not _UNSET:
            value = str(origin_trace_id).strip() if origin_trace_id is not None else None
            tokens.append(("origin_trace", _origin_trace_id_var.set(value or None)))
        yield
    finally:
        for key, token in reversed(tokens):
            if key == "session":
                _session_id_var.reset(token)
            elif key == "turn":
                _turn_index_var.reset(token)
            elif key == "request":
                _request_type_var.reset(token)
            elif key == "origin_trace":
                _origin_trace_id_var.reset(token)


def current_request_type(default: str = "unknown") -> str:
    value = str(_request_type_var.get() or "").strip()
    return value or default


def current_session_id() -> uuid.UUID | None:
    return _coerce_uuid(_session_id_var.get())


def current_turn_index() -> int | None:
    value = _coerce_int(_turn_index_var.get())
    if value is None:
        return None
    return max(value, 0)


def current_origin_trace_id() -> str | None:
    value = str(_origin_trace_id_var.get() or "").strip()
    return value or None


def _coerce_token(raw: Any) -> int | None:
    value = _coerce_int(raw)
    if value is None:
        return None
    return max(value, 0)


def record_llm_telemetry(
    *,
    provider: str,
    model_name: str,
    latency_ms: int,
    status: str,
    status_code: int | None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    error_type: str | None = None,
    meta_json: dict[str, Any] | None = None,
    request_type: str | None = None,
    session_id: str | uuid.UUID | None = None,
    turn_index: int | None = None,
    trace_id: str | None = None,
    origin_trace_id: str | None = None,
) -> None:
    global _telemetry_disabled_until
    now = time.monotonic()
    if now < _telemetry_disabled_until:
        return

    provider_name = str(provider or "").strip()
    model = str(model_name or "").strip()
    if not provider_name or not model:
        return

    normalized_prompt = _coerce_token(prompt_tokens)
    normalized_completion = _coerce_token(completion_tokens)
    normalized_total = _coerce_token(total_tokens)
    if normalized_total is None and (normalized_prompt is not None or normalized_completion is not None):
        normalized_total = (normalized_prompt or 0) + (normalized_completion or 0)

    sid = _coerce_uuid(session_id) or current_session_id()
    turn = _coerce_int(turn_index)
    if turn is None:
        turn = current_turn_index()
    if turn is not None:
        turn = max(turn, 0)

    resolved_request_type = str(request_type or "").strip() or current_request_type()
    resolved_trace_id = str(trace_id or "").strip() or (get_trace_id() or None)
    resolved_origin_trace_id = str(origin_trace_id or "").strip() or current_origin_trace_id()
    normalized_status = "ok" if str(status).strip().lower() == "ok" else "error"
    normalized_latency = max(int(latency_ms or 0), 0)

    cost_snapshot = estimate_cost_cents(
        model_name=model,
        prompt_tokens=normalized_prompt,
        completion_tokens=normalized_completion,
    )
    pricing_rev = pricing_revision() if cost_snapshot is not None else None

    db = SessionLocal()
    try:
        with db.begin():
            db.add(
                models.LlmTelemetryModel(
                    trace_id=resolved_trace_id,
                    origin_trace_id=resolved_origin_trace_id,
                    session_id=sid,
                    turn_index=turn,
                    provider=provider_name,
                    model_name=model,
                    request_type=resolved_request_type,
                    prompt_tokens=normalized_prompt,
                    completion_tokens=normalized_completion,
                    total_tokens=normalized_total,
                    cost_cents=cost_snapshot,
                    pricing_revision=pricing_rev,
                    latency_ms=normalized_latency,
                    status_code=status_code,
                    status=normalized_status,
                    error_type=(str(error_type).strip() or None) if error_type is not None else None,
                    meta_json=dict(meta_json or {}),
                )
            )
    except Exception:  # noqa: BLE001
        _telemetry_disabled_until = time.monotonic() + 30.0
        logger.exception(
            "Failed to persist llm telemetry row provider=%s model=%s request_type=%s",
            provider_name,
            model,
            resolved_request_type,
        )
    finally:
        db.close()


__all__ = [
    "telemetry_context",
    "current_request_type",
    "record_llm_telemetry",
]
