from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
import uuid
from typing import Any, Literal

from fastapi import HTTPException, status
from sqlalchemy import Date, case, cast, func, select
from sqlalchemy.orm import Session

from . import models, prompt_registry
from .llm_pricing import estimate_cost_cents

CostMode = Literal["snapshot", "current", "both"]


def _coerce_cost_mode(raw: str | None) -> CostMode:
    text = str(raw or "both").strip().lower()
    if text in {"snapshot", "current", "both"}:
        return text  # type: ignore[return-value]
    return "both"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _apply_cost_mode(
    *,
    snapshot: float | None,
    current: float | None,
    cost_mode: CostMode,
) -> tuple[float | None, float | None, float | None]:
    if cost_mode == "snapshot":
        return snapshot, None, None
    if cost_mode == "current":
        return None, current, None

    delta = None
    if snapshot is not None and current is not None:
        delta = round(current - snapshot, 4)
    return snapshot, current, delta


def _require_session_exists(db: Session, session_id: uuid.UUID) -> None:
    exists = db.execute(
        select(models.SessionModel.id)
        .where(models.SessionModel.id == session_id)
        .limit(1)
    ).scalar_one_or_none()
    if exists is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")


def _append_summary_bucket(
    buckets: dict[tuple[str, str, str], dict[str, Any]],
    *,
    provider: str,
    model_name: str,
    request_type: str,
    calls_total: int,
    errors_total: int,
    prompt_tokens_total: int,
    completion_tokens_total: int,
    total_tokens_total: int,
    cost_cents_total_snapshot: float,
    latency_ms_avg: float,
) -> None:
    key = (provider, model_name, request_type)
    bucket = buckets.setdefault(
        key,
        {
            "provider": provider,
            "model_name": model_name,
            "request_type": request_type,
            "calls_total": 0,
            "errors_total": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "cost_cents_total_snapshot": 0.0,
            "_latency_weighted_sum": 0.0,
        },
    )
    bucket["calls_total"] += max(int(calls_total), 0)
    bucket["errors_total"] += max(int(errors_total), 0)
    bucket["prompt_tokens_total"] += max(int(prompt_tokens_total), 0)
    bucket["completion_tokens_total"] += max(int(completion_tokens_total), 0)
    bucket["total_tokens_total"] += max(int(total_tokens_total), 0)
    bucket["cost_cents_total_snapshot"] += float(cost_cents_total_snapshot)
    bucket["_latency_weighted_sum"] += max(float(latency_ms_avg), 0.0) * max(int(calls_total), 0)


def _list_daily_raw_telemetry_rows(
    db: Session,
    *,
    cutoff: datetime,
    now_utc: datetime,
) -> list[Any]:
    day_utc = cast(func.timezone("UTC", models.LlmTelemetryModel.created_at), Date).label("day_utc")
    return db.execute(
        select(
            day_utc,
            models.LlmTelemetryModel.provider,
            models.LlmTelemetryModel.model_name,
            models.LlmTelemetryModel.request_type,
            func.count().label("calls_total"),
            func.sum(case((models.LlmTelemetryModel.status == "error", 1), else_=0)).label("errors_total"),
            func.coalesce(func.sum(models.LlmTelemetryModel.prompt_tokens), 0).label("prompt_tokens_total"),
            func.coalesce(func.sum(models.LlmTelemetryModel.completion_tokens), 0).label("completion_tokens_total"),
            func.coalesce(func.sum(models.LlmTelemetryModel.total_tokens), 0).label("total_tokens_total"),
            func.coalesce(func.sum(models.LlmTelemetryModel.cost_cents), 0).label("cost_cents_total_snapshot"),
            func.coalesce(func.avg(models.LlmTelemetryModel.latency_ms), 0).label("latency_ms_avg"),
        )
        .where(
            models.LlmTelemetryModel.created_at >= cutoff,
            models.LlmTelemetryModel.created_at <= now_utc,
        )
        .group_by(
            day_utc,
            models.LlmTelemetryModel.provider,
            models.LlmTelemetryModel.model_name,
            models.LlmTelemetryModel.request_type,
        )
    ).all()


def _list_daily_rollup_rows(
    db: Session,
    *,
    cutoff_day: date,
    now_day: date,
) -> list[Any]:
    return db.execute(
        select(
            models.LlmTelemetryDailyRollupModel.day_utc,
            models.LlmTelemetryDailyRollupModel.provider,
            models.LlmTelemetryDailyRollupModel.model_name,
            models.LlmTelemetryDailyRollupModel.request_type,
            models.LlmTelemetryDailyRollupModel.calls_total,
            models.LlmTelemetryDailyRollupModel.errors_total,
            models.LlmTelemetryDailyRollupModel.prompt_tokens_total,
            models.LlmTelemetryDailyRollupModel.completion_tokens_total,
            models.LlmTelemetryDailyRollupModel.total_tokens_total,
            models.LlmTelemetryDailyRollupModel.cost_cents_total_snapshot,
            models.LlmTelemetryDailyRollupModel.latency_ms_avg,
        )
        .where(
            models.LlmTelemetryDailyRollupModel.day_utc >= cutoff_day,
            models.LlmTelemetryDailyRollupModel.day_utc <= now_day,
        )
    ).all()


def _is_full_utc_day_in_window(
    day_utc: date,
    *,
    cutoff: datetime,
    now_utc: datetime,
) -> bool:
    day_start = datetime.combine(day_utc, time.min, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    return day_start >= cutoff and day_end <= now_utc


def get_llm_telemetry_summary(
    db: Session,
    *,
    days: int = 7,
    cost_mode: str | None = None,
) -> dict[str, Any]:
    normalized_mode = _coerce_cost_mode(cost_mode)
    window_days = max(int(days), 1)
    now_utc = datetime.now(UTC)
    cutoff = now_utc - timedelta(days=window_days)

    # Retention keeps an exact sliding raw window, while historical full UTC
    # days are preserved in daily rollups. Use raw rows for partial edge days
    # of the requested window and rollups for full UTC days inside the window.
    # If the preferred source is missing, fall back to the other one.
    raw_rows = _list_daily_raw_telemetry_rows(
        db,
        cutoff=cutoff,
        now_utc=now_utc,
    )
    rollup_rows = _list_daily_rollup_rows(
        db,
        cutoff_day=cutoff.date(),
        now_day=now_utc.date(),
    )
    by_bucket: list[dict[str, Any]] = []
    totals = {
        "calls_total": 0,
        "errors_total": 0,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "total_tokens_total": 0,
        "cost_cents_total_snapshot": 0.0,
        "cost_cents_total_current_estimate": 0.0,
    }

    raw_daily_rows: dict[tuple[date, str, str, str], Any] = {}
    for row in raw_rows:
        day_utc = getattr(row, "day_utc", None)
        provider = str(getattr(row, "provider", "") or "")
        model_name = str(getattr(row, "model_name", "") or "")
        request_type = str(getattr(row, "request_type", "") or "")
        if not isinstance(day_utc, date) or not provider or not model_name or not request_type:
            continue
        raw_daily_rows[(day_utc, provider, model_name, request_type)] = row
    rollup_daily_rows: dict[tuple[date, str, str, str], Any] = {}
    for row in rollup_rows:
        day_utc = getattr(row, "day_utc", None)
        provider = str(getattr(row, "provider", "") or "")
        model_name = str(getattr(row, "model_name", "") or "")
        request_type = str(getattr(row, "request_type", "") or "")
        if not isinstance(day_utc, date) or not provider or not model_name or not request_type:
            continue
        rollup_daily_rows[(day_utc, provider, model_name, request_type)] = row

    bucket_totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    all_daily_keys = set(raw_daily_rows) | set(rollup_daily_rows)
    for daily_key in all_daily_keys:
        day_utc, _provider, _model_name, _request_type = daily_key
        prefer_rollup = _is_full_utc_day_in_window(
            day_utc,
            cutoff=cutoff,
            now_utc=now_utc,
        )
        primary_row = rollup_daily_rows.get(daily_key) if prefer_rollup else raw_daily_rows.get(daily_key)
        fallback_row = raw_daily_rows.get(daily_key) if prefer_rollup else rollup_daily_rows.get(daily_key)
        row = primary_row or fallback_row
        if row is None:
            continue
        _append_summary_bucket(
            bucket_totals,
            provider=str(getattr(row, "provider", "") or ""),
            model_name=str(getattr(row, "model_name", "") or ""),
            request_type=str(getattr(row, "request_type", "") or ""),
            calls_total=int(getattr(row, "calls_total", 0) or 0),
            errors_total=int(getattr(row, "errors_total", 0) or 0),
            prompt_tokens_total=int(getattr(row, "prompt_tokens_total", 0) or 0),
            completion_tokens_total=int(getattr(row, "completion_tokens_total", 0) or 0),
            total_tokens_total=int(getattr(row, "total_tokens_total", 0) or 0),
            cost_cents_total_snapshot=_to_float(getattr(row, "cost_cents_total_snapshot", 0.0)) or 0.0,
            latency_ms_avg=_to_float(getattr(row, "latency_ms_avg", 0.0)) or 0.0,
        )

    for row in sorted(
        bucket_totals.values(),
        key=lambda item: (
            str(item.get("provider") or ""),
            str(item.get("model_name") or ""),
            str(item.get("request_type") or ""),
        ),
    ):
        prompt_tokens = int(row["prompt_tokens_total"] or 0)
        completion_tokens = int(row["completion_tokens_total"] or 0)
        current_estimate = estimate_cost_cents(
            model_name=str(row["model_name"]),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        snapshot_total = _to_float(row["cost_cents_total_snapshot"]) or 0.0
        snapshot, current, delta = _apply_cost_mode(
            snapshot=snapshot_total,
            current=current_estimate,
            cost_mode=normalized_mode,
        )

        calls_total = int(row["calls_total"] or 0)
        totals["calls_total"] += calls_total
        totals["errors_total"] += int(row["errors_total"] or 0)
        totals["prompt_tokens_total"] += prompt_tokens
        totals["completion_tokens_total"] += completion_tokens
        totals["total_tokens_total"] += int(row["total_tokens_total"] or 0)
        totals["cost_cents_total_snapshot"] += snapshot_total
        totals["cost_cents_total_current_estimate"] += float(current_estimate or 0.0)
        latency_weighted_sum = float(row.get("_latency_weighted_sum") or 0.0)
        latency_ms_avg = 0.0 if calls_total <= 0 else round(latency_weighted_sum / calls_total, 2)

        by_bucket.append(
            {
                "provider": row["provider"],
                "model_name": row["model_name"],
                "request_type": row["request_type"],
                "calls_total": calls_total,
                "errors_total": int(row["errors_total"] or 0),
                "prompt_tokens_total": prompt_tokens,
                "completion_tokens_total": completion_tokens,
                "total_tokens_total": int(row["total_tokens_total"] or 0),
                "cost_cents_snapshot": snapshot,
                "cost_cents_current_estimate": current,
                "cost_delta_cents": delta,
                "latency_ms_avg": latency_ms_avg,
            }
        )

    total_snapshot, total_current, total_delta = _apply_cost_mode(
        snapshot=round(float(totals["cost_cents_total_snapshot"]), 4),
        current=round(float(totals["cost_cents_total_current_estimate"]), 4),
        cost_mode=normalized_mode,
    )

    return {
        "window_days": window_days,
        "cost_mode": normalized_mode,
        "totals": {
            "calls_total": totals["calls_total"],
            "errors_total": totals["errors_total"],
            "prompt_tokens_total": totals["prompt_tokens_total"],
            "completion_tokens_total": totals["completion_tokens_total"],
            "total_tokens_total": totals["total_tokens_total"],
            "cost_cents_snapshot": total_snapshot,
            "cost_cents_current_estimate": total_current,
            "cost_delta_cents": total_delta,
        },
        "by_bucket": by_bucket,
    }


def list_session_llm_telemetry(
    db: Session,
    *,
    session_id: uuid.UUID,
    limit: int = 100,
    offset: int = 0,
    turn_index: int | None = None,
    request_type: str | None = None,
    provider: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    cost_mode: str | None = None,
) -> dict[str, Any]:
    _require_session_exists(db, session_id)
    normalized_mode = _coerce_cost_mode(cost_mode)

    where_clauses = [models.LlmTelemetryModel.session_id == session_id]
    if turn_index is not None:
        where_clauses.append(models.LlmTelemetryModel.turn_index == int(turn_index))
    if request_type:
        where_clauses.append(models.LlmTelemetryModel.request_type == str(request_type).strip())
    if provider:
        where_clauses.append(models.LlmTelemetryModel.provider == str(provider).strip())
    if from_ts is not None:
        where_clauses.append(models.LlmTelemetryModel.created_at >= from_ts)
    if to_ts is not None:
        where_clauses.append(models.LlmTelemetryModel.created_at <= to_ts)

    total = db.execute(
        select(func.count())
        .select_from(models.LlmTelemetryModel)
        .where(*where_clauses)
    ).scalar_one() or 0

    rows = db.execute(
        select(models.LlmTelemetryModel)
        .where(*where_clauses)
        .order_by(models.LlmTelemetryModel.created_at.desc(), models.LlmTelemetryModel.id.desc())
        .offset(max(int(offset), 0))
        .limit(min(max(int(limit), 1), 500))
    ).scalars().all()

    items: list[dict[str, Any]] = []
    for row in rows:
        current_estimate = estimate_cost_cents(
            model_name=row.model_name,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
        )
        snapshot, current, delta = _apply_cost_mode(
            snapshot=_to_float(row.cost_cents),
            current=current_estimate,
            cost_mode=normalized_mode,
        )
        items.append(
            {
                "id": int(row.id),
                "created_at": row.created_at,
                "trace_id": row.trace_id,
                "origin_trace_id": row.origin_trace_id,
                "session_id": row.session_id,
                "turn_index": row.turn_index,
                "provider": row.provider,
                "model_name": row.model_name,
                "request_type": row.request_type,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "total_tokens": row.total_tokens,
                "cost_cents_snapshot": snapshot,
                "cost_cents_current_estimate": current,
                "cost_delta_cents": delta,
                "pricing_revision": row.pricing_revision,
                "latency_ms": row.latency_ms,
                "status_code": row.status_code,
                "status": row.status,
                "error_type": row.error_type,
                "meta_json": dict(row.meta_json or {}),
            }
        )

    return {
        "session_id": session_id,
        "cost_mode": normalized_mode,
        "total": int(total),
        "limit": min(max(int(limit), 1), 500),
        "offset": max(int(offset), 0),
        "items": items,
    }


def list_outbox_events(
    db: Session,
    *,
    status_filter: str | None = None,
    event_type: str | None = None,
    session_id: uuid.UUID | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    clauses = []
    if status_filter:
        clauses.append(models.OutboxEventModel.status == str(status_filter).strip())
    if event_type:
        clauses.append(models.OutboxEventModel.event_type == str(event_type).strip())
    if session_id is not None:
        clauses.append(models.OutboxEventModel.session_id == session_id)

    total = db.execute(
        select(func.count())
        .select_from(models.OutboxEventModel)
        .where(*clauses)
    ).scalar_one() or 0

    rows = db.execute(
        select(models.OutboxEventModel)
        .where(*clauses)
        .order_by(models.OutboxEventModel.created_at.desc())
        .offset(max(int(offset), 0))
        .limit(min(max(int(limit), 1), 500))
    ).scalars().all()

    items = [
        {
            "event_id": row.event_id,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "available_at": row.available_at,
            "locked_at": row.locked_at,
            "processed_at": row.processed_at,
            "status": row.status,
            "event_type": row.event_type,
            "session_id": row.session_id,
            "turn_index": row.turn_index,
            "attempts": row.attempts,
            "max_attempts": row.max_attempts,
            "last_error": row.last_error,
            "trace_id": row.trace_id,
            "dedupe_key": row.dedupe_key,
            "payload": dict(row.payload or {}),
        }
        for row in rows
    ]

    return {
        "total": int(total),
        "limit": min(max(int(limit), 1), 500),
        "offset": max(int(offset), 0),
        "items": items,
    }


def list_active_system_prompts(db: Session) -> list[models.SystemPromptRegistryModel]:
    return prompt_registry.list_active_prompts(db)


def activate_system_prompt(
    db: Session,
    *,
    module: str,
    version: int,
) -> models.SystemPromptRegistryModel:
    return prompt_registry.activate_prompt_version(db, module=module, version=version)


__all__ = [
    "activate_system_prompt",
    "get_llm_telemetry_summary",
    "list_active_system_prompts",
    "list_outbox_events",
    "list_session_llm_telemetry",
]
