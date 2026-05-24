from __future__ import annotations

import hashlib
import uuid
from typing import Any

from sqlalchemy import select

from . import models, outbox_runtime
from .crud_consequences import PRIORITY_WEIGHT, select_due_consequence_window_payloads
from .crud_shared import _acquire_session_turn_lock, _require_session, _session_turn_runtime_lock
from .db import SessionLocal, USE_WORLD_DIRECTOR
from .observability import get_trace_id

DIRECTOR_AGENDA_OBJECT_TYPE = "__director_agenda"
DIRECTOR_ACTION_OBJECT_TYPE = "__director_action"
DIRECTOR_AGENDA_NAME_PREFIX = "director_agenda"
DIRECTOR_ACTION_NAME_PREFIX = "director_action"
MAX_DIRECTOR_CHAIN_DEPTH = 3
WORLD_TICK_DEFAULT_BUDGET = 2


def _agenda_key_from_window(window_key: str) -> str:
    digest = hashlib.sha256(f"agenda:{window_key}".encode("utf-8")).hexdigest()
    return f"agenda:{digest}"


def _action_key_from_agenda(agenda_key: str, turn_index: int) -> str:
    digest = hashlib.sha256(f"action:{agenda_key}:{turn_index}".encode("utf-8")).hexdigest()
    return f"director_action:{digest}"


def enqueue_world_tick(
    db: Any,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    root_turn_index: int,
    triggered_by_turn_index: int,
    chain_depth: int,
    source_window_keys: list[str] | None = None,
    agenda_keys: list[str] | None = None,
    budget: int = WORLD_TICK_DEFAULT_BUDGET,
) -> None:
    if not USE_WORLD_DIRECTOR:
        return
    payload = {
        "root_turn_index": int(root_turn_index),
        "triggered_by_turn_index": int(triggered_by_turn_index),
        "chain_depth": max(int(chain_depth), 0),
        "source_window_keys": list(source_window_keys or []),
        "agenda_keys": list(agenda_keys or []),
        "budget": max(int(budget), 1),
    }
    outbox_runtime.enqueue_outbox_event(
        db,
        event_type=outbox_runtime.EVENT_WORLD_TICK,
        payload=payload,
        session_id=session_id,
        turn_index=turn_index,
        trace_id=get_trace_id(),
        dedupe_key=f"world_tick:{session_id}:{triggered_by_turn_index}:{max(int(chain_depth), 0)}",
    )


def _coerce_agenda_payload(raw_payload: Any) -> dict[str, Any] | None:
    if not isinstance(raw_payload, dict):
        return None
    agenda_key = str(raw_payload.get("agenda_key") or "").strip()
    if not agenda_key:
        return None
    status = str(raw_payload.get("status") or "open").strip().lower()
    if status not in {"open", "running", "closed", "suppressed"}:
        status = "open"
    payload: dict[str, Any] = {
        "agenda_key": agenda_key,
        "actor_object_id": str(raw_payload.get("actor_object_id") or "").strip() or None,
        "agenda_kind": str(raw_payload.get("agenda_kind") or "consequence_window").strip() or "consequence_window",
        "status": status,
        "priority_score": float(raw_payload.get("priority_score") or 0.0),
        "source_window_keys": [
            str(item).strip()
            for item in list(raw_payload.get("source_window_keys") or [])
            if str(item).strip()
        ],
        "source_obligation_keys": [
            str(item).strip()
            for item in list(raw_payload.get("source_obligation_keys") or [])
            if str(item).strip()
        ],
        "economy_pressure_keys": [
            str(item).strip()
            for item in list(raw_payload.get("economy_pressure_keys") or [])
            if str(item).strip()
        ],
        "target_object_ids": [
            str(item).strip()
            for item in list(raw_payload.get("target_object_ids") or [])
            if str(item).strip()
        ],
        "active_dedupe_key": str(raw_payload.get("active_dedupe_key") or "").strip() or None,
        "running_since_turn": raw_payload.get("running_since_turn"),
        "retry_count": int(raw_payload.get("retry_count") or 0),
        "last_evaluated_turn": raw_payload.get("last_evaluated_turn"),
    }
    suppressed_until_turn = raw_payload.get("suppressed_until_turn")
    if isinstance(suppressed_until_turn, int) and not isinstance(suppressed_until_turn, bool):
        payload["suppressed_until_turn"] = suppressed_until_turn
    return payload


def _list_agenda_rows(db: Any, session_id: uuid.UUID) -> list[models.ObjectModel]:
    return db.execute(
        select(models.ObjectModel).where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == DIRECTOR_AGENDA_OBJECT_TYPE,
        )
    ).scalars().all()


def _find_outbox_event_by_dedupe_key(db: Any, dedupe_key: str) -> models.OutboxEventModel | None:
    return db.execute(
        select(models.OutboxEventModel).where(models.OutboxEventModel.dedupe_key == dedupe_key).limit(1)
    ).scalar_one_or_none()


def _load_active_economy_pressures(
    db: Any,
    session_id: uuid.UUID,
    *,
    current_turn: int,
) -> list[dict[str, Any]]:
    turn_row = db.execute(
        select(models.TurnModel.ai_json).where(
            models.TurnModel.session_id == session_id,
            models.TurnModel.turn_index <= current_turn,
        ).order_by(models.TurnModel.turn_index.desc()).limit(1)
    ).scalar_one_or_none()
    ai_json = dict(turn_row or {}) if isinstance(turn_row, dict) else {}
    raw_pressures = ((((ai_json.get("economy") or {}) if isinstance(ai_json, dict) else {}).get("after") or {}).get("pressures") or [])
    pressures: list[dict[str, Any]] = []
    for item in list(raw_pressures):
        if not isinstance(item, dict):
            continue
        pressure_key = str(item.get("pressure_key") or "").strip()
        if not pressure_key:
            continue
        status = str(item.get("status") or "").strip().lower()
        if status not in {"active", "critical"}:
            continue
        pressures.append(
            {
                "pressure_key": pressure_key,
                "entity_id": str(item.get("entity_id") or "").strip() or None,
                "severity": float(item.get("severity") or 0.0),
                "status": status,
            }
        )
    pressures.sort(key=lambda item: (-float(item.get("severity") or 0.0), str(item.get("pressure_key") or "")))
    return pressures


def _refresh_or_materialize_agendas(
    db: Any,
    session_id: uuid.UUID,
    *,
    current_turn: int,
    budget: int,
) -> list[dict[str, Any]]:
    agenda_rows = _list_agenda_rows(db, session_id)
    agenda_by_key = {
        str(dict(row.data or {}).get("agenda_key") or "").strip(): row
        for row in agenda_rows
        if str(dict(row.data or {}).get("agenda_key") or "").strip()
    }
    active_pressures = _load_active_economy_pressures(
        db,
        session_id,
        current_turn=current_turn,
    )
    due_windows = select_due_consequence_window_payloads(
        db,
        session_id,
        turn_index=current_turn,
        budget=max(budget * 4, budget),
    )
    for window in due_windows:
        window_key = str(window.get("window_key") or "").strip()
        if not window_key:
            continue
        agenda_key = _agenda_key_from_window(window_key)
        target_object_ids = list(window.get("target_object_ids") or [])
        matched_pressures = [
            pressure
            for pressure in active_pressures
            if str(pressure.get("entity_id") or "").strip()
            and str(pressure.get("entity_id") or "").strip() in set(target_object_ids)
        ]
        pressure_keys = [str(item.get("pressure_key") or "").strip() for item in matched_pressures if str(item.get("pressure_key") or "").strip()]
        pressure_boost = max((float(item.get("severity") or 0.0) for item in matched_pressures), default=0.0)
        priority_score = float(PRIORITY_WEIGHT.get(str(window.get("priority") or "med"), PRIORITY_WEIGHT["med"])) + pressure_boost
        payload = {
            "agenda_key": agenda_key,
            "actor_object_id": next(iter(target_object_ids), None),
            "agenda_kind": "consequence_window",
            "status": "open",
            "priority_score": priority_score,
            "source_window_keys": [window_key],
            "source_obligation_keys": list(window.get("opened_obligation_keys") or []),
            "economy_pressure_keys": pressure_keys,
            "target_object_ids": target_object_ids,
            "active_dedupe_key": None,
            "running_since_turn": None,
            "retry_count": 0,
            "last_evaluated_turn": current_turn,
        }
        existing_row = agenda_by_key.get(agenda_key)
        if existing_row is None:
            db.add(
                models.ObjectModel(
                    session_id=session_id,
                    type=DIRECTOR_AGENDA_OBJECT_TYPE,
                    name=f"{DIRECTOR_AGENDA_NAME_PREFIX}:{agenda_key}",
                    data=payload,
                )
            )
            continue
        current_payload = _coerce_agenda_payload(dict(existing_row.data or {})) or {}
        status = str(current_payload.get("status") or "open").strip()
        if status == "suppressed":
            suppressed_until_turn = current_payload.get("suppressed_until_turn")
            if isinstance(suppressed_until_turn, int) and current_turn <= suppressed_until_turn:
                payload["status"] = "suppressed"
                payload["suppressed_until_turn"] = suppressed_until_turn
            else:
                payload["status"] = "open"
        elif status in {"running", "closed"}:
            payload["status"] = status
        payload["active_dedupe_key"] = current_payload.get("active_dedupe_key")
        payload["running_since_turn"] = current_payload.get("running_since_turn")
        payload["retry_count"] = int(current_payload.get("retry_count") or 0)
        existing_row.name = f"{DIRECTOR_AGENDA_NAME_PREFIX}:{agenda_key}"
        existing_row.data = payload
    db.flush()

    refreshed_payloads: list[dict[str, Any]] = []
    for row in _list_agenda_rows(db, session_id):
        payload = _coerce_agenda_payload(dict(row.data or {}))
        if payload is None:
            continue
        if payload.get("status") == "running":
            dedupe_key = str(payload.get("active_dedupe_key") or "").strip()
            if dedupe_key:
                outbox_row = _find_outbox_event_by_dedupe_key(db, dedupe_key)
                if outbox_row is not None and outbox_row.status == outbox_runtime.OUTBOX_STATUS_FAILED:
                    payload["status"] = "suppressed"
                    payload["suppressed_until_turn"] = current_turn + 1
                    payload["active_dedupe_key"] = None
                    payload["running_since_turn"] = None
                    payload["retry_count"] = int(payload.get("retry_count") or 0) + 1
                    row.data = payload
        refreshed_payloads.append(payload)
    db.flush()
    return refreshed_payloads


def _schedule_due_agendas(
    db: Any,
    session_id: uuid.UUID,
    *,
    root_turn_index: int,
    triggered_by_turn_index: int,
    chain_depth: int,
    current_turn: int,
    budget: int,
) -> int:
    agendas = _refresh_or_materialize_agendas(
        db,
        session_id,
        current_turn=current_turn,
        budget=budget,
    )
    open_agendas = [
        payload
        for payload in agendas
        if str(payload.get("status") or "").strip() == "open"
    ]
    open_agendas.sort(
        key=lambda item: (
            -float(item.get("priority_score") or 0.0),
            str(item.get("agenda_key") or ""),
        )
    )
    scheduled = 0
    agenda_rows = {
        str(dict(row.data or {}).get("agenda_key") or "").strip(): row
        for row in _list_agenda_rows(db, session_id)
    }
    for payload in open_agendas[: max(min(budget, 2), 0)]:
        agenda_key = str(payload.get("agenda_key") or "").strip()
        row = agenda_rows.get(agenda_key)
        if row is None:
            continue
        dedupe_key = f"npc_agenda_tick:{session_id}:{agenda_key}:{triggered_by_turn_index}"
        payload["status"] = "running"
        payload["active_dedupe_key"] = dedupe_key
        payload["running_since_turn"] = current_turn
        payload["last_evaluated_turn"] = current_turn
        row.data = payload
        outbox_runtime.enqueue_outbox_event(
            db,
            event_type=outbox_runtime.EVENT_NPC_AGENDA_TICK,
            payload={
                "root_turn_index": int(root_turn_index),
                "triggered_by_turn_index": int(triggered_by_turn_index),
                "chain_depth": max(int(chain_depth), 0) + 1,
                "source_window_keys": list(payload.get("source_window_keys") or []),
                "agenda_keys": [agenda_key],
                "budget": max(int(budget), 1),
                "agenda_key": agenda_key,
            },
            session_id=session_id,
            turn_index=current_turn,
            trace_id=get_trace_id(),
            dedupe_key=dedupe_key,
        )
        scheduled += 1
    db.flush()
    return scheduled


def _run_world_tick_outbox_event(
    *,
    session_id: uuid.UUID,
    payload: dict[str, Any],
) -> int:
    if not USE_WORLD_DIRECTOR:
        return 0
    db = SessionLocal()
    try:
        with _session_turn_runtime_lock(db, session_id):
            with db.begin():
                _acquire_session_turn_lock(db, session_id)
                session_row = _require_session(db, session_id, for_update=True)
                current_turn = max(int(getattr(session_row, "turn_index", 0) or 0), 0)
                chain_depth = max(int(payload.get("chain_depth") or 0), 0)
                if chain_depth >= MAX_DIRECTOR_CHAIN_DEPTH:
                    return 0
                return _schedule_due_agendas(
                    db,
                    session_id,
                    root_turn_index=max(int(payload.get("root_turn_index") or current_turn), 1),
                    triggered_by_turn_index=max(int(payload.get("triggered_by_turn_index") or current_turn), 1),
                    chain_depth=chain_depth,
                    current_turn=current_turn,
                    budget=max(int(payload.get("budget") or WORLD_TICK_DEFAULT_BUDGET), 1),
                )
    finally:
        if db.in_transaction():
            db.rollback()
        db.close()


def _run_npc_agenda_tick_outbox_event(
    *,
    session_id: uuid.UUID,
    payload: dict[str, Any],
) -> int:
    if not USE_WORLD_DIRECTOR:
        return 0
    agenda_key = str(payload.get("agenda_key") or "").strip()
    if not agenda_key:
        raise RuntimeError("npc_agenda_tick payload missing agenda_key")

    db = SessionLocal()
    try:
        with _session_turn_runtime_lock(db, session_id):
            with db.begin():
                _acquire_session_turn_lock(db, session_id)
                agenda_row = db.execute(
                    select(models.ObjectModel).where(
                        models.ObjectModel.session_id == session_id,
                        models.ObjectModel.type == DIRECTOR_AGENDA_OBJECT_TYPE,
                        models.ObjectModel.data["agenda_key"].astext == agenda_key,
                    )
                ).scalar_one_or_none()
                if agenda_row is None:
                    return 0
                agenda_payload = _coerce_agenda_payload(dict(agenda_row.data or {}))
                if agenda_payload is None or str(agenda_payload.get("status") or "").strip() != "running":
                    return 0
                actor_raw = str(agenda_payload.get("actor_object_id") or "").strip()
                actor_object_id = None
                if actor_raw:
                    try:
                        actor_object_id = uuid.UUID(actor_raw)
                    except ValueError:
                        actor_object_id = None

            from .application.turn_services import turn_application_service

            turn_row = turn_application_service.run_director_turn_locked(
                db,
                session_id,
                actor_object_id=actor_object_id,
                root_turn_index=max(int(payload.get("root_turn_index") or 1), 1),
                triggered_by_turn_index=max(int(payload.get("triggered_by_turn_index") or 1), 1),
                chain_depth=max(int(payload.get("chain_depth") or 1), 1),
                source_window_keys=list(payload.get("source_window_keys") or []),
                economy_pressure_keys=list(agenda_payload.get("economy_pressure_keys") or []),
                agenda_key=agenda_key,
                budget=max(int(payload.get("budget") or WORLD_TICK_DEFAULT_BUDGET), 1),
            )

            with db.begin():
                _acquire_session_turn_lock(db, session_id)
                agenda_row = db.execute(
                    select(models.ObjectModel).where(
                        models.ObjectModel.session_id == session_id,
                        models.ObjectModel.type == DIRECTOR_AGENDA_OBJECT_TYPE,
                        models.ObjectModel.data["agenda_key"].astext == agenda_key,
                    )
                ).scalar_one_or_none()
                if agenda_row is not None:
                    agenda_payload = _coerce_agenda_payload(dict(agenda_row.data or {})) or {"agenda_key": agenda_key}
                    agenda_payload["status"] = "closed"
                    agenda_payload["active_dedupe_key"] = None
                    agenda_payload["running_since_turn"] = None
                    agenda_payload["last_evaluated_turn"] = int(turn_row.turn_index)
                    agenda_row.data = agenda_payload
                action_key = _action_key_from_agenda(agenda_key, int(turn_row.turn_index))
                action_payload = {
                    "action_key": action_key,
                    "turn_index": int(turn_row.turn_index),
                    "root_turn_index": max(int(payload.get("root_turn_index") or turn_row.turn_index), 1),
                    "triggered_by_turn_index": max(int(payload.get("triggered_by_turn_index") or turn_row.turn_index), 1),
                    "actor_object_id": actor_raw or None,
                    "agenda_key": agenda_key,
                    "source_window_keys": list(payload.get("source_window_keys") or []),
                    "target_object_ids": list((agenda_payload or {}).get("target_object_ids") or []),
                    "applied_ops_summary": [
                        str(item.get("op") or "").strip()
                        for item in list(((turn_row.ai_json or {}) if isinstance(turn_row.ai_json, dict) else {}).get("applied_ops") or [])[:8]
                        if isinstance(item, dict) and str(item.get("op") or "").strip()
                    ],
                    "result_status": str((((turn_row.ai_json or {}) if isinstance(turn_row.ai_json, dict) else {}).get("validator") or {}).get("status") or "ok"),
                    "chronicle_embedding_snippet": str(((turn_row.ai_json or {}) if isinstance(turn_row.ai_json, dict) else {}).get("chronicle_embedding_snippet") or "").strip(),
                }
                db.add(
                    models.ObjectModel(
                        session_id=session_id,
                        type=DIRECTOR_ACTION_OBJECT_TYPE,
                        name=f"{DIRECTOR_ACTION_NAME_PREFIX}:{action_key}",
                        data=action_payload,
                    )
                )
                if max(int(payload.get("chain_depth") or 0), 0) < MAX_DIRECTOR_CHAIN_DEPTH:
                    enqueue_world_tick(
                        db,
                        session_id=session_id,
                        turn_index=int(turn_row.turn_index),
                        root_turn_index=max(int(payload.get("root_turn_index") or turn_row.turn_index), 1),
                        triggered_by_turn_index=int(turn_row.turn_index),
                        chain_depth=max(int(payload.get("chain_depth") or 0), 0),
                        source_window_keys=list(payload.get("source_window_keys") or []),
                        agenda_keys=[agenda_key],
                        budget=max(int(payload.get("budget") or WORLD_TICK_DEFAULT_BUDGET), 1),
                    )
                db.flush()
            return 1
    finally:
        if db.in_transaction():
            db.rollback()
        db.close()


__all__ = [
    "DIRECTOR_ACTION_OBJECT_TYPE",
    "DIRECTOR_AGENDA_OBJECT_TYPE",
    "MAX_DIRECTOR_CHAIN_DEPTH",
    "WORLD_TICK_DEFAULT_BUDGET",
    "enqueue_world_tick",
    "_run_world_tick_outbox_event",
    "_run_npc_agenda_tick_outbox_event",
]
