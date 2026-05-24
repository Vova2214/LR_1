"""Dynamic graph relationship health checks via DeepSeek.

Periodically analyzes recent events and proposes link.create / link.close
operations that are stored in ``state_json["pending_graph_ops"]`` until
an explicit apply request is made. Selected read/write helpers remain
deprecated compatibility shims over the graph repository.
"""
from __future__ import annotations

from contextlib import nullcontext
import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from fastapi import HTTPException, status

from . import models, schemas
from .architecture_contracts import COMPATIBILITY_MODULE_CONTRACTS
from . import crud_entities as _entities
from .crud_planning import (
    _coerce_patch_op,
    _validate_patch_ops,
)
from .crud_patch_apply import apply_patch_ops
from .crud_shared import (
    TurnApplyExternalArtifacts,
    TurnApplyExternalPreparationRequired,
    _prepare_turn_apply_external_request,
    _acquire_session_turn_lock,
    _normalize_json_preview,
    _recover_abandoned_pending_turn_locked,
    _require_session,
    _safe_int,
    _session_turn_runtime_lock,
    turn_apply_external_artifacts_context,
)
from .db import OPENROUTER_CHAT_MODEL, SessionLocal, TURN_CONTEXT_MAX_CHARS
from .llm import openrouter_chat
from .llm_telemetry import telemetry_context
from .persistence.graph_repository import graph_repository

logger = logging.getLogger(__name__)
COMPATIBILITY_MODULE_CONTRACT = COMPATIBILITY_MODULE_CONTRACTS[__name__]

GRAPH_HEALTH_CHECK_INTERVAL_TURNS = 5
GRAPH_HEALTH_MAX_EVENTS = 20

_GRAPH_HEALTH_CHECK_SYSTEM = (
    "You are a graph consistency auditor for a text RPG game engine. "
    "Given recent events, a list of zone entities with their existing relationships, and asymmetric social links, "
    "propose link.create and link.close operations to keep the relationship graph accurate.\n\n"
    "RULES:\n"
    "- Only propose relationships that are clearly implied by the events.\n"
    "- Do NOT infer from a single ambiguous event. Require at least two corroborating facts.\n"
    "- Allowed link types: knows, friends_with, allied_with, hostile_to, family, "
    "employer, employee, rival.\n"
    "- Never use link.create type='located_in'.\n"
    "- Never use link.close type='located_in' or type='tracking_quest'.\n"
    "- When a relationship flips (for example hostile_to -> friends_with), emit link.close for obsolete links.\n"
    "- asymmetric_relationships are one-way active social links missing the reverse edge; "
    "you may repair with link.create (bidirectional=true is allowed) or intentionally keep one-way if justified.\n"
    "- Use existing object UUIDs (not tmp: refs) since all entities already exist.\n"
    "- If no new relationships are warranted, return {\"proposed_ops\": []}.\n"
    "- Return ONLY valid JSON: {\"proposed_ops\": [...]}.\n"
    "- link.create must have: {\"op\": \"link.create\", \"from\": \"<uuid>\", \"to\": \"<uuid>\", "
    "\"type\": \"<link_type>\", \"data\": {}}.\n"
    "- link.close must have: {\"op\": \"link.close\", \"from\": \"<uuid>\", \"to\": \"<uuid>\", "
    "\"type\": \"<link_type>\", \"bidirectional\": false}."
)


def _normalize_pending_graph_op(raw_op: Any) -> dict[str, Any] | None:
    return graph_repository.normalize_pending_graph_op(raw_op)


def _list_recent_events_for_zone(
    db: Session,
    session_id: uuid.UUID,
    zone_id: uuid.UUID | None,
    *,
    limit: int = GRAPH_HEALTH_MAX_EVENTS,
) -> list[dict[str, Any]]:
    """List recent events scoped to the given zone."""
    return graph_repository.list_recent_events_for_zone(
        db,
        session_id=session_id,
        zone_id=zone_id,
        limit=limit,
    )


def _list_zone_entities_with_links(
    db: Session,
    session_id: uuid.UUID,
    zone_id: uuid.UUID | None,
) -> list[dict[str, Any]]:
    """List entities located in the given zone with their active links."""
    return graph_repository.list_zone_entities_with_links(
        db,
        session_id=session_id,
        zone_id=zone_id,
    )


def _list_asymmetric_relationships(
    db: Session,
    session_id: uuid.UUID,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List active one-way reciprocal social links missing reverse direction."""
    return graph_repository.list_asymmetric_relationships(
        db,
        session_id=session_id,
        limit=limit,
    )


def run_graph_health_check(
    db: Session,
    session_id: uuid.UUID,
    *,
    zone_id: uuid.UUID | None,
    current_turn: int,
) -> list[dict[str, Any]]:
    """Run DeepSeek graph health check and store proposed ops in state_json.

    Returns the list of proposed ops (may be empty).
    """
    if current_turn < 1:
        return []

    had_active_tx = db.in_transaction()
    payload = _build_graph_health_payload(
        db=db,
        session_id=session_id,
        zone_id=zone_id,
    )
    if payload is None:
        return []

    if not had_active_tx and db.in_transaction():
        db.rollback()

    proposed_ops = _propose_graph_health_ops(payload)
    if not proposed_ops:
        return []

    return _store_pending_graph_ops(
        db=db,
        session_id=session_id,
        zone_id=zone_id,
        proposed_ops=proposed_ops,
    )


def _build_graph_health_payload(
    *,
    db: Session,
    session_id: uuid.UUID,
    zone_id: uuid.UUID | None,
) -> dict[str, Any] | None:
    from . import crud as crud_runtime

    events = crud_runtime._list_recent_events_for_zone(db, session_id, zone_id)
    entities = crud_runtime._list_zone_entities_with_links(db, session_id, zone_id)
    asymmetric_relationships = crud_runtime._list_asymmetric_relationships(db, session_id)
    if not events and not asymmetric_relationships:
        return None

    return {
        "recent_events": events,
        "zone_entities": entities,
        "asymmetric_relationships": asymmetric_relationships,
    }


def _propose_graph_health_ops(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from . import crud as crud_runtime

    try:
        with telemetry_context(request_type="graph_health_check"):
            result = openrouter_chat.generate_json(
                model=OPENROUTER_CHAT_MODEL,
                system_prompt=_GRAPH_HEALTH_CHECK_SYSTEM,
                user_prompt=_normalize_json_preview(payload, max(TURN_CONTEXT_MAX_CHARS, 1)),
                max_tokens=400,
            )
    except Exception:  # noqa: BLE001
        logger.warning("Graph health check DeepSeek call failed", exc_info=True)
        raise

    raw_ops = result.get("proposed_ops")
    if not isinstance(raw_ops, list) or not raw_ops:
        return []

    # Coerce and filter valid ops
    coerced_ops: list[dict[str, Any]] = []
    for idx, raw_op in enumerate(raw_ops[:16]):
        coerced = crud_runtime._coerce_patch_op(raw_op, idx)
        if coerced is None:
            continue
        normalized = _normalize_pending_graph_op(coerced)
        if normalized is not None:
            coerced_ops.append(normalized)

    if not coerced_ops:
        return []

    return coerced_ops


def _build_graph_health_turn_text(
    *,
    applied_count: int,
    rejected_count: int,
    reasons: list[str],
) -> str:
    if applied_count > 0:
        if rejected_count > 0:
            return f"(graph health) applied {applied_count} repairs, rejected {rejected_count}"
        return f"(graph health) applied {applied_count} repairs"

    reason_preview = ", ".join(str(reason).strip() for reason in reasons if str(reason).strip())
    if reason_preview:
        return f"(graph health) rejected {rejected_count} repairs: {reason_preview}"
    return f"(graph health) rejected {rejected_count} repairs"


def _store_pending_graph_ops(
    *,
    db: Session,
    session_id: uuid.UUID,
    zone_id: uuid.UUID | None,
    proposed_ops: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not proposed_ops:
        return []

    from . import crud as crud_runtime

    tx_context = nullcontext() if db.in_transaction() else db.begin()
    with tx_context:
        session_row = crud_runtime._require_session(db, session_id, for_update=True)
        graph_repository.store_pending_graph_ops(
            db,
            session_id=session_id,
            proposed_ops=proposed_ops,
            session_row=session_row,
        )

    logger.info(
        "Graph health check proposed %d ops for session=%s zone=%s",
        len(proposed_ops), session_id, zone_id,
    )
    return proposed_ops


def _run_graph_health_snapshot_outbox_event(
    *,
    session_id: uuid.UUID,
    current_turn: int,
    zone_id: uuid.UUID | None,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if current_turn < 1 or current_turn % GRAPH_HEALTH_CHECK_INTERVAL_TURNS != 0:
        return []

    normalized_payload = payload if isinstance(payload, dict) else {}
    proposed_ops = _propose_graph_health_ops(normalized_payload)
    if not proposed_ops:
        return []

    db = SessionLocal()
    try:
        return _store_pending_graph_ops(
            db=db,
            session_id=session_id,
            zone_id=zone_id,
            proposed_ops=proposed_ops,
        )
    finally:
        if db.in_transaction():
            db.rollback()
        db.close()


def maybe_run_graph_health_check(
    db: Session,
    session_id: uuid.UUID,
    *,
    zone_id: uuid.UUID | None,
    current_turn: int,
) -> list[dict[str, Any]]:
    """Run graph health check if it's due (every N turns). Returns proposed ops."""
    if current_turn < 1 or current_turn % GRAPH_HEALTH_CHECK_INTERVAL_TURNS != 0:
        return []
    return run_graph_health_check(
        db, session_id,
        zone_id=zone_id,
        current_turn=current_turn,
    )


def get_pending_graph_ops(
    db: Session,
    session_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """Return the current pending graph ops from state_json."""
    from . import crud as crud_runtime

    session_row = crud_runtime._require_session(db, session_id)
    return graph_repository.get_pending_graph_ops(
        db,
        session_id=session_id,
        session_row=session_row,
    )


def apply_pending_graph_ops(
    db: Session,
    session_id: uuid.UUID,
) -> schemas.PendingGraphOpsApplyOut:
    """Validate and apply pending graph ops via _validate_patch_ops pipeline.

    Clears pending_graph_ops from state_json after processing.

    Acquires the runtime advisory lock to prevent turn_index collisions
    with concurrent ``run_turn`` or ``move_player`` calls.
    """
    with _session_turn_runtime_lock(db, session_id):
        return _apply_pending_graph_ops_locked(db, session_id)


def _apply_pending_graph_ops_locked(
    db: Session,
    session_id: uuid.UUID,
) -> schemas.PendingGraphOpsApplyOut:
    from . import crud as crud_runtime

    def _has_active_transaction() -> bool:
        in_transaction_attr = getattr(db, "in_transaction", None)
        in_transaction_value = in_transaction_attr() if callable(in_transaction_attr) else in_transaction_attr
        return in_transaction_value if isinstance(in_transaction_value, bool) else False

    artifacts = TurnApplyExternalArtifacts()
    while True:
        try:
            with turn_apply_external_artifacts_context(artifacts):
                with db.begin():
                    crud_runtime._acquire_session_turn_lock(db, session_id)
                    session_row = crud_runtime._require_session(db, session_id, for_update=True)
                    state_payload = dict(session_row.state_json or {})

                    # Guard: refuse to create a synthetic turn while a real turn is
                    # in progress — prevents turn_index collisions.
                    state_payload = _recover_abandoned_pending_turn_locked(
                        db=db,
                        session_id=session_id,
                        session_row=session_row,
                        state_payload=state_payload,
                    )

                    raw_ops = state_payload.get("pending_graph_ops")

                    if not isinstance(raw_ops, list) or not raw_ops:
                        return schemas.PendingGraphOpsApplyOut(
                            applied_count=0,
                            rejected_count=0,
                            reasons=["no pending graph ops"],
                        )

                    # Validate through the standard pipeline
                    validation = crud_runtime._validate_patch_ops(raw_ops)

                    current_turn = max(int(getattr(session_row, "turn_index", 0) or 0), 0)
                    synthetic_turn = current_turn + 1
                    time_data = state_payload.get("time") or {}
                    in_game_day = _safe_int(time_data.get("day")) or 0
                    in_game_minute = _safe_int(time_data.get("minute")) or 0

                    validated_updates: list[dict[str, Any]] = []
                    applied_ops: list[dict[str, Any]] = []
                    applied_count = 0
                    if validation.status == "ok" and validation.parsed_ops:
                        validated_updates = [
                            op.model_dump(mode="json", by_alias=True)
                            for op in validation.parsed_ops
                        ]
                        patch_result = crud_runtime.apply_patch_ops(
                            db=db,
                            session_id=session_id,
                            new_turn=synthetic_turn,
                            in_game_day=in_game_day,
                            in_game_minute=in_game_minute,
                            ops=validation.parsed_ops,
                        )
                        applied_ops = list(patch_result.applied_ops)
                        applied_count = patch_result.applied_input_count

                    rejected_count = max(len(raw_ops) - applied_count, 0)
                    synthetic_ai_text = _build_graph_health_turn_text(
                        applied_count=applied_count,
                        rejected_count=rejected_count,
                        reasons=list(validation.reasons or []),
                    )
                    synthetic_ai_json: dict[str, Any] = {
                        "status": "graph_health_check_applied" if applied_count > 0 else "graph_health_check_rejected",
                        "source": "pending_graph_ops",
                        "applied_ops": applied_ops,
                        "validated_updates": validated_updates,
                        "validator": {
                            "status": validation.status,
                            "reasons": validation.reasons,
                        },
                        "pending_graph_ops_input_count": len(raw_ops),
                        "applied_count": applied_count,
                        "rejected_count": rejected_count,
                        "turn_weight": 0.1 if applied_count > 0 else 0.0,
                        "in_game_time": {"day": in_game_day, "minute": in_game_minute},
                    }
                    db.add(
                        models.TurnModel(
                            session_id=session_id,
                            turn_index=synthetic_turn,
                            user_input="[graph_health_check]",
                            ai_text=synthetic_ai_text,
                            ai_json=synthetic_ai_json,
                        )
                    )
                    session_row.turn_index = synthetic_turn

                    # Clear pending ops regardless of outcome
                    state_payload.pop("pending_graph_ops", None)
                    session_row.state_json = state_payload
                    db.flush()
                    _entities._enqueue_turn_chronicle_sync_event(
                        db,
                        session_id=session_id,
                        turn_index=synthetic_turn,
                    )

                return schemas.PendingGraphOpsApplyOut(
                    applied_count=applied_count,
                    rejected_count=rejected_count,
                    reasons=validation.reasons if validation.status != "ok" else [],
                )
        except TurnApplyExternalPreparationRequired as exc:
            if _has_active_transaction():
                db.rollback()
            _prepare_turn_apply_external_request(artifacts, exc)


__all__ = [
    "run_graph_health_check",
    "maybe_run_graph_health_check",
    "get_pending_graph_ops",
    "apply_pending_graph_ops",
]
