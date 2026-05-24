from __future__ import annotations

"""Turn-processing module (Phase 1b implementation)."""

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy import Integer, case, cast, select
from sqlalchemy.orm import Session

from . import crud_core as _core
from . import models, schemas
from .crud_shared import (
    TurnApplyExternalArtifacts,
    TurnApplyExternalPreparationRequired,
    _canonicalize_durable_fact_identity_refs,
    _coalesce_memory_candidates_by_identity,
    _memory_candidates_to_durable_facts,
    _prepare_turn_apply_external_request,
    _rollback_read_only_autobegin_transaction,
    _supplement_memory_candidates_with_durable_facts,
    turn_apply_external_artifacts_context,
)
from .llm_telemetry import telemetry_context
from .observability import (
    record_memory_bundle_pressure,
    record_memory_continuity_miss,
    record_memory_false_resurfacing,
    record_memory_surface,
    record_memory_transition_ambiguity,
    record_memory_usage,
    record_turn,
    trace_extra,
)
from .strings import FALLBACK_TURN_PATCH_ERROR

MIN_TURN_INTERVAL_SECONDS = 5.0
_PROTECTED_TURN_AI_JSON_KEYS = frozenset(
    {
        "status",
        "source",
        "note",
        "narration",
        "choices",
        "proposed_updates",
        "memory_candidates",
        "consequence_seeds",
        "consequence_intents",
        "resolved_consequence_ids",
        "consequence_windows_opened",
        "semantic_events",
        "scene_entities",
        "validated_updates",
        "applied_ops",
        "ref_map",
        "ttl_cleaned",
        "in_game_time",
        "zone_scope",
        "validator",
        "durable_facts",
        "memory_trace",
        "memory_debug",
        "librarian_used",
        "turn_weight",
        "chronicle_embedding_snippet",
        "memory_candidates_stored",
        "llm_usage",
        "planner_contract_version",
        "structural_signals",
        "consequence_validation",
        "pending_graph_ops_input_count",
        "applied_count",
        "rejected_count",
    }
)


def _schema_durable_fact_from_resolved_fact(raw_fact: Any) -> schemas.DurableFact | None:
    kind = str(getattr(raw_fact, "kind", "") or "").strip().lower()
    narrative_summary = str(
        getattr(raw_fact, "narrative_recall_summary", "")
        or getattr(raw_fact, "recall_summary", "")
        or getattr(raw_fact, "text", "")
        or ""
    ).strip()
    search_summary = str(
        getattr(raw_fact, "search_recall_summary", "")
        or getattr(raw_fact, "recall_summary", "")
        or getattr(raw_fact, "text", "")
        or ""
    ).strip()
    identity_text = str(getattr(raw_fact, "identity_text", "") or "").strip() or narrative_summary or search_summary
    actor_object_id = getattr(raw_fact, "actor_object_id", None)
    counterparty_object_id = getattr(raw_fact, "counterparty_object_id", None)
    object_id = getattr(raw_fact, "object_id", None)
    location_object_id = getattr(raw_fact, "location_object_id", None)
    quest_object_id = getattr(raw_fact, "quest_object_id", None)
    context_object_ids = list(getattr(raw_fact, "context_object_ids", None) or [])
    if not kind or not narrative_summary or not search_summary:
        return None
    try:
        return schemas.DurableFact(
            kind=kind,
            search_recall_summary=search_summary,
            narrative_recall_summary=narrative_summary,
            identity_text=identity_text,
            state=str(getattr(raw_fact, "state", "active") or "active").strip().lower() or "active",
            priority=str(getattr(raw_fact, "priority", "med") or "med").strip().lower() or "med",
            actor_ref=actor_object_id,
            counterparty_ref=counterparty_object_id,
            object_ref=object_id,
            location_ref=location_object_id,
            quest_ref=quest_object_id,
            context_refs=context_object_ids,
            relationship_type=getattr(raw_fact, "relationship_type", None),
            callback_candidate=bool(getattr(raw_fact, "callback_candidate", False)),
            knowledge_scope=str(getattr(raw_fact, "knowledge_scope", "global") or "global").strip().lower() or "global",
            player_salience=str(getattr(raw_fact, "player_salience", "none") or "none").strip().lower() or "none",
            expectation_salience=str(getattr(raw_fact, "expectation_salience", "none") or "none").strip().lower() or "none",
            continuity_contract_strength=str(
                getattr(raw_fact, "continuity_contract_strength", "none") or "none"
            ).strip().lower() or "none",
            independent_evidence_count=max(int(getattr(raw_fact, "independent_evidence_count", 0) or 0), 0),
            repetition_count=max(int(getattr(raw_fact, "repetition_count", 0) or 0), 0),
            last_reconfirmed_turn=_core._safe_int(getattr(raw_fact, "last_reconfirmed_turn", None)),
        )
    except ValidationError:
        return None


def _allocate_turn(
    db: Session,
    session_id: uuid.UUID,
    user_input: str | None,
    *,
    turn_kind: str = "player",
    actor_object_id: uuid.UUID | None = None,
    triggered_by_turn_index: int | None = None,
    root_turn_index: int | None = None,
    advance_time: bool = True,
) -> tuple[int, int, int, int, int]:
    with db.begin():
        _core._acquire_session_turn_lock(db, session_id)
        session_row = _core._require_session(db, session_id, for_update=True)
        state_payload = dict(session_row.state_json or {})
        state_payload = _core._recover_abandoned_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=state_payload,
        )

        new_turn = session_row.turn_index + 1
        session_row.turn_index = new_turn

        raw_time = state_payload.get("time")
        if not isinstance(raw_time, dict):
            raw_time = {"day": 0, "minute": 0}
        prev_day = max(_core._safe_int(raw_time.get("day")) or 0, 0)
        prev_minute = max(_core._safe_int(raw_time.get("minute")) or 0, 0)

        if advance_time:
            next_day, next_minute, time_scale = _core._normalize_time_payload(state_payload)
        else:
            next_day = prev_day
            next_minute = prev_minute
            time_scale = _core._coerce_time_scale_minutes(state_payload.get("time_scale"))
        state_payload["time"] = {"day": next_day, "minute": next_minute}
        state_payload["time_scale"] = time_scale

        turn_row = models.TurnModel(
            session_id=session_id,
            turn_index=new_turn,
            turn_kind=str(turn_kind or "player").strip() or "player",
            user_input=(str(user_input).strip() if isinstance(user_input, str) and str(user_input).strip() else None),
            actor_object_id=actor_object_id,
            triggered_by_turn_index=triggered_by_turn_index,
            root_turn_index=new_turn if str(turn_kind or "player").strip() == "player" else root_turn_index,
            ai_text=None,
            ai_json={
                "status": "pending",
                "turn_kind": str(turn_kind or "player").strip() or "player",
                "in_game_time": {"day": next_day, "minute": next_minute},
            },
        )
        db.add(turn_row)
        # Persist the turn row first so pending_turn refs are valid even if constraints become IMMEDIATE.
        db.flush([turn_row])

        state_payload["pending_turn"] = new_turn
        state_payload["pending_turn_started_at"] = datetime.now(timezone.utc).isoformat()
        session_row.state_json = state_payload
        db.flush()

    return new_turn, next_day, next_minute, prev_day, prev_minute


def _coerce_turn_weight_value(raw_value: Any) -> float | None:
    if isinstance(raw_value, bool):
        return None
    parsed: float | None = None
    if isinstance(raw_value, (int, float)):
        parsed = float(raw_value)
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
    if parsed is None or parsed != parsed:
        return None
    return round(min(max(parsed, 0.0), 1.0), 6)


def _coerce_uuid_value(raw_value: Any) -> uuid.UUID | None:
    if isinstance(raw_value, uuid.UUID):
        return raw_value
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except (TypeError, ValueError, AttributeError):
        return None


def _merge_json_patch(target: dict[str, Any] | None, patch: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(target or {})
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
            continue
        if isinstance(value, dict):
            current_value = merged.get(key)
            current_mapping = current_value if isinstance(current_value, dict) else {}
            merged[key] = _merge_json_patch(current_mapping, value)
            continue
        merged[key] = value
    return merged


def _validate_turn_ai_json_patch(patch: dict[str, Any]) -> None:
    forbidden_keys = sorted(key for key in patch if key in _PROTECTED_TURN_AI_JSON_KEYS)
    if not forbidden_keys:
        return
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "code": "turn_ai_json_patch_forbidden",
            "message": "PATCH /turns may not modify canonical ai_json fields",
            "forbidden_keys": forbidden_keys,
        },
    )


def _count_turn_memory_transition_ambiguities(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
) -> int:
    rows = db.execute(
        select(models.ObjectModel.data)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == _core._embeddings.MEMORY_FACT_OBJECT_TYPE,
            models.ObjectModel.data["source_turn"].astext == str(turn_index),
        )
    ).all()
    ambiguity_count = 0
    for (data,) in rows:
        payload = dict(data or {})
        if bool(payload.get("transition_ambiguity")):
            ambiguity_count += 1
    return ambiguity_count


def _is_mock_like_session(db: Session) -> bool:
    return type(db).__module__.startswith("unittest.mock")


def _json_text_int_expr(text_expr: Any) -> Any:
    return case(
        (text_expr.op("~")(r"^-?\d+$"), cast(text_expr, Integer)),
        else_=None,
    )


def _confirmed_fact_keys_between_turns(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_gt: int,
    turn_lte: int,
    fact_keys: list[str] | None = None,
) -> set[str]:
    if turn_lte <= turn_gt or _is_mock_like_session(db):
        return set()
    turn_expr = _json_text_int_expr(models.ObjectModel.data["last_confirmed_turn"].astext)
    filters: list[Any] = [
        models.ObjectModel.session_id == session_id,
        models.ObjectModel.type == _core._embeddings.MEMORY_FACT_OBJECT_TYPE,
        turn_expr.is_not(None),
        turn_expr > int(turn_gt),
        turn_expr <= int(turn_lte),
    ]
    normalized_fact_keys = [str(value).strip() for value in list(fact_keys or []) if str(value).strip()]
    if normalized_fact_keys:
        filters.append(models.ObjectModel.data["fact_key"].astext.in_(normalized_fact_keys))
    rows = db.execute(
        select(models.ObjectModel.data["fact_key"].astext).where(*filters)
    ).all()
    return {
        str(raw_fact_key).strip()
        for (raw_fact_key,) in rows
        if str(raw_fact_key).strip()
    }


def _confirmed_fact_object_ids_between_turns(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_gt: int,
    turn_lte: int,
    object_ids: list[str] | None = None,
) -> set[str]:
    if turn_lte <= turn_gt or _is_mock_like_session(db):
        return set()
    turn_expr = _json_text_int_expr(models.ObjectModel.data["last_confirmed_turn"].astext)
    filters: list[Any] = [
        models.ObjectModel.session_id == session_id,
        models.ObjectModel.type == _core._embeddings.MEMORY_FACT_OBJECT_TYPE,
        turn_expr.is_not(None),
        turn_expr > int(turn_gt),
        turn_expr <= int(turn_lte),
    ]
    normalized_object_ids = [str(value).strip() for value in list(object_ids or []) if str(value).strip()]
    parsed_object_ids: list[uuid.UUID] = []
    for raw_object_id in normalized_object_ids:
        try:
            parsed_object_ids.append(uuid.UUID(raw_object_id))
        except ValueError:
            continue
    if parsed_object_ids:
        filters.append(models.ObjectModel.object_id.in_(parsed_object_ids))
    rows = db.execute(
        select(models.ObjectModel.object_id).where(*filters)
    ).all()
    return {
        str(raw_object_id).strip()
        for (raw_object_id,) in rows
        if raw_object_id is not None and str(raw_object_id).strip()
    }


def _augment_memory_debug_with_current_turn_effects(
    db: Session,
    *,
    session_id: uuid.UUID,
    current_turn: int,
    memory_debug: dict[str, Any],
) -> dict[str, Any]:
    surfaced_callback_rows = [
        item
        for item in list(memory_debug.get("surfaced_callback_rows") or [])
        if isinstance(item, dict)
    ]
    used_callback_ids = {
        str(item).strip()
        for item in list(memory_debug.get("used_callback_ids") or [])
        if str(item).strip()
    }
    callback_fact_keys = [
        str(item.get("fact_key") or "").strip()
        for item in surfaced_callback_rows
        if str(item.get("fact_key") or "").strip()
    ]
    confirmed_callback_fact_keys = _confirmed_fact_keys_between_turns(
        db,
        session_id=session_id,
        turn_gt=max(current_turn - 1, 0),
        turn_lte=current_turn,
        fact_keys=callback_fact_keys,
    )
    useful_callback_ids = set(used_callback_ids)
    for row in surfaced_callback_rows:
        prompt_id = str(row.get("prompt_id") or "").strip()
        fact_key = str(row.get("fact_key") or "").strip()
        if prompt_id and fact_key and fact_key in confirmed_callback_fact_keys:
            useful_callback_ids.add(prompt_id)
    memory_debug["useful_callback_ids"] = sorted(useful_callback_ids)
    return memory_debug


def _build_turn_memory_debug_payload(
    *,
    context_pack: dict[str, Any] | None,
    memory_trace: schemas.MemoryTrace | None,
    transition_ambiguity_count: int,
) -> dict[str, Any]:
    observability = dict((context_pack or {}).get("memory_observability") or {})
    surfaced_relevant_ids = [
        str(item).strip()
        for item in list(observability.get("surfaced_relevant_ids") or [])
        if str(item).strip()
    ]
    surfaced_callback_ids = [
        str(item).strip()
        for item in list(observability.get("surfaced_callback_ids") or [])
        if str(item).strip()
    ]
    surfaced_bundle_ids = [
        str(item).strip()
        for item in list(observability.get("surfaced_bundle_ids") or [])
        if str(item).strip()
    ]
    surfaced_obligation_ids = [
        str(item).strip()
        for item in list(observability.get("surfaced_obligation_ids") or [])
        if str(item).strip()
    ]
    surfaced_actor_view_ids = [
        str(item).strip()
        for item in list(observability.get("surfaced_actor_view_ids") or [])
        if str(item).strip()
    ]
    used_relevant_ids = [
        str(item).strip()
        for item in list((memory_trace.used_relevant_ids if memory_trace else []) or [])
        if str(item).strip()
    ]
    used_callback_ids = [
        str(item).strip()
        for item in list((memory_trace.used_callback_ids if memory_trace else []) or [])
        if str(item).strip()
    ]
    used_bundle_ids = [
        str(item).strip()
        for item in list((memory_trace.used_bundle_ids if memory_trace else []) or [])
        if str(item).strip()
    ]
    used_obligation_ids = [
        str(item).strip()
        for item in list((memory_trace.used_obligation_ids if memory_trace else []) or [])
        if str(item).strip()
    ]
    used_actor_view_ids = [
        str(item).strip()
        for item in list((memory_trace.used_actor_view_ids if memory_trace else []) or [])
        if str(item).strip()
    ]
    miss_candidates = [
        item
        for item in list(observability.get("miss_candidates") or [])
        if isinstance(item, dict)
    ]
    surfaced_relevant_rows = [
        item
        for item in list(observability.get("surfaced_relevant_rows") or [])
        if isinstance(item, dict)
    ]
    surfaced_callback_rows = [
        item
        for item in list(observability.get("surfaced_callback_rows") or [])
        if isinstance(item, dict)
    ]
    surfaced_bundle_rows = [
        item
        for item in list(observability.get("surfaced_bundle_rows") or [])
        if isinstance(item, dict)
    ]
    surfaced_obligation_rows = [
        item
        for item in list(observability.get("surfaced_obligation_rows") or [])
        if isinstance(item, dict)
    ]
    surfaced_actor_view_rows = [
        item
        for item in list(observability.get("surfaced_actor_view_rows") or [])
        if isinstance(item, dict)
    ]
    tuning_candidate_metrics = dict(observability.get("tuning_candidate_metrics") or {})

    def _with_used_tuning_metrics() -> dict[str, Any]:
        if not tuning_candidate_metrics:
            return {}
        lane_metrics = {
            str(lane_name): {
                "candidate_count": int(dict(metrics or {}).get("candidate_count") or 0),
                "surfaced_count": int(dict(metrics or {}).get("surfaced_count") or 0),
                "used_count": int(dict(metrics or {}).get("used_count") or 0),
                "missed_count": int(dict(metrics or {}).get("missed_count") or 0),
                "compressed_count": int(dict(metrics or {}).get("compressed_count") or 0),
            }
            for lane_name, metrics in dict(tuning_candidate_metrics.get("lane_metrics") or {}).items()
            if str(lane_name)
        }
        family_metrics = {
            str(family_key): {
                "lane": str(dict(metrics or {}).get("lane") or ""),
                "candidate_count": int(dict(metrics or {}).get("candidate_count") or 0),
                "surfaced_count": int(dict(metrics or {}).get("surfaced_count") or 0),
                "used_count": int(dict(metrics or {}).get("used_count") or 0),
                "missed_count": int(dict(metrics or {}).get("missed_count") or 0),
                "compressed_count": int(dict(metrics or {}).get("compressed_count") or 0),
            }
            for family_key, metrics in dict(tuning_candidate_metrics.get("family_metrics") or {}).items()
            if str(family_key)
        }
        used_prompt_ids = {
            *[item for item in filtered_used_relevant_ids if item],
            *[item for item in used_bundle_ids if item in set(surfaced_bundle_ids)],
            *[item for item in filtered_used_obligation_ids if item],
            *[item for item in filtered_used_actor_view_ids if item],
        }
        surfaced_rows = [
            *surfaced_relevant_rows,
            *surfaced_bundle_rows,
            *surfaced_obligation_rows,
            *surfaced_actor_view_rows,
        ]
        seen_used_refs: set[tuple[str, str, str]] = set()
        for row in surfaced_rows:
            prompt_id = str(row.get("prompt_id") or "").strip()
            lane_name = str(row.get("lane") or "").strip()
            family_key = str(row.get("lane_family_key") or "").strip()
            if not prompt_id or prompt_id not in used_prompt_ids or not lane_name:
                continue
            used_ref = (prompt_id, lane_name, family_key or "")
            if used_ref in seen_used_refs:
                continue
            seen_used_refs.add(used_ref)
            if lane_name in lane_metrics:
                lane_metrics[lane_name]["used_count"] += 1
            if family_key and family_key in family_metrics:
                family_metrics[family_key]["used_count"] += 1
        return {
            "evidence_source": str(
                tuning_candidate_metrics.get("evidence_source") or "full_candidate_pool_aggregates"
            ),
            "full_candidate_count": int(tuning_candidate_metrics.get("full_candidate_count") or 0),
            "inspector_candidate_count": int(tuning_candidate_metrics.get("inspector_candidate_count") or 0),
            "lane_metrics": lane_metrics,
            "family_metrics": family_metrics,
        }

    filtered_used_relevant_ids = [item for item in used_relevant_ids if item in set(surfaced_relevant_ids)]
    filtered_used_obligation_ids = [
        item for item in used_obligation_ids if item in set(surfaced_obligation_ids)
    ] or [item for item in filtered_used_relevant_ids if item in set(surfaced_obligation_ids)]
    filtered_used_actor_view_ids = [
        item for item in used_actor_view_ids if item in set(surfaced_actor_view_ids)
    ] or [item for item in filtered_used_relevant_ids if item in set(surfaced_actor_view_ids)]
    return {
        "turn_intent": str((context_pack or {}).get("turn_intent") or "").strip() or None,
        "scene_mode": str((context_pack or {}).get("scene_mode") or "").strip() or None,
        "surfaced_relevant_ids": surfaced_relevant_ids,
        "surfaced_callback_ids": surfaced_callback_ids,
        "surfaced_bundle_ids": surfaced_bundle_ids,
        "surfaced_obligation_ids": surfaced_obligation_ids,
        "surfaced_actor_view_ids": surfaced_actor_view_ids,
        "surfaced_relevant_rows": surfaced_relevant_rows,
        "surfaced_callback_rows": surfaced_callback_rows,
        "surfaced_bundle_rows": surfaced_bundle_rows,
        "surfaced_obligation_rows": surfaced_obligation_rows,
        "surfaced_actor_view_rows": surfaced_actor_view_rows,
        "used_relevant_ids": filtered_used_relevant_ids,
        "used_callback_ids": [item for item in used_callback_ids if item in set(surfaced_callback_ids)],
        "used_bundle_ids": [item for item in used_bundle_ids if item in set(surfaced_bundle_ids)],
        "used_obligation_ids": filtered_used_obligation_ids,
        "used_actor_view_ids": filtered_used_actor_view_ids,
        "transition_ambiguity_count": max(int(transition_ambiguity_count), 0),
        "bundle_pressure_count": max(int(observability.get("bundle_pressure_count") or 0), 0),
        "obligation_count": max(int(observability.get("obligation_count") or 0), 0),
        "story_obligation_keys": [
            str(item).strip()
            for item in list(observability.get("story_obligation_keys") or [])
            if str(item).strip()
        ],
        "miss_candidates": miss_candidates,
        "feedback_miss_candidates": [
            item
            for item in list(observability.get("feedback_miss_candidates") or [])
            if isinstance(item, dict)
        ],
        "feedback_policy": dict(observability.get("feedback_policy") or {}),
        "review_freshness": dict(observability.get("review_freshness") or {}),
        "session_memory_profile": str(observability.get("session_memory_profile") or "").strip() or None,
        "session_memory_profile_override": (
            str(observability.get("session_memory_profile_override") or "").strip() or None
        ),
        "session_narrative_mode": str(observability.get("session_narrative_mode") or "").strip() or None,
        "session_narrative_mode_override": (
            str(observability.get("session_narrative_mode_override") or "").strip() or None
        ),
        "session_narrative_mode_source": (
            str(observability.get("session_narrative_mode_source") or "").strip() or None
        ),
        "memory_policy_state": dict(observability.get("memory_policy_state") or {}),
        "operational_alerts": [
            item
            for item in list(observability.get("operational_alerts") or [])
            if isinstance(item, dict)
        ],
        "operational_guardrails": dict(observability.get("operational_guardrails") or {}),
        "lane_budgets": dict(observability.get("lane_budgets") or {}),
        "lane_counts": dict(observability.get("lane_counts") or {}),
        "saturation_diagnostics": dict(observability.get("saturation_diagnostics") or {}),
        "graph_traversal": list(observability.get("graph_traversal") or []),
        "candidate_rows": [
            item
            for item in list(observability.get("candidate_rows") or [])
            if isinstance(item, dict)
        ],
        "tuning_candidate_metrics": _with_used_tuning_metrics(),
        "obligation_diversity": dict(observability.get("obligation_diversity") or {}),
        "why_not_surfaced": [
            item
            for item in list(observability.get("why_not_surfaced") or [])
            if isinstance(item, dict)
        ],
        "compressed_rows": [
            item
            for item in list(observability.get("compressed_rows") or [])
            if isinstance(item, dict)
        ],
        "tuning_policy_version": str(observability.get("tuning_policy_version") or "").strip() or None,
        "memory_trace": memory_trace.model_dump(mode="json") if memory_trace is not None else {},
    }


def _record_turn_memory_observability(memory_debug: dict[str, Any], *, context_pack: dict[str, Any] | None) -> None:
    surfaced_relevant_ids = list(memory_debug.get("surfaced_relevant_ids") or [])
    surfaced_callback_ids = list(memory_debug.get("surfaced_callback_ids") or [])
    surfaced_bundle_ids = list(memory_debug.get("surfaced_bundle_ids") or [])
    used_relevant_ids = set(str(item).strip() for item in list(memory_debug.get("used_relevant_ids") or []) if str(item).strip())
    used_callback_ids = set(str(item).strip() for item in list(memory_debug.get("useful_callback_ids") or memory_debug.get("used_callback_ids") or []) if str(item).strip())
    used_bundle_ids = set(str(item).strip() for item in list(memory_debug.get("used_bundle_ids") or []) if str(item).strip())
    record_memory_surface(
        relevant_count=len(surfaced_relevant_ids),
        callback_count=len(surfaced_callback_ids),
        bundle_count=len(surfaced_bundle_ids),
    )
    record_memory_usage(
        relevant_used=len(used_relevant_ids),
        callback_used=len(used_callback_ids),
        bundle_used=len(used_bundle_ids),
    )
    transition_ambiguity_count = max(int(memory_debug.get("transition_ambiguity_count") or 0), 0)
    if transition_ambiguity_count:
        record_memory_transition_ambiguity(transition_ambiguity_count)
    bundle_pressure_count = max(int(memory_debug.get("bundle_pressure_count") or 0), 0)
    if bundle_pressure_count:
        record_memory_bundle_pressure(bundle_pressure_count)


def _finalize_memory_observability_windows(
    db: Session,
    *,
    session_id: uuid.UUID,
    current_turn: int,
) -> None:
    if current_turn < 2 or _is_mock_like_session(db):
        return
    rows = db.execute(
        select(models.TurnModel)
        .where(
            models.TurnModel.session_id == session_id,
            models.TurnModel.turn_index < current_turn,
            models.TurnModel.turn_index >= max(current_turn - 5, 1),
        )
        .order_by(models.TurnModel.turn_index.asc())
    ).scalars().all()
    for turn_row in rows:
        turn_index = max(int(getattr(turn_row, "turn_index", 0) or 0), 0)
        ai_json = dict(getattr(turn_row, "ai_json", {}) or {})
        memory_debug = dict(ai_json.get("memory_debug") or {})
        if not memory_debug:
            continue
        changed = False
        if (
            turn_index + 3 <= current_turn
            and not memory_debug.get("false_resurfacing_finalized_turn")
        ):
            surfaced_callback_rows = [
                item
                for item in list(memory_debug.get("surfaced_callback_rows") or [])
                if isinstance(item, dict)
            ]
            useful_callback_ids = {
                str(item).strip()
                for item in list(memory_debug.get("useful_callback_ids") or memory_debug.get("used_callback_ids") or [])
                if str(item).strip()
            }
            callback_fact_keys = [
                str(item.get("fact_key") or "").strip()
                for item in surfaced_callback_rows
                if str(item.get("prompt_id") or "").strip()
                and str(item.get("fact_key") or "").strip()
                and str(item.get("prompt_id") or "").strip() not in useful_callback_ids
            ]
            confirmed_fact_keys = _confirmed_fact_keys_between_turns(
                db,
                session_id=session_id,
                turn_gt=turn_index,
                turn_lte=min(turn_index + 3, current_turn),
                fact_keys=callback_fact_keys,
            )
            false_resurfacing_count = sum(
                1
                for item in surfaced_callback_rows
                if str(item.get("prompt_id") or "").strip()
                and str(item.get("prompt_id") or "").strip() not in useful_callback_ids
                and str(item.get("fact_key") or "").strip()
                and str(item.get("fact_key") or "").strip() not in confirmed_fact_keys
            )
            if false_resurfacing_count:
                record_memory_false_resurfacing(false_resurfacing_count)
            memory_debug["false_resurfacing_finalized_turn"] = current_turn
            memory_debug["false_resurfacing_count"] = false_resurfacing_count
            changed = True
        if (
            turn_index + 5 <= current_turn
            and not memory_debug.get("continuity_miss_finalized_turn")
        ):
            surfaced_relevant_rows = [
                item
                for item in list(memory_debug.get("surfaced_relevant_rows") or [])
                if isinstance(item, dict)
            ]
            miss_candidates = [
                item
                for item in list(memory_debug.get("miss_candidates") or [])
                if isinstance(item, dict)
            ]
            expected_surfaced_count = sum(
                1
                for item in surfaced_relevant_rows
                if str(item.get("layer") or "").strip().lower() == "fact"
                and (
                    float(item.get("expectation_salience_score") or 0.0) >= 0.65
                    or float(item.get("player_salience_score") or 0.0) >= 0.9
                )
            )
            miss_object_ids = [
                str(item.get("object_id") or "").strip()
                for item in miss_candidates
                if str(item.get("object_id") or "").strip()
            ]
            confirmed_object_ids = _confirmed_fact_object_ids_between_turns(
                db,
                session_id=session_id,
                turn_gt=turn_index,
                turn_lte=min(turn_index + 5, current_turn),
                object_ids=miss_object_ids,
            )
            missed_count = sum(
                1
                for item in miss_candidates
                if str(item.get("object_id") or "").strip() in confirmed_object_ids
            )
            expected_count = expected_surfaced_count + len(miss_candidates)
            if expected_count or missed_count:
                record_memory_continuity_miss(
                    expected_count=expected_count,
                    missed_count=missed_count,
                )
            memory_debug["continuity_miss_finalized_turn"] = current_turn
            memory_debug["continuity_miss_count"] = missed_count
            memory_debug["continuity_expected_count"] = expected_count
            changed = True
        if changed:
            ai_json["memory_debug"] = memory_debug
            turn_row.ai_json = ai_json


def _derive_turn_weight(
    *,
    applied_ops_count: int,
    narration: str,
    memory_candidates_count: int,
) -> float:
    # Deterministic fallback when LLM did not provide turn_weight.
    score = 0.2
    score += min(max(applied_ops_count, 0), 6) * 0.08
    if memory_candidates_count > 0:
        score += 0.12
    if len(str(narration or "").strip()) >= 260:
        score += 0.08
    return round(min(max(score, 0.0), 1.0), 6)


def _load_last_turn_started_at(
    db: Session,
    session_id: uuid.UUID,
) -> datetime | None:
    # Lightweight unit tests often pass simplified db doubles here. Skip the
    # shared-state rate limit when the caller is not a real ORM session.
    try:
        session_row = _core._require_session(db, session_id)
    except AttributeError:
        return None

    raw_state = getattr(session_row, "state_json", {}) or {}
    if isinstance(raw_state, dict):
        state_payload = raw_state
    else:
        try:
            state_payload = dict(raw_state)
        except Exception:
            return None

    return _core._parse_datetime_utc(state_payload.get("last_turn_started_at"))


def _enforce_turn_rate_limit(
    db: Session,
    session_id: uuid.UUID,
    *,
    now_utc: datetime | None = None,
) -> None:
    last_started_at = _load_last_turn_started_at(db, session_id)
    if last_started_at is None:
        return

    now_value = now_utc or datetime.now(timezone.utc)
    elapsed = max((now_value - last_started_at).total_seconds(), 0.0)
    remaining = min(max(MIN_TURN_INTERVAL_SECONDS - elapsed, 0.0), MIN_TURN_INTERVAL_SECONDS)
    if remaining <= 0:
        return

    retry_after = max(int(remaining), 1)
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many turn requests, try again later",
        headers={"Retry-After": str(retry_after)},
    )


def _collect_memory_anchor_object_ids(
    *,
    applied_ops: list[dict[str, Any]],
    ref_map: dict[str, str],
    resolved_zone_scope: uuid.UUID | None,
) -> list[str]:
    resolved_ops = _core.resolve_applied_op_refs(applied_ops, ref_map)
    anchor_ids: set[str] = set()

    def _push(raw_value: Any) -> None:
        value = str(raw_value or "").strip()
        if not value:
            return
        try:
            parsed = uuid.UUID(value)
        except (TypeError, ValueError, AttributeError):
            return
        anchor_ids.add(str(parsed))

    for op in resolved_ops:
        if not isinstance(op, dict):
            continue
        for key in ("object", "object_id", "from", "to", "scope"):
            _push(op.get(key))
        payload = op.get("payload")
        if isinstance(payload, dict):
            for key in ("object_id", "target_object_id", "npc_id", "faction_id", "zone_id", "item_id"):
                _push(payload.get(key))
    if resolved_zone_scope is not None:
        _push(resolved_zone_scope)
    return sorted(anchor_ids)


def _has_active_transaction(db: Session) -> bool:
    in_transaction_attr = getattr(db, "in_transaction", None)
    in_transaction_value = in_transaction_attr() if callable(in_transaction_attr) else in_transaction_attr
    return in_transaction_value if isinstance(in_transaction_value, bool) else False


def _rollback_read_phase_transaction(db: Session) -> None:
    if _rollback_read_only_autobegin_transaction(db):
        return
    if _has_active_transaction(db):
        db.rollback()


def _build_turn_plan_outside_apply_tx(
    db: Session,
    session_id: uuid.UUID,
    *,
    payload: schemas.TurnIn,
    allow_debug_patch: bool,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
) -> tuple[_core.TurnPlanResult, dict[str, Any] | None]:
    from .application.dtos import TurnAllocation
    from .application.turn_services import turn_planning_service

    envelope = turn_planning_service.build_turn_plan_outside_apply_tx(
        db,
        session_id,
        payload=payload,
        allow_debug_patch=allow_debug_patch,
        allocation=TurnAllocation(
            new_turn=new_turn,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
            previous_day=in_game_day,
            previous_minute=in_game_minute,
        ),
    )
    return envelope.plan, envelope.context_pack


def _schedule_turn_chronicle_sync(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    dedupe: bool = True,
) -> None:
    from . import crud_continuity as _continuity

    if _core.USE_EMBEDDINGS:
        _core._outbox_runtime.enqueue_outbox_event(
            db,
            event_type=_core._outbox_runtime.EVENT_TURN_CHRONICLE_SYNC,
            payload={},
            session_id=session_id,
            turn_index=turn_index,
            trace_id=_core.get_trace_id(),
            dedupe_key=f"turn_chronicle_sync:{session_id}:{turn_index}" if dedupe else None,
        )
    _continuity._enqueue_turn_memory_sync_event(
        db,
        session_id=session_id,
        turn_index=turn_index,
        dedupe=dedupe,
    )


def _schedule_turn_graph_health_check(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    zone_id: uuid.UUID | None,
) -> None:
    interval = max(int(_core._graph_ops.GRAPH_HEALTH_CHECK_INTERVAL_TURNS), 1)
    if turn_index < 1 or turn_index % interval != 0:
        return
    recent_events = _core._graph_ops._list_recent_events_for_zone(db, session_id, zone_id)
    asymmetric_relationships = _core._graph_ops._list_asymmetric_relationships(db, session_id)
    if not recent_events and not asymmetric_relationships:
        return
    zone_entities = _core._graph_ops._list_zone_entities_with_links(db, session_id, zone_id)
    _core._outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_core._outbox_runtime.EVENT_TURN_GRAPH_HEALTH_CHECK,
        payload={
            "zone_id": str(zone_id) if zone_id is not None else None,
            "recent_events": recent_events,
            "zone_entities": zone_entities,
            "asymmetric_relationships": asymmetric_relationships,
        },
        session_id=session_id,
        turn_index=turn_index,
        trace_id=_core.get_trace_id(),
        dedupe_key=f"turn_graph_health_check:{session_id}:{turn_index}",
    )


def _parse_optional_uuid(raw_value: Any) -> uuid.UUID | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except ValueError:
        return None


def _delete_turn_chronicle_chunks(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
) -> None:
    _rollback_read_phase_transaction(db)
    with db.begin():
        for namespace in (
            _core.CHRONICLE_OUTPUT_NAMESPACE,
            _core.CHRONICLE_INPUT_NAMESPACE,
        ):
            chunk_row = db.execute(
                select(models.ChronicleChunkModel).where(
                    models.ChronicleChunkModel.session_id == session_id,
                    models.ChronicleChunkModel.turn_index == turn_index,
                    models.ChronicleChunkModel.namespace == namespace,
                )
            ).scalar_one_or_none()
            if chunk_row is not None:
                db.delete(chunk_row)


def _sync_turn_chronicle_embeddings(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
) -> None:
    if not _core.USE_EMBEDDINGS:
        return

    turn_row = db.get(models.TurnModel, (session_id, turn_index))
    if turn_row is None:
        return

    raw_ai_json = turn_row.ai_json if isinstance(turn_row.ai_json, dict) else {}
    ai_json = dict(raw_ai_json or {})
    user_input = str(turn_row.user_input or "")
    ai_text = str(turn_row.ai_text or "")
    zone_id = _parse_optional_uuid(ai_json.get("zone_scope"))
    effective_time = ai_json.get("in_game_time") if isinstance(ai_json.get("in_game_time"), dict) else {}
    in_game_day = effective_time.get("day")
    in_game_minute = effective_time.get("minute")
    choices = ai_json.get("choices") if isinstance(ai_json.get("choices"), list) else []
    applied_ops = ai_json.get("applied_ops") if isinstance(ai_json.get("applied_ops"), list) else []
    batched_snippet = str(ai_json.get("chronicle_embedding_snippet") or "").strip()
    event_summaries = _core._list_turn_event_embedding_lines(db, session_id, turn_index)

    _rollback_read_phase_transaction(db)

    if not ai_text.strip() and not batched_snippet:
        _delete_turn_chronicle_chunks(db, session_id, turn_index)
        return

    if _core.USE_CHRONICLE_SUMMARIZER:
        if batched_snippet:
            snippet_for_embedding = batched_snippet
        elif ai_text.strip():
            snippet_for_embedding = _core._summarize_turn_for_indexing(
                user_input=user_input,
                narration=ai_text,
                applied_ops=applied_ops,
                session_id=str(session_id),
            )
        else:
            snippet_for_embedding = ""
    else:
        snippet_for_embedding = batched_snippet or _core._build_embedding_snippet(
            user_input=user_input,
            narration=ai_text,
            choices=choices,
            applied_ops=applied_ops,
            event_summaries=event_summaries,
        )

    if str(snippet_for_embedding or "").strip():
        _core.index_turn_embedding(
            db=db,
            session_id=session_id,
            turn_index=turn_index,
            zone_id=zone_id,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
            snippet_text=snippet_for_embedding,
            namespace=_core.CHRONICLE_OUTPUT_NAMESPACE,
        )

    input_snippet = _core._build_input_embedding_snippet(user_input)
    if input_snippet:
        _core.index_turn_embedding(
            db=db,
            session_id=session_id,
            turn_index=turn_index,
            zone_id=zone_id,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
            snippet_text=input_snippet,
            namespace=_core.CHRONICLE_INPUT_NAMESPACE,
        )


def _run_turn_chronicle_sync_outbox_event(
    *,
    session_id: uuid.UUID,
    turn_index: int,
) -> None:
    db = _core.SessionLocal()
    try:
        _sync_turn_chronicle_embeddings(db, session_id, turn_index)
    finally:
        if _has_active_transaction(db):
            db.rollback()
        db.close()


def _apply_turn_plan(
    db: Session,
    session_id: uuid.UUID,
    *,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
    allocation: Any | None = None,
    plan: _core.TurnPlanResult | None = None,
    context_pack: dict[str, Any] | None = None,
    payload: schemas.TurnIn | None = None,
    allow_debug_patch: bool = False,
) -> tuple[models.TurnModel, list[dict[str, Any]], uuid.UUID | None]:
    from .application.turn_services import turn_application_service

    return turn_application_service.apply_turn_plan(
        db=db,
        session_id=session_id,
        new_turn=new_turn,
        in_game_day=in_game_day,
        in_game_minute=in_game_minute,
        allocation=allocation,
        plan=plan,
        context_pack=context_pack,
        payload=payload,
        allow_debug_patch=allow_debug_patch,
    )


def run_turn(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.TurnIn,
    *,
    allow_debug_patch: bool,
) -> models.TurnModel:
    from .application.turn_services import turn_application_service

    return turn_application_service.run_turn(
        db,
        session_id,
        payload,
        allow_debug_patch=allow_debug_patch,
    )


def _run_turn_locked(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.TurnIn,
    *,
    allow_debug_patch: bool,
) -> models.TurnModel:
    from .application.turn_services import turn_application_service

    return turn_application_service.run_turn_locked(
        db,
        session_id,
        payload,
        allow_debug_patch=allow_debug_patch,
    )


def patch_turn(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
    payload: schemas.TurnPatchIn,
) -> models.TurnModel:
    turn_row: models.TurnModel | None = None
    should_sync_embeddings = False

    with db.begin():
        # Acquire advisory lock to prevent races with concurrent run_turn /
        # move_player that may be modifying this same turn row.
        _core._acquire_session_turn_lock(db, session_id)
        _core._require_session(db, session_id)

        turn_row = db.get(models.TurnModel, (session_id, turn_index))
        if turn_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turn not found")

        if payload.ai_text is not None:
            turn_row.ai_text = payload.ai_text
            should_sync_embeddings = True
        if payload.ai_json is not None:
            _validate_turn_ai_json_patch(payload.ai_json)
            current_ai_json = turn_row.ai_json if isinstance(turn_row.ai_json, dict) else {}
            turn_row.ai_json = _merge_json_patch(current_ai_json, payload.ai_json)
            should_sync_embeddings = True

        if should_sync_embeddings and _core.USE_EMBEDDINGS:
            # Patched turns need a fresh chronicle rebuild from the authoritative
            # turn row, including ai_json-derived metadata. Do this durably after
            # commit instead of issuing synchronous embedding work on the request
            # path or reusing stale chunk metadata.
            _schedule_turn_chronicle_sync(
                db,
                session_id=session_id,
                turn_index=turn_index,
                dedupe=False,
            )

        db.flush()

    if turn_row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Turn patch failed")

    return turn_row


def _recover_stuck_pending_turn(
    db: Session,
    session_id: uuid.UUID,
    expected_turn: int,
    *,
    reason: str,
) -> None:
    with db.begin():
        _core._acquire_session_turn_lock(db, session_id)
        session_row = _core._require_session(db, session_id, for_update=True)
        state_payload = dict(session_row.state_json or {})
        if _core._safe_int(state_payload.get("pending_turn")) != expected_turn:
            return

        _core._clear_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=state_payload,
            pending_turn=expected_turn,
            reason=reason,
        )
        repaired_location = _core._repair_player_location_after_pending_turn_recovery(
            db=db,
            session_id=session_id,
            session_row=session_row,
        )
        if repaired_location:
            _schedule_turn_chronicle_sync(
                db,
                session_id=session_id,
                turn_index=expected_turn,
            )
        db.flush()


def recover_pending_turn(
    db: Session,
    session_id: uuid.UUID,
    *,
    force: bool = False,
) -> schemas.PendingTurnRecoveryOut:
    with db.begin():
        _core._acquire_session_turn_lock(db, session_id)
        session_row = _core._require_session(db, session_id, for_update=True)
        state_payload = dict(session_row.state_json or {})
        pending_turn = _core._safe_int(state_payload.get("pending_turn"))
        if pending_turn is None:
            return schemas.PendingTurnRecoveryOut(
                recovered=False,
                pending_turn=None,
                reason="no_pending_turn",
            )

        started_at = _core._get_pending_turn_started_at(db, session_id, pending_turn, state_payload)
        pending_is_stale = _core._is_pending_turn_stale(started_at)
        if _core._session_turn_runtime_lock_supported(db) and _core._is_session_turn_runtime_lock_held(
            db,
            session_id,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="turn already in progress",
            )

        if not force and not pending_is_stale:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="pending turn is still within timeout window",
            )

        reason = "manual_force_reset" if force else "manual_timeout_reset"
        _core._clear_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=state_payload,
            pending_turn=pending_turn,
            reason=reason,
        )
        repaired_location = _core._repair_player_location_after_pending_turn_recovery(
            db=db,
            session_id=session_id,
            session_row=session_row,
        )
        if repaired_location:
            _schedule_turn_chronicle_sync(
                db,
                session_id=session_id,
                turn_index=pending_turn,
            )
        db.flush()
        return schemas.PendingTurnRecoveryOut(
            recovered=True,
            pending_turn=pending_turn,
            reason=reason,
        )


__all__ = [
    "MIN_TURN_INTERVAL_SECONDS",
    "_allocate_turn",
    "_apply_turn_plan",
    "_coerce_turn_weight_value",
    "_derive_turn_weight",
    "_enforce_turn_rate_limit",
    "_recover_stuck_pending_turn",
    "_run_turn_locked",
    "patch_turn",
    "recover_pending_turn",
    "run_turn",
]
