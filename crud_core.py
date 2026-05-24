"""Legacy CRUD core helpers.

Selected session-read entrypoints remain as deprecated compatibility shims over
repository-backed persistence reads. Other symbols in this module remain owned
runtime helpers until they are extracted or retired explicitly.
"""

from __future__ import annotations

import hashlib
import json as _json
import asyncio
import logging
import uuid
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from . import crud_context as _context
from . import crud_embeddings_ops as _embeddings
from . import crud_entities as _entities
from . import crud_graph_ops as _graph_ops
from . import outbox_runtime as _outbox_runtime
from . import crud_telemetry as _telemetry_ops
from . import crud_patch_apply as _patch_apply
from . import crud_planning as _planning
from . import crud_lore_adaptation as _lore_adaptation
from . import crud_profiles as _profiles
from . import crud_shared as _shared
from . import models, schemas
from .constants import (
    LOCATED_IN_LINK_TYPE,
    NPC_SOCIAL_LINK_TYPES,
    QUEST_TERMINAL_STATUSES,
    REACTION_CONFLICT_LINK_TYPES,
    REACTION_SUPPORT_LINK_TYPES,
)
from .db import (
    DEDUP_ARBITER_MIN_SIM,
    DEDUP_SIM_THRESHOLD,
    EMBED_SNIPPET_MAX_CHARS,
    OPENROUTER_CHAT_MODEL,
    PENDING_TURN_TIMEOUT_SECONDS,
    RETRIEVAL_TOP_K,
    SessionLocal,
    TURN_CONTEXT_MAX_CHARS,
    TURN_CONTEXT_MAX_TOKENS,
    TURN_CONTEXT_TOKEN_RESERVE,
    TURN_CONTEXT_SEMANTIC_TURNS_LIMIT,
    TURN_CONTEXT_TURNS_LIMIT,
    CTX_WEIGHT_DECAY_LAMBDA,
    ELASTIC_MIN_RELEVANCE_THRESHOLD,
    USE_CHRONICLE_SUMMARIZER,
    USE_CTX_WEIGHT_DECAY,
    USE_CONTEXT_COMPRESSOR,
    USE_CONTEXT_TRIMMER,
    USE_CONSEQUENCES,
    USE_DEDUP_ARBITER,
    USE_ELASTIC_ENTROPY_THRESHOLD,
    USE_EMBEDDINGS,
    USE_PROMPT_CACHE_LAYOUT,
    USE_PROFILE_SYNTHESIZER,
    USE_REACTION_ENRICHER,
    USE_STATE_FIRST_PIPELINE,
    USE_WORLD_PROMPT_SUMMARIZER,
    USE_WORLD_DIRECTOR,
    WORLD_PROMPT_CHUNK_MAX_CHARS,
    WORLD_PROMPT_FALLBACK_MAX_CHARS,
    WORLD_PROMPT_TOP_K,
    OPENROUTER_API_KEY,
    OPENROUTER_LIBRARIAN_MODEL,
    OPENROUTER_NARRATOR_MODEL,
    ZONE_GLOBAL_DEDUP_THRESHOLD,
    DEFAULT_SPAWN_ZONE_NAME,
)
from .llm_telemetry import telemetry_context
from .llm import openrouter_chat
from ..db import OPENROUTER_ASSISTANT_MODEL
from .observability import get_trace_id
from .architecture_contracts import COMPATIBILITY_MODULE_CONTRACTS
from .persistence.session_read_repository import extract_applied_ops_from_ai_json, session_read_repository
from .resilience import CircuitOpenError
from .strings import FALLBACK_DEGRADED_NARRATION

logger = logging.getLogger(__name__)
COMPATIBILITY_MODULE_CONTRACT = COMPATIBILITY_MODULE_CONTRACTS[__name__]

LINK_TEXT_DATA_KEYS = ("description", "reason", "note", "details", "summary")

# ---------------------------------------------------------------------------
from .crud_shared import (
    PatchValidationResult,
    PreparedObjectCreateOp,
    TurnPlanResult,
    _acquire_session_turn_lock,
    _build_session_snapshot_dump,
    _clear_pending_turn_locked,
    _close_player_active_located_in_links,
    _coerce_time_scale_minutes,
    _count_json_tokens,
    _count_text_tokens,
    _get_active_link,
    _get_active_located_in_links,
    _get_latest_located_in_link,
    _get_object,
    _get_pending_turn_started_at,
    _get_player_current_zone_id,
    _get_session_player_object_id,
    _infer_actor_zone_id,
    _is_pending_turn_stale,
    _is_session_turn_runtime_lock_held,
    _is_true,
    _normalize_claim_id_list,
    _normalize_json_preview,
    _normalize_json_preview_by_tokens,
    _normalize_time_payload,
    _parse_datetime_utc,
    _require_object,
    _require_session,
    _resolve_object_ref,
    _safe_int,
    _sanitize_object_data_for_context,
    _session_turn_lock_key,
    _session_turn_runtime_lock,
    _session_turn_runtime_lock_key,
    _session_turn_runtime_lock_supported,
    _recover_abandoned_pending_turn_locked,
    _to_jsonable,
    _truncate_text,
    _truncate_text_to_tokens,
)
from .crud_profiles import (
    _build_faction_profile_text,
    _build_item_profile_text,
    _build_npc_profile_text,
    _build_player_profile_text,
    _build_quest_profile_text,
    _build_zone_profile_text,
    _clear_profile_synth_cache,
    _should_refresh_item_or_faction_profile_embedding,
    _should_refresh_npc_profile_embedding,
    _should_refresh_player_profile_embedding,
    _should_refresh_quest_profile_embedding,
    _should_refresh_zone_profile_embedding,
)
from .crud_embeddings_ops import (
    MEMORY_EVENT_OBJECT_TYPE,
    MEMORY_FACT_OBJECT_TYPE,
    _build_link_context_snippet,
    _coerce_importance,
    _extract_claim_text,
    _extract_link_context_text,
    _list_active_link_context_snippets,
    _maybe_embed_texts,
    _refresh_link_context_embedding,
    _store_memory_candidates,
    _upsert_claim_text_embedding,
    _upsert_faction_profile_embedding,
    _upsert_item_profile_embedding,
    _upsert_link_context_embedding,
    _upsert_npc_profile_embedding,
    _upsert_object_embedding,
    _upsert_player_profile_embedding,
    _upsert_quest_profile_embedding,
    _upsert_zone_profile_embedding,
)
from .crud_context import (
    NARRATIVE_SPINE_OBJECT_TYPE,
    SESSION_SUMMARY_LIVE_TURNS,
    SESSION_SUMMARY_OBJECT_TYPE,
    _SPINE_UPDATER_MAX_TOKENS,
    _apply_elastic_field_budgets,
    _apply_unified_context_scoring,
    _build_embedding_snippet,
    _build_event_embedding_line,
    _build_input_embedding_snippet,
    _build_reaction_hints,
    _build_relevance_query_text,
    _build_turn_context_pack,
    _build_zone_npc_knowledge,
    _collect_embedding_candidates,
    _embed_query_for_relevance,
    _enrich_reaction_hints,
    _ensure_world_prompt_chunks_indexed,
    _get_latest_narrative_spine_row,
    _get_player_inventory,
    _get_player_location_history,
    _get_recent_ai_text_for_relevance,
    _get_relevant_player_for_input,
    _list_active_npc_claim_links_for_knowledge,
    _list_active_zone_actor_ids,
    _list_one_hop_link_candidates_for_context,
    _list_orphaned_items_for_context,
    _list_recent_turn_payload_for_spine,
    _list_relevant_claims_for_input,
    _list_relevant_factions_for_input,
    _list_relevant_items_for_input,
    _list_relevant_links_for_input,
    _list_relevant_memories_for_input,
    _list_relevant_npcs_for_input,
    _list_relevant_objects_for_input,
    _list_relevant_quests_for_input,
    _list_relevant_world_prompt_chunks,
    _list_semantically_relevant_turn_indices,
    _list_turn_event_embedding_lines,
    _list_zone_npcs_with_relationships,
    _list_zone_recent_claims,
    _merge_npc_knowledge_subjects,
    _render_turn_ai_text,
    _resolve_exact_name_object_ids_for_context,
    _serialize_patch_ops,
    _split_world_prompt_chunks,
    _summarize_world_prompt_chunks,
    _update_narrative_spine,
)
from .crud_planning import (
    PATCH_OP_LIST_ADAPTER,
    _build_plan_from_debug_patch,
    _call_deepseek_patch_generator,
    _call_librarian,
    _call_narrator,
    _call_post_apply_narrator,
    _call_narrator_text_only,
    _collect_refs,
    _parse_narrator_response,
    _parse_world_intent_response,
    _resolve_travel_turn_plan,
    _resolve_turn_plan,
    _resolve_turn_plan_from_request,
    _resolve_turn_plan_legacy,
    _resolve_turn_plan_state_first,
    _resolve_turn_plan_split,
    _toposort_patch_ops,
    _validate_patch_ops,
)
from .crud_patch_apply import (
    _find_ephemeral_npc_dedup_candidate,
    _find_global_object_dedup_candidate,
    _find_global_zone_dedup_candidate,
    _find_persistent_npc_dedup_candidate,
    _prepare_object_create_chunk,
    apply_patch_ops,
)
from .crud_entities import (
    create_link,
    create_object,
    create_session_snapshot,
    create_session_with_defaults,
    delete_session,
    get_object,
    get_session,
    get_session_snapshot,
    get_session_token_stats,
    index_turn_embedding,
    list_events,
    list_links,
    list_objects,
    list_session_snapshots,
    reindex_world_prompt,
    semantic_retrieve,
)
from .crud_telemetry import (
    activate_system_prompt,
    get_llm_telemetry_summary,
    list_active_system_prompts,
    list_outbox_events,
    list_session_llm_telemetry,
)
from .crud_consequences import (
    COOLDOWN_TURNS,
    CONSEQUENCE_WINDOW_OBJECT_TYPE,
    MAX_CONSEQUENCE_CHAIN_DEPTH,
    MAX_CONSEQUENCE_WINDOW_SPAN,
    PRIORITY_WEIGHT,
    apply_consequence_results,
    detect_structural_signals,
    ensure_persisted_consequence_windows,
    enqueue_consequences,
    get_object_by_type,
    list_consequence_window_payloads,
    materialize_consequence_windows,
    resolve_applied_op_refs,
    select_due_consequences,
    select_due_consequences_from_state_json,
    select_due_consequence_window_payloads,
)

# Constants that are canonically defined here and used by crud_core functions.
CHRONICLE_OUTPUT_NAMESPACE = "chronicle_output"
CHRONICLE_INPUT_NAMESPACE = "chronicle_input"
RELEVANCE_QUERY_EMBED_INSTRUCTION = (
    "Retrieve relevant game history and world facts for the current player action"
)
WORLD_PROMPT_EMBED_INSTRUCTION = "Represent this world lore and setting rules for retrieval"

_SUMMARIZER_SYSTEM = (
    "You are a game chronicle indexer. "
    "Extract ONLY factual events from this RPG turn for semantic search indexing. "
    "One sentence per fact. No atmosphere, no adjectives. "
    "Include: who did what, with whom, what changed, what was learned. "
    "Max 6 sentences. Output plain text only."
)

def _summarize_turn_for_indexing(
    *,
    user_input: str,
    narration: str,
    applied_ops: list[dict[str, Any]],
    session_id: str | None = None,
) -> str:
    """Chronicle summarizer. Falls back to deterministic base snippet on any error."""
    fallback_snippet = _context._build_embedding_snippet(
        user_input=user_input,
        narration=narration,
        choices=[],
        applied_ops=applied_ops,
    )
    op_summary = ", ".join(
        f"{op.get('op')}:{op.get('name') or op.get('object') or ''}"
        for op in applied_ops[:8]
    )
    user_prompt = (
        f"Player: {user_input}\n"
        f"Narration: {narration[:800]}\n"
        f"Applied ops: {op_summary}"
    )
    try:
        with telemetry_context(request_type="chronicle_summarizer"):
            messages = [
                {"role": "system", "content": _SUMMARIZER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            content, _ = asyncio.run(
                openrouter_chat(
                    model=OPENROUTER_ASSISTANT_MODEL,
                    messages=messages,
                    max_tokens=200
                )
            )
            summary = content
    except Exception:  # noqa: BLE001
        logger.warning("Chronicle summarizer failed, using base snippet", exc_info=True)
        return fallback_snippet
    summary_text = str(summary or "").strip()
    if not summary_text:
        logger.warning("Chronicle summarizer returned empty output, using base snippet")
        return fallback_snippet
    return summary_text


def _allocate_turn(
    db: Session,
    session_id: uuid.UUID,
    user_input: str,
) -> tuple[int, int, int, int, int]:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic._allocate_turn(db, session_id, user_input)


def _coerce_turn_weight_value(raw_value: Any) -> float | None:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic._coerce_turn_weight_value(raw_value)


def _derive_turn_weight(
    *,
    applied_ops_count: int,
    narration: str,
    memory_candidates_count: int,
) -> float:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic._derive_turn_weight(
        applied_ops_count=applied_ops_count,
        narration=narration,
        memory_candidates_count=memory_candidates_count,
    )


class _PlanBuildFailed(RuntimeError):
    """Raised when turn plan construction fails inside the apply transaction."""


def _apply_turn_plan(
    db: Session,
    session_id: uuid.UUID,
    *,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
    allocation: Any | None = None,
    plan: TurnPlanResult | None = None,
    context_pack: dict[str, Any] | None = None,
    payload: schemas.TurnIn | None = None,
    allow_debug_patch: bool = False,
) -> tuple[models.TurnModel, list[dict[str, Any]], uuid.UUID | None]:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic._apply_turn_plan(
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


def _coerce_ttl_cleanup_result(value: _entities.TtlCleanupResult | int | None) -> _entities.TtlCleanupResult:
    return _entities._coerce_ttl_cleanup_result(value)


def cleanup_ephemeral_npcs(
    db: Session,
    session_id: uuid.UUID,
    current_turn: int,
    *,
    in_game_day: int | None = None,
    in_game_minute: int | None = None,
) -> _entities.TtlCleanupResult:
    return _entities._coerce_ttl_cleanup_result(
        _entities._cleanup_ephemeral_npcs(
            db,
            session_id,
            current_turn,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )
    )


def patch_turn(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
    payload: schemas.TurnPatchIn,
) -> models.TurnModel:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic.patch_turn(db, session_id, turn_index, payload)


def _recover_stuck_pending_turn(
    db: Session,
    session_id: uuid.UUID,
    expected_turn: int,
    *,
    reason: str,
) -> None:
    from . import crud_turns_logic as _turns_logic

    _turns_logic._recover_stuck_pending_turn(
        db=db,
        session_id=session_id,
        expected_turn=expected_turn,
        reason=reason,
    )


def recover_pending_turn(
    db: Session,
    session_id: uuid.UUID,
    *,
    force: bool = False,
) -> schemas.PendingTurnRecoveryOut:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic.recover_pending_turn(db, session_id, force=force)


def _repair_player_location_after_pending_turn_recovery(
    db: Session,
    session_id: uuid.UUID,
    session_row: models.SessionModel,
) -> bool:
    try:
        player_object_id = _get_session_player_object_id(db, session_id)
    except HTTPException as exc:
        if exc.status_code in {status.HTTP_404_NOT_FOUND, status.HTTP_409_CONFLICT}:
            return False
        raise

    active_locations = _shared._get_active_located_in_links(db, session_id, player_object_id)
    if len(active_locations) == 1:
        return False

    anchor_link = (
        active_locations[-1]
        if active_locations
        else _shared._get_latest_located_in_link(db, session_id, player_object_id)
    )
    anchor_zone = (
        _shared._get_object(db, session_id, anchor_link.to_object_id)
        if anchor_link is not None
        else None
    )
    created_anchor_zone = False
    if anchor_zone is None:
        anchor_zone = db.execute(
            select(models.ObjectModel)
            .where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == "zone",
            )
            .order_by(models.ObjectModel.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()
    if anchor_zone is None:
        anchor_zone = models.ObjectModel(
            session_id=session_id,
            type="zone",
            name=DEFAULT_SPAWN_ZONE_NAME,
            data={},
        )
        db.add(anchor_zone)
        db.flush()
        created_anchor_zone = True

    repair_turn = max(int(getattr(session_row, "turn_index", 0) or 0), 0)
    in_game_day, in_game_minute = _shared._extract_in_game_time(dict(getattr(session_row, "state_json", {}) or {}))
    turn_row = db.get(models.TurnModel, (session_id, repair_turn))
    if turn_row is None:
        raise RuntimeError(
            f"missing recovery turn row for session_id={session_id} turn_index={repair_turn}"
        )

    player_object = _get_object(db, session_id, player_object_id)
    if player_object is None:
        return False

    applied_ops: list[dict[str, Any]] = []
    if created_anchor_zone:
        applied_ops.append(
            {
                "op": "object.create",
                "ref": str(anchor_zone.object_id),
                "type": "zone",
                "name": anchor_zone.name,
                "data": dict(anchor_zone.data or {}),
            }
        )
        _entities._add_internal_object_created_event(
            db,
            session_id=session_id,
            turn_index=repair_turn,
            object_row=anchor_zone,
            object_data=dict(anchor_zone.data or {}),
        )
        if USE_EMBEDDINGS:
            _entities._enqueue_zone_profile_refresh_event(
                db,
                session_id=session_id,
                object_id=anchor_zone.object_id,
            )

    closed_links = _shared._close_player_active_located_in_links(
        db,
        session_id,
        player_object_id,
        closed_at_turn=repair_turn,
    )
    for closed_link in closed_links:
        applied_ops.append(
            {
                "op": "link.close",
                "from": str(closed_link.from_object_id),
                "to": str(closed_link.to_object_id),
                "type": closed_link.type,
            }
        )
        _entities._add_internal_link_event(
            db,
            session_id=session_id,
            turn_index=repair_turn,
            event_type="LINK_CLOSED",
            from_object_id=closed_link.from_object_id,
            to_object_id=closed_link.to_object_id,
            link_type=closed_link.type,
            from_object=_get_object(db, session_id, closed_link.from_object_id),
            to_object=_get_object(db, session_id, closed_link.to_object_id),
            valid_to_turn=repair_turn,
            source="pending_turn_recovery",
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )

    new_link = models.LinkModel(
        session_id=session_id,
        from_object_id=player_object_id,
        to_object_id=anchor_zone.object_id,
        type=LOCATED_IN_LINK_TYPE,
        data={},
        valid_from_turn=repair_turn,
        valid_to_turn=None,
    )
    db.add(new_link)
    applied_ops.append(
        {
            "op": "link.create",
            "from": str(player_object_id),
            "to": str(anchor_zone.object_id),
            "type": LOCATED_IN_LINK_TYPE,
            "data": {},
            "valid_from_turn": repair_turn,
            "valid_to_turn": None,
        }
    )
    _entities._add_internal_link_event(
        db,
        session_id=session_id,
        turn_index=repair_turn,
        event_type="LINK_CREATED",
        from_object_id=player_object_id,
        to_object_id=anchor_zone.object_id,
        link_type=LOCATED_IN_LINK_TYPE,
        from_object=player_object,
        to_object=anchor_zone,
        link_data={},
        valid_from_turn=repair_turn,
        valid_to_turn=None,
        source="pending_turn_recovery",
        in_game_day=in_game_day,
        in_game_minute=in_game_minute,
    )

    ai_json = dict(turn_row.ai_json or {})
    existing_applied_ops = [
        dict(op) for op in list(ai_json.get("applied_ops") or []) if isinstance(op, dict)
    ]
    existing_validated_updates = [
        dict(op) for op in list(ai_json.get("validated_updates") or []) if isinstance(op, dict)
    ]
    ai_json["applied_ops"] = existing_applied_ops + applied_ops
    ai_json["validated_updates"] = existing_validated_updates + applied_ops
    ai_json.setdefault("in_game_time", {"day": in_game_day, "minute": in_game_minute})
    turn_row.ai_json = ai_json

    logger.warning(
        "Recovered missing player location after pending turn reset in session_id=%s player_object_id=%s turn=%s",
        session_id,
        player_object_id,
        repair_turn,
    )
    return True


def _classify_plan_degradation(exc: BaseException) -> tuple[str, str, bool] | None:
    if isinstance(exc, CircuitOpenError):
        return "circuit_open", str(exc.provider), True

    provider = str(getattr(exc, "provider", "") or "").strip().lower()
    status_code = getattr(exc, "status_code", None)
    provider_availability_failure = getattr(exc, "provider_availability_failure", None)
    error_type = str(getattr(exc, "error_type", "") or type(exc).__name__).strip().lower()

    if provider and isinstance(provider_availability_failure, bool):
        if "timeout" in error_type:
            return "provider_timeout", provider, True
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return "provider_5xx", provider, True
        if provider_availability_failure:
            return "retry_exhausted", provider, True
        return None

    message = str(exc or "").strip().lower()
    if not message:
        return None

    provider = "unknown"
    if "openrouter" in message or "qwen" in message:
        provider = "openrouter"
    elif "xai" in message or "grok" in message or "deepseek" in message:
        provider = "openrouter"

    if "timeout" in message:
        return "provider_timeout", provider, True
    if " 5" in message or "status 5" in message or "service unavailable" in message:
        return "provider_5xx", provider, True
    if "retry" in message and ("exhausted" in message or "failed" in message):
        return "retry_exhausted", provider, True
    if "circuit open" in message:
        return "circuit_open", provider, True
    return None


def _build_degraded_turn_plan(
    *,
    cause: BaseException,
    in_game_day: int,
    in_game_minute: int,
) -> TurnPlanResult | None:
    degraded_meta = _classify_plan_degradation(cause)
    if degraded_meta is None:
        return None
    reason, provider, retryable = degraded_meta
    trace_id = get_trace_id()
    fallback_narration = FALLBACK_DEGRADED_NARRATION
    return TurnPlanResult(
        narration=fallback_narration,
        choices=[],
        zone_scope=None,
        parsed_ops=[],
        validator_status="reject",
        validator_reasons=[f"degraded:{reason}"],
        raw_response={
            "narration": fallback_narration,
            "choices": [],
            "proposed_updates": [],
            "semantic_events": [],
            "scene_entities": [],
            "consequence_seeds": [],
            "resolved_consequence_ids": [],
            "zone_scope": None,
            "in_game_time": {"day": in_game_day, "minute": in_game_minute},
            "degraded": {
                "active": True,
                "reason": reason,
                "provider": provider,
                "retryable": retryable,
                "trace_id": trace_id,
            },
        },
        librarian_used=False,
    )


def run_turn(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.TurnIn,
    *,
    allow_debug_patch: bool,
) -> models.TurnModel:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic.run_turn(
        db=db,
        session_id=session_id,
        payload=payload,
        allow_debug_patch=allow_debug_patch,
    )


def upload_lore(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.LoreUploadIn,
) -> dict[str, Any]:
    return _lore_adaptation.upload_lore(db, session_id, payload)


def get_lore_adaptation(
    db: Session,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    return _lore_adaptation.get_lore_adaptation(db, session_id)


def answer_lore_gap(
    db: Session,
    session_id: uuid.UUID,
    gap_id: str,
    payload: schemas.LoreGapAnswerIn,
) -> dict[str, Any]:
    return _lore_adaptation.answer_lore_gap(db, session_id, gap_id, payload)


def auto_resolve_lore_gaps(
    db: Session,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    return _lore_adaptation.auto_resolve_lore_gaps(db, session_id)


def get_lore_turn_blocker(
    db: Session,
    session_id: uuid.UUID,
) -> dict[str, Any] | None:
    return _lore_adaptation.get_lore_turn_blocker(db, session_id)


def _run_turn_locked(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.TurnIn,
    *,
    allow_debug_patch: bool,
) -> models.TurnModel:
    from . import crud_turns_logic as _turns_logic

    return _turns_logic._run_turn_locked(
        db=db,
        session_id=session_id,
        payload=payload,
        allow_debug_patch=allow_debug_patch,
    )


def move_player(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.MoveIn,
) -> schemas.MoveOut:
    from . import crud_movement as _movement

    return _movement.move_player(db, session_id, payload)


def _move_player_locked(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.MoveIn,
) -> schemas.MoveOut:
    from .crud_movement import _move_player_locked as _movement_move_locked

    return _movement_move_locked(db, session_id, payload)


def _extract_applied_ops_from_ai_json(ai_json: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    return extract_applied_ops_from_ai_json(ai_json)


def _resolve_committed_turn_upper_bound(
    session_row: models.SessionModel,
    *,
    requested_to_turn: int | None = None,
) -> int:
    return session_read_repository.resolve_committed_turn_upper_bound(
        session_row,
        requested_to_turn=requested_to_turn,
    )


def get_session_timeline(
    db: Session,
    session_id: uuid.UUID,
    *,
    from_turn: int = 0,
    to_turn: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    from . import crud as crud_runtime

    session_row = crud_runtime._require_session(db, session_id)
    return session_read_repository.get_session_timeline(
        db,
        session_id=session_id,
        session_row=session_row,
        from_turn=from_turn,
        to_turn=to_turn,
        limit=limit,
        offset=offset,
    )


def get_session_diff(
    db: Session,
    session_id: uuid.UUID,
    *,
    from_turn: int,
    to_turn: int,
) -> dict[str, Any]:
    from . import crud as crud_runtime

    session_row = crud_runtime._require_session(db, session_id)
    return session_read_repository.get_session_diff(
        db,
        session_id=session_id,
        session_row=session_row,
        from_turn=from_turn,
        to_turn=to_turn,
    )


def get_relationship_graph(
    db: Session,
    session_id: uuid.UUID,
    *,
    zone_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    from . import crud as crud_runtime

    crud_runtime._require_session(db, session_id)
    return session_read_repository.get_relationship_graph(
        db,
        session_id=session_id,
        zone_id=zone_id,
    )


def create_claim(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.ClaimCreateIn,
) -> schemas.ClaimCreateOut:
    from . import crud_claims as _claims

    return _claims.create_claim(db, session_id, payload)


# Re-export graph ops functions so the crud.py facade picks them up.
get_pending_graph_ops = _graph_ops.get_pending_graph_ops
apply_pending_graph_ops = _graph_ops.apply_pending_graph_ops
run_graph_health_check = _graph_ops.run_graph_health_check
maybe_run_graph_health_check = _graph_ops.maybe_run_graph_health_check
