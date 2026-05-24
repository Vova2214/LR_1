"""Shared runtime helpers and compatibility shims.

The runtime lock/session helpers in this module remain owned here. The durable
fact and memory-candidate policy helpers listed in `COMPATIBILITY_MODULE_CONTRACT`
are deprecated forwards into `src.domain.memory_candidates`.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from typing import TYPE_CHECKING, Any, Iterator, Literal

from fastapi import HTTPException, status
from sqlalchemy import func, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, SessionTransactionOrigin

from . import models, schemas
from .architecture_contracts import COMPATIBILITY_MODULE_CONTRACTS
from .constants import LOCATED_IN_LINK_TYPE, RECIPROCAL_SOCIAL_LINK_TYPES, SESSION_PLAYER_REF
from .db import CONTEXT_TOKENIZER_ENCODING, PENDING_TURN_TIMEOUT_SECONDS
from .domain import memory_candidates as memory_candidate_domain
from .strings import FALLBACK_TURN_ERROR

if TYPE_CHECKING:
    from .application.turn_contracts import (
        TurnApplyExternalArtifacts,
        TurnApplyExternalPreparationRequired,
        TurnApplyExternalRequest,
        TurnPlanResult,
    )

try:
    import tiktoken
except ImportError as exc:
    raise RuntimeError("tiktoken is required for token-based context budgeting") from exc

try:
    _TOKEN_ENCODING = tiktoken.get_encoding(CONTEXT_TOKENIZER_ENCODING)
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        f"Failed to load tokenizer encoding '{CONTEXT_TOKENIZER_ENCODING}'"
    ) from exc

MINUTES_PER_DAY = 1440
DEFAULT_TIME_SCALE_MINUTES = 10
MAX_TIME_SCALE_MINUTES = MINUTES_PER_DAY
CONTEXT_OBJECT_ALLOWED_KEYS = ("short_desc", "status", "hp")
CONTEXT_OBJECT_DESCRIPTION_KEYS = (
    "short_desc",
    "description",
    "desc",
    "summary",
    "objective",
    "goal",
    "current_step",
    "stage",
    "lore",
    "описание",
)
CONTEXT_OBJECT_TECHNICAL_KEYS = {
    "spawn",
    "despawn_turn",
    "despawned_turn",
    "despawn_reason",
    "ephemeral",
    "pinned",
}

logger = logging.getLogger(__name__)
COMPATIBILITY_MODULE_CONTRACT = COMPATIBILITY_MODULE_CONTRACTS[__name__]
_RUNTIME_TURN_LOCK_XOR_MASK = 0x9E3779B97F4A7C15
_HELD_RUNTIME_TURN_LOCK_KEYS_VAR: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "held_runtime_turn_lock_keys",
    default=frozenset(),
)
_TURN_APPLY_EXTERNAL_ARTIFACTS_VAR: contextvars.ContextVar["TurnApplyExternalArtifacts | None"] = (
    contextvars.ContextVar("turn_apply_external_artifacts", default=None)
)


@dataclass(slots=True)
class PatchValidationResult:
    status: Literal["ok", "reject", "uncertain"]
    reasons: list[str]
    parsed_ops: list[schemas.PatchOp]


_DURABLE_FACT_PRIORITY_RANK = {"low": 0, "med": 1, "high": 2}
_DURABLE_FACT_SCOPE_RANK = {"npc_private": 0, "public": 1, "global": 2}
_MEMORY_CALLBACK_STRENGTH_RANK = {"none": 0, "soft": 1, "strong": 2}
_TEXT_DISTINGUISHED_DURABLE_FACT_KINDS = frozenset(
    {
        "home_detail",
        "location_fact",
        "recurring_prop",
        "emotional_scene",
        "injury",
        "quest_milestone",
        "decision",
    }
)
_IDENTITY_TEXT_REQUIRED_DURABLE_FACT_KINDS = frozenset(
    {
        "home_detail",
        "location_fact",
        "recurring_prop",
        "emotional_scene",
        "injury",
        "quest_milestone",
        "decision",
        "promise",
        "debt",
        "gift",
        "betrayal",
    }
)
_ROLE_INSENSITIVE_REF_IDENTITY_FACT_KINDS = frozenset(
    {
        "ownership",
        "gift",
        "trophy",
    }
)
_SUBJECT_ONLY_REF_IDENTITY_FACT_KINDS = frozenset(
    {
        "home_detail",
        "recurring_prop",
        "location_fact",
        "emotional_scene",
        "injury",
        "quest_milestone",
        "decision",
    }
)
_ROLE_INSENSITIVE_COUNTERPARTY_FACT_KINDS = frozenset({"gift", "ownership", "trophy"})
_DIRECTIONAL_COUNTERPARTY_FACT_KINDS = frozenset({"promise", "debt", "betrayal"})
_COUNTERPARTY_FACT_KINDS = frozenset(
    _ROLE_INSENSITIVE_COUNTERPARTY_FACT_KINDS | _DIRECTIONAL_COUNTERPARTY_FACT_KINDS
)
def _merge_durable_fact_priority(left: str, right: str) -> str:
    return memory_candidate_domain.merge_durable_fact_priority(left, right)


def _merge_durable_fact_scope(left: str, right: str) -> str:
    return memory_candidate_domain.merge_durable_fact_scope(left, right)


def _merge_callback_strength(left: str, right: str) -> str:
    return memory_candidate_domain.merge_callback_strength(left, right)


def _merge_unique_refs(*ref_lists: list[schemas.Ref] | tuple[schemas.Ref, ...]) -> list[schemas.Ref]:
    return memory_candidate_domain.merge_unique_refs(*ref_lists)


def _resolve_fact_identity_ref(
    raw_ref: schemas.Ref | None,
    *,
    ref_map: dict[str, str] | None = None,
    player_object_id: uuid.UUID | None = None,
) -> schemas.Ref | None:
    return memory_candidate_domain.resolve_fact_identity_ref(
        raw_ref,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )


def _canonicalize_durable_fact_identity_refs(
    fact: schemas.DurableFact,
    *,
    ref_map: dict[str, str] | None = None,
    player_object_id: uuid.UUID | None = None,
) -> schemas.DurableFact:
    return memory_candidate_domain.canonicalize_durable_fact_identity_refs(
        fact,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )


def _canonicalize_memory_candidate_identity_refs(
    candidate: schemas.MemoryCandidate,
    *,
    ref_map: dict[str, str] | None = None,
    player_object_id: uuid.UUID | None = None,
) -> schemas.MemoryCandidate:
    return memory_candidate_domain.canonicalize_memory_candidate_identity_refs(
        candidate,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )


def _normalized_fact_identity_text(kind: str, text: str) -> str:
    return memory_candidate_domain.normalized_fact_identity_text(kind, text)


def _fact_identity_text_value(fact: schemas.DurableFact) -> str | None:
    return memory_candidate_domain.fact_identity_text_value(fact)


def _with_fact_identity_text(fact: schemas.DurableFact) -> schemas.DurableFact:
    return memory_candidate_domain.with_fact_identity_text(fact)


def _fact_identity_ref_signature(
    *,
    kind: str,
    actor_ref: Any = None,
    counterparty_ref: Any | None = None,
    object_ref: Any = None,
    location_ref: Any = None,
    quest_ref: Any = None,
    relationship_type: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    return memory_candidate_domain.fact_identity_ref_signature(
        kind=kind,
        actor_ref=actor_ref,
        counterparty_ref=counterparty_ref,
        object_ref=object_ref,
        location_ref=location_ref,
        quest_ref=quest_ref,
        relationship_type=relationship_type,
    )


def _durable_fact_signature(fact: schemas.DurableFact) -> tuple[str, str, tuple[str, ...], str, str]:
    return memory_candidate_domain.durable_fact_signature(fact)


def _merge_durable_fact(existing: schemas.DurableFact, incoming: schemas.DurableFact) -> schemas.DurableFact:
    return memory_candidate_domain.merge_durable_fact(existing, incoming)


def _merge_fact_memory_candidates(
    existing: schemas.MemoryCandidate,
    incoming: schemas.MemoryCandidate,
) -> schemas.MemoryCandidate:
    return memory_candidate_domain.merge_fact_memory_candidates(existing, incoming)


def _merge_event_memory_candidates(
    existing: schemas.MemoryCandidate,
    incoming: schemas.MemoryCandidate,
) -> schemas.MemoryCandidate:
    return memory_candidate_domain.merge_event_memory_candidates(existing, incoming)


def _commit_event_scene_ref_signature(
    candidate: schemas.MemoryCandidate,
) -> tuple[str, ...]:
    return memory_candidate_domain.commit_event_scene_ref_signature(candidate)

def _memory_candidate_commit_scene_signature(
    candidate: schemas.MemoryCandidate,
) -> tuple[str, ...]:
    return memory_candidate_domain.memory_candidate_commit_scene_signature(candidate)


def _memory_candidate_identity_signature(
    candidate: schemas.MemoryCandidate,
) -> tuple[Any, ...]:
    return memory_candidate_domain.memory_candidate_identity_signature(candidate)


def _coalesce_memory_candidates_by_identity(
    memory_candidates: list[schemas.MemoryCandidate],
    *,
    ref_map: dict[str, str] | None = None,
    player_object_id: uuid.UUID | None = None,
) -> list[schemas.MemoryCandidate]:
    return memory_candidate_domain.coalesce_memory_candidates_by_identity(
        memory_candidates,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )


def _memory_candidates_to_durable_facts(
    memory_candidates: list[schemas.MemoryCandidate],
) -> list[schemas.DurableFact]:
    return memory_candidate_domain.memory_candidates_to_durable_facts(memory_candidates)


def _default_memory_candidate_durability(priority: str) -> float:
    return memory_candidate_domain.default_memory_candidate_durability(priority)


def _durable_fact_to_memory_candidate(
    fact: schemas.DurableFact,
    *,
    fallback_anchors: list[schemas.Ref] | None = None,
) -> schemas.MemoryCandidate | None:
    return memory_candidate_domain.durable_fact_to_memory_candidate(
        fact,
        fallback_anchors=fallback_anchors,
    )


def _committing_fact_memory_candidate(
    candidate: schemas.MemoryCandidate,
    *,
    fallback_anchors: list[schemas.Ref] | None = None,
) -> schemas.MemoryCandidate | None:
    return memory_candidate_domain.committing_fact_memory_candidate(
        candidate,
        fallback_anchors=fallback_anchors,
    )


def _supplement_memory_candidates_with_durable_facts(
    memory_candidates: list[schemas.MemoryCandidate],
    durable_facts: list[schemas.DurableFact],
    *,
    fallback_anchors: list[schemas.Ref] | None = None,
) -> list[schemas.MemoryCandidate]:
    return memory_candidate_domain.supplement_memory_candidates_with_durable_facts(
        memory_candidates,
        durable_facts,
        fallback_anchors=fallback_anchors,
    )


def _memory_fact_identity_key(
    *,
    kind: str,
    actor_object_id: uuid.UUID | None,
    counterparty_object_id: uuid.UUID | None,
    object_id: uuid.UUID | None,
    location_object_id: uuid.UUID | None,
    quest_object_id: uuid.UUID | None,
    identity_text: str,
    relationship_type: str | None,
) -> str:
    return memory_candidate_domain.memory_fact_identity_key(
        kind=kind,
        actor_ref=actor_object_id,
        counterparty_ref=counterparty_object_id,
        object_ref=object_id,
        location_ref=location_object_id,
        quest_ref=quest_object_id,
        relationship_type=relationship_type,
        identity_text=identity_text,
    )


def _effective_durable_facts(
    durable_facts: list[schemas.DurableFact],
    *,
    memory_candidates: list[schemas.MemoryCandidate] | None = None,
) -> list[schemas.DurableFact]:
    return memory_candidate_domain.effective_durable_facts(
        durable_facts,
        memory_candidates=memory_candidates,
    )


@dataclass(slots=True)
class PreparedObjectCreateOp:
    object_data: dict[str, Any]
    current_zone_id: uuid.UUID | None
    npc_profile_text: str | None = None
    npc_profile_embedding: list[float] | None = None
    zone_profile_text: str | None = None
    zone_profile_embedding: list[float] | None = None
    item_profile_text: str | None = None
    item_profile_embedding: list[float] | None = None
    faction_profile_text: str | None = None
    faction_profile_embedding: list[float] | None = None
    quest_profile_text: str | None = None
    quest_profile_embedding: list[float] | None = None
    player_profile_text: str | None = None
    player_profile_embedding: list[float] | None = None


def _session_transaction_origin(db: Session) -> SessionTransactionOrigin | None:
    get_transaction = getattr(db, "get_transaction", None)
    if not callable(get_transaction):
        return None
    transaction = get_transaction()
    origin = getattr(transaction, "origin", None)
    return origin if isinstance(origin, SessionTransactionOrigin) else None


def _session_has_pending_state(db: Session) -> bool:
    for attr_name in ("new", "dirty", "deleted"):
        collection = getattr(db, attr_name, None)
        if collection is None:
            continue
        try:
            if len(collection) > 0:
                return True
        except Exception:  # noqa: BLE001
            if bool(collection):
                return True
    return False


def _rollback_read_only_autobegin_transaction(db: Session) -> bool:
    in_transaction_attr = getattr(db, "in_transaction", None)
    in_transaction_value = in_transaction_attr() if callable(in_transaction_attr) else in_transaction_attr
    if not isinstance(in_transaction_value, bool) or not in_transaction_value:
        return False
    if _session_transaction_origin(db) != SessionTransactionOrigin.AUTOBEGIN:
        return False
    if _session_has_pending_state(db):
        return False
    db.rollback()
    return True


def _turn_apply_embedding_key(*, instruction: str | None, text: str) -> tuple[str | None, str]:
    normalized_instruction = str(instruction or "").strip() or None
    return normalized_instruction, str(text)


def _turn_apply_dedup_arbiter_key(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def current_turn_apply_external_artifacts() -> TurnApplyExternalArtifacts | None:
    return _TURN_APPLY_EXTERNAL_ARTIFACTS_VAR.get()


@contextmanager
def turn_apply_external_artifacts_context(artifacts: TurnApplyExternalArtifacts) -> Iterator[None]:
    token = _TURN_APPLY_EXTERNAL_ARTIFACTS_VAR.set(artifacts)
    try:
        yield
    finally:
        _TURN_APPLY_EXTERNAL_ARTIFACTS_VAR.reset(token)


@contextmanager
def suspend_turn_apply_external_artifacts() -> Iterator[None]:
    token = _TURN_APPLY_EXTERNAL_ARTIFACTS_VAR.set(None)
    try:
        yield
    finally:
        _TURN_APPLY_EXTERNAL_ARTIFACTS_VAR.reset(token)


def store_turn_apply_embedding_vectors(
    artifacts: TurnApplyExternalArtifacts,
    *,
    instruction: str | None,
    texts: list[str],
    vectors: list[list[float]],
) -> None:
    for text, vector in zip(texts, vectors, strict=True):
        artifacts.embedding_vectors[_turn_apply_embedding_key(instruction=instruction, text=text)] = list(vector)


def store_turn_apply_dedup_arbiter_decision(
    artifacts: TurnApplyExternalArtifacts,
    *,
    payload: dict[str, Any],
    decision: bool,
) -> None:
    artifacts.dedup_arbiter_decisions[_turn_apply_dedup_arbiter_key(payload)] = bool(decision)


def store_turn_apply_profile_text(
    artifacts: TurnApplyExternalArtifacts,
    *,
    cache_key: str,
    text: str,
) -> None:
    normalized_key = str(cache_key).strip()
    if not normalized_key:
        raise ValueError("cache_key is required for profile text artifacts")
    artifacts.profile_texts[normalized_key] = str(text)


def _prepare_turn_apply_external_request(
    artifacts: TurnApplyExternalArtifacts,
    request: TurnApplyExternalRequest | TurnApplyExternalPreparationRequired,
) -> None:
    from .application.turn_contracts import TurnApplyExternalPreparationRequired

    requested = request.request if isinstance(request, TurnApplyExternalPreparationRequired) else request
    with suspend_turn_apply_external_artifacts():
        if requested.kind == "embedding":
            from .crud_embeddings_ops import _maybe_embed_texts

            texts = list(requested.texts)
            vectors = _maybe_embed_texts(texts, instruction=requested.instruction)
            store_turn_apply_embedding_vectors(
                artifacts,
                instruction=requested.instruction,
                texts=texts,
                vectors=vectors,
            )
            return

        if requested.kind == "dedup_arbiter":
            from .crud_patch_apply import _call_dedup_arbiter

            payload = dict(requested.dedup_payload or {})
            decision = _call_dedup_arbiter(payload)
            store_turn_apply_dedup_arbiter_decision(
                artifacts,
                payload=payload,
                decision=decision,
            )
            return

        if requested.kind == "profile_text":
            from .crud_profiles import _maybe_synthesize_profile_text

            cache_key = str(requested.profile_cache_key or "").strip()
            if not cache_key:
                raise RuntimeError("profile_text request is missing profile_cache_key")
            synthesized = _maybe_synthesize_profile_text(
                object_type=str(requested.profile_object_type or ""),
                name=str(requested.profile_name or ""),
                data=dict(requested.profile_data or {}),
                fallback_text=str(requested.profile_fallback_text or ""),
            )
            store_turn_apply_profile_text(
                artifacts,
                cache_key=cache_key,
                text=synthesized,
            )
            return

    raise RuntimeError(f"Unsupported turn apply external request kind: {requested.kind}")


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _coerce_time_scale_minutes(raw_value: Any) -> int:
    parsed = _safe_int(raw_value)
    if parsed is None or parsed <= 0:
        return DEFAULT_TIME_SCALE_MINUTES
    return min(parsed, MAX_TIME_SCALE_MINUTES)


def _parse_datetime_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _get_pending_turn_started_at(
    db: Session,
    session_id: uuid.UUID,
    pending_turn: int,
    state_payload: dict[str, Any],
) -> datetime | None:
    started_at = _parse_datetime_utc(state_payload.get("pending_turn_started_at"))
    if started_at is not None:
        return started_at

    turn_row = db.get(models.TurnModel, (session_id, pending_turn))
    if turn_row is None:
        return None
    return _parse_datetime_utc(turn_row.created_at)


def _is_pending_turn_stale(started_at: datetime | None, *, now_utc: datetime | None = None) -> bool:
    if started_at is None:
        return False
    now_value = now_utc or datetime.now(timezone.utc)
    timeout_seconds = max(PENDING_TURN_TIMEOUT_SECONDS, 1)
    return (now_value - started_at).total_seconds() >= timeout_seconds


def _clear_pending_turn_locked(
    db: Session,
    session_id: uuid.UUID,
    session_row: models.SessionModel,
    state_payload: dict[str, Any],
    pending_turn: int,
    *,
    reason: str,
) -> None:
    # Contract: turn indices are monotonic and never decremented during recovery.
    # Failed pending turns are kept as tombstones to avoid PK/index reuse and
    # preserve absolute turn semantics for downstream systems.
    state_payload.pop("pending_turn", None)
    state_payload.pop("pending_turn_started_at", None)
    state_payload.pop("pending_controlled_ops_turn", None)
    state_payload.pop("pending_controlled_ops", None)

    time_scale = _coerce_time_scale_minutes(state_payload.get("time_scale"))
    state_payload["time_scale"] = time_scale

    raw_time = state_payload.get("time")
    if isinstance(raw_time, dict):
        day = max(_safe_int(raw_time.get("day")) or 0, 0)
        minute = max(_safe_int(raw_time.get("minute")) or 0, 0)
        total_minutes = max(day * MINUTES_PER_DAY + minute, 0)
        rollback_total = max(total_minutes - time_scale, 0)
        normalized_day, normalized_minute = divmod(rollback_total, MINUTES_PER_DAY)
        state_payload["time"] = {"day": normalized_day, "minute": normalized_minute}

    session_row.state_json = state_payload

    in_game_day, in_game_minute = _extract_in_game_time(state_payload)
    turn_row = db.get(models.TurnModel, (session_id, pending_turn))
    if turn_row is not None:
        ai_json = dict(turn_row.ai_json or {})
        if ai_json.get("status") == "pending":
            ai_json["status"] = "error"
        ai_json.setdefault("error", reason)
        ai_json.setdefault("source", "pending_turn_recovery")
        ai_json.setdefault("note", "pending_turn_recovery")
        ai_json.setdefault("applied_ops", [])
        ai_json.setdefault("validated_updates", [])
        ai_json.setdefault("in_game_time", {"day": in_game_day, "minute": in_game_minute})
        if not turn_row.ai_text:
            turn_row.ai_text = FALLBACK_TURN_ERROR
        turn_row.ai_json = ai_json
        return

    db.add(
        models.TurnModel(
            session_id=session_id,
            turn_index=pending_turn,
            user_input="[pending_turn_recovery]",
            ai_text=FALLBACK_TURN_ERROR,
            ai_json=_build_internal_turn_ai_json(
                note="pending_turn_recovery",
                applied_ops=[],
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
                extra_ai_json={
                    "status": "error",
                    "source": "pending_turn_recovery",
                    "error": reason,
                },
            ),
        )
    )


def _normalize_time_payload(state_payload: dict[str, Any]) -> tuple[int, int, int]:
    """Advance in-game time by ``time_scale`` minutes and return (day, minute, scale).

    Every caller (_allocate_turn, move_player) advances time by one
    ``time_scale`` step.  This is intentional: each action consumes
    in-game time.  Two sequential actions (e.g. move + turn) therefore
    advance time by 2 × time_scale.
    """
    raw_time = state_payload.get("time")
    if not isinstance(raw_time, dict):
        raw_time = {"day": 0, "minute": 0}

    day = _safe_int(raw_time.get("day"))
    minute = _safe_int(raw_time.get("minute"))
    if day is None or day < 0:
        day = 0
    if minute is None or minute < 0 or minute >= MINUTES_PER_DAY:
        minute = 0

    time_scale = _coerce_time_scale_minutes(state_payload.get("time_scale"))

    total_minutes = day * MINUTES_PER_DAY + minute + time_scale
    next_day = total_minutes // MINUTES_PER_DAY
    next_minute = total_minutes % MINUTES_PER_DAY
    # Cap the day counter to prevent unbounded growth over very long sessions.
    next_day = min(next_day, 999_999)
    return next_day, next_minute, time_scale


def _coerce_state_payload(raw_state: Any) -> dict[str, Any]:
    if isinstance(raw_state, dict):
        return dict(raw_state)
    return {}


def _extract_in_game_time(state_payload: dict[str, Any]) -> tuple[int | None, int | None]:
    raw_time = state_payload.get("time")
    if not isinstance(raw_time, dict):
        return None, None
    return _safe_int(raw_time.get("day")), _safe_int(raw_time.get("minute"))


def _build_internal_turn_ai_json(
    *,
    note: str,
    applied_ops: list[dict[str, Any]],
    in_game_day: int | None,
    in_game_minute: int | None,
    extra_ai_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    serialized_ops = [dict(op) for op in applied_ops if isinstance(op, dict)]
    ai_json: dict[str, Any] = {
        "status": "completed",
        "source": "internal_api",
        "note": str(note or "").strip() or "internal_mutation",
        "applied_ops": serialized_ops,
        "validated_updates": serialized_ops,
        "in_game_time": {"day": in_game_day, "minute": in_game_minute},
    }
    if isinstance(extra_ai_json, dict):
        ai_json.update({str(key): value for key, value in extra_ai_json.items()})
    return ai_json


def _create_internal_turn_row(
    db: Session,
    session_id: uuid.UUID,
    session_row: models.SessionModel,
    *,
    turn_index: int,
    user_input: str,
    ai_text: str | None,
    note: str,
    applied_ops: list[dict[str, Any]],
    in_game_day: int | None,
    in_game_minute: int | None,
    extra_ai_json: dict[str, Any] | None = None,
) -> models.TurnModel:
    turn_row = models.TurnModel(
        session_id=session_id,
        turn_index=max(int(turn_index), 0),
        user_input=str(user_input or "").strip() or "[internal_mutation]",
        ai_text=str(ai_text).strip() if isinstance(ai_text, str) and ai_text.strip() else None,
        ai_json=_build_internal_turn_ai_json(
            note=note,
            applied_ops=applied_ops,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
            extra_ai_json=extra_ai_json,
        ),
    )
    session_row.turn_index = turn_row.turn_index
    db.add(turn_row)
    db.flush([turn_row])
    return turn_row


def _require_session(db: Session, session_id: uuid.UUID, *, for_update: bool = False) -> models.SessionModel:
    query = select(models.SessionModel).where(models.SessionModel.id == session_id)
    if for_update:
        query = query.with_for_update()

    session_row = db.execute(query).scalar_one_or_none()
    if session_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session_row


def _session_turn_lock_key(session_id: uuid.UUID) -> int:
    # Keep the lock key stable and deterministic per session.
    key = session_id.int & ((1 << 64) - 1)
    if key >= (1 << 63):
        key -= (1 << 64)
    return key


def _session_turn_runtime_lock_key(session_id: uuid.UUID) -> int:
    key = (session_id.int & ((1 << 64) - 1)) ^ _RUNTIME_TURN_LOCK_XOR_MASK
    if key >= (1 << 63):
        key -= (1 << 64)
    return key


def _current_context_holds_session_turn_runtime_lock(session_id: uuid.UUID) -> bool:
    lock_key = _session_turn_runtime_lock_key(session_id)
    return lock_key in _HELD_RUNTIME_TURN_LOCK_KEYS_VAR.get()


def _session_turn_runtime_lock_supported(db: Session) -> bool:
    bind = db.get_bind()
    dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "")).lower()
    return dialect_name == "postgresql"


def _acquire_session_turn_lock(db: Session, session_id: uuid.UUID) -> None:
    if not _session_turn_runtime_lock_supported(db):
        return
    db.execute(select(func.pg_advisory_xact_lock(_session_turn_lock_key(session_id))))


def _open_lock_connection(bind: Any) -> Any:
    if isinstance(bind, Engine):
        return bind.connect()
    if isinstance(bind, Connection):
        return bind.engine.connect()

    engine = getattr(bind, "engine", None)
    if engine is not None:
        engine_connect = getattr(engine, "connect", None)
        if callable(engine_connect):
            return engine_connect()

    connect = getattr(bind, "connect", None)
    if callable(connect):
        return connect()

    raise RuntimeError(f"Unsupported bind type for advisory lock connection: {type(bind).__name__}")


def _recover_abandoned_pending_turn_locked(
    db: Session,
    session_id: uuid.UUID,
    session_row: models.SessionModel,
    state_payload: dict[str, Any],
    *,
    reason_if_recovered: str = "owner_lock_missing_auto_recovery",
) -> dict[str, Any]:
    pending_turn = _safe_int(state_payload.get("pending_turn"))
    if pending_turn is None:
        if state_payload.get("pending_turn") is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="turn already in progress",
            )
        return state_payload

    if _session_turn_runtime_lock_supported(db):
        if (
            _is_session_turn_runtime_lock_held(db, session_id)
            and not _current_context_holds_session_turn_runtime_lock(session_id)
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="turn already in progress",
            )
        _clear_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=state_payload,
            pending_turn=pending_turn,
            reason=reason_if_recovered,
        )
        return dict(session_row.state_json or {})

    started_at = _get_pending_turn_started_at(db, session_id, pending_turn, state_payload)
    if not _is_pending_turn_stale(started_at):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="turn already in progress",
        )

    _clear_pending_turn_locked(
        db=db,
        session_id=session_id,
        session_row=session_row,
        state_payload=state_payload,
        pending_turn=pending_turn,
        reason="timeout_auto_recovery",
    )
    return dict(session_row.state_json or {})


@contextmanager
def _session_turn_runtime_lock(db: Session, session_id: uuid.UUID) -> Iterator[None]:
    if not _session_turn_runtime_lock_supported(db):
        yield
        return

    bind = db.get_bind()
    lock_key = _session_turn_runtime_lock_key(session_id)
    lock_conn = _open_lock_connection(bind)
    lock_acquired = False
    lock_token: contextvars.Token[frozenset[int]] | None = None
    try:
        lock_acquired = bool(lock_conn.execute(select(func.pg_try_advisory_lock(lock_key))).scalar_one())
        if not lock_acquired:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="turn already in progress",
            )

        held_lock_keys = _HELD_RUNTIME_TURN_LOCK_KEYS_VAR.get()
        lock_token = _HELD_RUNTIME_TURN_LOCK_KEYS_VAR.set(held_lock_keys | {lock_key})
        yield
    finally:
        if lock_acquired:
            try:
                unlocked = lock_conn.execute(select(func.pg_advisory_unlock(lock_key))).scalar_one_or_none()
                if unlocked is False:
                    logger.warning(
                        "Session runtime turn lock was not held during unlock for session_id=%s",
                        session_id,
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to release runtime advisory lock for session_id=%s; "
                    "the lock will be released when the connection closes",
                    session_id,
                )
        if lock_token is not None:
            _HELD_RUNTIME_TURN_LOCK_KEYS_VAR.reset(lock_token)
        try:
            lock_conn.close()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to close lock connection for session_id=%s",
                session_id,
            )


def _is_session_turn_runtime_lock_held(db: Session, session_id: uuid.UUID) -> bool:
    if not _session_turn_runtime_lock_supported(db):
        # Unknown lock state on non-Postgres; keep conservative behavior.
        return True

    bind = db.get_bind()
    lock_key = _session_turn_runtime_lock_key(session_id)
    # Query pg_locks catalog instead of acquiring/releasing the lock,
    # which eliminates the race window in the old try-then-unlock pattern.
    unsigned_key = lock_key & 0xFFFFFFFFFFFFFFFF
    classid = int((unsigned_key >> 32) & 0xFFFFFFFF)
    objid = int(unsigned_key & 0xFFFFFFFF)
    held = db.execute(
        text(
            "SELECT count(*) > 0 FROM pg_locks "
            "WHERE locktype = 'advisory' "
            "AND classid = :classid AND objid = :objid "
            "AND granted = true"
        ),
        {"classid": classid, "objid": objid},
    ).scalar_one()
    return bool(held)


def _get_object(db: Session, session_id: uuid.UUID, object_id: uuid.UUID) -> models.ObjectModel | None:
    return db.execute(
        select(models.ObjectModel).where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.object_id == object_id,
        )
    ).scalar_one_or_none()


def _require_object(db: Session, session_id: uuid.UUID, object_id: uuid.UUID) -> models.ObjectModel:
    object_row = _get_object(db, session_id, object_id)
    if object_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object {object_id} not found in session",
        )
    return object_row


def _get_session_player_object_id(db: Session, session_id: uuid.UUID) -> uuid.UUID:
    session_row = _require_session(db, session_id)
    player_object_id_raw = (session_row.state_json or {}).get("player_object_id")

    if player_object_id_raw is not None:
        try:
            player_object_id = uuid.UUID(str(player_object_id_raw))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Invalid player_object_id in session state",
            ) from exc
        player_object = _require_object(db, session_id, player_object_id)
        if player_object.type != "player":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="player_object_id in session state does not reference a player object",
            )
        return player_object_id

    player_object_id_candidate = db.execute(
        select(models.ObjectModel.object_id)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "player",
        )
        .order_by(models.ObjectModel.created_at.asc())
        .limit(1)
    ).scalar_one_or_none()
    if player_object_id_candidate is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session has no player object",
        )
    return player_object_id_candidate


def _resolve_object_ref(
    db: Session,
    session_id: uuid.UUID,
    ref: schemas.Ref,
    ref_map: dict[str, uuid.UUID],
) -> uuid.UUID:
    if isinstance(ref, uuid.UUID):
        object_id = ref
    elif ref == SESSION_PLAYER_REF:
        object_id = _get_session_player_object_id(db, session_id)
    else:
        mapped_object_id = ref_map.get(ref)
        if mapped_object_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown temporary ref: {ref}",
            )
        object_id = mapped_object_id

    _require_object(db, session_id, object_id)
    return object_id


def _get_active_located_in_links(
    db: Session,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
) -> list[models.LinkModel]:
    return list(
        db.execute(
            select(models.LinkModel)
            .where(
                models.LinkModel.session_id == session_id,
                models.LinkModel.from_object_id == from_object_id,
                models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                models.LinkModel.valid_to_turn.is_(None),
            )
            .order_by(models.LinkModel.created_at.asc())
        ).scalars().all()
    )


def _get_active_link(
    db: Session,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
    link_type: str,
) -> models.LinkModel | None:
    row = db.execute(
        select(models.LinkModel)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == from_object_id,
            models.LinkModel.to_object_id == to_object_id,
            models.LinkModel.type == link_type,
            models.LinkModel.valid_to_turn.is_(None),
        )
        .order_by(models.LinkModel.created_at.asc())
        .limit(1)
    ).scalar_one_or_none()
    if not isinstance(row, models.LinkModel):
        return None
    return row


def _get_latest_located_in_link(
    db: Session,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
) -> models.LinkModel | None:
    return db.execute(
        select(models.LinkModel)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == from_object_id,
            models.LinkModel.type == LOCATED_IN_LINK_TYPE,
        )
        .order_by(
            models.LinkModel.valid_from_turn.desc(),
            models.LinkModel.created_at.desc(),
        )
        .limit(1)
    ).scalar_one_or_none()


def _close_player_active_located_in_links(
    db: Session,
    session_id: uuid.UUID,
    player_object_id: uuid.UUID,
    *,
    closed_at_turn: int,
) -> list[models.LinkModel]:
    """Close all active ``located_in`` links for the player.

    .. warning::

        This function only **closes** existing links; it does NOT create
        a replacement.  Every caller MUST add a new ``located_in`` link
        immediately after calling this function, otherwise the player
        will have no active location.

    Returns the list of links that were closed (for logging/diagnostics).
    """
    active_links = _get_active_located_in_links(db, session_id, player_object_id)
    if len(active_links) > 1:
        logger.warning(
            "Repairing %s active located_in links for session_id=%s player_object_id=%s",
            len(active_links),
            session_id,
            player_object_id,
        )
    for link in active_links:
        close_turn = closed_at_turn
        link_valid_from = getattr(link, "valid_from_turn", close_turn)
        if isinstance(link_valid_from, int) and close_turn < link_valid_from:
            logger.warning(
                "Clamping link close turn %s to valid_from_turn %s for link_id=%s "
                "in session_id=%s; this creates a zero-width validity window",
                close_turn,
                link_valid_from,
                getattr(link, "link_id", "?"),
                session_id,
            )
            close_turn = link_valid_from
        link.valid_to_turn = close_turn
    return active_links


def _get_player_current_zone_id(db: Session, session_id: uuid.UUID) -> uuid.UUID | None:
    try:
        player_object_id = _get_session_player_object_id(db, session_id)
    except HTTPException:
        return None

    active_links = _get_active_located_in_links(db, session_id, player_object_id)
    if not active_links:
        return None
    return active_links[0].to_object_id


def _infer_actor_zone_id(
    db: Session,
    session_id: uuid.UUID,
    actor_object_id: uuid.UUID,
) -> uuid.UUID | None:
    active_links = _get_active_located_in_links(db, session_id, actor_object_id)
    if len(active_links) > 1:
        logger.warning(
            "Multiple active located_in links for session_id=%s actor=%s; using first link deterministically",
            session_id,
            actor_object_id,
        )
    if active_links:
        return active_links[0].to_object_id

    latest_link = _get_latest_located_in_link(db, session_id, actor_object_id)
    if latest_link is not None:
        return latest_link.to_object_id
    return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _build_session_snapshot_dump(
    db: Session,
    session_row: models.SessionModel,
) -> dict[str, Any]:
    session_id = session_row.id

    object_rows = db.execute(
        select(models.ObjectModel)
        .where(models.ObjectModel.session_id == session_id)
        .order_by(models.ObjectModel.created_at.asc())
    ).scalars().all()
    link_rows = db.execute(
        select(models.LinkModel)
        .where(models.LinkModel.session_id == session_id)
        .order_by(models.LinkModel.created_at.asc())
    ).scalars().all()
    turn_rows = db.execute(
        select(models.TurnModel)
        .where(models.TurnModel.session_id == session_id)
        .order_by(models.TurnModel.turn_index.asc())
    ).scalars().all()
    event_rows = db.execute(
        select(models.EventModel)
        .where(models.EventModel.session_id == session_id)
        .order_by(models.EventModel.turn_index.asc(), models.EventModel.created_at.asc())
    ).scalars().all()

    dump_payload: dict[str, Any] = {
        "meta": {
            "session_id": str(session_id),
            "captured_turn_index": session_row.turn_index,
            "captured_at": _to_jsonable(datetime.now(timezone.utc)),
        },
        "session": {
            "id": session_id,
            "created_at": session_row.created_at,
            "updated_at": session_row.updated_at,
            "world_prompt": session_row.world_prompt,
            "state_json": session_row.state_json,
            "turn_index": session_row.turn_index,
        },
        "objects": [
            {
                "session_id": row.session_id,
                "object_id": row.object_id,
                "type": row.type,
                "name": row.name,
                "data": row.data,
                "created_at": row.created_at,
            }
            for row in object_rows
        ],
        "links": [
            {
                "session_id": row.session_id,
                "link_id": row.link_id,
                "from_object_id": row.from_object_id,
                "to_object_id": row.to_object_id,
                "type": row.type,
                "data": row.data,
                "valid_from_turn": row.valid_from_turn,
                "valid_to_turn": row.valid_to_turn,
                "created_at": row.created_at,
            }
            for row in link_rows
        ],
        "turns": [
            {
                "session_id": row.session_id,
                "turn_index": row.turn_index,
                "user_input": row.user_input,
                "ai_text": row.ai_text,
                "ai_json": row.ai_json,
                "created_at": row.created_at,
            }
            for row in turn_rows
        ],
        "events": [
            {
                "session_id": row.session_id,
                "event_id": row.event_id,
                "turn_index": row.turn_index,
                "type": row.type,
                "scope_object_id": row.scope_object_id,
                "payload": row.payload,
                "created_at": row.created_at,
            }
            for row in event_rows
        ],
    }
    return _to_jsonable(dump_payload)


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return value[:max_chars]


def _normalize_json_preview(payload: Any, max_chars: int) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        text = str(payload)
    return _truncate_text(text, max_chars)


def _count_text_tokens(text: str) -> int:
    if not isinstance(text, str):
        text = str(text or "")
    if not text:
        return 0
    return len(_TOKEN_ENCODING.encode(text, disallowed_special=()))


def _count_json_tokens(payload: Any) -> int:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        text = str(payload)
    return _count_text_tokens(text)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if not isinstance(text, str):
        text = str(text or "")
    if not text:
        return ""
    encoded = _TOKEN_ENCODING.encode(text, disallowed_special=())
    if len(encoded) <= max_tokens:
        return text
    return _TOKEN_ENCODING.decode(encoded[:max_tokens])


def _normalize_json_preview_by_tokens(payload: Any, max_tokens: int) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        text = str(payload)
    return _truncate_text_to_tokens(text, max_tokens)


def _normalize_claim_id_list(raw_value: Any, *, limit: int = 24) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    max_items = max(limit, 0)
    for raw_item in raw_value:
        if len(normalized) >= max_items:
            break
        try:
            claim_id = str(uuid.UUID(str(raw_item).strip()))
        except (TypeError, ValueError, AttributeError):
            continue
        if claim_id in seen:
            continue
        seen.add(claim_id)
        normalized.append(claim_id)
    return normalized


def _sanitize_object_data_for_context(raw_data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_data, dict):
        return {}

    result: dict[str, Any] = {}

    short_desc_value: Any = None
    if "short_desc" in raw_data:
        short_desc_value = raw_data.get("short_desc")
    else:
        for key in CONTEXT_OBJECT_DESCRIPTION_KEYS:
            value = raw_data.get(key)
            if isinstance(value, str) and value.strip():
                short_desc_value = value.strip()
                break

    if isinstance(short_desc_value, str):
        trimmed = short_desc_value.strip()
        if trimmed:
            result["short_desc"] = _truncate_text(trimmed, 220)

    status_value = raw_data.get("status")
    if isinstance(status_value, str) and status_value.strip():
        result["status"] = _truncate_text(status_value.strip(), 40)

    hp_value = raw_data.get("hp")
    if isinstance(hp_value, (int, float)):
        result["hp"] = hp_value
    elif isinstance(hp_value, str):
        hp_trimmed = hp_value.strip()
        if hp_trimmed:
            result["hp"] = _truncate_text(hp_trimmed, 24)

    for key in CONTEXT_OBJECT_ALLOWED_KEYS:
        if key in CONTEXT_OBJECT_TECHNICAL_KEYS:
            continue
        if key in result:
            continue
        value = raw_data.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[key] = value

    return result


def __getattr__(name: str) -> Any:
    if name in {
        "TurnPlanResult",
        "TurnApplyExternalRequest",
        "TurnApplyExternalArtifacts",
        "TurnApplyExternalPreparationRequired",
    }:
        module = import_module(".application.turn_contracts", __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "PatchValidationResult",
    "TurnPlanResult",
    "PreparedObjectCreateOp",
    "_is_true",
    "_safe_int",
    "_coerce_time_scale_minutes",
    "_parse_datetime_utc",
    "_get_pending_turn_started_at",
    "_is_pending_turn_stale",
    "_clear_pending_turn_locked",
    "_normalize_time_payload",
    "_coerce_state_payload",
    "_extract_in_game_time",
    "_build_internal_turn_ai_json",
    "_create_internal_turn_row",
    "_session_turn_lock_key",
    "_session_turn_runtime_lock_key",
    "_session_turn_runtime_lock_supported",
    "_acquire_session_turn_lock",
    "_recover_abandoned_pending_turn_locked",
    "_session_turn_runtime_lock",
    "_is_session_turn_runtime_lock_held",
    "_require_session",
    "_get_object",
    "_require_object",
    "_get_session_player_object_id",
    "_resolve_object_ref",
    "_get_active_located_in_links",
    "_get_active_link",
    "_get_latest_located_in_link",
    "_close_player_active_located_in_links",
    "_get_player_current_zone_id",
    "_infer_actor_zone_id",
    "_to_jsonable",
    "_build_session_snapshot_dump",
    "_truncate_text",
    "_normalize_json_preview",
    "_count_text_tokens",
    "_count_json_tokens",
    "_truncate_text_to_tokens",
    "_normalize_json_preview_by_tokens",
    "_normalize_claim_id_list",
    "_sanitize_object_data_for_context",
]
