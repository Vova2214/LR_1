"""Embedding and memory persistence runtime helpers.

Selected lookup helpers remain deprecated compatibility shims over the
memory-write repository while the wider module still owns live embedding and
memory persistence behavior.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from . import models, schemas
from .architecture_contracts import COMPATIBILITY_MODULE_CONTRACTS
from .constants import SESSION_PLAYER_REF
from .db import (
    EMBED_SNIPPET_MAX_CHARS,
    OPENROUTER_CHAT_MODEL,
    USE_EMBEDDINGS,
)
from .embeddings import openrouter
from .llm_telemetry import telemetry_context
from .crud_profiles import (
    _build_faction_profile_text,
    _build_item_profile_text,
    _build_npc_profile_text,
    _build_player_profile_text,
    _build_quest_profile_text,
    _build_zone_profile_text,
)
from .crud_shared import (
    TurnApplyExternalPreparationRequired,
    TurnApplyExternalRequest,
    _coalesce_memory_candidates_by_identity,
    _memory_candidate_commit_scene_signature,
    _memory_candidates_to_durable_facts,
    _memory_fact_identity_key,
    _safe_int,
    _supplement_memory_candidates_with_durable_facts,
    _turn_apply_embedding_key,
    _truncate_text,
    current_turn_apply_external_artifacts,
)
from .domain.memory_policy import (
    bundle_family_for_kind,
    bundle_key,
    bundle_fact_rank_tuple,
    coalesce_fact_state,
    continuity_contract_strength_score,
    derive_dormancy_transition,
    expectation_debt_score_for_payload,
    derive_compression_mode,
    event_identity_key as policy_event_identity_key,
    expectation_salience_score,
    fact_identity_key as policy_fact_identity_key,
    fact_slot_signature,
    merge_causal_links,
    merge_callback_strength as policy_merge_callback_strength,
    normalize_continuity_contract_strength,
    normalize_event_consequence_role,
    normalize_event_outcome,
    normalize_memory_certainty,
    normalize_player_salience,
    normalize_expectation_salience,
    normalize_state,
    persistence_memory_candidate_importance,
    player_salience_score,
    resolve_fact_transition,
    same_turn_obligation_causal_links,
    SUPERSEDED_FACT_KINDS,
    state_allows_callback,
    transition_causal_links,
)
from .persistence.memory_write_repository import memory_write_repository

MEMORY_EVENT_OBJECT_TYPE = "__memory_event"
MEMORY_FACT_OBJECT_TYPE = "__memory_fact"
MEMORY_BUNDLE_OBJECT_TYPE = "__memory_bundle"
MEMORY_EVENT_NAMESPACE = "memory_event_search_summary"
MEMORY_FACT_NAMESPACE = "memory_fact_search_summary"
MEMORY_BUNDLE_NAMESPACE = "memory_bundle_search_summary"
MAX_MEMORY_CANDIDATES_PER_TURN = 6
NPC_PROFILE_EMBED_INSTRUCTION = "Represent this NPC identity for deduplication"
PLAYER_PROFILE_EMBED_INSTRUCTION = "Represent this player profile for retrieval and continuity"
ZONE_PROFILE_EMBED_INSTRUCTION = "Represent this game location for deduplication"
ITEM_PROFILE_EMBED_INSTRUCTION = "Represent this game item for deduplication"
FACTION_PROFILE_EMBED_INSTRUCTION = "Represent this faction for deduplication"
QUEST_PROFILE_EMBED_INSTRUCTION = "Represent this quest for deduplication"
CLAIM_TEXT_EMBED_INSTRUCTION = "Represent this in-game claim or rumor"
MEMORY_EVENT_EMBED_INSTRUCTION = "Represent this episodic in-game memory search summary for semantic recall"
MEMORY_FACT_EMBED_INSTRUCTION = "Represent this durable in-game memory fact search summary for semantic recall"
MEMORY_BUNDLE_EMBED_INSTRUCTION = "Represent this coherent in-game memory bundle search summary for semantic recall"
LINK_CONTEXT_EMBED_INSTRUCTION = "Represent this relationship and power dynamic between entities"
LINK_CONTEXT_NAMESPACE = "link_context"
LINK_CONTENT_TYPES = {"controls", "allied_with", "hostile_to", "owns", "guards"}
LINK_TEXT_DATA_KEYS = ("description", "reason", "note", "details", "summary")
_MEMORY_CALLBACK_STRENGTH_RANK = {"none": 0, "soft": 1, "strong": 2}
_MEMORY_SCOPE_RANK = {"npc_private": 0, "public": 1, "global": 2}

logger = logging.getLogger(__name__)
COMPATIBILITY_MODULE_CONTRACT = COMPATIBILITY_MODULE_CONTRACTS[__name__]
LEGACY_MEMORY_FACT_MATCH_DISTANCE_THRESHOLD = 0.14


def _use_embeddings_enabled() -> bool:
    from . import crud as crud_runtime

    return bool(crud_runtime.USE_EMBEDDINGS)


def _maybe_embed_texts(
    texts: list[str],
    *,
    instruction: str | None = None,
) -> list[list[float]]:
    if not _use_embeddings_enabled():
        return []
    artifacts = current_turn_apply_external_artifacts()
    if artifacts is not None:
        resolved_vectors: list[list[float]] = []
        missing_texts: list[str] = []
        normalized_instruction = str(instruction or "").strip() or None
        for text in texts:
            key = _turn_apply_embedding_key(instruction=normalized_instruction, text=text)
            vector = artifacts.embedding_vectors.get(key)
            if vector is None:
                missing_texts.append(text)
                continue
            resolved_vectors.append(list(vector))
        if missing_texts:
            raise TurnApplyExternalPreparationRequired(
                TurnApplyExternalRequest(
                    kind="embedding",
                    instruction=normalized_instruction,
                    texts=tuple(dict.fromkeys(missing_texts)),
                )
            )
        return resolved_vectors
    request_type = "embedding"
    instruction_text = str(instruction or "").strip()
    if instruction_text == NPC_PROFILE_EMBED_INSTRUCTION:
        request_type = "embedding_npc_profile"
    elif instruction_text == PLAYER_PROFILE_EMBED_INSTRUCTION:
        request_type = "embedding_player_profile"
    elif instruction_text == ZONE_PROFILE_EMBED_INSTRUCTION:
        request_type = "embedding_zone_profile"
    elif instruction_text == ITEM_PROFILE_EMBED_INSTRUCTION:
        request_type = "embedding_item_profile"
    elif instruction_text == FACTION_PROFILE_EMBED_INSTRUCTION:
        request_type = "embedding_faction_profile"
    elif instruction_text == QUEST_PROFILE_EMBED_INSTRUCTION:
        request_type = "embedding_quest_profile"
    elif instruction_text == CLAIM_TEXT_EMBED_INSTRUCTION:
        request_type = "embedding_claim_text"
    elif instruction_text == MEMORY_EVENT_EMBED_INSTRUCTION:
        request_type = "embedding_memory_event"
    elif instruction_text == MEMORY_FACT_EMBED_INSTRUCTION:
        request_type = "embedding_memory_fact"
    elif instruction_text == LINK_CONTEXT_EMBED_INSTRUCTION:
        request_type = "embedding_link_context"
    with telemetry_context(request_type=request_type):
        if instruction is None:
            return openrouter.embed_texts(texts)
        return openrouter.embed_texts(texts, instruction=instruction)


def _upsert_object_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    namespace: str,
    text_hash: str,
    embedding: list[float],
) -> None:
    existing = db.get(models.ObjectEmbeddingModel, (session_id, object_id, namespace))
    if existing is None:
        db.add(
            models.ObjectEmbeddingModel(
                session_id=session_id,
                object_id=object_id,
                namespace=namespace,
                text_hash=text_hash,
                embedding=embedding,
            )
        )
        return

    existing.text_hash = text_hash
    existing.embedding = embedding


def _upsert_npc_profile_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    npc_name: str,
    npc_data: dict[str, Any],
    *,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    profile_text = precomputed_text or _build_npc_profile_text(npc_name, npc_data)
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [profile_text],
            instruction=NPC_PROFILE_EMBED_INSTRUCTION,
        )[0]
    profile_hash = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace="npc_profile",
        text_hash=profile_hash,
        embedding=embedding,
    )


def _upsert_player_profile_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    player_name: str,
    player_data: dict[str, Any],
    *,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    profile_text = precomputed_text or _build_player_profile_text(player_name, player_data)
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [profile_text],
            instruction=PLAYER_PROFILE_EMBED_INSTRUCTION,
        )[0]
    profile_hash = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace="player_profile",
        text_hash=profile_hash,
        embedding=embedding,
    )


def _upsert_zone_profile_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    zone_name: str,
    zone_data: dict[str, Any],
    *,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    profile_text = precomputed_text or _build_zone_profile_text(zone_name, zone_data)
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [profile_text],
            instruction=ZONE_PROFILE_EMBED_INSTRUCTION,
        )[0]
    profile_hash = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace="zone_profile",
        text_hash=profile_hash,
        embedding=embedding,
    )


def _upsert_item_profile_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    item_name: str,
    item_data: dict[str, Any],
    *,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    profile_text = precomputed_text or _build_item_profile_text(item_name, item_data)
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [profile_text],
            instruction=ITEM_PROFILE_EMBED_INSTRUCTION,
        )[0]
    profile_hash = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace="item_profile",
        text_hash=profile_hash,
        embedding=embedding,
    )


def _upsert_faction_profile_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    faction_name: str,
    faction_data: dict[str, Any],
    *,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    profile_text = precomputed_text or _build_faction_profile_text(faction_name, faction_data)
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [profile_text],
            instruction=FACTION_PROFILE_EMBED_INSTRUCTION,
        )[0]
    profile_hash = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace="faction_profile",
        text_hash=profile_hash,
        embedding=embedding,
    )


def _upsert_quest_profile_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    quest_name: str,
    quest_data: dict[str, Any],
    *,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    profile_text = precomputed_text or _build_quest_profile_text(quest_name, quest_data)
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [profile_text],
            instruction=QUEST_PROFILE_EMBED_INSTRUCTION,
        )[0]
    profile_hash = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace="quest_profile",
        text_hash=profile_hash,
        embedding=embedding,
    )


def _extract_claim_text(claim_data: dict[str, Any]) -> str:
    raw_text = claim_data.get("text")
    if raw_text is None:
        return ""
    return _truncate_text(str(raw_text).strip(), 800)


def _upsert_claim_text_embedding(
    db: Session,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    claim_data: dict[str, Any],
    *,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    claim_text = precomputed_text if precomputed_text is not None else _extract_claim_text(claim_data)
    if not claim_text:
        return

    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [claim_text],
            instruction=CLAIM_TEXT_EMBED_INSTRUCTION,
        )[0]
    claim_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace="claim_text",
        text_hash=claim_hash,
        embedding=embedding,
    )


def _normalize_memory_text(text: str) -> str:
    return _truncate_text(" ".join(str(text or "").split()).strip(), 200)


def _normalize_memory_summary(text: str) -> str:
    return _normalize_memory_text(text)


def _coerce_importance(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    parsed: float | None = None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
    if parsed is None:
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return round(parsed, 6)


def _coerce_seed_priority(value: Any) -> str:
    priority = str(value or "med").strip().lower()
    if priority not in {"low", "med", "high"}:
        return "med"
    return priority


def _merge_seed_priority(existing_value: Any, incoming_value: Any) -> str:
    if existing_value is None or not str(existing_value).strip():
        return _coerce_seed_priority(incoming_value)
    order = {"low": 0, "med": 1, "high": 2}
    existing = _coerce_seed_priority(existing_value)
    incoming = _coerce_seed_priority(incoming_value)
    return existing if order[existing] >= order[incoming] else incoming


def _coerce_memory_scope(value: Any) -> str:
    scope = str(value or "global").strip().lower()
    if scope not in _MEMORY_SCOPE_RANK:
        return "global"
    return scope


def _merge_memory_scope(existing_value: Any, incoming_value: Any) -> str:
    if existing_value is None or not str(existing_value).strip():
        return _coerce_memory_scope(incoming_value)
    existing = _coerce_memory_scope(existing_value)
    incoming = _coerce_memory_scope(incoming_value)
    return existing if _MEMORY_SCOPE_RANK[existing] >= _MEMORY_SCOPE_RANK[incoming] else incoming


def _coerce_callback_strength(value: Any) -> str:
    strength = str(value or "none").strip().lower()
    if strength not in _MEMORY_CALLBACK_STRENGTH_RANK:
        return "none"
    return strength


def _merge_callback_strength(existing_value: Any, incoming_value: Any) -> str:
    return policy_merge_callback_strength(existing_value, incoming_value)


def _normalize_anchor_object_ids(raw_anchor_object_ids: Any, *, max_items: int = 24) -> list[str]:
    if not isinstance(raw_anchor_object_ids, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_anchor_object_ids:
        value = str(raw_item or "").strip()
        if not value:
            continue
        try:
            parsed = uuid.UUID(value)
        except (TypeError, ValueError, AttributeError):
            continue
        object_id = str(parsed)
        if object_id in seen:
            continue
        seen.add(object_id)
        normalized.append(object_id)
        if len(normalized) >= max(max_items, 1):
            break
    return normalized


def _merge_anchor_object_ids(existing_ids: Any, incoming_ids: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for object_id in _normalize_anchor_object_ids(existing_ids) + _normalize_anchor_object_ids(incoming_ids):
        if object_id in seen:
            continue
        seen.add(object_id)
        merged.append(object_id)
    return merged


def _resolve_memory_ref(
    raw_value: Any,
    *,
    ref_map: dict[str, str] | None,
    player_object_id: uuid.UUID | None,
) -> uuid.UUID | None:
    if isinstance(raw_value, uuid.UUID):
        return raw_value
    text = str(raw_value or "").strip()
    if not text:
        return None
    if text == SESSION_PLAYER_REF:
        return player_object_id
    try:
        return uuid.UUID(text)
    except (TypeError, ValueError, AttributeError):
        pass
    if text.startswith("tmp:"):
        mapped = str((ref_map or {}).get(text) or "").strip()
        if not mapped:
            return None
        try:
            return uuid.UUID(mapped)
        except (TypeError, ValueError, AttributeError):
            return None
    return None


def _resolve_memory_candidate_identity_refs(
    candidate: schemas.MemoryCandidate,
    *,
    ref_map: dict[str, str] | None,
    player_object_id: uuid.UUID | None,
) -> tuple[uuid.UUID | None, uuid.UUID | None, uuid.UUID | None, uuid.UUID | None, uuid.UUID | None, list[uuid.UUID]]:
    canonical_fact = candidate.canonical_fact if candidate.layer == "fact" or candidate.requires_commit else None
    actor_ref = canonical_fact.actor_ref if canonical_fact is not None else candidate.actor_ref
    counterparty_ref = canonical_fact.counterparty_ref if canonical_fact is not None else candidate.counterparty_ref
    object_ref = canonical_fact.object_ref if canonical_fact is not None else candidate.object_ref
    location_ref = canonical_fact.location_ref if canonical_fact is not None else candidate.location_ref
    quest_ref = canonical_fact.quest_ref if canonical_fact is not None else candidate.quest_ref
    context_refs = list(canonical_fact.context_refs) if canonical_fact is not None else list(candidate.context_refs)

    actor_object_id = _resolve_memory_ref(
        actor_ref,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )
    object_id = _resolve_memory_ref(
        object_ref,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )
    counterparty_object_id = _resolve_memory_ref(
        counterparty_ref,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )
    location_object_id = _resolve_memory_ref(
        location_ref,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )
    quest_object_id = _resolve_memory_ref(
        quest_ref,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )
    context_object_ids: list[uuid.UUID] = []
    for raw_value in context_refs:
        resolved = _resolve_memory_ref(
            raw_value,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        if resolved is None or resolved in context_object_ids:
            continue
        context_object_ids.append(resolved)
    return actor_object_id, counterparty_object_id, object_id, location_object_id, quest_object_id, context_object_ids


def _memory_fact_key_from_row_data(data: dict[str, Any]) -> str | None:
    kind = str(data.get("kind") or "").strip().lower()
    identity_text = str(data.get("identity_text") or "").strip()
    if not kind or not identity_text:
        return None
    actor_object_id = _resolve_memory_ref(data.get("actor_object_id"), ref_map=None, player_object_id=None)
    counterparty_object_id = _resolve_memory_ref(
        data.get("counterparty_object_id"),
        ref_map=None,
        player_object_id=None,
    )
    object_id = _resolve_memory_ref(data.get("object_id"), ref_map=None, player_object_id=None)
    location_object_id = _resolve_memory_ref(data.get("location_object_id"), ref_map=None, player_object_id=None)
    quest_object_id = _resolve_memory_ref(data.get("quest_object_id"), ref_map=None, player_object_id=None)
    relationship_type = str(data.get("relationship_type") or "").strip().lower() or None
    return _memory_fact_identity_key(
        kind=kind,
        actor_object_id=actor_object_id,
        counterparty_object_id=counterparty_object_id,
        object_id=object_id,
        location_object_id=location_object_id,
        quest_object_id=quest_object_id,
        identity_text=identity_text,
        relationship_type=relationship_type,
    )


def _memory_fact_slot_signature(
    *,
    kind: str,
    actor_object_id: uuid.UUID | None,
    counterparty_object_id: uuid.UUID | None,
    object_id: uuid.UUID | None,
    location_object_id: uuid.UUID | None,
    quest_object_id: uuid.UUID | None,
    relationship_type: str | None,
) -> tuple[Any, ...]:
    return fact_slot_signature(
        kind=kind,
        actor_ref=actor_object_id,
        counterparty_ref=counterparty_object_id,
        object_ref=object_id,
        location_ref=location_object_id,
        quest_ref=quest_object_id,
        relationship_type=relationship_type,
    )


def _memory_fact_slot_signature_from_row_data(data: dict[str, Any]) -> tuple[Any, ...]:
    actor_object_id = _resolve_memory_ref(data.get("actor_object_id"), ref_map=None, player_object_id=None)
    counterparty_object_id = _resolve_memory_ref(data.get("counterparty_object_id"), ref_map=None, player_object_id=None)
    object_id = _resolve_memory_ref(data.get("object_id"), ref_map=None, player_object_id=None)
    location_object_id = _resolve_memory_ref(data.get("location_object_id"), ref_map=None, player_object_id=None)
    quest_object_id = _resolve_memory_ref(data.get("quest_object_id"), ref_map=None, player_object_id=None)
    return _memory_fact_slot_signature(
        kind=str(data.get("kind") or "").strip().lower(),
        actor_object_id=actor_object_id,
        counterparty_object_id=counterparty_object_id,
        object_id=object_id,
        location_object_id=location_object_id,
        quest_object_id=quest_object_id,
        relationship_type=str(data.get("relationship_type") or "").strip().lower() or None,
    )


def _iter_memory_fact_rows_for_kind(
    db: Session,
    *,
    session_id: uuid.UUID,
    fact_kind: str,
) -> list[models.ObjectModel]:
    return memory_write_repository.list_memory_fact_rows_for_kind(
        db,
        session_id=session_id,
        object_type=MEMORY_FACT_OBJECT_TYPE,
        fact_kind=fact_kind,
    )


def _iter_memory_fact_rows_for_identity(
    rows: list[models.ObjectModel],
    *,
    fact_key: str,
) -> list[models.ObjectModel]:
    matched: list[models.ObjectModel] = []
    for row in rows:
        if _memory_fact_key_from_row_data(dict(row.data or {})) != fact_key:
            continue
        matched.append(row)
    return matched


def _active_memory_fact_rows(
    rows: list[models.ObjectModel],
) -> list[models.ObjectModel]:
    return [
        row
        for row in rows
        if normalize_state(dict(row.data or {}).get("state")) == "active"
        and str(dict(row.data or {}).get("status") or "active").strip().lower() == "active"
    ]


def _iter_active_memory_fact_rows_for_slot(
    db: Session,
    *,
    session_id: uuid.UUID,
    fact_kind: str,
    slot_signature: tuple[Any, ...],
) -> list[models.ObjectModel]:
    candidate_rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == MEMORY_FACT_OBJECT_TYPE,
            models.ObjectModel.data["kind"].astext == fact_kind,
        )
    ).scalars().all()
    matched: list[models.ObjectModel] = []
    for row in candidate_rows:
        data = dict(row.data or {})
        if normalize_state(data.get("state")) != "active":
            continue
        if str(data.get("status") or "active").strip().lower() != "active":
            continue
        if _memory_fact_slot_signature_from_row_data(data) != slot_signature:
            continue
        matched.append(row)
    return matched


def _supersede_competing_fact_rows(
    db: Session,
    *,
    session_id: uuid.UUID,
    fact_kind: str,
    slot_signature: tuple[Any, ...],
    incoming_fact_key: str,
) -> None:
    for row in _iter_active_memory_fact_rows_for_slot(
        db,
        session_id=session_id,
        fact_kind=fact_kind,
        slot_signature=slot_signature,
    ):
        data = dict(row.data or {})
        existing_fact_key = str(data.get("fact_key") or "")
        if not existing_fact_key or existing_fact_key == incoming_fact_key:
            continue
        data["state"] = "superseded"
        data["callback_candidate"] = False
        data["callback_strength"] = "none"
        row.data = data


def _memory_event_key(
    *,
    kind: str,
    search_recall_summary: str,
    turn_index: int,
    event_role: str,
    event_outcome: str = "asserted",
    actor_object_id: uuid.UUID | None,
    counterparty_object_id: uuid.UUID | None,
    object_id: uuid.UUID | None,
    location_object_id: uuid.UUID | None,
    quest_object_id: uuid.UUID | None,
    relationship_type: str | None = None,
    commit_fact_key: str | None = None,
    scene_ref_ids: list[str] | None = None,
) -> str:
    return policy_event_identity_key(
        kind=kind,
        search_recall_summary=search_recall_summary,
        turn_index=turn_index,
        event_role=event_role,
        actor_ref=actor_object_id,
        counterparty_ref=counterparty_object_id,
        object_ref=object_id,
        location_ref=location_object_id,
        quest_ref=quest_object_id,
        relationship_type=relationship_type,
        requires_commit=commit_fact_key is not None,
        commit_fact_key=commit_fact_key,
        scene_refs=_normalize_scene_ref_ids(scene_ref_ids),
        event_outcome=event_outcome,
    )


def _normalize_scene_ref_ids(values: Any) -> list[str]:
    return sorted(
        {
            str(value).strip()
            for value in list(values or [])
            if str(value).strip()
        }
    )


def _find_existing_memory_event_row(
    db: Session,
    *,
    session_id: uuid.UUID,
    event_key: str,
    commit_fact_key: str | None,
    turn_index: int,
    scene_ref_ids: list[str] | None = None,
) -> models.ObjectModel | None:
    return memory_write_repository.find_existing_memory_event_row(
        db,
        session_id=session_id,
        event_type=MEMORY_EVENT_OBJECT_TYPE,
        event_key=event_key,
        commit_fact_key=commit_fact_key,
        turn_index=turn_index,
        scene_ref_ids=scene_ref_ids,
    )


def _apply_same_turn_causal_links(
    db: Session,
    *,
    stored_fact_payloads: list[dict[str, Any]],
    turn_index: int,
) -> None:
    if len(stored_fact_payloads) < 2:
        return
    payloads_by_key = {
        str(payload.get("fact_key") or ""): payload
        for payload in stored_fact_payloads
        if str(payload.get("fact_key") or "")
    }
    if not payloads_by_key:
        return
    rows = memory_write_repository.list_memory_fact_rows_by_keys(
        db,
        object_type=MEMORY_FACT_OBJECT_TYPE,
        fact_keys=list(payloads_by_key),
    )
    row_by_fact_key = {
        str(dict(row.data or {}).get("fact_key") or ""): row
        for row in rows
        if str(dict(row.data or {}).get("fact_key") or "")
    }
    for source_fact_key, source_payload in payloads_by_key.items():
        row = row_by_fact_key.get(source_fact_key)
        if row is None:
            continue
        data = dict(row.data or {})
        links = same_turn_obligation_causal_links(
            source_fact=source_payload,
            candidate_facts=[payload for key, payload in payloads_by_key.items() if key != source_fact_key],
            turn_index=turn_index,
        )
        if not links:
            continue
        data["causal_links"] = merge_causal_links(data.get("causal_links"), links)
        row.data = data


def _memory_candidate_importance(
    *,
    priority: str,
    durability: float,
    emotional_weight: float,
    obligation_weight: float,
    sentimental_weight: float,
    routine_weight: float,
    player_salience: float = 0.0,
) -> float:
    return persistence_memory_candidate_importance(
        priority=priority,
        durability=durability,
        emotional_weight=emotional_weight,
        obligation_weight=obligation_weight,
        sentimental_weight=sentimental_weight,
        routine_weight=routine_weight,
        player_salience=player_salience,
    )


def _upsert_memory_summary_embedding(
    db: Session,
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    namespace: str,
    summary: str,
    instruction: str,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return
    normalized_summary = _normalize_memory_summary(summary)
    if not normalized_summary:
        return
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts([normalized_summary], instruction=instruction)[0]
    text_hash = hashlib.sha256(normalized_summary.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=object_id,
        namespace=namespace,
        text_hash=text_hash,
        embedding=embedding,
    )


def _extract_link_context_text(link_data: dict[str, Any]) -> str:
    if not isinstance(link_data, dict):
        return ""
    for key in LINK_TEXT_DATA_KEYS:
        raw_value = link_data.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            text = raw_value.strip()
        else:
            text = str(raw_value).strip()
        if text:
            return text
    return ""


def _build_link_context_snippet(
    *,
    from_name: str,
    link_type: str,
    to_name: str,
    text: str,
) -> str:
    from_value = from_name.strip() or "Unknown"
    relation = link_type.strip() or "related_to"
    to_value = to_name.strip() or "Unknown"
    content = text.strip()
    if not content:
        return ""
    return f"{from_value} {relation} {to_value}: {content}"


def _list_active_link_context_snippets(
    db: Session,
    session_id: uuid.UUID,
    *,
    from_object_id: uuid.UUID,
    from_name: str,
) -> list[str]:
    rows = db.execute(
        select(models.LinkModel, models.ObjectModel)
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.LinkModel.session_id,
                models.ObjectModel.object_id == models.LinkModel.to_object_id,
            ),
        )
        .where(
            models.LinkModel.session_id == session_id,
            # link_context is an outgoing-profile index for a single source object.
            models.LinkModel.from_object_id == from_object_id,
            models.LinkModel.valid_to_turn.is_(None),
            models.LinkModel.type.in_(tuple(LINK_CONTENT_TYPES)),
        )
        .order_by(models.LinkModel.created_at.asc())
    ).all()

    snippets: list[str] = []
    for link_row, to_object in rows:
        content_text = _extract_link_context_text(dict(link_row.data or {}))
        if not content_text:
            continue
        snippet = _build_link_context_snippet(
            from_name=from_name,
            link_type=link_row.type,
            to_name=to_object.name,
            text=content_text,
        )
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    return snippets


def _upsert_link_context_embedding(
    db: Session,
    session_id: uuid.UUID,
    *,
    from_object_id: uuid.UUID,
    from_name: str,
    to_name: str,
    link_type: str,
    link_data: dict[str, Any],
) -> None:
    if not _use_embeddings_enabled():
        return
    if link_type not in LINK_CONTENT_TYPES:
        return

    link_text = _extract_link_context_text(link_data)
    if not link_text:
        return

    snippets = _list_active_link_context_snippets(
        db=db,
        session_id=session_id,
        from_object_id=from_object_id,
        from_name=from_name,
    )
    new_snippet = _build_link_context_snippet(
        from_name=from_name,
        link_type=link_type,
        to_name=to_name,
        text=link_text,
    )
    if new_snippet and new_snippet not in snippets:
        snippets.append(new_snippet)

    embedding_text = _truncate_text("\n".join(snippets), EMBED_SNIPPET_MAX_CHARS)
    if not embedding_text:
        return
    embedding = _maybe_embed_texts(
        [embedding_text],
        instruction=LINK_CONTEXT_EMBED_INSTRUCTION,
    )[0]
    text_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()

    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=from_object_id,
        namespace=LINK_CONTEXT_NAMESPACE,
        text_hash=text_hash,
        embedding=embedding,
    )


def _refresh_link_context_embedding(
    db: Session,
    session_id: uuid.UUID,
    *,
    from_object_id: uuid.UUID,
    precomputed_text: str | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    if not _use_embeddings_enabled():
        return

    from_object = db.get(models.ObjectModel, (session_id, from_object_id))
    existing = db.get(models.ObjectEmbeddingModel, (session_id, from_object_id, LINK_CONTEXT_NAMESPACE))
    if from_object is None:
        if existing is not None:
            db.delete(existing)
        return

    if precomputed_text is None:
        snippets = _list_active_link_context_snippets(
            db=db,
            session_id=session_id,
            from_object_id=from_object_id,
            from_name=from_object.name,
        )
        if not snippets:
            if existing is not None:
                db.delete(existing)
            return
        embedding_text = _truncate_text("\n".join(snippets), EMBED_SNIPPET_MAX_CHARS)
    else:
        embedding_text = _truncate_text(precomputed_text, EMBED_SNIPPET_MAX_CHARS)
    if not embedding_text:
        if existing is not None:
            db.delete(existing)
        return
    embedding = precomputed_embedding
    if embedding is None:
        embedding = _maybe_embed_texts(
            [embedding_text],
            instruction=LINK_CONTEXT_EMBED_INSTRUCTION,
        )[0]
    text_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()
    _upsert_object_embedding(
        db=db,
        session_id=session_id,
        object_id=from_object_id,
        namespace=LINK_CONTEXT_NAMESPACE,
        text_hash=text_hash,
        embedding=embedding,
    )


def _store_memory_candidates(
    db: Session,
    session_id: uuid.UUID,
    *,
    turn_index: int,
    zone_scope_id: uuid.UUID | None,
    in_game_day: int | None,
    in_game_minute: int | None,
    memory_candidates: list[schemas.MemoryCandidate],
    source_ops_count: int | None = None,
    anchor_object_ids: list[str] | None = None,
    ref_map: dict[str, str] | None = None,
    player_object_id: uuid.UUID | None = None,
) -> dict[str, int]:
    resolved_source_ops_count = max(_safe_int(source_ops_count) or 0, 0)
    turn_anchor_ids = _normalize_anchor_object_ids(anchor_object_ids)
    if zone_scope_id is not None:
        turn_anchor_ids = _merge_anchor_object_ids(turn_anchor_ids, [str(zone_scope_id)])
    memory_candidates = _coalesce_memory_candidates_by_identity(
        memory_candidates,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )
    effective_durable_facts = _memory_candidates_to_durable_facts(memory_candidates)
    memory_candidates = _supplement_memory_candidates_with_durable_facts(
        memory_candidates,
        effective_durable_facts,
        fallback_anchors=turn_anchor_ids,
    )
    memory_candidates = _coalesce_memory_candidates_by_identity(
        memory_candidates,
        ref_map=ref_map,
        player_object_id=player_object_id,
    )

    stored_event_count = 0
    stored_fact_count = 0
    stored_fact_payloads: list[dict[str, Any]] = []
    for candidate in memory_candidates:
        search_recall_summary = _normalize_memory_summary(candidate.search_recall_summary)
        narrative_recall_summary = _normalize_memory_summary(candidate.narrative_recall_summary)
        if not search_recall_summary or not narrative_recall_summary:
            continue

        canonical_fact = candidate.canonical_fact
        (
            actor_object_id,
            counterparty_object_id,
            object_id,
            location_object_id,
            quest_object_id,
            context_object_ids,
        ) = _resolve_memory_candidate_identity_refs(
            candidate,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        candidate_anchor_ids = [
            str(resolved)
            for resolved in (
                _resolve_memory_ref(raw_value, ref_map=ref_map, player_object_id=player_object_id)
                for raw_value in [*candidate.anchors, *candidate.scene_refs]
            )
            if resolved is not None
        ]
        combined_anchor_ids = _merge_anchor_object_ids(
            turn_anchor_ids,
            candidate_anchor_ids
            + ([str(actor_object_id)] if actor_object_id is not None else [])
            + ([str(counterparty_object_id)] if counterparty_object_id is not None else [])
            + ([str(object_id)] if object_id is not None else [])
            + ([str(location_object_id)] if location_object_id is not None else [])
            + ([str(quest_object_id)] if quest_object_id is not None else [])
            + [str(context_object_id) for context_object_id in context_object_ids],
        )
        if not combined_anchor_ids and actor_object_id is None and object_id is None and quest_object_id is None:
            continue

        priority = _coerce_seed_priority(candidate.priority)
        knowledge_scope = _coerce_memory_scope(candidate.knowledge_scope)
        callback_strength = _coerce_callback_strength(candidate.callback_strength)
        player_salience = player_salience_score(candidate.player_salience)
        expectation_salience = expectation_salience_score(candidate.expectation_salience)
        continuity_contract_strength = normalize_continuity_contract_strength(candidate.continuity_contract_strength)
        continuity_contract_strength_value = continuity_contract_strength_score(continuity_contract_strength)
        certainty = normalize_memory_certainty(getattr(canonical_fact, "certainty", "confirmed") if canonical_fact is not None else "confirmed")
        durability = _coerce_importance(candidate.durability) or 0.0
        emotional_weight = _coerce_importance(candidate.emotional_weight) or 0.0
        obligation_weight = _coerce_importance(candidate.obligation_weight) or 0.0
        sentimental_weight = _coerce_importance(candidate.sentimental_weight) or 0.0
        routine_weight = _coerce_importance(candidate.routine_weight) or 0.0
        importance = _memory_candidate_importance(
            priority=priority,
            durability=durability,
            emotional_weight=emotional_weight,
            obligation_weight=obligation_weight,
            sentimental_weight=sentimental_weight,
            routine_weight=routine_weight,
            player_salience=player_salience,
        )
        event_role = candidate.event_role if candidate.layer == "event" else "supporting"
        event_outcome = normalize_event_outcome(candidate.event_outcome if candidate.layer == "event" else "asserted")
        event_consequence_role = normalize_event_consequence_role(
            candidate.event_consequence_role if candidate.layer == "event" else "setup"
        )

        if candidate.layer == "event":
            commit_fact_key: str | None = None
            commit_identity_text: str | None = None
            commit_relationship_type: str | None = None
            scene_ref_ids: list[str] = []
            if candidate.requires_commit and canonical_fact is not None:
                commit_identity_text = str(
                    canonical_fact.identity_text
                    or canonical_fact.narrative_recall_summary
                    or canonical_fact.search_recall_summary
                ).strip()
                commit_relationship_type = str(canonical_fact.relationship_type or "").strip().lower() or None
                commit_fact_key = _memory_fact_identity_key(
                    kind=canonical_fact.kind,
                    actor_object_id=actor_object_id,
                    counterparty_object_id=counterparty_object_id,
                    object_id=object_id,
                    location_object_id=location_object_id,
                    quest_object_id=quest_object_id,
                    identity_text=commit_identity_text,
                    relationship_type=commit_relationship_type,
                )
                scene_ref_ids = _normalize_scene_ref_ids(
                    _memory_candidate_commit_scene_signature(candidate)
                )
            event_key = _memory_event_key(
                kind=candidate.kind,
                search_recall_summary=search_recall_summary,
                turn_index=turn_index,
                event_role=event_role,
                event_outcome=event_outcome,
                actor_object_id=actor_object_id,
                counterparty_object_id=counterparty_object_id,
                object_id=object_id,
                location_object_id=location_object_id,
                quest_object_id=quest_object_id,
                commit_fact_key=commit_fact_key,
                relationship_type=commit_relationship_type,
                scene_ref_ids=scene_ref_ids,
            )
            row = _find_existing_memory_event_row(
                db,
                session_id=session_id,
                event_key=event_key,
                commit_fact_key=commit_fact_key,
                turn_index=turn_index,
                scene_ref_ids=scene_ref_ids,
            )
            existing_data = dict(row.data or {}) if row is not None else {}
            merged_scene_ref_ids = _normalize_scene_ref_ids(scene_ref_ids)
            event_expectation_debt_score = expectation_debt_score_for_payload(
                {
                    "kind": candidate.kind,
                    "event_outcome": event_outcome,
                    "event_consequence_role": event_consequence_role,
                    "continuity_contract_strength_score": continuity_contract_strength_value,
                    "expectation_salience_score": expectation_salience,
                    "player_salience_score": player_salience,
                    "obligation_weight": obligation_weight,
                }
            )
            event_last_salient_turn = (
                turn_index
                if importance >= 0.65 or expectation_salience >= 0.65
                else _safe_int(existing_data.get("last_salient_turn")) or turn_index
            )
            event_dormancy_transition = derive_dormancy_transition(
                reference_turn=turn_index,
                state="active",
                status="active",
                deadline_turn=None,
                resolved_at_turn=None,
                last_salient_turn=event_last_salient_turn,
                dormant_since_turn=existing_data.get("dormant_since_turn"),
                compression_mode=existing_data.get("compression_mode"),
                obligation_pressure=obligation_weight,
                expectation_debt=event_expectation_debt_score,
                continuity_strength=continuity_contract_strength_value,
                player_salience=player_salience,
                expectation_salience=expectation_salience,
            )
            event_data: dict[str, Any] = {
                "event_key": event_key,
                "layer": "event",
                "kind": candidate.kind,
                "event_role": event_role,
                "event_outcome": event_outcome,
                "event_consequence_role": event_consequence_role,
                "search_recall_summary": search_recall_summary,
                "narrative_recall_summary": narrative_recall_summary,
                "priority": _merge_seed_priority(existing_data.get("priority"), priority),
                "actor_object_id": str(actor_object_id) if actor_object_id is not None else None,
                "counterparty_object_id": str(counterparty_object_id) if counterparty_object_id is not None else None,
                "object_id": str(object_id) if object_id is not None else None,
                "location_object_id": str(location_object_id) if location_object_id is not None else None,
                "quest_object_id": str(quest_object_id) if quest_object_id is not None else None,
                "context_object_ids": [str(context_object_id) for context_object_id in context_object_ids],
                "anchor_object_ids": combined_anchor_ids,
                "scene_ref_ids": merged_scene_ref_ids,
                "knowledge_scope": _merge_memory_scope(existing_data.get("knowledge_scope"), knowledge_scope),
                "callback_strength": _merge_callback_strength(existing_data.get("callback_strength"), callback_strength),
                "commit_fact_key": commit_fact_key or existing_data.get("commit_fact_key"),
                "identity_text": commit_identity_text or existing_data.get("identity_text"),
                "relationship_type": commit_relationship_type or existing_data.get("relationship_type"),
                "player_salience": normalize_player_salience(candidate.player_salience),
                "expectation_salience": normalize_expectation_salience(candidate.expectation_salience),
                "continuity_contract_strength": continuity_contract_strength,
                "certainty": certainty,
                "player_salience_score": max(
                    _coerce_importance(existing_data.get("player_salience_score")) or 0.0,
                    player_salience,
                ),
                "expectation_salience_score": max(
                    _coerce_importance(existing_data.get("expectation_salience_score")) or 0.0,
                    expectation_salience,
                ),
                "continuity_contract_strength_score": max(
                    _coerce_importance(existing_data.get("continuity_contract_strength_score")) or 0.0,
                    continuity_contract_strength_value,
                ),
                "durability": max(_coerce_importance(existing_data.get("durability")) or 0.0, durability),
                "emotional_weight": max(_coerce_importance(existing_data.get("emotional_weight")) or 0.0, emotional_weight),
                "obligation_weight": max(_coerce_importance(existing_data.get("obligation_weight")) or 0.0, obligation_weight),
                "sentimental_weight": max(_coerce_importance(existing_data.get("sentimental_weight")) or 0.0, sentimental_weight),
                "routine_weight": max(_coerce_importance(existing_data.get("routine_weight")) or 0.0, routine_weight),
                "importance": max(_coerce_importance(existing_data.get("importance")) or 0.0, importance),
                "created_turn": _safe_int(existing_data.get("created_turn")) or turn_index,
                "source_turn": _safe_int(existing_data.get("source_turn")) or turn_index,
                "last_seen_turn": turn_index,
                "source_ops_count": max(_safe_int(existing_data.get("source_ops_count")) or 0, resolved_source_ops_count),
                "independent_evidence_count": max(_safe_int(existing_data.get("independent_evidence_count")) or 0, 1),
                "repetition_count": max(_safe_int(existing_data.get("repetition_count")) or 0, 0),
                "last_salient_turn": event_last_salient_turn,
                "dormant_since_turn": event_dormancy_transition.dormant_since_turn,
                "persisted_dormancy_state": event_dormancy_transition.dormancy_state,
                "dormancy_transition": event_dormancy_transition.transition,
                "dormancy_transition_turn": event_dormancy_transition.transition_turn,
                "dormancy_reason_flags": list(event_dormancy_transition.reason_flags),
                "expectation_debt_score": event_expectation_debt_score,
                "zone_scope_id": str(zone_scope_id) if zone_scope_id else existing_data.get("zone_scope_id"),
                "in_game_time": {"day": in_game_day, "minute": in_game_minute},
                "status": "active",
            }
            row = memory_write_repository.upsert_memory_object_row(
                db,
                session_id=session_id,
                object_type=MEMORY_EVENT_OBJECT_TYPE,
                object_name=f"memory_event:{candidate.kind}",
                data=event_data,
                row=row,
            )
            _upsert_memory_summary_embedding(
                db=db,
                    session_id=session_id,
                    object_id=row.object_id,
                    namespace=MEMORY_EVENT_NAMESPACE,
                    summary=search_recall_summary,
                    instruction=MEMORY_EVENT_EMBED_INSTRUCTION,
                )
            stored_event_count += 1
            continue

        if canonical_fact is None:
            continue
        fact_kind = canonical_fact.kind
        fact_priority = _merge_seed_priority(priority, canonical_fact.priority)
        fact_scope = _coerce_memory_scope(canonical_fact.knowledge_scope or knowledge_scope)
        relationship_type = str(canonical_fact.relationship_type or "").strip().lower() or None
        fact_state = normalize_state(canonical_fact.state)
        fact_slot_signature = _memory_fact_slot_signature(
            kind=fact_kind,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            relationship_type=relationship_type,
        )
        fact_key = _memory_fact_identity_key(
            kind=fact_kind,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            identity_text=canonical_fact.identity_text
            or canonical_fact.narrative_recall_summary
            or canonical_fact.search_recall_summary,
            relationship_type=relationship_type,
        )
        fact_rows_for_kind = _iter_memory_fact_rows_for_kind(
            db,
            session_id=session_id,
            fact_kind=fact_kind,
        )
        identity_rows = _iter_memory_fact_rows_for_identity(
            fact_rows_for_kind,
            fact_key=fact_key,
        )
        active_identity_rows = _active_memory_fact_rows(identity_rows)
        transition_resolution = resolve_fact_transition(
            requested_state=fact_state,
            active_match_count=len(active_identity_rows),
        )
        if fact_kind in SUPERSEDED_FACT_KINDS and fact_state == "active":
            _supersede_competing_fact_rows(
                db,
                session_id=session_id,
                fact_kind=fact_kind,
                slot_signature=fact_slot_signature,
                incoming_fact_key=fact_key,
            )
        if transition_resolution.ambiguous:
            logger.warning(
                "memory_fact_transition_ambiguous session=%s turn=%s kind=%s fact_key=%s active_matches=%s requested_state=%s",
                session_id,
                turn_index,
                fact_kind,
                fact_key,
                transition_resolution.active_match_count,
                transition_resolution.requested_state,
            )
        row = active_identity_rows[0] if transition_resolution.action == "update_existing" else None
        existing_data = dict(row.data or {}) if row is not None else {}
        merged_state = (
            coalesce_fact_state(existing_data.get("state"), transition_resolution.persisted_state)
            if row is not None
            else transition_resolution.persisted_state
        )
        fact_callback_candidate = bool(canonical_fact.callback_candidate and state_allows_callback(fact_state))
        if not state_allows_callback(merged_state):
            fact_callback_candidate = False
        fact_callback_strength = (
            _merge_callback_strength(existing_data.get("callback_strength"), callback_strength)
            if fact_callback_candidate
            else "none"
        )
        if fact_callback_candidate and fact_callback_strength == "none":
            fact_callback_strength = "soft"
        support_count = max(_safe_int(existing_data.get("support_count")) or 0, 0) + 1
        existing_independent_evidence_count = max(_safe_int(existing_data.get("independent_evidence_count")) or 0, 0)
        existing_repetition_count = max(_safe_int(existing_data.get("repetition_count")) or 0, 0)
        incoming_repetition_count = max(int(canonical_fact.repetition_count or 0), 0)
        independent_evidence_count = existing_independent_evidence_count + 1
        repetition_count = existing_repetition_count + incoming_repetition_count
        stored_durability = max(_coerce_importance(existing_data.get("durability")) or 0.0, durability)
        confidence = round(min(1.0, 0.35 + 0.15 * support_count + 0.20 * stored_durability), 6)
        fact_search_summary = search_recall_summary
        fact_narrative_summary = narrative_recall_summary
        causal_links = merge_causal_links(existing_data.get("causal_links"))
        transition_links = transition_causal_links(
            requested_state=fact_state,
            matched_fact_key=(str(existing_data.get("fact_key") or "") or fact_key) if row is not None else None,
            turn_index=turn_index,
            source_fact_key=fact_key,
        )
        causal_links = merge_causal_links(causal_links, transition_links)
        last_reconfirmed_turn = turn_index if row is not None else (_safe_int(existing_data.get("last_reconfirmed_turn")) or None)
        expectation_debt_score = expectation_debt_score_for_payload(
            {
                "kind": fact_kind,
                "state": merged_state,
                "continuity_contract_strength_score": continuity_contract_strength_value,
                "expectation_salience_score": expectation_salience,
                "player_salience_score": player_salience,
                "obligation_weight": obligation_weight,
            }
        )
        fact_last_salient_turn = (
            turn_index
            if importance >= 0.65 or expectation_salience >= 0.65 or obligation_weight >= 0.5
            else _safe_int(existing_data.get("last_salient_turn")) or turn_index
        )
        compression_mode = derive_compression_mode(
            state=merged_state,
            callback_strength=fact_callback_strength,
            continuity_contract_strength=continuity_contract_strength,
            last_reconfirmed_turn=last_reconfirmed_turn,
            current_turn=turn_index,
            has_bundle=False,
            expectation_debt_score=expectation_debt_score,
        )
        dormancy_transition = derive_dormancy_transition(
            reference_turn=turn_index,
            state=merged_state,
            status="active",
            deadline_turn=None,
            resolved_at_turn=turn_index if merged_state in {"fulfilled", "broken", "superseded", "contradicted"} else None,
            last_salient_turn=fact_last_salient_turn,
            dormant_since_turn=existing_data.get("dormant_since_turn"),
            compression_mode=compression_mode,
            obligation_pressure=obligation_weight,
            expectation_debt=expectation_debt_score,
            continuity_strength=continuity_contract_strength_value,
            player_salience=player_salience,
            expectation_salience=expectation_salience,
        )
        fact_data: dict[str, Any] = {
            "fact_key": fact_key,
            "layer": "fact",
            "kind": fact_kind,
            "state": merged_state,
            "search_recall_summary": fact_search_summary,
            "narrative_recall_summary": fact_narrative_summary,
            "identity_text": str(
                canonical_fact.identity_text
                or canonical_fact.narrative_recall_summary
                or canonical_fact.search_recall_summary
            ).strip(),
            "priority": _merge_seed_priority(existing_data.get("priority"), fact_priority),
            "actor_object_id": str(actor_object_id) if actor_object_id is not None else None,
            "counterparty_object_id": str(counterparty_object_id) if counterparty_object_id is not None else None,
            "object_id": str(object_id) if object_id is not None else None,
            "location_object_id": str(location_object_id) if location_object_id is not None else None,
            "quest_object_id": str(quest_object_id) if quest_object_id is not None else None,
            "context_object_ids": [str(context_object_id) for context_object_id in context_object_ids],
            "anchor_object_ids": _merge_anchor_object_ids(existing_data.get("anchor_object_ids"), combined_anchor_ids),
            "knowledge_scope": _merge_memory_scope(existing_data.get("knowledge_scope"), fact_scope),
            "callback_strength": fact_callback_strength,
            "callback_candidate": fact_callback_candidate,
            "relationship_type": relationship_type,
            "player_salience": normalize_player_salience(candidate.player_salience),
            "expectation_salience": normalize_expectation_salience(candidate.expectation_salience),
            "continuity_contract_strength": continuity_contract_strength,
            "certainty": certainty,
            "player_salience_score": max(
                _coerce_importance(existing_data.get("player_salience_score")) or 0.0,
                player_salience,
            ),
            "expectation_salience_score": max(
                _coerce_importance(existing_data.get("expectation_salience_score")) or 0.0,
                expectation_salience,
            ),
            "continuity_contract_strength_score": max(
                _coerce_importance(existing_data.get("continuity_contract_strength_score")) or 0.0,
                continuity_contract_strength_value,
            ),
            "durability": stored_durability,
            "emotional_weight": max(_coerce_importance(existing_data.get("emotional_weight")) or 0.0, emotional_weight),
            "obligation_weight": max(_coerce_importance(existing_data.get("obligation_weight")) or 0.0, obligation_weight),
            "sentimental_weight": max(_coerce_importance(existing_data.get("sentimental_weight")) or 0.0, sentimental_weight),
            "routine_weight": max(_coerce_importance(existing_data.get("routine_weight")) or 0.0, routine_weight),
            "importance": max(_coerce_importance(existing_data.get("importance")) or 0.0, importance),
            "source_turn": _safe_int(existing_data.get("source_turn")) or turn_index,
            "started_at_turn": _safe_int(existing_data.get("started_at_turn")) or turn_index,
            "resolved_at_turn": turn_index if merged_state in {"fulfilled", "broken", "superseded", "contradicted"} else None,
            "last_confirmed_turn": turn_index,
            "last_reconfirmed_turn": last_reconfirmed_turn,
            "last_salient_turn": fact_last_salient_turn,
            "dormant_since_turn": dormancy_transition.dormant_since_turn,
            "persisted_dormancy_state": dormancy_transition.dormancy_state,
            "dormancy_transition": dormancy_transition.transition,
            "dormancy_transition_turn": dormancy_transition.transition_turn,
            "dormancy_reason_flags": list(dormancy_transition.reason_flags),
            "source_ops_count": max(_safe_int(existing_data.get("source_ops_count")) or 0, resolved_source_ops_count),
            "support_count": support_count,
            "independent_evidence_count": independent_evidence_count,
            "repetition_count": repetition_count,
            "confidence": confidence,
            "causal_links": causal_links,
            "transition_ambiguity": transition_resolution.ambiguous,
            "transition_requested_state": (
                transition_resolution.requested_state
                if transition_resolution.ambiguous
                else None
            ),
            "expectation_debt_score": expectation_debt_score,
            "compression_mode": compression_mode,
            "status": "active",
        }
        row = memory_write_repository.upsert_memory_object_row(
            db,
            session_id=session_id,
            object_type=MEMORY_FACT_OBJECT_TYPE,
            object_name=f"memory_fact:{fact_kind}",
            data=fact_data,
            row=row,
        )
        _upsert_memory_summary_embedding(
            db=db,
                    session_id=session_id,
                    object_id=row.object_id,
                    namespace=MEMORY_FACT_NAMESPACE,
                    summary=fact_search_summary,
                    instruction=MEMORY_FACT_EMBED_INSTRUCTION,
        )
        stored_fact_payloads.append(fact_data)
        stored_fact_count += 1

    if stored_fact_payloads:
        _apply_same_turn_causal_links(db, stored_fact_payloads=stored_fact_payloads, turn_index=turn_index)
    return {"event_count": stored_event_count, "fact_count": stored_fact_count}


__all__ = [
    "MEMORY_EVENT_OBJECT_TYPE",
    "MEMORY_FACT_OBJECT_TYPE",
    "MEMORY_BUNDLE_OBJECT_TYPE",
    "MEMORY_EVENT_NAMESPACE",
    "MEMORY_FACT_NAMESPACE",
    "MEMORY_BUNDLE_NAMESPACE",
    "MEMORY_BUNDLE_EMBED_INSTRUCTION",
    "_maybe_embed_texts",
    "_upsert_object_embedding",
    "_upsert_npc_profile_embedding",
    "_upsert_player_profile_embedding",
    "_upsert_zone_profile_embedding",
    "_upsert_item_profile_embedding",
    "_upsert_faction_profile_embedding",
    "_upsert_quest_profile_embedding",
    "_extract_claim_text",
    "_upsert_claim_text_embedding",
    "_normalize_memory_text",
    "_coerce_importance",
    "_extract_link_context_text",
    "_build_link_context_snippet",
    "_list_active_link_context_snippets",
    "_upsert_link_context_embedding",
    "_refresh_link_context_embedding",
    "_store_memory_candidates",
]
