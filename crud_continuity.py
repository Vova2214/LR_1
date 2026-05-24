"""Continuity runtime helpers and compatibility shims.

This module still owns continuity runtime orchestration. The pure policy
helpers listed in `COMPATIBILITY_MODULE_CONTRACT` are deprecated forwards into
`src.domain.continuity_policy`.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from . import models, schemas
from .architecture_contracts import COMPATIBILITY_MODULE_CONTRACTS
from .constants import (
    CARRIED_BY_LINK_TYPE,
    LOCATED_IN_LINK_TYPE,
    NPC_SOCIAL_LINK_TYPES,
    RECIPROCAL_SOCIAL_LINK_TYPES,
    SESSION_PLAYER_REF,
    TRACKING_QUEST_LINK_TYPE,
)
from .crud_embeddings_ops import (
    MEMORY_BUNDLE_EMBED_INSTRUCTION,
    MEMORY_BUNDLE_NAMESPACE,
    MEMORY_BUNDLE_OBJECT_TYPE,
    MEMORY_FACT_OBJECT_TYPE,
    _maybe_embed_texts,
    _store_memory_candidates,
    _upsert_object_embedding,
)
from .crud_shared import (
    _coalesce_memory_candidates_by_identity,
    _count_json_tokens,
    _get_object,
    _get_player_current_zone_id,
    _get_session_player_object_id,
    _memory_fact_identity_key,
    _safe_int,
    _truncate_text,
)
from .db import SessionLocal, USE_EMBEDDINGS
from .domain import continuity_policy as continuity_policy_domain
from .domain.memory_policy import (
    ACTOR_SCENE_FACT_KINDS,
    NARRATIVE_PACKET_ROLE_BY_BUNDLE_FAMILY,
    bundle_health_recovery_context,
    bundle_current_relevance_reason,
    callback_validation_plan,
    continuity_contract_strength_score,
    DESCRIPTIVE_LOCATION_FACT_KINDS,
    derive_compression_mode,
    derive_dormancy_transition,
    derive_surprise_weight as policy_derive_surprise_weight,
    expectation_debt_score_for_payload,
    expectation_salience_score,
    ITEM_POSSESSION_FACT_KINDS,
    ITEM_TRANSFER_FACT_KINDS,
    QUEST_FACT_KINDS,
    bundle_family_for_kind,
    bundle_fact_rank_tuple,
    bundle_link_payloads,
    bundle_key,
    coalesce_fact_state,
    is_soft_callback_kind as policy_is_soft_callback_kind,
    memory_candidate_importance as policy_memory_candidate_importance,
    merge_callback_strength,
    merge_priority,
    merge_resolved_durable_fact_payload,
    merge_scope,
    normalize_expectation_salience,
    normalize_continuity_contract_strength,
    normalize_memory_certainty,
    normalize_player_salience,
    normalize_state,
    player_salience_score,
    state_allows_callback,
)
from .observability import record_callback_decision

logger = logging.getLogger(__name__)
COMPATIBILITY_MODULE_CONTRACT = COMPATIBILITY_MODULE_CONTRACTS[__name__]

ENTITY_MEMORY_OBJECT_TYPE = "__entity_memory"
CALLBACK_MEMORY_OBJECT_TYPE = "__callback_memory"
CALLBACK_MEMORY_EMBED_NAMESPACE = "callback_memory"
CALLBACK_MEMORY_EMBED_INSTRUCTION = "Represent this long-term callback memory for narrative resurfacing"
MEMORY_BUNDLE_MAX_ITEMS = 3
MEMORY_BUNDLE_MAX_TOKENS = 240

HARD_MEMORY_MAX_ITEMS = 8
HARD_MEMORY_MAX_TOKENS = 350
ENTITY_HISTORY_MAX_ITEMS = 3
ENTITY_HISTORY_MAX_BULLETS = 6
ENTITY_HISTORY_MAX_TOKENS = 600
CALLBACK_MEMORY_MAX_ITEMS = 2
CALLBACK_MEMORY_MAX_TOKENS = 250
CALLBACK_SOFT_COOLDOWN_TURNS = 12
CALLBACK_STRONG_COOLDOWN_TURNS = 25
CALLBACK_MAX_ANCHORS = 6
MEMORY_SYNC_VERSION = 1
MEMORY_REVIEW_OBJECT_TYPE = "__memory_review_report"
MEMORY_REVIEW_OBJECT_NAME = "memory_review:latest"

HARD_CALLBACK_KINDS = frozenset(
    {
        "promise",
        "debt",
        "betrayal",
        "injury",
        "quest_milestone",
        "decision",
    }
)
SALIENT_CALLBACK_KINDS = frozenset(
    {
        "home_detail",
        "gift",
        "trophy",
        "recurring_prop",
        "relationship",
        "location_fact",
        "emotional_scene",
    }
)
NON_SALIENT_LINK_TYPES = frozenset(
    {
        LOCATED_IN_LINK_TYPE,
        CARRIED_BY_LINK_TYPE,
        TRACKING_QUEST_LINK_TYPE,
        "heard",
        "asserted",
        "adjacent",
    }
)
RARE_ITEM_TIERS = frozenset({"rare", "epic", "legendary", "unique", "artifact"})


@dataclass(slots=True)
class _ResolvedDurableFact:
    fact_key: str
    kind: str
    search_recall_summary: str
    narrative_recall_summary: str
    priority: str
    actor_object_id: uuid.UUID | None
    counterparty_object_id: uuid.UUID | None
    object_id: uuid.UUID | None
    quest_object_id: uuid.UUID | None
    context_object_ids: list[uuid.UUID]
    anchor_object_ids: list[uuid.UUID]
    callback_candidate: bool
    knowledge_scope: str
    relationship_type: str | None
    state: str
    importance: float
    surprise_weight: float
    source_turn: int
    soft_callback: bool
    location_object_id: uuid.UUID | None = None
    callback_strength: str = "soft"
    confidence: float = 0.7
    durability: float = 0.85
    emotional_weight: float = 0.0
    obligation_weight: float = 0.0
    sentimental_weight: float = 0.0
    routine_weight: float = 0.0
    player_salience: str = "none"
    player_salience_score: float = 0.0
    expectation_salience: str = "none"
    expectation_salience_score: float = 0.0
    continuity_contract_strength: str = "none"
    continuity_contract_strength_score: float = 0.0
    independent_evidence_count: int = 0
    repetition_count: int = 0
    compression_mode: str = "direct"
    last_reconfirmed_turn: int | None = None
    certainty: str = "confirmed"
    started_at_turn: int | None = None
    resolved_at_turn: int | None = None
    last_salient_turn: int | None = None
    dormant_since_turn: int | None = None
    causal_links: list[dict[str, Any]] = None  # type: ignore[assignment]

    @property
    def subject_object_id(self) -> uuid.UUID | None:
        if self.kind in ITEM_TRANSFER_FACT_KINDS | ITEM_POSSESSION_FACT_KINDS:
            return self.object_id
        if self.kind in QUEST_FACT_KINDS:
            return self.quest_object_id
        if self.kind in DESCRIPTIVE_LOCATION_FACT_KINDS:
            return self.object_id or self.actor_object_id or self.location_object_id
        return self.actor_object_id

    def __post_init__(self) -> None:
        if self.causal_links is None:
            self.causal_links = []

    @property
    def related_object_ids(self) -> list[uuid.UUID]:
        values: list[uuid.UUID] = []
        if self.counterparty_object_id is not None:
            values.append(self.counterparty_object_id)
        for object_id in self.context_object_ids:
            if object_id not in values:
                values.append(object_id)
        return values


def _merge_uuid_values(left: list[uuid.UUID], right: list[uuid.UUID]) -> list[uuid.UUID]:
    merged: list[uuid.UUID] = []
    for value in [*left, *right]:
        if value not in merged:
            merged.append(value)
    return merged


def _merge_resolved_durable_fact_rows(
    existing: _ResolvedDurableFact,
    incoming: _ResolvedDurableFact,
) -> _ResolvedDurableFact:
    merged_payload = merge_resolved_durable_fact_payload(
        asdict(existing),
        asdict(incoming),
    )
    return _ResolvedDurableFact(**merged_payload)


def _coerce_uuid(raw_value: Any) -> uuid.UUID | None:
    if isinstance(raw_value, uuid.UUID):
        return raw_value
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except ValueError:
        return None


def _resolve_fact_ref(
    raw_value: Any,
    *,
    ref_map: dict[str, str],
    player_object_id: uuid.UUID | None,
) -> uuid.UUID | None:
    if raw_value is None:
        return None
    if raw_value == SESSION_PLAYER_REF:
        return player_object_id
    parsed = _coerce_uuid(raw_value)
    if parsed is not None:
        return parsed
    text = str(raw_value or "").strip()
    if not text:
        return None
    if text == SESSION_PLAYER_REF:
        return player_object_id
    if text.startswith("tmp:"):
        return _coerce_uuid(ref_map.get(text))
    return _coerce_uuid(ref_map.get(text))


def _coerce_priority_score(priority: str) -> float:
    return continuity_policy_domain.coerce_priority_score(priority)


def _memory_candidate_importance(
    *,
    priority: str,
    durability: float,
    emotional_weight: float,
    obligation_weight: float,
    sentimental_weight: float,
    routine_weight: float,
) -> float:
    return policy_memory_candidate_importance(
        priority=priority,
        durability=durability,
        emotional_weight=emotional_weight,
        obligation_weight=obligation_weight,
        sentimental_weight=sentimental_weight,
        routine_weight=routine_weight,
    )


def _derive_surprise_weight(kind: str, priority: str, *, callback_candidate: bool) -> float:
    return policy_derive_surprise_weight(kind, priority, callback_candidate=callback_candidate)


def _is_soft_callback_kind(kind: str) -> bool:
    return policy_is_soft_callback_kind(kind)


def _stable_fact_key(
    *,
    kind: str,
    actor_object_id: uuid.UUID | None,
    counterparty_object_id: uuid.UUID | None,
    object_id: uuid.UUID | None,
    location_object_id: uuid.UUID | None,
    quest_object_id: uuid.UUID | None,
    identity_text: str,
    relationship_type: str | None = None,
) -> str:
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


def _normalize_relationship_type(raw_value: Any) -> str | None:
    text = str(raw_value or "").strip().lower()
    return text or None


def _infer_relationship_type_from_applied_ops(
    applied_ops: list[dict[str, Any]] | None,
    *,
    subject_object_id: uuid.UUID | None,
    related_object_ids: list[uuid.UUID],
) -> str | None:
    if subject_object_id is None or not related_object_ids or not applied_ops:
        return None
    related_set = set(related_object_ids)
    for op in applied_ops:
        if not isinstance(op, dict):
            continue
        if str(op.get("op") or "").strip() not in {"link.create", "link.close"}:
            continue
        relationship_type = _normalize_relationship_type(op.get("type"))
        if relationship_type is None:
            continue
        from_object_id = _coerce_uuid(op.get("from_object_id") or op.get("from"))
        to_object_id = _coerce_uuid(op.get("to_object_id") or op.get("to"))
        if from_object_id is None or to_object_id is None:
            continue
        if (
            subject_object_id == from_object_id and to_object_id in related_set
        ) or (
            subject_object_id == to_object_id and from_object_id in related_set
        ):
            return relationship_type
    return None


def _coerce_durable_fact_rows(
    raw_items: Any,
    *,
    ref_map: dict[str, str],
    player_object_id: uuid.UUID | None,
    source_turn: int,
    applied_ops: list[dict[str, Any]] | None = None,
) -> list[_ResolvedDurableFact]:
    if not isinstance(raw_items, list):
        return []

    resolved: list[_ResolvedDurableFact] = []
    index_by_fact_key: dict[str, int] = {}
    for raw_item in raw_items:
        try:
            fact = schemas.DurableFact.model_validate(raw_item)
        except Exception:  # noqa: BLE001
            continue

        actor_object_id = _resolve_fact_ref(
            fact.actor_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        counterparty_object_id = _resolve_fact_ref(
            fact.counterparty_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        object_id = _resolve_fact_ref(
            fact.object_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        location_object_id = _resolve_fact_ref(
            fact.location_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        quest_object_id = _resolve_fact_ref(
            fact.quest_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        context_object_ids: list[uuid.UUID] = []
        for raw_related in fact.context_refs[:CALLBACK_MAX_ANCHORS]:
            resolved_related = _resolve_fact_ref(
                raw_related,
                ref_map=ref_map,
                player_object_id=player_object_id,
            )
            if resolved_related is None or resolved_related in context_object_ids:
                continue
            context_object_ids.append(resolved_related)

        anchor_object_ids: list[uuid.UUID] = []
        for principal_id in (
            actor_object_id,
            counterparty_object_id,
            object_id,
            location_object_id,
            quest_object_id,
        ):
            if principal_id is not None and principal_id not in anchor_object_ids:
                anchor_object_ids.append(principal_id)
        if location_object_id is not None and location_object_id not in anchor_object_ids:
            anchor_object_ids.append(location_object_id)
        for related_object_id in context_object_ids:
            if related_object_id not in anchor_object_ids:
                anchor_object_ids.append(related_object_id)
        if not anchor_object_ids:
            continue

        relationship_type = _normalize_relationship_type(fact.relationship_type)
        if relationship_type is None and fact.kind == "relationship":
            relationship_type = _infer_relationship_type_from_applied_ops(
                applied_ops,
                subject_object_id=actor_object_id,
                related_object_ids=[counterparty_object_id] if counterparty_object_id is not None else [],
            )

        fact_key = _stable_fact_key(
            kind=fact.kind,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            identity_text=fact.identity_text or fact.narrative_recall_summary or fact.search_recall_summary,
            relationship_type=relationship_type,
        )
        priority = str(fact.priority or "med").strip().lower()
        importance = _memory_candidate_importance(
            priority=priority,
            durability=0.85,
            emotional_weight=0.0,
            obligation_weight=0.0,
            sentimental_weight=0.0,
            routine_weight=0.0,
        )
        soft_callback = _is_soft_callback_kind(str(fact.kind))
        candidate_row = _ResolvedDurableFact(
            fact_key=fact_key,
            kind=str(fact.kind),
            search_recall_summary=_truncate_text(fact.search_recall_summary, 240),
            narrative_recall_summary=_truncate_text(fact.narrative_recall_summary, 240),
            priority=priority,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            context_object_ids=list(context_object_ids),
            anchor_object_ids=list(anchor_object_ids),
            callback_candidate=bool(fact.callback_candidate),
            knowledge_scope=str(fact.knowledge_scope or "global").strip().lower() or "global",
            relationship_type=relationship_type,
            state=normalize_state(fact.state),
            callback_strength="soft" if bool(fact.callback_candidate) else "none",
            confidence=round(min(1.0, 0.35 + 0.15 * 1 + 0.20 * 0.85), 6),
            durability=0.85,
            emotional_weight=0.0,
            obligation_weight=0.0,
            sentimental_weight=0.0,
            routine_weight=0.0,
            player_salience=fact.player_salience,
            player_salience_score=player_salience_score(fact.player_salience),
            expectation_salience=fact.expectation_salience,
            expectation_salience_score=expectation_salience_score(fact.expectation_salience),
            continuity_contract_strength=fact.continuity_contract_strength,
            continuity_contract_strength_score=continuity_contract_strength_score(fact.continuity_contract_strength),
            independent_evidence_count=max(int(fact.independent_evidence_count or 0), 0),
            repetition_count=max(int(fact.repetition_count or 0), 0),
            compression_mode="direct",
            last_reconfirmed_turn=fact.last_reconfirmed_turn,
            importance=importance,
            surprise_weight=_derive_surprise_weight(
                str(fact.kind),
                priority,
                callback_candidate=bool(fact.callback_candidate),
            ),
            source_turn=max(int(source_turn), 0),
            soft_callback=soft_callback,
        )
        existing_index = index_by_fact_key.get(fact_key)
        if existing_index is None:
            index_by_fact_key[fact_key] = len(resolved)
            resolved.append(candidate_row)
            continue
        resolved[existing_index] = _merge_resolved_durable_fact_rows(
            resolved[existing_index],
            candidate_row,
        )
    return resolved


def _coerce_memory_candidate_rows(
    raw_items: Any,
    *,
    ref_map: dict[str, str],
    player_object_id: uuid.UUID | None,
    source_turn: int,
) -> list[_ResolvedDurableFact]:
    if not isinstance(raw_items, list):
        return []

    resolved: list[_ResolvedDurableFact] = []
    index_by_fact_key: dict[str, int] = {}
    for raw_item in raw_items:
        try:
            candidate = schemas.MemoryCandidate.model_validate(raw_item)
        except Exception:  # noqa: BLE001
            continue
        if candidate.canonical_fact is None:
            continue
        if candidate.layer != "fact" and not candidate.requires_commit:
            continue

        canonical_fact = candidate.canonical_fact
        actor_object_id = _resolve_fact_ref(
            canonical_fact.actor_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        counterparty_object_id = _resolve_fact_ref(
            canonical_fact.counterparty_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        object_id = _resolve_fact_ref(
            canonical_fact.object_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        location_object_id = _resolve_fact_ref(
            canonical_fact.location_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        quest_object_id = _resolve_fact_ref(
            canonical_fact.quest_ref,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        context_object_ids: list[uuid.UUID] = []
        for raw_related in canonical_fact.context_refs[:CALLBACK_MAX_ANCHORS]:
            resolved_related = _resolve_fact_ref(
                raw_related,
                ref_map=ref_map,
                player_object_id=player_object_id,
            )
            if resolved_related is None or resolved_related in context_object_ids:
                continue
            context_object_ids.append(resolved_related)

        anchor_object_ids: list[uuid.UUID] = []
        for principal_id in (
            actor_object_id,
            counterparty_object_id,
            object_id,
            location_object_id,
            quest_object_id,
        ):
            if principal_id is not None and principal_id not in anchor_object_ids:
                anchor_object_ids.append(principal_id)
        for context_object_id in context_object_ids:
            if context_object_id not in anchor_object_ids:
                anchor_object_ids.append(context_object_id)
        for raw_anchor in candidate.anchors[:CALLBACK_MAX_ANCHORS]:
            resolved_anchor = _resolve_fact_ref(
                raw_anchor,
                ref_map=ref_map,
                player_object_id=player_object_id,
            )
            if resolved_anchor is None or resolved_anchor in anchor_object_ids:
                continue
            anchor_object_ids.append(resolved_anchor)
        if not anchor_object_ids:
            continue

        relationship_type = _normalize_relationship_type(canonical_fact.relationship_type)
        knowledge_scope = str(canonical_fact.knowledge_scope or "global").strip().lower() or "global"
        fact_key = _stable_fact_key(
            kind=canonical_fact.kind,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            identity_text=canonical_fact.identity_text or canonical_fact.narrative_recall_summary or canonical_fact.search_recall_summary,
            relationship_type=relationship_type,
        )
        callback_strength = str(candidate.callback_strength or "none").strip().lower() or "none"
        callback_candidate = bool(canonical_fact.callback_candidate and state_allows_callback(canonical_fact.state))
        if callback_candidate and callback_strength == "none":
            callback_strength = "soft"
        if not callback_candidate:
            callback_strength = "none"
        effective_priority = merge_priority(candidate.priority, canonical_fact.priority)
        durability = _coerce_priority_score(candidate.priority) if candidate.durability is None else float(candidate.durability)
        confidence = round(min(1.0, 0.35 + 0.15 * 1 + 0.20 * durability), 6)
        candidate_row = _ResolvedDurableFact(
            fact_key=fact_key,
            kind=str(canonical_fact.kind),
            search_recall_summary=_truncate_text(candidate.search_recall_summary, 240),
            narrative_recall_summary=_truncate_text(candidate.narrative_recall_summary, 240),
            priority=effective_priority,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            context_object_ids=list(context_object_ids),
            anchor_object_ids=list(anchor_object_ids),
            callback_candidate=callback_candidate,
            knowledge_scope=knowledge_scope,
            relationship_type=relationship_type,
            state=normalize_state(canonical_fact.state),
            callback_strength=callback_strength,
            confidence=confidence,
            durability=float(candidate.durability),
            emotional_weight=float(candidate.emotional_weight),
            obligation_weight=float(candidate.obligation_weight),
            sentimental_weight=float(candidate.sentimental_weight),
            routine_weight=float(candidate.routine_weight),
            player_salience=normalize_player_salience(candidate.player_salience),
            player_salience_score=player_salience_score(candidate.player_salience),
            expectation_salience=normalize_expectation_salience(candidate.expectation_salience),
            expectation_salience_score=expectation_salience_score(candidate.expectation_salience),
            continuity_contract_strength=normalize_continuity_contract_strength(candidate.continuity_contract_strength),
            continuity_contract_strength_score=continuity_contract_strength_score(candidate.continuity_contract_strength),
            independent_evidence_count=max(int((canonical_fact.independent_evidence_count or 0)), 0),
            repetition_count=max(int((canonical_fact.repetition_count or 0)), 0),
            compression_mode="direct",
            last_reconfirmed_turn=canonical_fact.last_reconfirmed_turn,
            certainty=normalize_memory_certainty(canonical_fact.certainty),
            started_at_turn=max(int(source_turn), 0),
            resolved_at_turn=None,
            last_salient_turn=max(int(source_turn), 0),
            dormant_since_turn=None,
            importance=_memory_candidate_importance(
                priority=effective_priority,
                durability=float(candidate.durability),
                emotional_weight=float(candidate.emotional_weight),
                obligation_weight=float(candidate.obligation_weight),
                sentimental_weight=float(candidate.sentimental_weight),
                routine_weight=float(candidate.routine_weight),
            ),
            surprise_weight=_derive_surprise_weight(
                str(canonical_fact.kind),
                effective_priority,
                callback_candidate=callback_candidate,
            ),
            source_turn=max(int(source_turn), 0),
            soft_callback=callback_strength != "strong",
        )
        existing_index = index_by_fact_key.get(fact_key)
        if existing_index is None:
            index_by_fact_key[fact_key] = len(resolved)
            resolved.append(candidate_row)
            continue
        resolved[existing_index] = _merge_resolved_durable_fact_rows(
            resolved[existing_index],
            candidate_row,
        )
    return resolved


def _is_internal_object_row(row: models.ObjectModel | None) -> bool:
    return row is None or str(row.type or "").startswith("__")


def _is_salient_item_name(name: Any) -> bool:
    return continuity_policy_domain.is_salient_item_name(name)


def _is_rare_item_row(row: models.ObjectModel | None) -> bool:
    if row is None or str(row.type or "").strip() != "item":
        return False
    data = dict(getattr(row, "data", None) or {})
    rarity = str(data.get("rarity") or "").strip().lower()
    return rarity in RARE_ITEM_TIERS


def _make_resolved_fact(
    *,
    kind: str,
    search_recall_summary: str,
    narrative_recall_summary: str,
    identity_text: str | None = None,
    priority: str,
    actor_object_id: uuid.UUID | None,
    counterparty_object_id: uuid.UUID | None,
    object_id: uuid.UUID | None,
    location_object_id: uuid.UUID | None,
    quest_object_id: uuid.UUID | None,
    context_object_ids: list[uuid.UUID],
    anchor_object_ids: list[uuid.UUID],
    relationship_type: str | None,
    callback_candidate: bool,
    knowledge_scope: str,
    source_turn: int,
    callback_strength: str = "soft",
    state: str = "active",
    confidence: float | None = None,
    durability: float = 0.85,
    emotional_weight: float = 0.0,
    obligation_weight: float = 0.0,
    sentimental_weight: float = 0.0,
    routine_weight: float = 0.0,
    player_salience: str = "none",
    expectation_salience: str = "none",
    continuity_contract_strength: str = "none",
    certainty: str = "confirmed",
    independent_evidence_count: int = 0,
    repetition_count: int = 0,
    compression_mode: str = "direct",
) -> _ResolvedDurableFact:
    normalized_priority = str(priority or "med").strip().lower() or "med"
    resolved_confidence = confidence
    if resolved_confidence is None:
        resolved_confidence = round(min(1.0, 0.35 + 0.15 * 1 + 0.20 * durability), 6)
    normalized_state = normalize_state(state)
    normalized_player_salience = normalize_player_salience(player_salience)
    normalized_expectation_salience = normalize_expectation_salience(expectation_salience)
    normalized_continuity_contract_strength = normalize_continuity_contract_strength(continuity_contract_strength)
    return _ResolvedDurableFact(
        fact_key=_stable_fact_key(
            kind=kind,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            identity_text=identity_text or narrative_recall_summary or search_recall_summary,
            relationship_type=relationship_type,
        ),
        kind=kind,
        search_recall_summary=_truncate_text(search_recall_summary, 240),
        narrative_recall_summary=_truncate_text(narrative_recall_summary, 240),
        priority=normalized_priority,
        actor_object_id=actor_object_id,
        counterparty_object_id=counterparty_object_id,
        object_id=object_id,
        location_object_id=location_object_id,
        quest_object_id=quest_object_id,
        context_object_ids=list(context_object_ids),
        anchor_object_ids=list(anchor_object_ids),
        callback_candidate=callback_candidate,
        knowledge_scope=knowledge_scope,
        relationship_type=relationship_type,
        state=normalized_state,
        callback_strength=callback_strength if callback_candidate else "none",
        confidence=resolved_confidence,
        durability=durability,
        emotional_weight=emotional_weight,
        obligation_weight=obligation_weight,
        sentimental_weight=sentimental_weight,
        routine_weight=routine_weight,
        importance=_memory_candidate_importance(
            priority=normalized_priority,
            durability=durability,
            emotional_weight=emotional_weight,
            obligation_weight=obligation_weight,
            sentimental_weight=sentimental_weight,
            routine_weight=routine_weight,
        ),
        surprise_weight=_derive_surprise_weight(
            kind,
            normalized_priority,
            callback_candidate=callback_candidate,
        ),
        source_turn=max(int(source_turn), 0),
        soft_callback=_is_soft_callback_kind(kind),
        player_salience=normalized_player_salience,
        player_salience_score=player_salience_score(normalized_player_salience),
        expectation_salience=normalized_expectation_salience,
        expectation_salience_score=expectation_salience_score(normalized_expectation_salience),
        continuity_contract_strength=normalized_continuity_contract_strength,
        continuity_contract_strength_score=continuity_contract_strength_score(normalized_continuity_contract_strength),
        certainty=normalize_memory_certainty(certainty),
        independent_evidence_count=max(int(independent_evidence_count or 0), 0),
        repetition_count=max(int(repetition_count or 0), 0),
        compression_mode=compression_mode,
        started_at_turn=max(int(source_turn), 0),
        resolved_at_turn=max(int(source_turn), 0) if normalized_state in {"fulfilled", "broken", "superseded", "contradicted"} else None,
        last_salient_turn=max(int(source_turn), 0),
    )


def _build_anchor_object_ids(
    subject_object_id: uuid.UUID | None,
    related_object_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    return continuity_policy_domain.build_anchor_object_ids(subject_object_id, related_object_ids)

def _semantic_fact_signature(
    *,
    kind: str,
    text: str,
    actor_object_id: uuid.UUID | None,
    counterparty_object_id: uuid.UUID | None = None,
    object_id: uuid.UUID | None = None,
    location_object_id: uuid.UUID | None = None,
    quest_object_id: uuid.UUID | None = None,
    relationship_type: str | None = None,
) -> tuple[str, str, str]:
    return (
        _memory_fact_identity_key(
            kind=str(kind or "").strip().lower(),
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            identity_text=text,
            relationship_type=relationship_type,
        ),
        normalize_state("active"),
        "",
    )


def _fact_signature_from_resolved(fact: _ResolvedDurableFact) -> tuple[str, str, str]:
    return _semantic_fact_signature(
        kind=fact.kind,
        text=fact.narrative_recall_summary,
        actor_object_id=fact.actor_object_id,
        counterparty_object_id=fact.counterparty_object_id,
        object_id=fact.object_id,
        location_object_id=fact.location_object_id,
        quest_object_id=fact.quest_object_id,
        relationship_type=fact.relationship_type,
    )


def _merge_resolved_durable_facts(
    *groups: list[_ResolvedDurableFact],
) -> list[_ResolvedDurableFact]:
    merged: list[_ResolvedDurableFact] = []
    index_by_fact_key: dict[str, int] = {}
    for group in groups:
        for fact in group:
            existing_index = index_by_fact_key.get(fact.fact_key)
            if existing_index is None:
                index_by_fact_key[fact.fact_key] = len(merged)
                merged.append(fact)
                continue
            merged[existing_index] = _merge_resolved_durable_fact_rows(
                merged[existing_index],
                fact,
            )
    return merged


def _derive_durable_facts_from_applied_ops(
    db: Session,
    *,
    session_id: uuid.UUID,
    applied_ops: list[dict[str, Any]],
    source_turn: int,
    player_object_id: uuid.UUID | None,
) -> list[_ResolvedDurableFact]:
    if not applied_ops:
        return []

    object_cache: dict[uuid.UUID, models.ObjectModel | None] = {}

    def _cached_object(object_id: uuid.UUID | None) -> models.ObjectModel | None:
        if object_id is None:
            return None
        if object_id not in object_cache:
            object_cache[object_id] = _get_object(db, session_id, object_id)
        return object_cache[object_id]

    derived: list[_ResolvedDurableFact] = []
    for raw_op in applied_ops:
        if not isinstance(raw_op, dict):
            continue
        op_name = str(raw_op.get("op") or "").strip()
        if op_name == "link.create":
            link_type = _normalize_relationship_type(raw_op.get("type"))
            from_object_id = _coerce_uuid(raw_op.get("from_object_id") or raw_op.get("from"))
            to_object_id = _coerce_uuid(raw_op.get("to_object_id") or raw_op.get("to"))
            from_row = _cached_object(from_object_id)
            to_row = _cached_object(to_object_id)
            if _is_internal_object_row(from_row) or _is_internal_object_row(to_row):
                continue
            assert from_row is not None
            assert to_row is not None

            if (
                link_type == CARRIED_BY_LINK_TYPE
                and from_row.type == "item"
                and player_object_id is not None
                and to_object_id == player_object_id
                and _is_salient_item_name(from_row.name)
            ):
                derived.append(
                    _make_resolved_fact(
                        kind="ownership",
                        search_recall_summary=f"{from_row.name} now belongs with the player.",
                        narrative_recall_summary=f"{from_row.name} now belongs with the player.",
                        identity_text=f"{from_row.name} with the player",
                        priority="med",
                        actor_object_id=None,
                        counterparty_object_id=to_object_id,
                        object_id=from_object_id,
                        location_object_id=None,
                        quest_object_id=None,
                        context_object_ids=[],
                        anchor_object_ids=[from_object_id, to_object_id],
                        relationship_type=None,
                        callback_candidate=True,
                        knowledge_scope="global",
                        source_turn=source_turn,
                    )
                )

            if (
                link_type == CARRIED_BY_LINK_TYPE
                and from_row.type == "item"
                and player_object_id is not None
                and to_object_id == player_object_id
                and _is_rare_item_row(from_row)
                and _is_salient_item_name(from_row.name)
            ):
                derived.append(
                    _make_resolved_fact(
                        kind="trophy",
                        search_recall_summary=f"{from_row.name} became one of the player's notable possessions.",
                        narrative_recall_summary=f"{from_row.name} became one of the player's notable possessions.",
                        identity_text=f"{from_row.name} notable possession of the player",
                        priority="high",
                        actor_object_id=None,
                        counterparty_object_id=to_object_id,
                        object_id=from_object_id,
                        location_object_id=None,
                        quest_object_id=None,
                        context_object_ids=[],
                        anchor_object_ids=[from_object_id, to_object_id],
                        relationship_type=None,
                        callback_candidate=True,
                        knowledge_scope="global",
                        source_turn=source_turn,
                    )
                )
                continue

            if (
                link_type == LOCATED_IN_LINK_TYPE
                and from_row.type == "item"
                and to_row.type == "zone"
                and _is_salient_item_name(from_row.name)
            ):
                derived.append(
                    _make_resolved_fact(
                        kind="recurring_prop",
                        search_recall_summary=f"{from_row.name} is kept in {to_row.name}.",
                        narrative_recall_summary=f"{from_row.name} is kept in {to_row.name}.",
                        identity_text=f"{from_row.name} kept in {to_row.name}",
                        priority="med",
                        actor_object_id=from_object_id,
                        counterparty_object_id=None,
                        location_object_id=to_object_id,
                        object_id=None,
                        quest_object_id=None,
                        context_object_ids=[],
                        anchor_object_ids=[from_object_id, to_object_id],
                        relationship_type=None,
                        callback_candidate=True,
                        knowledge_scope="global",
                        source_turn=source_turn,
                    )
                )
                continue

            if (
                link_type in NPC_SOCIAL_LINK_TYPES
                and link_type not in NON_SALIENT_LINK_TYPES
                and from_object_id is not None
                and to_object_id is not None
                and {from_row.type, to_row.type} & {"player", "npc", "faction"}
            ):
                derived.append(
                    _make_resolved_fact(
                        kind="relationship",
                        search_recall_summary=f"{from_row.name} is now {link_type.replace('_', ' ')} with {to_row.name}.",
                        narrative_recall_summary=f"{from_row.name} is now {link_type.replace('_', ' ')} with {to_row.name}.",
                        identity_text=f"{link_type}:{from_row.name}:{to_row.name}",
                        priority="high" if player_object_id in {from_object_id, to_object_id} else "med",
                        actor_object_id=from_object_id,
                        counterparty_object_id=to_object_id,
                        location_object_id=None,
                        object_id=None,
                        quest_object_id=None,
                        context_object_ids=[],
                        anchor_object_ids=[from_object_id, to_object_id],
                        relationship_type=link_type,
                        callback_candidate=True,
                        knowledge_scope="global",
                        source_turn=source_turn,
                    )
                )
            continue

        if op_name == "object.update":
            object_id = _coerce_uuid(raw_op.get("object") or raw_op.get("object_id"))
            object_row = _cached_object(object_id)
            if _is_internal_object_row(object_row):
                continue
            assert object_row is not None
            patch = dict(raw_op.get("patch") or {})
            if object_row.type == "quest" and patch.get("status") is not None:
                derived.append(
                    _make_resolved_fact(
                        kind="quest_milestone",
                        search_recall_summary=f"{object_row.name} is now {patch.get('status')}.",
                        narrative_recall_summary=f"{object_row.name} is now {patch.get('status')}.",
                        identity_text=f"{object_row.name}:{patch.get('status')}",
                        priority="high",
                        actor_object_id=None,
                        counterparty_object_id=None,
                        location_object_id=None,
                        object_id=None,
                        quest_object_id=object_id,
                        context_object_ids=[],
                        anchor_object_ids=[object_id],
                        relationship_type=None,
                        callback_candidate=True,
                        knowledge_scope="global",
                        source_turn=source_turn,
                    )
                )

    return _merge_resolved_durable_facts(derived)


def _resolve_turn_memory_facts(
    db: Session,
    *,
    session_row: models.SessionModel,
    turn_row: models.TurnModel,
    turn_index: int,
) -> tuple[list[dict[str, Any]], list[_ResolvedDurableFact]]:
    ai_json = dict(turn_row.ai_json or {})
    applied_ops = [
        dict(raw_op)
        for raw_op in list(ai_json.get("applied_ops") or [])
        if isinstance(raw_op, dict)
    ]
    ref_map = {
        str(key): str(value)
        for key, value in dict(ai_json.get("ref_map") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    player_object_id = _coerce_uuid((session_row.state_json or {}).get("player_object_id"))
    candidate_facts = _coerce_memory_candidate_rows(
        ai_json.get("memory_candidates"),
        ref_map=ref_map,
        player_object_id=player_object_id,
        source_turn=turn_index,
    )
    derived_facts = _derive_durable_facts_from_applied_ops(
        db,
        session_id=session_row.id,
        applied_ops=applied_ops,
        source_turn=turn_index,
        player_object_id=player_object_id,
    )
    return applied_ops, _merge_resolved_durable_facts(candidate_facts, derived_facts)


def _resolved_durable_fact_from_memory_fact_row(
    row: models.ObjectModel | None,
) -> _ResolvedDurableFact | None:
    if row is None:
        return None
    data = dict(row.data or {})
    status = str(data.get("status") or "").strip().lower()
    if status in {"archived", "stale"}:
        return None
    fact_key = str(data.get("fact_key") or "").strip()
    kind = str(data.get("kind") or "").strip().lower()
    search_recall_summary = _truncate_text(str(data.get("search_recall_summary") or "").strip(), 240)
    narrative_recall_summary = _truncate_text(
        str(data.get("narrative_recall_summary") or data.get("search_recall_summary") or "").strip(),
        240,
    )
    if not fact_key or not kind or not search_recall_summary or not narrative_recall_summary:
        return None

    actor_object_id = _coerce_uuid(data.get("actor_object_id"))
    counterparty_object_id = _coerce_uuid(data.get("counterparty_object_id"))
    object_id = _coerce_uuid(data.get("object_id"))
    location_object_id = _coerce_uuid(data.get("location_object_id"))
    quest_object_id = _coerce_uuid(data.get("quest_object_id"))
    context_object_ids: list[uuid.UUID] = []
    for raw_related in list(data.get("context_object_ids") or []):
        resolved_related = _coerce_uuid(raw_related)
        if resolved_related is None or resolved_related in context_object_ids:
            continue
        context_object_ids.append(resolved_related)

    anchor_object_ids: list[uuid.UUID] = []
    for raw_anchor in list(data.get("anchor_object_ids") or []):
        resolved_anchor = _coerce_uuid(raw_anchor)
        if resolved_anchor is None or resolved_anchor in anchor_object_ids:
            continue
        anchor_object_ids.append(resolved_anchor)
    if not anchor_object_ids:
        for principal_id in (
            actor_object_id,
            counterparty_object_id,
            object_id,
            location_object_id,
            quest_object_id,
        ):
            if principal_id is not None and principal_id not in anchor_object_ids:
                anchor_object_ids.append(principal_id)
        for context_object_id in context_object_ids:
            if context_object_id not in anchor_object_ids:
                anchor_object_ids.append(context_object_id)
    if not anchor_object_ids:
        return None

    priority = str(data.get("priority") or "med").strip().lower() or "med"
    callback_candidate = bool(data.get("callback_candidate"))
    callback_strength = str(data.get("callback_strength") or "none").strip().lower() or "none"
    if callback_candidate:
        if callback_strength == "none":
            callback_strength = "soft"
    else:
        callback_strength = "none"
    confidence = data.get("confidence")
    durability = data.get("durability")
    emotional_weight = data.get("emotional_weight")
    obligation_weight = data.get("obligation_weight")
    sentimental_weight = data.get("sentimental_weight")
    routine_weight = data.get("routine_weight")
    importance = data.get("importance")
    surprise_weight = data.get("surprise_weight")
    return _ResolvedDurableFact(
        fact_key=fact_key,
        kind=kind,
        search_recall_summary=search_recall_summary,
        narrative_recall_summary=narrative_recall_summary,
        priority=priority,
        actor_object_id=actor_object_id,
        counterparty_object_id=counterparty_object_id,
        object_id=object_id,
        location_object_id=location_object_id,
        quest_object_id=quest_object_id,
        context_object_ids=context_object_ids,
        anchor_object_ids=anchor_object_ids,
        callback_candidate=callback_candidate,
        knowledge_scope=str(data.get("knowledge_scope") or "global").strip().lower() or "global",
        relationship_type=_normalize_relationship_type(data.get("relationship_type")),
        state=normalize_state(data.get("state")),
        importance=float(importance) if importance is not None else _memory_candidate_importance(
            priority=priority,
            durability=float(durability or 0.0),
            emotional_weight=float(emotional_weight or 0.0),
            obligation_weight=float(obligation_weight or 0.0),
            sentimental_weight=float(sentimental_weight or 0.0),
            routine_weight=float(routine_weight or 0.0),
        ),
        surprise_weight=float(surprise_weight)
        if surprise_weight is not None
        else _derive_surprise_weight(kind, priority, callback_candidate=callback_candidate),
        source_turn=max(_safe_int(data.get("source_turn")) or 0, 0),
        soft_callback=callback_strength != "strong",
        callback_strength=callback_strength,
        confidence=float(confidence)
        if confidence is not None
        else round(min(1.0, 0.35 + 0.15 * max(_safe_int(data.get("support_count")) or 1, 1) + 0.20 * float(durability or 0.0)), 6),
        durability=float(durability or 0.0),
        emotional_weight=float(emotional_weight or 0.0),
        obligation_weight=float(obligation_weight or 0.0),
        sentimental_weight=float(sentimental_weight or 0.0),
        routine_weight=float(routine_weight or 0.0),
        player_salience=normalize_player_salience(data.get("player_salience")),
        player_salience_score=player_salience_score(data.get("player_salience")),
        expectation_salience=normalize_expectation_salience(data.get("expectation_salience")),
        expectation_salience_score=float(data.get("expectation_salience_score") or 0.0),
        continuity_contract_strength=normalize_continuity_contract_strength(data.get("continuity_contract_strength")),
        continuity_contract_strength_score=float(data.get("continuity_contract_strength_score") or 0.0),
        certainty=normalize_memory_certainty(data.get("certainty")),
        independent_evidence_count=max(_safe_int(data.get("independent_evidence_count")) or 0, 0),
        repetition_count=max(_safe_int(data.get("repetition_count")) or 0, 0),
        compression_mode=str(data.get("compression_mode") or "direct"),
        last_reconfirmed_turn=_safe_int(data.get("last_reconfirmed_turn")),
        started_at_turn=_safe_int(data.get("started_at_turn")),
        resolved_at_turn=_safe_int(data.get("resolved_at_turn")),
        last_salient_turn=_safe_int(data.get("last_salient_turn")),
        dormant_since_turn=_safe_int(data.get("dormant_since_turn")),
        causal_links=list(data.get("causal_links") or []),
    )


def _resolve_callback_sync_facts(
    db: Session,
    *,
    session_id: uuid.UUID,
    durable_facts: list[_ResolvedDurableFact],
) -> list[_ResolvedDurableFact]:
    if not durable_facts:
        return []
    fact_keys = [fact.fact_key for fact in durable_facts if str(fact.fact_key or "").strip()]
    persisted_by_key: dict[str, _ResolvedDurableFact] = {}
    if fact_keys:
        persisted_rows = db.execute(
            select(models.ObjectModel)
            .where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == MEMORY_FACT_OBJECT_TYPE,
                models.ObjectModel.data["fact_key"].astext.in_(fact_keys),
            )
        ).scalars().all()
        for row in persisted_rows:
            resolved = _resolved_durable_fact_from_memory_fact_row(row)
            if resolved is None:
                continue
            persisted_by_key[resolved.fact_key] = resolved

    callback_facts: list[_ResolvedDurableFact] = []
    seen_fact_keys: set[str] = set()
    for fact in durable_facts:
        resolved = persisted_by_key.get(fact.fact_key, fact)
        if resolved.fact_key in seen_fact_keys:
            continue
        seen_fact_keys.add(resolved.fact_key)
        callback_facts.append(resolved)
    return callback_facts


def _find_memory_object_by_json_key(
    db: Session,
    *,
    session_id: uuid.UUID,
    object_type: str,
    json_key: str,
    value: str,
) -> models.ObjectModel | None:
    return db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == object_type,
            models.ObjectModel.data[json_key].astext == value,
        )
        .limit(1)
    ).scalar_one_or_none()


def _callback_row_active(row: models.ObjectModel) -> bool:
    data = dict(row.data or {})
    if not data:
        return False
    return bool(data.get("active", True))


def _upsert_callback_memory_rows(
    db: Session,
    *,
    session_id: uuid.UUID,
    durable_facts: list[_ResolvedDurableFact],
    embeddings_by_fact_key: dict[str, list[float]] | None,
) -> list[uuid.UUID]:
    touched_anchor_ids: list[uuid.UUID] = []
    for fact in durable_facts:
        row = _find_memory_object_by_json_key(
            db,
            session_id=session_id,
            object_type=CALLBACK_MEMORY_OBJECT_TYPE,
            json_key="fact_key",
            value=fact.fact_key,
        )
        existing_data = dict(row.data or {}) if row is not None else {}
        callback_data: dict[str, Any] = {
            "memory_sync_version": MEMORY_SYNC_VERSION,
            "fact_key": fact.fact_key,
            "kind": fact.kind,
            "search_recall_summary": fact.search_recall_summary,
            "narrative_recall_summary": fact.narrative_recall_summary,
            "text": fact.narrative_recall_summary,
            "priority": fact.priority,
            "actor_object_id": str(fact.actor_object_id) if fact.actor_object_id else None,
            "counterparty_object_id": str(fact.counterparty_object_id) if fact.counterparty_object_id else None,
            "object_id": str(fact.object_id) if fact.object_id else None,
            "location_object_id": str(fact.location_object_id) if fact.location_object_id else None,
            "quest_object_id": str(fact.quest_object_id) if fact.quest_object_id else None,
            "context_object_ids": [str(value) for value in fact.context_object_ids],
            "anchor_object_ids": [str(value) for value in fact.anchor_object_ids],
            "anchor_object_id": str(fact.anchor_object_ids[0]),
            "callback_candidate": fact.callback_candidate,
            "callback_strength": fact.callback_strength,
            "knowledge_scope": fact.knowledge_scope,
            "relationship_type": fact.relationship_type,
            "state": fact.state,
            "confidence": fact.confidence,
            "durability": fact.durability,
            "emotional_weight": fact.emotional_weight,
            "obligation_weight": fact.obligation_weight,
            "sentimental_weight": fact.sentimental_weight,
            "routine_weight": fact.routine_weight,
            "player_salience": fact.player_salience,
            "player_salience_score": fact.player_salience_score,
            "expectation_salience": fact.expectation_salience,
            "expectation_salience_score": fact.expectation_salience_score,
            "continuity_contract_strength": fact.continuity_contract_strength,
            "continuity_contract_strength_score": fact.continuity_contract_strength_score,
            "independent_evidence_count": fact.independent_evidence_count,
            "repetition_count": fact.repetition_count,
            "compression_mode": fact.compression_mode,
            "last_reconfirmed_turn": fact.last_reconfirmed_turn,
            "causal_links": list(fact.causal_links),
            "importance": fact.importance,
            "surprise_weight": fact.surprise_weight,
            "source_turn": fact.source_turn,
            "soft_callback": fact.soft_callback,
            "active": True,
            "last_recalled_turn": existing_data.get("last_recalled_turn"),
            "recall_count": max(_safe_int(existing_data.get("recall_count")) or 0, 0),
            "cooldown_until_turn": existing_data.get("cooldown_until_turn"),
        }
        if row is None:
            row = models.ObjectModel(
                session_id=session_id,
                type=CALLBACK_MEMORY_OBJECT_TYPE,
                name=f"callback:{fact.kind}",
                data=callback_data,
            )
            db.add(row)
            db.flush([row])
        else:
            row.name = f"callback:{fact.kind}"
            row.data = callback_data

        if fact.callback_candidate and USE_EMBEDDINGS:
            embedding = (embeddings_by_fact_key or {}).get(fact.fact_key)
            if embedding is not None:
                text_hash = hashlib.sha256(fact.search_recall_summary.encode("utf-8")).hexdigest()
                _upsert_object_embedding(
                    db=db,
                    session_id=session_id,
                    object_id=row.object_id,
                    namespace=CALLBACK_MEMORY_EMBED_NAMESPACE,
                    text_hash=text_hash,
                    embedding=embedding,
                )

        for anchor_object_id in fact.anchor_object_ids:
            if anchor_object_id not in touched_anchor_ids:
                touched_anchor_ids.append(anchor_object_id)
    return touched_anchor_ids


def _upsert_callback_memory_embeddings(
    db: Session,
    *,
    session_id: uuid.UUID,
    durable_facts: list[_ResolvedDurableFact],
    embeddings_by_fact_key: dict[str, list[float]],
) -> None:
    if not embeddings_by_fact_key:
        return
    for fact in durable_facts:
        if not fact.callback_candidate:
            continue
        embedding = embeddings_by_fact_key.get(fact.fact_key)
        if embedding is None:
            continue
        row = _find_memory_object_by_json_key(
            db,
            session_id=session_id,
            object_type=CALLBACK_MEMORY_OBJECT_TYPE,
            json_key="fact_key",
            value=fact.fact_key,
        )
        if row is None:
            continue
        text_hash = hashlib.sha256(fact.search_recall_summary.encode("utf-8")).hexdigest()
        _upsert_object_embedding(
            db=db,
            session_id=session_id,
            object_id=row.object_id,
            namespace=CALLBACK_MEMORY_EMBED_NAMESPACE,
            text_hash=text_hash,
            embedding=embedding,
        )


def _iter_callback_rows_for_anchors(
    db: Session,
    *,
    session_id: uuid.UUID,
    anchor_object_ids: list[uuid.UUID],
) -> list[models.ObjectModel]:
    if not anchor_object_ids:
        return []
    anchor_id_texts = sorted({str(value) for value in anchor_object_ids if value is not None})
    if not anchor_id_texts:
        return []
    anchor_filters = [
        models.ObjectModel.data.contains({"anchor_object_ids": [anchor_id_text]})
        for anchor_id_text in anchor_id_texts
    ]
    return db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == CALLBACK_MEMORY_OBJECT_TYPE,
            or_(*anchor_filters),
        )
    ).scalars().all()


def _callback_similarity_scores_for_rows(
    *,
    db: Session,
    session_id: uuid.UUID,
    callback_rows: list[models.ObjectModel],
    query_embedding: list[float] | None,
) -> dict[str, float]:
    if query_embedding is None or not USE_EMBEDDINGS or not callback_rows:
        return {}
    callback_object_ids = [
        row.object_id
        for row in callback_rows
        if isinstance(getattr(row, "object_id", None), uuid.UUID)
    ]
    if not callback_object_ids:
        return {}
    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
    rows = db.execute(
        select(
            models.ObjectEmbeddingModel.object_id,
            distance_expr.label("distance"),
        ).where(
            models.ObjectEmbeddingModel.session_id == session_id,
            models.ObjectEmbeddingModel.namespace == CALLBACK_MEMORY_EMBED_NAMESPACE,
            models.ObjectEmbeddingModel.object_id.in_(callback_object_ids),
        )
    ).all()
    scores: dict[str, float] = {}
    for object_id, distance in rows:
        if object_id is None or distance is None:
            continue
        scores[str(object_id)] = round(max(1.0 - float(distance), 0.0), 6)
    return scores


def _active_link_exists(
    db: Session,
    *,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
    link_type: str,
) -> bool:
    return (
        db.execute(
            select(models.LinkModel.link_id)
            .where(
                models.LinkModel.session_id == session_id,
                models.LinkModel.from_object_id == from_object_id,
                models.LinkModel.to_object_id == to_object_id,
                models.LinkModel.type == link_type,
                models.LinkModel.valid_to_turn.is_(None),
            )
            .limit(1)
        ).scalar_one_or_none()
        is not None
    )


def _active_link_exists_any_type(
    db: Session,
    *,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
) -> bool:
    return (
        db.execute(
            select(models.LinkModel.link_id)
            .where(
                models.LinkModel.session_id == session_id,
                models.LinkModel.valid_to_turn.is_(None),
                or_(
                    and_(
                        models.LinkModel.from_object_id == from_object_id,
                        models.LinkModel.to_object_id == to_object_id,
                    ),
                    and_(
                        models.LinkModel.from_object_id == to_object_id,
                        models.LinkModel.to_object_id == from_object_id,
                    ),
                ),
            )
            .limit(1)
        ).scalar_one_or_none()
        is not None
    )


def _invalidate_callback_rows(
    db: Session,
    *,
    session_id: uuid.UUID,
    anchor_object_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    touched: list[uuid.UUID] = []
    rows = _iter_callback_rows_for_anchors(
        db,
        session_id=session_id,
        anchor_object_ids=anchor_object_ids,
    )
    for row in rows:
        data = dict(row.data or {})
        if not _callback_row_active(row):
            continue
        invalidation_reason: str | None = None
        fact_kind = str(data.get("kind") or "").strip().lower()
        actor_object_id = _coerce_uuid(data.get("actor_object_id"))
        counterparty_object_id = _coerce_uuid(data.get("counterparty_object_id"))
        object_id = _coerce_uuid(data.get("object_id"))
        location_object_id = _coerce_uuid(data.get("location_object_id"))
        quest_object_id = _coerce_uuid(data.get("quest_object_id"))
        context_object_ids = [
            parsed
            for parsed in (_coerce_uuid(raw_value) for raw_value in list(data.get("context_object_ids") or []))
            if parsed is not None
        ]
        validation_plan = callback_validation_plan(
            kind=fact_kind,
            actor_object_id=actor_object_id,
            counterparty_object_id=counterparty_object_id,
            object_id=object_id,
            location_object_id=location_object_id,
            quest_object_id=quest_object_id,
            relationship_type=data.get("relationship_type"),
        )
        subject_object_id = validation_plan.subject_object_id
        if subject_object_id is None:
            data["active"] = False
            data["inactive_reason"] = "missing_subject"
            row.data = data
            for anchor_object_id in anchor_object_ids:
                if anchor_object_id not in touched:
                    touched.append(anchor_object_id)
            continue
        subject_row = _get_object(db, session_id, subject_object_id)
        if subject_row is None:
            data["active"] = False
            data["inactive_reason"] = "subject_deleted"
            row.data = data
            continue

        deactivate = False
        if validation_plan.mode == "located_in":
            if validation_plan.target_object_id is not None:
                deactivate = not _active_link_exists(
                    db,
                    session_id=session_id,
                    from_object_id=subject_object_id,
                    to_object_id=validation_plan.target_object_id,
                    link_type=LOCATED_IN_LINK_TYPE,
                )
                if deactivate:
                    invalidation_reason = "location_changed"
        elif validation_plan.mode == "carried_or_located":
            if validation_plan.target_object_id is None:
                deactivate = True
                invalidation_reason = "missing_counterparty"
            else:
                carried = _active_link_exists(
                    db,
                    session_id=session_id,
                    from_object_id=subject_object_id,
                    to_object_id=validation_plan.target_object_id,
                    link_type=CARRIED_BY_LINK_TYPE,
                )
                located = _active_link_exists(
                    db,
                    session_id=session_id,
                    from_object_id=subject_object_id,
                    to_object_id=validation_plan.target_object_id,
                    link_type=LOCATED_IN_LINK_TYPE,
                )
                if not carried and not located:
                    deactivate = True
                    invalidation_reason = "holder_or_location_mismatch"
        elif validation_plan.mode == "relationship":
            if validation_plan.target_object_id is None:
                deactivate = True
                invalidation_reason = "missing_relationship_target"
            elif validation_plan.relationship_type is not None:
                relationship_still_active = (
                    _active_link_exists(
                        db,
                        session_id=session_id,
                        from_object_id=subject_object_id,
                        to_object_id=validation_plan.target_object_id,
                        link_type=validation_plan.relationship_type,
                    )
                    or (
                        validation_plan.relationship_type in RECIPROCAL_SOCIAL_LINK_TYPES
                        and _active_link_exists(
                            db,
                            session_id=session_id,
                            from_object_id=validation_plan.target_object_id,
                            to_object_id=subject_object_id,
                            link_type=validation_plan.relationship_type,
                        )
                    )
                )
            else:
                relationship_still_active = (
                    _active_link_exists_any_type(
                        db,
                        session_id=session_id,
                        from_object_id=subject_object_id,
                        to_object_id=validation_plan.target_object_id,
                    )
                )
            if not relationship_still_active:
                deactivate = True
                invalidation_reason = "relationship_inactive"
        elif validation_plan.mode == "counterparty_required":
            if validation_plan.target_object_id is None:
                deactivate = True
                invalidation_reason = "missing_counterparty"

        if deactivate:
            data["active"] = False
            data["inactive_reason"] = invalidation_reason or validation_plan.mode
            row.data = data
            for anchor_object_id in anchor_object_ids:
                if anchor_object_id not in touched:
                    touched.append(anchor_object_id)
    return touched


def _state_bullets_for_anchor(
    db: Session,
    *,
    session_id: uuid.UUID,
    anchor_row: models.ObjectModel,
) -> list[str]:
    data = dict(anchor_row.data or {})
    bullets: list[str] = []
    object_type = str(anchor_row.type or "").strip()
    if object_type == "npc":
        status_text = str(data.get("status") or "active").strip()
        if status_text:
            bullets.append(f"status: {anchor_row.name} is {status_text}")
        zone_id = _get_player_current_zone_id(db, session_id) if anchor_row.type == "player" else None
        if zone_id is None:
            zone_id = db.execute(
                select(models.LinkModel.to_object_id)
                .where(
                    models.LinkModel.session_id == session_id,
                    models.LinkModel.from_object_id == anchor_row.object_id,
                    models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                    models.LinkModel.valid_to_turn.is_(None),
                )
                .limit(1)
            ).scalar_one_or_none()
        if zone_id is not None:
            zone_row = _get_object(db, session_id, zone_id)
            if zone_row is not None:
                bullets.append(f"location: {anchor_row.name} is tied to {zone_row.name}")
    elif object_type == "player":
        zone_id = _get_player_current_zone_id(db, session_id)
        if zone_id is not None:
            zone_row = _get_object(db, session_id, zone_id)
            if zone_row is not None:
                bullets.append(f"location: player is at {zone_row.name}")
    elif object_type == "quest":
        status_text = str(data.get("status") or "").strip()
        if status_text:
            bullets.append(f"quest status: {status_text}")
        short_desc = str(data.get("short_desc") or data.get("objective") or "").strip()
        if short_desc:
            bullets.append(f"quest detail: {_truncate_text(short_desc, 120)}")
    elif object_type == "item":
        carrier_id = db.execute(
            select(models.LinkModel.to_object_id)
            .where(
                models.LinkModel.session_id == session_id,
                models.LinkModel.from_object_id == anchor_row.object_id,
                models.LinkModel.type == CARRIED_BY_LINK_TYPE,
                models.LinkModel.valid_to_turn.is_(None),
            )
            .limit(1)
        ).scalar_one_or_none()
        if carrier_id is not None:
            carrier_row = _get_object(db, session_id, carrier_id)
            if carrier_row is not None:
                bullets.append(f"held by: {carrier_row.name}")
        else:
            zone_id = db.execute(
                select(models.LinkModel.to_object_id)
                .where(
                    models.LinkModel.session_id == session_id,
                    models.LinkModel.from_object_id == anchor_row.object_id,
                    models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                    models.LinkModel.valid_to_turn.is_(None),
                )
                .limit(1)
            ).scalar_one_or_none()
            if zone_id is not None:
                zone_row = _get_object(db, session_id, zone_id)
                if zone_row is not None:
                    bullets.append(f"located in: {zone_row.name}")
    else:
        status_text = str(data.get("status") or "").strip()
        if status_text:
            bullets.append(f"status: {status_text}")
    return bullets[:3]


def _compact_fact_payload(data: dict[str, Any]) -> dict[str, Any]:
    knowledge_scope = str(data.get("knowledge_scope") or "global").strip().lower()
    if knowledge_scope not in {"global", "public", "npc_private"}:
        knowledge_scope = "global"
    return {
        "fact_key": str(data.get("fact_key") or ""),
        "kind": str(data.get("kind") or ""),
        "text": _truncate_text(str(data.get("narrative_recall_summary") or data.get("search_recall_summary") or ""), 160),
        "priority": str(data.get("priority") or "med"),
        "source_turn": _safe_int(data.get("source_turn")),
        "knowledge_scope": knowledge_scope,
    }


def _bundle_role_payload_from_fact_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor_object_id": _coerce_uuid(data.get("actor_object_id")),
        "counterparty_object_id": _coerce_uuid(data.get("counterparty_object_id")),
        "object_id": _coerce_uuid(data.get("object_id")),
        "location_object_id": _coerce_uuid(data.get("location_object_id")),
        "quest_object_id": _coerce_uuid(data.get("quest_object_id")),
        "relationship_type": _normalize_relationship_type(data.get("relationship_type")),
    }


def _build_memory_bundle_summary(
    *,
    fact_payloads: list[dict[str, Any]],
    recent_event_summaries: list[str],
) -> tuple[str, str]:
    sorted_facts = sorted(
        fact_payloads,
        key=bundle_fact_rank_tuple,
        reverse=True,
    )
    narrative_parts: list[str] = []
    search_parts: list[str] = []
    for payload in sorted_facts[:2]:
        search_text = _truncate_text(str(payload.get("search_recall_summary") or ""), 140)
        narrative_text = _truncate_text(
            str(payload.get("narrative_recall_summary") or payload.get("search_recall_summary") or ""),
            140,
        )
        if search_text and search_text not in search_parts:
            search_parts.append(search_text)
        if narrative_text and narrative_text not in narrative_parts:
            narrative_parts.append(narrative_text)
    for summary in recent_event_summaries[:1]:
        normalized = _truncate_text(str(summary or ""), 140)
        if normalized and normalized not in search_parts:
            search_parts.append(normalized)
        if normalized and normalized not in narrative_parts:
            narrative_parts.append(normalized)
    return (
        _truncate_text(" | ".join(search_parts), 240),
        _truncate_text(" | ".join(narrative_parts), 240),
    )


def _rebuild_memory_bundle_rows(
    db: Session,
    *,
    session_id: uuid.UUID,
    current_turn: int,
) -> None:
    latest_review_payload = dict(
        db.execute(
            select(models.ObjectModel.data).where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == MEMORY_REVIEW_OBJECT_TYPE,
                models.ObjectModel.name == MEMORY_REVIEW_OBJECT_NAME,
            )
        ).scalar_one_or_none()
        or {}
    )
    fact_rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == MEMORY_FACT_OBJECT_TYPE,
        )
    ).scalars().all()
    grouped: dict[str, dict[str, Any]] = {}
    all_fact_keys: set[str] = set()
    for row in fact_rows:
        data = dict(row.data or {})
        if normalize_state(data.get("state")) in {"superseded", "contradicted"}:
            continue
        if str(data.get("status") or "active").strip().lower() != "active":
            continue
        family = bundle_family_for_kind(str(data.get("kind") or ""))
        if family is None:
            continue
        role_payload = _bundle_role_payload_from_fact_data(data)
        row_bundle_key = bundle_key(
            family=family,
            actor_ref=role_payload["actor_object_id"],
            counterparty_ref=role_payload["counterparty_object_id"],
            object_ref=role_payload["object_id"],
            location_ref=role_payload["location_object_id"],
            quest_ref=role_payload["quest_object_id"],
            relationship_type=role_payload["relationship_type"],
        )
        bucket = grouped.setdefault(
            row_bundle_key,
            {
                "bundle_key": row_bundle_key,
                "bundle_family": family,
                **role_payload,
                "fact_payloads": [],
                "fact_rows": [],
                "fact_keys": [],
                "recent_event_ids": [],
                "anchor_object_ids": [],
                "knowledge_scope": "global",
                "importance": 0.0,
                "player_salience_score": 0.0,
                "expectation_salience_score": 0.0,
                "continuity_contract_strength_score": 0.0,
                "expectation_debt_score": 0.0,
                "obligation_pressure_score": 0.0,
                "last_support_turn": 0,
            },
        )
        bucket["fact_payloads"].append(data)
        bucket["fact_rows"].append(row)
        fact_key = str(data.get("fact_key") or "")
        if fact_key:
            bucket["fact_keys"].append(fact_key)
            all_fact_keys.add(fact_key)
        for anchor_object_id in list(data.get("anchor_object_ids") or []):
            anchor_text = str(anchor_object_id).strip()
            if anchor_text and anchor_text not in bucket["anchor_object_ids"]:
                bucket["anchor_object_ids"].append(anchor_text)
        bucket["knowledge_scope"] = merge_scope(
            bucket["knowledge_scope"],
            str(data.get("knowledge_scope") or "global"),
        )
        bucket["importance"] = max(bucket["importance"], float(data.get("importance") or 0.0))
        bucket["player_salience_score"] = max(
            bucket["player_salience_score"],
            float(data.get("player_salience_score") or 0.0),
        )
        bucket["expectation_salience_score"] = max(
            bucket["expectation_salience_score"],
            float(data.get("expectation_salience_score") or 0.0),
        )
        bucket["continuity_contract_strength_score"] = max(
            bucket["continuity_contract_strength_score"],
            float(data.get("continuity_contract_strength_score") or 0.0),
        )
        bucket["expectation_debt_score"] = max(
            bucket["expectation_debt_score"],
            float(data.get("expectation_debt_score") or expectation_debt_score_for_payload(data)),
        )
        bucket["obligation_pressure_score"] = max(
            bucket["obligation_pressure_score"],
            float(data.get("obligation_pressure_score") or data.get("continuity_pressure_score") or data.get("obligation_weight") or 0.0),
        )
        last_support_turn = max(
            _safe_int(data.get("last_confirmed_turn")) or 0,
            _safe_int(data.get("source_turn")) or 0,
        )
        bucket["last_support_turn"] = max(bucket["last_support_turn"], last_support_turn)

    event_summaries_by_fact_key: dict[str, list[str]] = {}
    if all_fact_keys:
        event_rows = db.execute(
            select(models.ObjectModel)
            .where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == "__memory_event",
                models.ObjectModel.data["commit_fact_key"].astext.in_(sorted(all_fact_keys)),
            )
        ).scalars().all()
        for row in event_rows:
            data = dict(row.data or {})
            fact_key = str(data.get("commit_fact_key") or "")
            summary = _truncate_text(
                str(data.get("narrative_recall_summary") or data.get("search_recall_summary") or ""),
                140,
            )
            if not fact_key or not summary:
                continue
            event_summaries_by_fact_key.setdefault(fact_key, [])
            if summary not in event_summaries_by_fact_key[fact_key]:
                event_summaries_by_fact_key[fact_key].append(summary)
            row_object_id = getattr(row, "object_id", None)
            if isinstance(row_object_id, uuid.UUID):
                bucket_keys = [
                    bundle_key(
                        family=group["bundle_family"],
                        actor_ref=group["actor_object_id"],
                        counterparty_ref=group["counterparty_object_id"],
                        object_ref=group["object_id"],
                        location_ref=group["location_object_id"],
                        quest_ref=group["quest_object_id"],
                        relationship_type=group["relationship_type"],
                    )
                    for group in grouped.values()
                    if fact_key in group["fact_keys"]
                ]
                for row_bundle_key in bucket_keys:
                    grouped[row_bundle_key]["recent_event_ids"].append(str(row_object_id))

    existing_rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == MEMORY_BUNDLE_OBJECT_TYPE,
        )
    ).scalars().all()
    existing_by_key = {
        str(dict(row.data or {}).get("bundle_key") or ""): row
        for row in existing_rows
        if str(dict(row.data or {}).get("bundle_key") or "")
    }
    seen_bundle_keys: set[str] = set()
    for bundle_row_key, bucket in grouped.items():
        seen_bundle_keys.add(bundle_row_key)
        recent_event_summaries: list[str] = []
        for fact_key in bucket["fact_keys"]:
            recent_event_summaries.extend(event_summaries_by_fact_key.get(fact_key, []))
        search_summary, narrative_summary = _build_memory_bundle_summary(
            fact_payloads=bucket["fact_payloads"],
            recent_event_summaries=recent_event_summaries,
        )
        if not search_summary or not narrative_summary:
            continue
        ranked_fact_payloads = sorted(bucket["fact_payloads"], key=bundle_fact_rank_tuple, reverse=True)
        core_fact_key = str(ranked_fact_payloads[0].get("fact_key") or "") if ranked_fact_payloads else ""
        supporting_fact_keys = [
            str(payload.get("fact_key") or "")
            for payload in ranked_fact_payloads[1:3]
            if str(payload.get("fact_key") or "")
        ]
        callback_candidate = bool(ranked_fact_payloads[0].get("callback_candidate")) if ranked_fact_payloads else False
        recent_reconfirmation = any(
            (_safe_int(payload.get("last_reconfirmed_turn")) or 0) >= max(current_turn - 3, 0)
            for payload in ranked_fact_payloads
        )
        state_conflict = any(normalize_state(payload.get("state")) != "active" for payload in ranked_fact_payloads)
        health_recovery_context = bundle_health_recovery_context(
            session_memory_health_score_value=latest_review_payload.get("session_memory_health_score"),
            finding_counts=dict(latest_review_payload.get("finding_counts") or {}),
            saturation_diagnostics=dict(latest_review_payload.get("saturation_diagnostics") or {}),
            continuity_contract_strength=float(bucket["continuity_contract_strength_score"] or 0.0),
            callback_candidate=callback_candidate,
            recent_reconfirmation=recent_reconfirmation,
            obligation_pressure=float(bucket["obligation_pressure_score"] or 0.0),
            expectation_debt=float(bucket["expectation_debt_score"] or 0.0),
            state_conflict=state_conflict,
        )
        current_relevance_reason = bundle_current_relevance_reason(
            anchor_overlap=False,
            recent_event_count=len(bucket["recent_event_ids"]),
            continuity_contract_strength=float(bucket["continuity_contract_strength_score"] or 0.0),
            callback_candidate=callback_candidate,
            bundle_pressure=len(bucket["fact_payloads"]) > 3,
            state_conflict=state_conflict,
            recent_reconfirmation=recent_reconfirmation,
            obligation_pressure=float(bucket["obligation_pressure_score"] or 0.0),
            expectation_debt=float(bucket["expectation_debt_score"] or 0.0),
            health_recovery=bool(health_recovery_context.get("eligible")),
        )
        row_data = {
            "bundle_key": bundle_row_key,
            "bundle_family": bucket["bundle_family"],
            "search_recall_summary": search_summary,
            "narrative_recall_summary": narrative_summary,
            "layer": "bundle",
            "kind": bucket["bundle_family"],
            "fact_keys": bucket["fact_keys"],
            "core_fact_key": core_fact_key or None,
            "supporting_fact_keys": supporting_fact_keys,
            "recent_event_ids": bucket["recent_event_ids"][:2],
            "callback_candidate": callback_candidate,
            "narrative_packet_role": NARRATIVE_PACKET_ROLE_BY_BUNDLE_FAMILY.get(bucket["bundle_family"], "conflict"),
            "actor_object_id": str(bucket["actor_object_id"]) if bucket["actor_object_id"] is not None else None,
            "counterparty_object_id": str(bucket["counterparty_object_id"]) if bucket["counterparty_object_id"] is not None else None,
            "object_id": str(bucket["object_id"]) if bucket["object_id"] is not None else None,
            "location_object_id": str(bucket["location_object_id"]) if bucket["location_object_id"] is not None else None,
            "quest_object_id": str(bucket["quest_object_id"]) if bucket["quest_object_id"] is not None else None,
            "relationship_type": bucket["relationship_type"],
            "anchor_object_ids": bucket["anchor_object_ids"],
            "knowledge_scope": bucket["knowledge_scope"],
            "importance": round(float(bucket["importance"]), 6),
            "player_salience_score": round(float(bucket["player_salience_score"]), 6),
            "expectation_salience_score": round(float(bucket["expectation_salience_score"]), 6),
            "continuity_contract_strength_score": round(float(bucket["continuity_contract_strength_score"]), 6),
            "expectation_debt_score": round(float(bucket["expectation_debt_score"]), 6),
            "current_relevance_reason": current_relevance_reason,
            "current_relevance_reason_context": (
                dict(health_recovery_context or {})
                if current_relevance_reason == "health_recovery"
                else None
            ),
            "packet_pressure_score": round(
                min(
                    max(
                        float(bucket["obligation_pressure_score"] or 0.0),
                        float(bucket["expectation_debt_score"] or 0.0),
                        float(bucket["continuity_contract_strength_score"] or 0.0),
                    ),
                    1.0,
                ),
                6,
            ),
            "why_packet_became_core": (
                "highest_fact_rank"
                if core_fact_key
                else "no_core_fact"
            ),
            "supporting_fact_payloads": ranked_fact_payloads[:3],
            "source_turn": bucket["last_support_turn"],
            "updated_turn": current_turn,
            "status": "active",
        }
        row = existing_by_key.get(bundle_row_key)
        if row is None:
            row = models.ObjectModel(
                session_id=session_id,
                type=MEMORY_BUNDLE_OBJECT_TYPE,
                name=f"memory_bundle:{bucket['bundle_family']}",
                data=row_data,
            )
            db.add(row)
            db.flush([row])
        else:
            row.name = f"memory_bundle:{bucket['bundle_family']}"
            row.data = row_data
        existing_by_key[bundle_row_key] = row
        if USE_EMBEDDINGS:
            text_hash = hashlib.sha256(search_summary.encode("utf-8")).hexdigest()
            embedding = _maybe_embed_texts(
                [search_summary],
                instruction=MEMORY_BUNDLE_EMBED_INSTRUCTION,
            )
            if embedding:
                _upsert_object_embedding(
                    db=db,
                    session_id=session_id,
                    object_id=row.object_id,
                    namespace=MEMORY_BUNDLE_NAMESPACE,
                    text_hash=text_hash,
                    embedding=embedding[0],
                )
    bundle_rows_for_links = [
        dict(row.data or {})
        for key, row in existing_by_key.items()
        if key in seen_bundle_keys
    ]
    bundle_relationships_by_key = bundle_link_payloads(bundle_rows_for_links)
    for bundle_row_key in seen_bundle_keys:
        row = existing_by_key.get(bundle_row_key)
        if row is None:
            row = _find_memory_object_by_json_key(
                db,
                session_id=session_id,
                object_type=MEMORY_BUNDLE_OBJECT_TYPE,
                json_key="bundle_key",
                value=bundle_row_key,
            )
        if row is None:
            continue
        data = dict(row.data or {})
        data["bundle_relationships"] = list(bundle_relationships_by_key.get(bundle_row_key, []))
        row.data = data
    for bucket in grouped.values():
        has_bundle = bucket["bundle_key"] in seen_bundle_keys
        for fact_row in bucket.get("fact_rows", []):
            data = dict(fact_row.data or {})
            data["compression_mode"] = derive_compression_mode(
                state=data.get("state"),
                callback_strength=data.get("callback_strength"),
                continuity_contract_strength=data.get("continuity_contract_strength"),
                last_reconfirmed_turn=_safe_int(data.get("last_reconfirmed_turn")),
                current_turn=current_turn,
                has_bundle=has_bundle,
                expectation_debt_score=data.get("expectation_debt_score"),
                obligation_pressure_score=data.get("obligation_pressure_score"),
            )
            dormancy_transition = derive_dormancy_transition(
                reference_turn=current_turn,
                state=data.get("state"),
                status=data.get("status"),
                deadline_turn=data.get("deadline_turn"),
                resolved_at_turn=data.get("resolved_at_turn"),
                last_salient_turn=data.get("last_salient_turn"),
                dormant_since_turn=data.get("dormant_since_turn"),
                compression_mode=data.get("compression_mode"),
                obligation_pressure=data.get("obligation_pressure_score"),
                expectation_debt=data.get("expectation_debt_score"),
                continuity_strength=data.get("continuity_contract_strength_score"),
                player_salience=data.get("player_salience_score"),
                expectation_salience=data.get("expectation_salience_score"),
                current_relevance_reason=data.get("current_relevance_reason"),
            )
            data["dormant_since_turn"] = dormancy_transition.dormant_since_turn
            data["persisted_dormancy_state"] = dormancy_transition.dormancy_state
            data["dormancy_transition"] = dormancy_transition.transition
            data["dormancy_transition_turn"] = dormancy_transition.transition_turn
            data["dormancy_reason_flags"] = list(dormancy_transition.reason_flags)
            fact_row.data = data
    for existing_key, row in existing_by_key.items():
        if existing_key in seen_bundle_keys:
            continue
        db.delete(row)


def _is_narrator_visible_knowledge_scope(raw_value: Any) -> bool:
    knowledge_scope = str(raw_value or "global").strip().lower()
    if knowledge_scope not in {"global", "public", "npc_private"}:
        knowledge_scope = "global"
    return knowledge_scope in {"global", "public"}


def _rebuild_entity_memory_rows(
    db: Session,
    *,
    session_id: uuid.UUID,
    anchor_object_ids: list[uuid.UUID],
    current_turn: int,
) -> None:
    for anchor_object_id in anchor_object_ids:
        anchor_row = _get_object(db, session_id, anchor_object_id)
        if anchor_row is None or str(anchor_row.type or "").startswith("__"):
            continue
        row = _find_memory_object_by_json_key(
            db,
            session_id=session_id,
            object_type=ENTITY_MEMORY_OBJECT_TYPE,
            json_key="anchor_object_id",
            value=str(anchor_object_id),
        )
        existing_data = dict(row.data or {}) if row is not None else {}
        callback_rows = _iter_callback_rows_for_anchors(
            db,
            session_id=session_id,
            anchor_object_ids=[anchor_object_id],
        )
        callback_payloads = [
            dict(row.data or {})
            for row in callback_rows
            if _callback_row_active(row)
            and _is_narrator_visible_knowledge_scope(dict(row.data or {}).get("knowledge_scope"))
        ]
        callback_payloads.sort(
            key=lambda payload: (
                float(payload.get("importance") or 0.0),
                _safe_int(payload.get("source_turn")) or 0,
            ),
            reverse=True,
        )
        pinned_facts = [
            _compact_fact_payload(payload)
            for payload in callback_payloads
            if str(payload.get("priority") or "").strip().lower() == "high"
        ][:ENTITY_HISTORY_MAX_BULLETS]
        open_threads = [
            _compact_fact_payload(payload)
            for payload in callback_payloads
            if str(payload.get("kind") or "").strip().lower() in HARD_CALLBACK_KINDS
        ][:ENTITY_HISTORY_MAX_BULLETS]
        relationship_facts = [
            _compact_fact_payload(payload)
            for payload in callback_payloads
            if str(payload.get("kind") or "").strip().lower() in {"relationship", "emotional_scene"}
        ][:ENTITY_HISTORY_MAX_BULLETS]
        sentimental_items = [
            _compact_fact_payload(payload)
            for payload in callback_payloads
            if str(payload.get("kind") or "").strip().lower() in {"gift", "trophy", "recurring_prop", "ownership"}
        ][:ENTITY_HISTORY_MAX_BULLETS]
        home_facts = [
            _compact_fact_payload(payload)
            for payload in callback_payloads
            if str(payload.get("kind") or "").strip().lower() in {"home_detail", "location_fact", "recurring_prop"}
        ][:ENTITY_HISTORY_MAX_BULLETS]
        last_material_turn = 0
        for payload in callback_payloads:
            source_turn = max(_safe_int(payload.get("source_turn")) or 0, 0)
            if source_turn > last_material_turn:
                last_material_turn = source_turn

        entity_data = {
            "memory_sync_version": MEMORY_SYNC_VERSION,
            "anchor_object_id": str(anchor_object_id),
            "anchor_type": str(anchor_row.type or ""),
            "anchor_name": str(anchor_row.name or ""),
            "state_bullets": _state_bullets_for_anchor(
                db,
                session_id=session_id,
                anchor_row=anchor_row,
            ),
            "pinned_facts": pinned_facts,
            "open_threads": open_threads,
            "relationship_facts": relationship_facts,
            "sentimental_items": sentimental_items,
            "home_facts": home_facts,
            "last_material_turn": last_material_turn,
            "updated_turn": current_turn,
        }
        if row is None:
            db.add(
                models.ObjectModel(
                    session_id=session_id,
                    type=ENTITY_MEMORY_OBJECT_TYPE,
                    name=f"entity_memory:{anchor_row.type}:{anchor_row.name}",
                    data=entity_data,
                )
            )
        else:
            row.name = f"entity_memory:{anchor_row.type}:{anchor_row.name}"
            row.data = entity_data


def _collect_anchor_ids_from_applied_ops(
    applied_ops: list[dict[str, Any]],
) -> list[uuid.UUID]:
    anchor_ids: list[uuid.UUID] = []
    for op in applied_ops:
        if not isinstance(op, dict):
            continue
        for key in ("object", "object_id", "scope_object_id", "player", "to", "from"):
            parsed = _coerce_uuid(op.get(key))
            if parsed is not None and parsed not in anchor_ids:
                anchor_ids.append(parsed)
        if str(op.get("op") or "").strip() in {"link.create", "link.close"}:
            for key in ("from_object_id", "to_object_id"):
                parsed = _coerce_uuid(op.get(key))
                if parsed is not None and parsed not in anchor_ids:
                    anchor_ids.append(parsed)
    return anchor_ids


def _format_entity_history(row: models.ObjectModel) -> dict[str, Any]:
    data = dict(row.data or {})
    payload = {
        "object_id": str(data.get("anchor_object_id") or ""),
        "name": str(data.get("anchor_name") or ""),
        "type": str(data.get("anchor_type") or ""),
        "state_bullets": list(data.get("state_bullets") or [])[:3],
        "pinned_facts": list(data.get("pinned_facts") or [])[:ENTITY_HISTORY_MAX_BULLETS],
        "open_threads": list(data.get("open_threads") or [])[:ENTITY_HISTORY_MAX_BULLETS],
        "relationship_facts": list(data.get("relationship_facts") or [])[:ENTITY_HISTORY_MAX_BULLETS],
        "sentimental_items": list(data.get("sentimental_items") or [])[:ENTITY_HISTORY_MAX_BULLETS],
        "home_facts": list(data.get("home_facts") or [])[:ENTITY_HISTORY_MAX_BULLETS],
        "last_material_turn": _safe_int(data.get("last_material_turn")),
    }
    return payload


def _infer_scene_mode(
    *,
    intent_tags: list[str] | None,
) -> str:
    return continuity_policy_domain.infer_scene_mode(intent_tags=intent_tags)


def _callback_similarity_score(
    *,
    db: Session,
    session_id: uuid.UUID,
    callback_row: models.ObjectModel,
    query_embedding: list[float] | None,
) -> float:
    score = 0.0
    if query_embedding is not None and USE_EMBEDDINGS:
        distance = db.execute(
            select(models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding))
            .where(
                models.ObjectEmbeddingModel.session_id == session_id,
                models.ObjectEmbeddingModel.object_id == callback_row.object_id,
                models.ObjectEmbeddingModel.namespace == CALLBACK_MEMORY_EMBED_NAMESPACE,
            )
            .limit(1)
        ).scalar_one_or_none()
        if distance is not None:
            score = max(1.0 - float(distance), 0.0)
    return round(score, 6)


def _trim_memory_rows(
    rows: list[dict[str, Any]],
    *,
    max_items: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    return continuity_policy_domain.trim_memory_rows(
        rows,
        max_items=max_items,
        max_tokens=max_tokens,
        count_json_tokens=_count_json_tokens,
    )


def _build_memory_context_blocks(
    db: Session,
    session_id: uuid.UUID,
    *,
    current_turn: int,
    player_object_id: uuid.UUID | None,
    current_zone_id: uuid.UUID | None,
    current_zone_name: str | None,
    user_input: str,
    recent_ai_text: str,
    intent_tags: list[str] | None,
    query_embedding: list[float] | None,
    relevant_npcs: list[dict[str, Any]],
    relevant_items: list[dict[str, Any]],
    relevant_quests: list[dict[str, Any]],
    relevant_factions: list[dict[str, Any]],
    explicit_anchor_object_ids: list[str],
) -> dict[str, Any]:
    anchor_object_ids: list[uuid.UUID] = []
    for raw_anchor_id in explicit_anchor_object_ids:
        parsed = _coerce_uuid(raw_anchor_id)
        if parsed is not None and parsed not in anchor_object_ids:
            anchor_object_ids.append(parsed)
    for raw_value in (player_object_id, current_zone_id):
        if raw_value is not None and raw_value not in anchor_object_ids:
            anchor_object_ids.append(raw_value)
    for source in (relevant_npcs[:2], relevant_quests[:1], relevant_items[:1], relevant_factions[:1]):
        for payload in source:
            parsed = _coerce_uuid(payload.get("object_id"))
            if parsed is not None and parsed not in anchor_object_ids:
                anchor_object_ids.append(parsed)
    anchor_object_ids = anchor_object_ids[: max(ENTITY_HISTORY_MAX_ITEMS + 3, 6)]

    relevant_npc_ids = {
        parsed
        for parsed in (_coerce_uuid(payload.get("object_id")) for payload in relevant_npcs)
        if parsed is not None
    }
    entity_history_rows: list[dict[str, Any]] = []
    hard_memory_candidates: list[tuple[float, dict[str, Any]]] = []
    for anchor_object_id in anchor_object_ids[:ENTITY_HISTORY_MAX_ITEMS]:
        row = _find_memory_object_by_json_key(
            db,
            session_id=session_id,
            object_type=ENTITY_MEMORY_OBJECT_TYPE,
            json_key="anchor_object_id",
            value=str(anchor_object_id),
        )
        if row is None:
            continue
        history_payload = _format_entity_history(row)
        entity_history_rows.append(history_payload)
        history_object_id = _coerce_uuid(history_payload.get("object_id"))
        is_player_anchor = history_object_id is not None and history_object_id == player_object_id
        is_zone_anchor = history_object_id is not None and history_object_id == current_zone_id
        is_relevant_npc_anchor = history_object_id is not None and history_object_id in relevant_npc_ids
        for fact in list(history_payload.get("pinned_facts") or []):
            hard_memory_candidates.append(
                (
                    3.0,
                    {
                        "anchor_object_id": history_payload.get("object_id"),
                        "anchor_name": history_payload.get("name"),
                        "kind": fact.get("kind"),
                        "text": fact.get("text"),
                        "priority": fact.get("priority"),
                        "source_turn": fact.get("source_turn"),
                    },
                )
            )
        for fact in list(history_payload.get("open_threads") or []):
            hard_memory_candidates.append(
                (
                    2.8,
                    {
                        "anchor_object_id": history_payload.get("object_id"),
                        "anchor_name": history_payload.get("name"),
                        "kind": fact.get("kind"),
                        "text": fact.get("text"),
                        "priority": fact.get("priority"),
                        "source_turn": fact.get("source_turn"),
                    },
                )
            )
        for fact in list(history_payload.get("relationship_facts") or []):
            if not (is_player_anchor or is_relevant_npc_anchor):
                continue
            hard_memory_candidates.append(
                (
                    2.45 if is_relevant_npc_anchor else 2.35,
                    {
                        "anchor_object_id": history_payload.get("object_id"),
                        "anchor_name": history_payload.get("name"),
                        "kind": fact.get("kind"),
                        "text": fact.get("text"),
                        "priority": fact.get("priority"),
                        "source_turn": fact.get("source_turn"),
                    },
                )
            )
        for fact in list(history_payload.get("sentimental_items") or []):
            if not (is_player_anchor or is_relevant_npc_anchor):
                continue
            hard_memory_candidates.append(
                (
                    2.2 if is_player_anchor else 2.0,
                    {
                        "anchor_object_id": history_payload.get("object_id"),
                        "anchor_name": history_payload.get("name"),
                        "kind": fact.get("kind"),
                        "text": fact.get("text"),
                        "priority": fact.get("priority"),
                        "source_turn": fact.get("source_turn"),
                    },
                )
            )
        for fact in list(history_payload.get("home_facts") or []):
            if not is_zone_anchor:
                continue
            hard_memory_candidates.append(
                (
                    2.25,
                    {
                        "anchor_object_id": history_payload.get("object_id"),
                        "anchor_name": history_payload.get("name"),
                        "kind": fact.get("kind"),
                        "text": fact.get("text"),
                        "priority": fact.get("priority"),
                        "source_turn": fact.get("source_turn"),
                    },
                )
            )

    hard_memory_candidates.sort(
        key=lambda item: (
            item[0],
            _safe_int(item[1].get("source_turn")) or 0,
        ),
        reverse=True,
    )
    hard_memory_rows = [payload for _score, payload in hard_memory_candidates]

    scene_mode = _infer_scene_mode(intent_tags=intent_tags)
    callback_candidates = _iter_callback_rows_for_anchors(
        db,
        session_id=session_id,
        anchor_object_ids=anchor_object_ids,
    )
    use_callback_embedding_batch = query_embedding is not None and USE_EMBEDDINGS
    callback_similarity_by_object_id = (
        _callback_similarity_scores_for_rows(
            db=db,
            session_id=session_id,
            callback_rows=callback_candidates,
            query_embedding=query_embedding,
        )
        if use_callback_embedding_batch
        else {}
    )
    selected_callbacks: list[tuple[float, dict[str, Any]]] = []
    strong_selected = 0
    soft_selected = 0
    for row in callback_candidates:
        data = dict(row.data or {})
        if not _callback_row_active(row):
            continue
        if not _is_narrator_visible_knowledge_scope(data.get("knowledge_scope")):
            record_callback_decision("scope_filter")
            continue
        if not bool(data.get("callback_candidate", False)):
            continue
        source_turn = max(_safe_int(data.get("source_turn")) or 0, 0)
        cooldown_until_turn = _safe_int(data.get("cooldown_until_turn"))
        if cooldown_until_turn is not None and cooldown_until_turn > current_turn:
            record_callback_decision("cooldown")
            continue
        soft_callback = bool(data.get("soft_callback", True))
        if scene_mode == "high_tension" and soft_callback:
            record_callback_decision("scene_filter")
            continue
        anchor_hits = sum(
            1
            for raw_anchor_id in list(data.get("anchor_object_ids") or [])
            if _coerce_uuid(raw_anchor_id) in anchor_object_ids
        )
        if anchor_hits <= 0:
            continue
        callback_anchor_ids = {
            parsed
            for parsed in (_coerce_uuid(raw_value) for raw_value in list(data.get("anchor_object_ids") or []))
            if parsed is not None
        }
        age_bonus = min(max(current_turn - source_turn, 0) / 600.0, 0.25)
        last_recalled_turn = _safe_int(data.get("last_recalled_turn"))
        if last_recalled_turn is None:
            recent_penalty = 0.0
        else:
            turns_since_recall = max(current_turn - last_recalled_turn, 0)
            recent_penalty = max(25 - min(turns_since_recall, 25), 0) / 100.0
        if use_callback_embedding_batch:
            relevance = callback_similarity_by_object_id.get(str(getattr(row, "object_id", "")), 0.0)
        else:
            relevance = _callback_similarity_score(
                db=db,
                session_id=session_id,
                callback_row=row,
                query_embedding=query_embedding,
            )
        kind = str(data.get("kind") or "").strip().lower()
        zone_resurfacing_boost = 0.0
        if (
            current_zone_id is not None
            and current_zone_id in callback_anchor_ids
            and kind in {"home_detail", "location_fact", "recurring_prop"}
        ):
            zone_resurfacing_boost = 0.24
        npc_revisit_boost = 0.0
        if relevant_npc_ids.intersection(callback_anchor_ids) and kind in {
            "relationship",
            "emotional_scene",
            "promise",
            "debt",
            "betrayal",
            "injury",
        }:
            npc_revisit_boost = 0.28
        player_emotional_boost = 0.0
        if (
            player_object_id is not None
            and player_object_id in callback_anchor_ids
            and kind == "emotional_scene"
        ):
            player_emotional_boost = 0.24
        player_sentimental_boost = 0.0
        if (
            player_object_id is not None
            and player_object_id in callback_anchor_ids
            and kind in {"gift", "trophy", "ownership"}
        ):
            player_sentimental_boost = 0.22
        if relevance <= 0.0:
            if zone_resurfacing_boost > 0.0:
                relevance = max(relevance, 0.12)
            if npc_revisit_boost > 0.0:
                relevance = max(relevance, 0.14)
            if player_emotional_boost > 0.0:
                relevance = max(relevance, 0.14)
            if player_sentimental_boost > 0.0:
                relevance = max(relevance, 0.12)
        score = (
            min(anchor_hits, 3) * 0.2
            + relevance * 0.25
            + float(data.get("importance") or 0.0) * 0.2
            + age_bonus
            + float(data.get("surprise_weight") or 0.0) * 0.12
            + zone_resurfacing_boost
            + npc_revisit_boost
            + player_emotional_boost
            + player_sentimental_boost
            - recent_penalty
        )
        payload = {
            "fact_key": str(data.get("fact_key") or ""),
            "kind": kind,
            "text": _truncate_text(str(data.get("text") or ""), 180),
            "priority": str(data.get("priority") or "med"),
            "source_turn": source_turn,
            "anchor_hits": anchor_hits,
            "anchor_object_ids": list(data.get("anchor_object_ids") or []),
            "relevance": round(relevance, 6),
            "scene_mode": scene_mode,
            "callback_strength": "soft" if soft_callback else "strong",
            "confidence": _coerce_priority_score(str(data.get("priority") or "med"))
            if data.get("confidence") is None
            else float(data.get("confidence")),
            "durability": float(data.get("durability") or 0.0),
            "emotional_weight": float(data.get("emotional_weight") or 0.0),
            "obligation_weight": float(data.get("obligation_weight") or 0.0),
            "sentimental_weight": float(data.get("sentimental_weight") or 0.0),
            "routine_weight": float(data.get("routine_weight") or 0.0),
            "zone_resurfacing_boost": round(zone_resurfacing_boost, 6),
            "npc_revisit_boost": round(npc_revisit_boost, 6),
            "player_emotional_boost": round(player_emotional_boost, 6),
            "player_sentimental_boost": round(player_sentimental_boost, 6),
        }
        selected_callbacks.append((round(score, 6), payload))

    selected_callbacks.sort(key=lambda item: (item[0], item[1].get("source_turn") or 0), reverse=True)
    callback_rows: list[dict[str, Any]] = []
    for _score_value, payload in selected_callbacks:
        if len(callback_rows) >= CALLBACK_MEMORY_MAX_ITEMS:
            break
        if payload["callback_strength"] == "strong":
            if strong_selected >= 1:
                continue
            strong_selected += 1
        else:
            if soft_selected >= 1:
                continue
            soft_selected += 1
        callback_rows.append(payload)
        record_callback_decision("selected")

    return {
        "scene_mode": scene_mode,
        "hard_memory": _trim_memory_rows(
            hard_memory_rows,
            max_items=HARD_MEMORY_MAX_ITEMS,
            max_tokens=HARD_MEMORY_MAX_TOKENS,
        ),
        "entity_histories": _trim_memory_rows(
            entity_history_rows,
            max_items=ENTITY_HISTORY_MAX_ITEMS,
            max_tokens=ENTITY_HISTORY_MAX_TOKENS,
        ),
        "callback_memories": _trim_memory_rows(
            callback_rows,
            max_items=CALLBACK_MEMORY_MAX_ITEMS,
            max_tokens=CALLBACK_MEMORY_MAX_TOKENS,
        ),
        "current_zone_name": current_zone_name,
    }


def _mark_callback_recalled(
    db: Session,
    *,
    session_id: uuid.UUID,
    fact_keys: list[str],
    current_turn: int,
) -> None:
    if not fact_keys:
        return
    rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == CALLBACK_MEMORY_OBJECT_TYPE,
            models.ObjectModel.data["fact_key"].astext.in_(fact_keys),
        )
    ).scalars().all()
    for row in rows:
        data = dict(row.data or {})
        data["last_recalled_turn"] = current_turn
        data["recall_count"] = max(_safe_int(data.get("recall_count")) or 0, 0) + 1
        soft_callback = bool(data.get("soft_callback", True))
        data["cooldown_until_turn"] = current_turn + (
            CALLBACK_SOFT_COOLDOWN_TURNS if soft_callback else CALLBACK_STRONG_COOLDOWN_TURNS
        )
        row.data = data


def _sync_turn_memory_artifacts(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    embeddings_by_fact_key: dict[str, list[float]] | None = None,
) -> list[_ResolvedDurableFact]:
    turn_row = db.get(models.TurnModel, (session_id, turn_index))
    if turn_row is None:
        return []
    session_row = db.get(models.SessionModel, session_id)
    if session_row is None:
        return []
    applied_ops, durable_facts = _resolve_turn_memory_facts(
        db,
        session_row=session_row,
        turn_row=turn_row,
        turn_index=turn_index,
    )
    callback_facts = _resolve_callback_sync_facts(
        db,
        session_id=session_id,
        durable_facts=durable_facts,
    )
    touched_anchor_ids = _collect_anchor_ids_from_applied_ops(applied_ops)
    for fact in callback_facts:
        for anchor_object_id in fact.anchor_object_ids:
            if anchor_object_id not in touched_anchor_ids:
                touched_anchor_ids.append(anchor_object_id)

    if callback_facts:
        updated_anchor_ids = _upsert_callback_memory_rows(
            db,
            session_id=session_id,
            durable_facts=callback_facts,
            embeddings_by_fact_key=embeddings_by_fact_key,
        )
        for anchor_object_id in updated_anchor_ids:
            if anchor_object_id not in touched_anchor_ids:
                touched_anchor_ids.append(anchor_object_id)

    if touched_anchor_ids:
        invalidated_anchor_ids = _invalidate_callback_rows(
            db,
            session_id=session_id,
            anchor_object_ids=touched_anchor_ids,
        )
        for anchor_object_id in invalidated_anchor_ids:
            if anchor_object_id not in touched_anchor_ids:
                touched_anchor_ids.append(anchor_object_id)
        _rebuild_entity_memory_rows(
            db,
            session_id=session_id,
            anchor_object_ids=touched_anchor_ids,
            current_turn=turn_index,
        )
    _rebuild_memory_bundle_rows(
        db,
        session_id=session_id,
        current_turn=turn_index,
    )
    return callback_facts


def _run_turn_memory_sync_outbox_event(
    *,
    session_id: uuid.UUID,
    turn_index: int,
) -> None:
    callback_facts: list[_ResolvedDurableFact] = []
    write_db = SessionLocal()
    try:
        with write_db.begin():
            callback_facts = _sync_turn_memory_artifacts(
                write_db,
                session_id=session_id,
                turn_index=turn_index,
                embeddings_by_fact_key={},
            )
    finally:
        if write_db.in_transaction():
            write_db.rollback()
        write_db.close()

    if not USE_EMBEDDINGS:
        return

    callback_candidates = [fact for fact in callback_facts if fact.callback_candidate]
    if not callback_candidates:
        return

    embeddings_by_fact_key: dict[str, list[float]] = {}
    text_to_fact_keys: dict[str, list[str]] = {}
    for fact in callback_candidates:
        text_to_fact_keys.setdefault(fact.narrative_recall_summary, []).append(fact.fact_key)
    unique_texts = list(text_to_fact_keys.keys())
    if not unique_texts:
        return

    try:
        vectors = _maybe_embed_texts(
            unique_texts,
            instruction=CALLBACK_MEMORY_EMBED_INSTRUCTION,
        )
        for text, vector in zip(unique_texts, vectors, strict=True):
            for fact_key in text_to_fact_keys.get(text, []):
                embeddings_by_fact_key[fact_key] = vector
    except Exception:
        logger.warning(
            "turn memory sync embedding enrichment failed for session %s turn %s",
            session_id,
            turn_index,
            exc_info=True,
        )
        return

    if not embeddings_by_fact_key:
        return

    enrich_db = SessionLocal()
    try:
        with enrich_db.begin():
            _upsert_callback_memory_embeddings(
                enrich_db,
                session_id=session_id,
                durable_facts=callback_candidates,
                embeddings_by_fact_key=embeddings_by_fact_key,
            )
    finally:
        if enrich_db.in_transaction():
            enrich_db.rollback()
        enrich_db.close()


def _enqueue_turn_memory_sync_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    dedupe: bool = True,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_TURN_MEMORY_SYNC,
        payload={},
        session_id=session_id,
        turn_index=turn_index,
        trace_id=get_trace_id(),
        dedupe_key=f"turn_memory_sync:{session_id}:{turn_index}" if dedupe else None,
    )


__all__ = [
    "ENTITY_MEMORY_OBJECT_TYPE",
    "CALLBACK_MEMORY_OBJECT_TYPE",
    "CALLBACK_MEMORY_EMBED_NAMESPACE",
    "_build_memory_context_blocks",
    "_enqueue_turn_memory_sync_event",
    "_mark_callback_recalled",
    "_run_turn_memory_sync_outbox_event",
    "_sync_turn_memory_artifacts",
]
