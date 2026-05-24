from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Literal

from fastapi import HTTPException, status
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models, schemas
from .constants import (
    LOCATED_IN_LINK_TYPE,
    QUEST_TERMINAL_STATUSES,
    SESSION_PLAYER_REF,
    TRACKING_QUEST_LINK_TYPE,
)
from .crud_consequences import MAX_CONSEQUENCE_WINDOW_SPAN
from .crud_context import MAX_WORLD_CONSTITUTION_CHARS, _build_turn_context_pack, _serialize_patch_ops
from .crud_shared import (
    PatchValidationResult,
    TurnPlanResult,
    _get_session_player_object_id,
    _memory_candidates_to_durable_facts,
    _normalize_json_preview,
    _normalize_json_preview_by_tokens,
    _rollback_read_only_autobegin_transaction,
    _truncate_text,
)
from .domain import geography_policy as geography_policy_domain
from .db import (
    OPENROUTER_CHAT_MODEL,
    TURN_CONTEXT_MAX_TOKENS,
    USE_CONSEQUENCES,
    USE_PROMPT_CACHE_LAYOUT,
    USE_SPLIT_NARRATOR_PATCHES,
    USE_STATE_FIRST_PIPELINE,
    OPENROUTER_API_KEY,
    OPENROUTER_LIBRARIAN_MODEL,
    OPENROUTER_NARRATOR_MODEL,
)
from .llm_telemetry import telemetry_context
from .llm import openrouter_chat
from ..db import (
    OPENROUTER_NARRATOR_MODEL,
    OPENROUTER_LIBRARIAN_MODEL,
    OPENROUTER_ASSISTANT_MODEL,
)
from .observability import record_canon_repair
from .prompt_registry import resolve_system_prompt
from .strings import FALLBACK_AI_UNAVAILABLE, FALLBACK_NO_RESPONSE

MAX_PATCH_OPS = 64

PATCH_OP_LIST_ADAPTER = TypeAdapter(list[schemas.PatchOp])
logger = logging.getLogger(__name__)

_NARRATOR_CONTEXT_KEY_ALIASES: dict[str, str] = {
    "session_summaries": "summaries",
    "recent_turns": "turns",
    "hard_memory": "hard_mem",
    "entity_histories": "entity_hist",
    "callback_memories": "callbacks",
    "relevant_npcs": "npcs",
    "relevant_items": "items",
    "orphaned_items": "orphans",
    "relevant_quests": "quests",
    "archived_quest_recall": "archived_quests",
    "relevant_factions": "factions",
    "relevant_links": "links",
    "relevant_claims": "claims",
    "zone_claims": "zclaims",
    "latent_consequences": "latent",
    "structural_signals": "signals",
    "player_inventory": "inv",
    "player_location_history": "loc_hist",
}
_PROMPT_CACHE_STATIC_KEYS = {
    "world_constitution_for_system",
    "world_prompt_for_system",
    "narrative_spine_for_system",
    "session_summaries",
    "summaries",
    "has_world_constitution",
}
_DESYNC_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-_'][0-9A-Za-zА-Яа-яЁё]+)*")
_TMP_REF_RE = re.compile(r"^tmp:[A-Za-z0-9_-]+$")
_DESYNC_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "but",
        "from",
        "in",
        "inside",
        "near",
        "of",
        "on",
        "or",
        "out",
        "the",
        "through",
        "to",
        "toward",
        "towards",
        "with",
        "без",
        "в",
        "во",
        "возле",
        "вы",
        "герой",
        "где",
        "его",
        "ее",
        "её",
        "и",
        "из",
        "игрок",
        "их",
        "к",
        "ко",
        "на",
        "но",
        "о",
        "об",
        "около",
        "он",
        "она",
        "они",
        "от",
        "перед",
        "по",
        "под",
        "при",
        "рядом",
        "с",
        "со",
        "ты",
        "у",
        "что",
        "you",
        "your",
    }
)
_SCENE_ENTITY_ROLE_ALIASES = {
    "current_location": "location",
    "landmark": "mentioned",
}
_SCENE_ENTITY_SALIENCE_ALIASES = {
    "low": 0.25,
    "med": 0.5,
    "medium": 0.5,
    "high": 0.9,
}
_DESYNC_CONTEXT_NAME_KEYS = (
    "relevant_npcs",
    "zone_npcs",
    "relevant_items",
    "orphaned_items",
    "relevant_factions",
    "relevant_quests",
)
_DESYNC_EVENT_NAME_KEYS = ("name", "victim", "victim_name", "target", "target_name", "entity", "entity_name", "npc_name")
_DESYNC_DEATH_STATUS_VALUES = frozenset(
    {
        "dead",
        "deceased",
        "killed",
        "slain",
        "inactive",
        "defeated",
        "destroyed",
        "мертв",
        "мертва",
        "убит",
        "убита",
        "погиб",
        "погибла",
        "неактивен",
        "неактивна",
    }
)
_DESYNC_RU_INFLECTION_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "его",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "иях",
    "ях",
    "ах",
    "ов",
    "ев",
    "ам",
    "ям",
    "ом",
    "ем",
    "ой",
    "ей",
    "ою",
    "ею",
    "ую",
    "юю",
    "ая",
    "яя",
    "ое",
    "ее",
    "ие",
    "ые",
    "ий",
    "ый",
    "а",
    "я",
    "е",
    "и",
    "ы",
    "о",
    "у",
    "ю",
)


def _coerce_choice(item: Any, idx: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {
            "id": str(idx + 1),
            "text": text,
            "intent": "act",
            "risk": "med",
        }
    if not isinstance(item, dict):
        return None

    text_value = str(item.get("text") or item.get("label") or "").strip()
    if not text_value:
        return None
    risk_raw = str(item.get("risk") or "med").strip().lower()
    risk = risk_raw if risk_raw in {"low", "med", "high"} else "med"
    return {
        "id": str(item.get("id") or idx + 1),
        "text": text_value,
        "intent": str(item.get("intent") or "act"),
        "risk": risk,
    }


def _coerce_memory_candidate(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    layer = str(item.get("layer") or "").strip().lower()
    if layer not in {"event", "fact"}:
        return None

    kind = str(item.get("kind") or "").strip().lower()
    if kind not in {
        "ownership",
        "home_detail",
        "gift",
        "trophy",
        "debt",
        "promise",
        "betrayal",
        "injury",
        "quest_milestone",
        "recurring_prop",
        "emotional_scene",
        "relationship",
        "decision",
        "location_fact",
        "other",
    }:
        return None

    search_recall_summary = str(item.get("search_recall_summary") or "").strip()
    narrative_recall_summary = str(item.get("narrative_recall_summary") or "").strip()
    if not search_recall_summary or not narrative_recall_summary:
        return None

    priority = str(item.get("priority") or "med").strip().lower()
    if priority not in {"low", "med", "high"}:
        priority = "med"
    knowledge_scope = str(item.get("knowledge_scope") or "global").strip().lower()
    if knowledge_scope not in {"global", "public", "npc_private"}:
        knowledge_scope = "global"

    callback_strength = str(item.get("callback_strength") or "none").strip().lower()
    if callback_strength not in {"none", "soft", "strong"}:
        callback_strength = "none"

    durability = _coerce_turn_weight(item.get("durability"))
    emotional_weight = _coerce_turn_weight(item.get("emotional_weight"))
    obligation_weight = _coerce_turn_weight(item.get("obligation_weight"))
    sentimental_weight = _coerce_turn_weight(item.get("sentimental_weight"))
    routine_weight = _coerce_turn_weight(item.get("routine_weight"))

    anchors: list[str] = []
    raw_anchors = item.get("anchors")
    if isinstance(raw_anchors, list):
        anchors = [value for value in (_coerce_ref(value) for value in raw_anchors[:6]) if value]
    scene_refs: list[str] = []
    raw_scene_refs = item.get("scene_refs")
    if isinstance(raw_scene_refs, list):
        scene_refs = [value for value in (_coerce_ref(value) for value in raw_scene_refs[:6]) if value]
    context_refs: list[str] = []
    raw_context_refs = item.get("context_refs")
    if isinstance(raw_context_refs, list):
        context_refs = [value for value in (_coerce_ref(value) for value in raw_context_refs[:6]) if value]
    player_salience = str(item.get("player_salience") or "none").strip().lower()
    if player_salience not in {"none", "low", "med", "high"}:
        player_salience = "none"
    expectation_salience = str(item.get("expectation_salience") or "none").strip().lower()
    if expectation_salience not in {"none", "low", "med", "high"}:
        expectation_salience = "none"
    continuity_contract_strength = str(item.get("continuity_contract_strength") or "none").strip().lower()
    if continuity_contract_strength not in {"none", "low", "med", "high"}:
        continuity_contract_strength = "none"
    event_role = str(item.get("event_role") or "supporting").strip().lower()
    if event_role not in {"scene", "supporting"}:
        event_role = "supporting"
    event_outcome = str(item.get("event_outcome") or "asserted").strip().lower()
    if event_outcome not in {"asserted", "failed_attempt", "counterfactual"}:
        event_outcome = "asserted"

    requires_commit = bool(item.get("requires_commit", False))
    if layer != "event":
        scene_refs = []
        event_role = "supporting"
        event_outcome = "asserted"
        requires_commit = False
    elif event_role == "scene" and not scene_refs:
        inferred_scene_refs: list[str] = []
        for candidate in (
            _coerce_ref(item.get("actor_ref")),
            _coerce_ref(item.get("counterparty_ref")),
            _coerce_ref(item.get("object_ref")),
            _coerce_ref(item.get("location_ref")),
            _coerce_ref(item.get("quest_ref")),
            *anchors,
            *context_refs,
        ):
            if not candidate or candidate in inferred_scene_refs:
                continue
            inferred_scene_refs.append(candidate)
            if len(inferred_scene_refs) >= 6:
                break
        scene_refs = inferred_scene_refs

    result: dict[str, Any] = {
        "layer": layer,
        "kind": kind,
        "search_recall_summary": _truncate_text(search_recall_summary, 200),
        "narrative_recall_summary": _truncate_text(narrative_recall_summary, 200),
        "priority": priority,
        "actor_ref": _coerce_ref(item.get("actor_ref")),
        "counterparty_ref": _coerce_ref(item.get("counterparty_ref")),
        "object_ref": _coerce_ref(item.get("object_ref")),
        "location_ref": _coerce_ref(item.get("location_ref")),
        "quest_ref": _coerce_ref(item.get("quest_ref")),
        "context_refs": context_refs,
        "anchors": anchors,
        "scene_refs": scene_refs,
        "event_role": event_role,
        "event_outcome": event_outcome,
        "player_salience": player_salience,
        "expectation_salience": expectation_salience,
        "continuity_contract_strength": continuity_contract_strength,
        "knowledge_scope": knowledge_scope,
        "callback_strength": callback_strength,
        "durability": 0.5 if durability is None else durability,
        "emotional_weight": 0.0 if emotional_weight is None else emotional_weight,
        "obligation_weight": 0.0 if obligation_weight is None else obligation_weight,
        "sentimental_weight": 0.0 if sentimental_weight is None else sentimental_weight,
        "routine_weight": 0.0 if routine_weight is None else routine_weight,
        "requires_commit": requires_commit,
    }
    canonical_fact_source = item.get("canonical_fact")
    if isinstance(canonical_fact_source, dict):
        canonical_fact_seed = dict(canonical_fact_source)
    else:
        canonical_fact_seed = {}
    if canonical_fact_seed:
        for field_name in (
            "kind",
            "search_recall_summary",
            "narrative_recall_summary",
            "identity_text",
            "priority",
        ):
            if canonical_fact_seed.get(field_name) in (None, "", []):
                canonical_fact_seed[field_name] = item.get(field_name)
        for field_name in (
            "actor_ref",
            "counterparty_ref",
            "object_ref",
            "location_ref",
            "quest_ref",
            "context_refs",
            "relationship_type",
            "knowledge_scope",
            "player_salience",
            "expectation_salience",
            "continuity_contract_strength",
        ):
            if canonical_fact_seed.get(field_name) in (None, "", []):
                canonical_fact_seed[field_name] = result.get(field_name)
    canonical_fact = _coerce_durable_fact(canonical_fact_seed or item.get("canonical_fact"))
    if canonical_fact is not None:
        result["canonical_fact"] = canonical_fact
    return result


def _coerce_durable_fact(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    kind = str(item.get("kind") or "").strip().lower()
    if kind not in {
        "ownership",
        "home_detail",
        "gift",
        "trophy",
        "debt",
        "promise",
        "betrayal",
        "injury",
        "quest_milestone",
        "recurring_prop",
        "emotional_scene",
        "relationship",
        "decision",
        "location_fact",
    }:
        return None

    search_recall_summary = str(item.get("search_recall_summary") or "").strip()
    narrative_recall_summary = str(item.get("narrative_recall_summary") or "").strip()
    if not search_recall_summary or not narrative_recall_summary:
        return None
    identity_text = str(item.get("identity_text") or "").strip()
    if not identity_text:
        return None

    priority = str(item.get("priority") or "med").strip().lower()
    if priority not in {"low", "med", "high"}:
        priority = "med"

    knowledge_scope = str(item.get("knowledge_scope") or "global").strip().lower()
    if knowledge_scope not in {"global", "public", "npc_private"}:
        knowledge_scope = "global"
    player_salience = str(item.get("player_salience") or "none").strip().lower()
    if player_salience not in {"none", "low", "med", "high"}:
        player_salience = "none"
    expectation_salience = str(item.get("expectation_salience") or "none").strip().lower()
    if expectation_salience not in {"none", "low", "med", "high"}:
        expectation_salience = "none"
    continuity_contract_strength = str(item.get("continuity_contract_strength") or "none").strip().lower()
    if continuity_contract_strength not in {"none", "low", "med", "high"}:
        continuity_contract_strength = "none"

    raw_context_refs = item.get("context_refs")
    context_refs: list[Any] = []
    if isinstance(raw_context_refs, list):
        context_refs = [value for value in (_coerce_ref(raw_value) for raw_value in raw_context_refs[:6]) if value]

    actor_ref = _coerce_ref(item.get("actor_ref"))
    counterparty_ref = _coerce_ref(item.get("counterparty_ref"))
    object_ref = _coerce_ref(item.get("object_ref"))
    location_ref = _coerce_ref(item.get("location_ref"))
    quest_ref = _coerce_ref(item.get("quest_ref"))
    relationship_type: str | None = None
    if kind == "relationship":
        raw_relationship_type = item.get("relationship_type")
        text_relationship_type = str(raw_relationship_type or "").strip().lower()
        if text_relationship_type:
            relationship_type = text_relationship_type
    if kind in {"promise", "debt", "betrayal"}:
        if actor_ref is None or counterparty_ref is None:
            return None
    elif kind == "gift":
        if object_ref is None or counterparty_ref is None:
            return None
        if location_ref is not None or quest_ref is not None:
            return None
    elif kind in {"ownership", "trophy"}:
        if object_ref is None or counterparty_ref is None:
            return None
        if actor_ref is not None or location_ref is not None or quest_ref is not None:
            return None
    elif kind == "relationship":
        if actor_ref is None or counterparty_ref is None or relationship_type is None:
            return None
        if object_ref is not None or location_ref is not None or quest_ref is not None:
            return None
    elif kind in {"home_detail", "recurring_prop", "location_fact"}:
        if actor_ref is None and object_ref is None and location_ref is None:
            return None
        if counterparty_ref is not None or quest_ref is not None:
            return None
    elif kind == "quest_milestone":
        if quest_ref is None:
            return None
        if actor_ref is not None and kind not in {"quest_milestone"}:
            return None
        if counterparty_ref is not None or object_ref is not None or location_ref is not None:
            return None
    else:
        if actor_ref is None:
            return None
        if counterparty_ref is not None or object_ref is not None or location_ref is not None or quest_ref is not None:
            return None
    return {
        "kind": kind,
        "search_recall_summary": _truncate_text(search_recall_summary, 240),
        "narrative_recall_summary": _truncate_text(narrative_recall_summary, 240),
        "identity_text": _truncate_text(identity_text, 160),
        "state": str(item.get("state") or "active").strip().lower() or "active",
        "priority": priority,
        "actor_ref": actor_ref,
        "counterparty_ref": counterparty_ref,
        "object_ref": object_ref,
        "location_ref": location_ref,
        "quest_ref": quest_ref,
        "context_refs": context_refs[:6],
        "relationship_type": relationship_type,
        "callback_candidate": bool(item.get("callback_candidate", True)),
        "knowledge_scope": knowledge_scope,
        "player_salience": player_salience,
        "expectation_salience": expectation_salience,
        "continuity_contract_strength": continuity_contract_strength,
        "independent_evidence_count": max(int(item.get("independent_evidence_count") or 0), 0),
        "repetition_count": max(int(item.get("repetition_count") or 0), 0),
    }


def _coerce_semantic_event(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    event_type = str(item.get("type") or item.get("event") or item.get("kind") or "").strip().lower()
    if event_type not in {
        "move",
        "death",
        "promise",
        "debt",
        "gift",
        "relationship_shift",
        "item_transfer",
        "other",
    }:
        return None

    polarity = str(item.get("polarity") or item.get("status") or "asserted").strip().lower()
    if polarity not in {"asserted", "negated", "attempted", "hypothetical", "uncertain"}:
        polarity = "asserted"
    requires_patch_raw = item.get("requires_patch")
    if not isinstance(requires_patch_raw, bool):
        for alias in ("requires_persistence", "requires_state_change", "needs_patch"):
            alias_value = item.get(alias)
            if isinstance(alias_value, bool):
                requires_patch_raw = alias_value
                break
    if not isinstance(requires_patch_raw, bool):
        return None

    result: dict[str, Any] = {
        "type": event_type,
        "polarity": polarity,
        "requires_patch": requires_patch_raw,
    }
    for field_name, aliases in {
        "subject": ("subject", "actor", "who"),
        "object": ("object", "patient", "item"),
        "source": ("source", "from", "origin"),
        "target": ("target", "to", "destination"),
    }.items():
        for alias in aliases:
            ref = _coerce_ref(item.get(alias))
            if ref:
                result[field_name] = ref
                break

    for field_name, aliases in {
        "subject_hint": ("subject_hint", "actor_hint", "who_hint"),
        "object_hint": ("object_hint", "item_hint", "patient_hint"),
        "source_hint": ("source_hint", "from_hint", "origin_hint"),
        "target_hint": ("target_hint", "to_hint", "destination_hint"),
        "note": ("note", "text", "summary"),
    }.items():
        text_value = ""
        for alias in aliases:
            text_value = str(item.get(alias) or "").strip()
            if text_value:
                break
        if text_value:
            limit = 180 if field_name == "note" else 120
            result[field_name] = _truncate_text(text_value, limit)

    if len(result) <= 2:
        return None
    return result


def _coerce_scene_entity(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    name = str(item.get("name") or item.get("label") or item.get("text") or "").strip()
    if not name:
        return None

    entity_type = str(item.get("entity_type") or item.get("type") or "other").strip().lower()
    if entity_type not in {"player", "npc", "zone", "item", "quest", "faction", "other"}:
        entity_type = "other"

    role = str(item.get("role") or item.get("kind") or "other").strip().lower()
    role = _SCENE_ENTITY_ROLE_ALIASES.get(role, role)
    if role not in {"focus", "speaker", "interlocutor", "target", "observer", "location", "mentioned", "other"}:
        role = "other"

    salience_raw = item.get("salience", 0.5)
    salience = 0.5
    if not isinstance(salience_raw, bool) and isinstance(salience_raw, (int, float)):
        salience = round(min(max(float(salience_raw), 0.0), 1.0), 6)
    elif isinstance(salience_raw, str):
        salience = _SCENE_ENTITY_SALIENCE_ALIASES.get(salience_raw.strip().lower(), 0.5)

    result: dict[str, Any] = {
        "name": _truncate_text(name, 120),
        "entity_type": entity_type,
        "role": role,
        "referent_candidate": bool(item.get("referent_candidate", False)),
        "salience": salience,
    }
    ref = _coerce_ref(item.get("ref") or item.get("object_id") or item.get("id"))
    if ref:
        result["ref"] = ref
    return result


def _coerce_consequence_seed(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    seed_id = str(item.get("id") or "").strip()
    text = str(item.get("text") or "").strip()
    if not seed_id or not text:
        return None

    priority = str(item.get("priority") or "med").strip().lower()
    if priority not in {"low", "med", "high"}:
        priority = "med"

    earliest_raw = item.get("earliest_turn")
    latest_raw = item.get("latest_turn")
    if not isinstance(earliest_raw, int):
        return None
    if not isinstance(latest_raw, int):
        return None
    if latest_raw < earliest_raw:
        return None

    max_shows_raw = item.get("max_shows", 2)
    if isinstance(max_shows_raw, bool):
        max_shows_raw = 2
    if not isinstance(max_shows_raw, int):
        max_shows_raw = 2
    max_shows = max(max_shows_raw, 0)

    shows_raw = item.get("shows", 0)
    if isinstance(shows_raw, bool):
        shows_raw = 0
    if not isinstance(shows_raw, int):
        shows_raw = 0
    shows = max(shows_raw, 0)

    last_shown_turn_raw = item.get("last_shown_turn")
    last_shown_turn: int | None = last_shown_turn_raw if isinstance(last_shown_turn_raw, int) else None

    raw_anchor_ids = item.get("anchor_object_ids")
    anchor_object_ids: list[str] = []
    if isinstance(raw_anchor_ids, list):
        for raw in raw_anchor_ids:
            text_value = str(raw).strip()
            if text_value:
                anchor_object_ids.append(text_value)

    created_turn_raw = item.get("created_turn", 0)
    if isinstance(created_turn_raw, bool):
        created_turn_raw = 0
    if not isinstance(created_turn_raw, int):
        created_turn_raw = 0
    created_turn = max(created_turn_raw, 0)

    depth_raw = item.get("depth", 0)
    if isinstance(depth_raw, bool):
        depth_raw = 0
    if not isinstance(depth_raw, int):
        depth_raw = 0
    depth = max(depth_raw, 0)

    return {
        "id": seed_id,
        "text": _truncate_text(text, 400),
        "priority": priority,
        "earliest_turn": earliest_raw,
        "latest_turn": latest_raw,
        "max_shows": max_shows,
        "shows": shows,
        "last_shown_turn": last_shown_turn,
        "anchor_object_ids": anchor_object_ids,
        "created_turn": created_turn,
        "depth": depth,
    }


def _coerce_consequence_intent(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    intent_key = str(item.get("intent_key") or item.get("id") or "").strip()
    if not intent_key:
        return None

    priority = str(item.get("priority") or "med").strip().lower()
    if priority not in {"low", "med", "high"}:
        priority = "med"

    earliest_raw = item.get("earliest_turn")
    latest_raw = item.get("latest_turn")
    if not isinstance(earliest_raw, int):
        return None
    if not isinstance(latest_raw, int) or latest_raw < earliest_raw:
        return None

    max_shows_raw = item.get("max_shows", 2)
    if isinstance(max_shows_raw, bool) or not isinstance(max_shows_raw, int):
        max_shows_raw = 2

    target_refs: list[str] = []
    raw_target_refs = item.get("target_refs")
    if isinstance(raw_target_refs, list):
        for raw_target in raw_target_refs:
            target_ref = _coerce_ref(raw_target)
            if target_ref is not None:
                target_refs.append(target_ref)

    anchor_refs: list[str] = []
    raw_anchor_refs = item.get("anchor_refs")
    if isinstance(raw_anchor_refs, list):
        for raw_anchor in raw_anchor_refs:
            anchor_ref = _coerce_ref(raw_anchor)
            if anchor_ref is not None:
                anchor_refs.append(anchor_ref)

    if not anchor_refs:
        raw_anchor_object_ids = item.get("anchor_object_ids")
        if isinstance(raw_anchor_object_ids, list):
            for raw_anchor in raw_anchor_object_ids:
                anchor_ref = _coerce_ref(raw_anchor)
                if anchor_ref is not None:
                    anchor_refs.append(anchor_ref)

    reason_kind = str(item.get("reason_kind") or item.get("reason") or "followup").strip()
    if not reason_kind:
        reason_kind = "followup"

    return {
        "intent_key": intent_key,
        "priority": priority,
        "earliest_turn": earliest_raw,
        "latest_turn": latest_raw,
        "max_shows": max(max_shows_raw, 0),
        "target_refs": target_refs,
        "anchor_refs": anchor_refs,
        "reason_kind": _truncate_text(reason_kind, 80),
    }


def _coerce_resolved_consequence_id(raw_value: Any) -> str | None:
    text = str(raw_value or "").strip()
    return text or None


def _coerce_ref(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text == SESSION_PLAYER_REF:
            return text
        if _TMP_REF_RE.fullmatch(text):
            return text
        try:
            return str(uuid.UUID(text))
        except (TypeError, ValueError, AttributeError):
            return None
    if isinstance(value, dict):
        for key in ("object", "object_id", "id", "ref", "from", "to", "scope"):
            ref = _coerce_ref(value.get(key))
            if ref:
                return ref
    return None


def _coerce_json_patch_to_object_update(item: dict[str, Any]) -> dict[str, Any] | None:
    path_raw = item.get("path")
    if not isinstance(path_raw, str):
        return None
    path = path_raw.strip()
    if not path:
        return None
    parts = [chunk for chunk in path.split("/") if chunk]
    if len(parts) < 3 or parts[0] != "objects":
        return None

    object_ref = parts[1]
    if not object_ref:
        return None

    key_index = 3 if len(parts) >= 4 and parts[2] == "data" else 2
    if key_index >= len(parts):
        return None
    # Only single-key object patches are supported here. Nested paths like
    # /objects/<id>/data/stats/hp cannot be represented as a flat object.update
    # patch without losing structure, so reject them instead of truncating.
    if len(parts) != key_index + 1:
        return None
    patch_key = parts[key_index]
    if not patch_key:
        return None

    op_name = str(item.get("op") or "").strip().lower()
    if op_name == "remove":
        patch_value: Any = None
    else:
        patch_value = item.get("value")
    return {
        "op": "object.update",
        "object": object_ref,
        "patch": {patch_key: patch_value},
    }


def _coerce_patch_op(item: Any, idx: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    op_name = str(item.get("op") or "").strip().lower()
    if not op_name:
        return None

    if op_name in {"add", "replace", "remove"}:
        return _coerce_json_patch_to_object_update(item)

    if op_name == "object.create":
        ref = _coerce_ref(item.get("ref") or item.get("tmp_ref"))
        if not ref:
            ref = f"tmp:auto_{idx}"
        if not isinstance(ref, str) or not ref.startswith("tmp:"):
            ref = f"tmp:auto_{idx}"
        object_type = str(item.get("type") or "").strip()
        data = item.get("data")
        if not isinstance(data, dict):
            data = {}
        name = str(item.get("name") or data.get("name") or "").strip()
        if not name and object_type == "world_constitution":
            name = "World Constitution"
        if object_type not in schemas.API_OBJECT_TYPE_VALUES or not name:
            return None
        return {
            "op": "object.create",
            "ref": ref,
            "type": object_type,
            "name": name,
            "data": data,
        }

    if op_name in {
        "object.update",
        "update.object",
        "update",
        "state.update",
        "entity.update",
        "npc.update",
        "object.patch",
    }:
        object_ref = _coerce_ref(
            item.get("object")
            or item.get("object_id")
            or item.get("id")
            or item.get("target")
            or item.get("ref")
        )
        patch = item.get("patch")
        if not isinstance(patch, dict):
            for key in ("data", "changes", "updates", "fields"):
                candidate_patch = item.get(key)
                if isinstance(candidate_patch, dict):
                    patch = candidate_patch
                    break
        if not isinstance(patch, dict):
            patch = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "op",
                    "object",
                    "object_id",
                    "id",
                    "target",
                    "ref",
                    "patch",
                    "data",
                    "changes",
                    "updates",
                    "fields",
                }
            }
        if not object_ref:
            converted = _coerce_json_patch_to_object_update(item)
            if converted is not None:
                return converted
            return None
        return {
            "op": "object.update",
            "object": object_ref,
            "patch": patch if isinstance(patch, dict) else {},
        }

    if op_name == "link.create":
        from_ref = _coerce_ref(item.get("from") or item.get("from_ref") or item.get("source") or item.get("from_object_id"))
        to_ref = _coerce_ref(item.get("to") or item.get("to_ref") or item.get("target") or item.get("to_object_id"))
        link_type = str(item.get("type") or item.get("link_type") or "").strip()
        if not from_ref or not to_ref or not link_type:
            return None
        data = item.get("data")
        if not isinstance(data, dict):
            data = {}
        return {
            "op": "link.create",
            "from": from_ref,
            "to": to_ref,
            "type": link_type,
            "data": data,
            "bidirectional": bool(item.get("bidirectional", False)),
        }

    if op_name in {"link.close", "link.delete"}:
        from_ref = _coerce_ref(item.get("from") or item.get("from_ref") or item.get("source") or item.get("from_object_id"))
        to_ref = _coerce_ref(item.get("to") or item.get("to_ref") or item.get("target") or item.get("to_object_id"))
        link_type = str(item.get("type") or item.get("link_type") or "").strip()
        if not from_ref or not to_ref or not link_type:
            return None
        return {
            "op": "link.close",
            "from": from_ref,
            "to": to_ref,
            "type": link_type,
            "bidirectional": bool(item.get("bidirectional", False)),
        }

    if op_name == "player.move":
        to_ref = _coerce_ref(item.get("to") or item.get("target") or item.get("to_object_id"))
        if not to_ref:
            return None
        player_ref = _coerce_ref(item.get("player") or SESSION_PLAYER_REF) or SESSION_PLAYER_REF
        return {
            "op": "player.move",
            "player": player_ref,
            "to": to_ref,
        }

    if op_name == "event.create":
        event_type = str(item.get("type") or item.get("event_type") or "").strip()
        if not event_type:
            return None
        payload = item.get("payload")
        if not isinstance(payload, dict):
            payload = {
                key: value
                for key, value in item.items()
                if key not in {"op", "type", "event_type", "scope", "scope_object_id"}
            }
        result: dict[str, Any] = {
            "op": "event.create",
            "type": event_type,
            "payload": payload if isinstance(payload, dict) else {},
        }
        scope_ref = _coerce_ref(item.get("scope") or item.get("scope_object_id") or item.get("object") or item.get("object_id"))
        if scope_ref:
            result["scope"] = scope_ref
        return result

    return None


def _coerce_in_game_time(raw_value: Any) -> dict[str, Any] | None:
    if isinstance(raw_value, dict):
        day_raw = raw_value.get("day")
        minute_raw = raw_value.get("minute")
        day = int(day_raw) if isinstance(day_raw, int) else None
        minute = int(minute_raw) if isinstance(minute_raw, int) else None
        return {"day": day, "minute": minute}
    if not isinstance(raw_value, str):
        return None

    text = raw_value.strip().lower()
    if not text:
        return None
    day_match = re.search(r"day\s*([0-9]+)", text)
    minute_match = re.search(r"minute\s*([0-9]+)", text)
    day = int(day_match.group(1)) if day_match else None
    minute = int(minute_match.group(1)) if minute_match else None
    if day is None and minute is None:
        return None
    return {"day": day, "minute": minute}


def _coerce_turn_weight(raw_value: Any) -> float | None:
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


def _coerce_zone_scope(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        for key in ("zone_scope", "zone_id", "id", "object", "object_id", "scope"):
            zone_scope = _coerce_zone_scope(raw_value.get(key))
            if zone_scope is not None:
                return zone_scope
        return None

    zone_ref = _coerce_ref(raw_value)
    if not zone_ref:
        return None
    try:
        return str(uuid.UUID(zone_ref))
    except (TypeError, ValueError, AttributeError):
        return None


def _coerce_narrator_payload(raw_payload: Any) -> dict[str, Any]:
    source = raw_payload if isinstance(raw_payload, dict) else {}
    narration = str(source.get("narration") or source.get("text") or source.get("message") or "").strip()
    if not narration:
        narration = FALLBACK_NO_RESPONSE

    raw_choices = source.get("choices")
    choices: list[dict[str, Any]] = []
    if isinstance(raw_choices, list):
        for idx, choice in enumerate(raw_choices):
            coerced_choice = _coerce_choice(choice, idx)
            if coerced_choice is not None:
                choices.append(coerced_choice)

    raw_updates = source.get("proposed_updates")
    proposed_updates: list[dict[str, Any]] = []
    if isinstance(raw_updates, list):
        for idx, op in enumerate(raw_updates):
            coerced_op = _coerce_patch_op(op, idx)
            if coerced_op is not None:
                proposed_updates.append(coerced_op)

    raw_memory_candidates = source.get("memory_candidates")
    memory_candidates: list[dict[str, Any]] = []
    if isinstance(raw_memory_candidates, list):
        for item in raw_memory_candidates:
            coerced_candidate = _coerce_memory_candidate(item)
            if coerced_candidate is not None:
                memory_candidates.append(coerced_candidate)

    semantic_events: list[dict[str, Any]] = []
    raw_semantic_events = source.get("semantic_events")
    if isinstance(raw_semantic_events, list):
        for item in raw_semantic_events:
            coerced_event = _coerce_semantic_event(item)
            if coerced_event is not None:
                semantic_events.append(coerced_event)

    scene_entities: list[dict[str, Any]] = []
    raw_scene_entities = source.get("scene_entities")
    if isinstance(raw_scene_entities, list):
        for item in raw_scene_entities:
            coerced_entity = _coerce_scene_entity(item)
            if coerced_entity is not None:
                scene_entities.append(coerced_entity)

    raw_consequences = source.get("consequence_seeds")
    consequence_seeds: list[dict[str, Any]] = []
    if isinstance(raw_consequences, list):
        for item in raw_consequences:
            coerced = _coerce_consequence_seed(item)
            if coerced is not None:
                consequence_seeds.append(coerced)

    raw_consequence_intents = source.get("consequence_intents")
    consequence_intents: list[dict[str, Any]] = []
    if isinstance(raw_consequence_intents, list):
        for item in raw_consequence_intents:
            coerced = _coerce_consequence_intent(item)
            if coerced is not None:
                consequence_intents.append(coerced)
    elif consequence_seeds:
        for item in consequence_seeds:
            coerced = _coerce_consequence_intent(
                {
                    "intent_key": item.get("id"),
                    "priority": item.get("priority"),
                    "earliest_turn": item.get("earliest_turn"),
                    "latest_turn": item.get("latest_turn"),
                    "max_shows": item.get("max_shows"),
                    "anchor_refs": item.get("anchor_object_ids", []),
                    "reason_kind": "legacy_seed",
                }
            )
            if coerced is not None:
                consequence_intents.append(coerced)

    raw_resolved = source.get("resolved_consequence_ids")
    resolved_consequence_ids: list[str] = []
    if isinstance(raw_resolved, list):
        for item in raw_resolved:
            parsed_id = _coerce_resolved_consequence_id(item)
            if parsed_id is not None:
                resolved_consequence_ids.append(parsed_id)

    result: dict[str, Any] = {
        "narration": narration,
        "choices": choices,
        "proposed_updates": proposed_updates,
        "memory_candidates": memory_candidates,
        "semantic_events": semantic_events,
        "scene_entities": scene_entities,
        "consequence_seeds": consequence_seeds,
        "consequence_intents": consequence_intents,
        "resolved_consequence_ids": resolved_consequence_ids,
        "zone_scope": _coerce_zone_scope(source.get("zone_scope")),
        "in_game_time": _coerce_in_game_time(source.get("in_game_time")),
        "turn_weight": _coerce_turn_weight(source.get("turn_weight")),
        "planner_contract_version": 2,
    }
    return result


def _coerce_world_intent_payload(raw_payload: Any) -> dict[str, Any]:
    source = raw_payload if isinstance(raw_payload, dict) else {}

    raw_updates = source.get("proposed_updates")
    proposed_updates: list[dict[str, Any]] = []
    if isinstance(raw_updates, list):
        for idx, op in enumerate(raw_updates):
            coerced_op = _coerce_patch_op(op, idx)
            if coerced_op is not None:
                proposed_updates.append(coerced_op)

    raw_memory_candidates = source.get("memory_candidates")
    memory_candidates: list[dict[str, Any]] = []
    if isinstance(raw_memory_candidates, list):
        for item in raw_memory_candidates:
            coerced_candidate = _coerce_memory_candidate(item)
            if coerced_candidate is not None:
                memory_candidates.append(coerced_candidate)

    semantic_events: list[dict[str, Any]] = []
    raw_semantic_events = source.get("semantic_events")
    if isinstance(raw_semantic_events, list):
        for item in raw_semantic_events:
            coerced_event = _coerce_semantic_event(item)
            if coerced_event is not None:
                semantic_events.append(coerced_event)

    scene_entities: list[dict[str, Any]] = []
    raw_scene_entities = source.get("scene_entities")
    if isinstance(raw_scene_entities, list):
        for item in raw_scene_entities:
            coerced_entity = _coerce_scene_entity(item)
            if coerced_entity is not None:
                scene_entities.append(coerced_entity)

    raw_consequences = source.get("consequence_intents")
    consequence_intents: list[dict[str, Any]] = []
    if isinstance(raw_consequences, list):
        for item in raw_consequences:
            coerced = _coerce_consequence_intent(item)
            if coerced is not None:
                consequence_intents.append(coerced)
    else:
        raw_legacy = source.get("consequence_seeds")
        if isinstance(raw_legacy, list):
            for item in raw_legacy:
                seed = _coerce_consequence_seed(item)
                if seed is None:
                    continue
                coerced = _coerce_consequence_intent(
                    {
                        "intent_key": seed.get("id"),
                        "priority": seed.get("priority"),
                        "earliest_turn": seed.get("earliest_turn"),
                        "latest_turn": seed.get("latest_turn"),
                        "max_shows": seed.get("max_shows"),
                        "anchor_refs": seed.get("anchor_object_ids", []),
                        "reason_kind": "legacy_seed",
                    }
                )
                if coerced is not None:
                    consequence_intents.append(coerced)

    raw_resolved = source.get("resolved_consequence_ids")
    resolved_consequence_ids: list[str] = []
    if isinstance(raw_resolved, list):
        for item in raw_resolved:
            parsed_id = _coerce_resolved_consequence_id(item)
            if parsed_id is not None:
                resolved_consequence_ids.append(parsed_id)

    return {
        "proposed_updates": proposed_updates,
        "memory_candidates": memory_candidates,
        "semantic_events": semantic_events,
        "scene_entities": scene_entities,
        "consequence_intents": consequence_intents,
        "resolved_consequence_ids": resolved_consequence_ids,
        "zone_scope": _coerce_zone_scope(source.get("zone_scope")),
        "planner_contract_version": 2,
    }


def _extract_xai_usage(raw_payload: Any) -> dict[str, int] | None:
    if not isinstance(raw_payload, dict):
        return None
    raw_usage = raw_payload.get("_xai_usage")
    if not isinstance(raw_usage, dict):
        return None

    usage: dict[str, int] = {}
    for key, value in raw_usage.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            usage[key] = value
            continue
        if isinstance(value, float) and value.is_integer():
            usage[key] = int(value)
    return usage or None


def _extract_openrouter_usage(raw_payload: Any) -> dict[str, int] | None:
    if not isinstance(raw_payload, dict):
        return None
    raw_usage = raw_payload.get("_openrouter_usage")
    if not isinstance(raw_usage, dict):
        return None

    usage: dict[str, int] = {}
    for key, value in raw_usage.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            usage[key] = value
            continue
        if isinstance(value, float) and value.is_integer():
            usage[key] = int(value)
    return usage or None


def _collect_refs(op: schemas.PatchOp) -> list[schemas.Ref]:
    if isinstance(op, schemas.ObjectCreateOp):
        return []
    if isinstance(op, schemas.ObjectUpdateOp):
        return [op.object]
    if isinstance(op, schemas.LinkCreateOp):
        return [op.from_ref, op.to]
    if isinstance(op, schemas.LinkCloseOp):
        return [op.from_ref, op.to]
    if isinstance(op, schemas.PlayerMoveOp):
        return [op.player, op.to]
    if isinstance(op, schemas.EventCreateOp):
        return [op.scope] if op.scope is not None else []
    return []


def _normalize_patch_ref(ref: schemas.Ref | None) -> str | None:
    if ref is None:
        return None
    if isinstance(ref, uuid.UUID):
        return str(ref)
    text = str(ref).strip()
    return text or None


def _collect_patch_op_ref_keys(op: schemas.PatchOp) -> set[str]:
    keys: set[str] = set()
    for ref in _collect_refs(op):
        normalized = _normalize_patch_ref(ref)
        if normalized:
            keys.add(normalized)
    if isinstance(op, schemas.ObjectCreateOp):
        keys.add(op.ref)
    if isinstance(op, schemas.EventCreateOp):
        scope_ref = _normalize_patch_ref(op.scope)
        if scope_ref:
            keys.add(scope_ref)
    return keys


def _durable_fact_expected_ops(fact_kind: str) -> tuple[str, ...]:
    if fact_kind == "relationship":
        return ("link.create", "link.close")
    if fact_kind == "location_fact":
        return ("player.move", "link.create", "link.close", "object.create")
    if fact_kind == "quest_milestone":
        return ("object.update", "object.create", "link.create", "event.create")
    return ("object.create", "object.update", "link.create", "link.close", "player.move", "event.create")


def _find_durable_fact_mismatch_reasons(
    durable_facts: list[schemas.DurableFact],
    parsed_ops: list[schemas.PatchOp],
) -> list[str]:
    if not durable_facts:
        return []
    if not parsed_ops:
        return ["durable_fact_missing_updates"]

    reasons: list[str] = []
    for fact in durable_facts:
        anchors: set[str] = set()
        for ref_value in (
            fact.actor_ref,
            fact.counterparty_ref,
            fact.object_ref,
            fact.location_ref,
            fact.quest_ref,
            *list(fact.context_refs),
        ):
            related_key = _normalize_patch_ref(ref_value)
            if related_key:
                anchors.add(related_key)
        if not anchors:
            continue

        expected_ops = _durable_fact_expected_ops(fact.kind)
        matched = False
        for op in parsed_ops:
            if op.op not in expected_ops:
                continue
            if _collect_patch_op_ref_keys(op) & anchors:
                matched = True
                break
        if matched:
            continue
        reasons.append(
            "durable_fact_missing_update: "
            f"kind={fact.kind} anchors={','.join(sorted(anchors))}"
        )
    return reasons


def _toposort_patch_ops(ops: list[schemas.PatchOp]) -> list[schemas.PatchOp]:
    if len(ops) <= 1:
        return ops

    create_index_by_ref: dict[str, int] = {}
    for idx, op in enumerate(ops):
        if isinstance(op, schemas.ObjectCreateOp):
            create_index_by_ref[op.ref] = idx
    if not create_index_by_ref:
        return ops

    edges: list[set[int]] = [set() for _ in ops]
    indegree = [0] * len(ops)

    for idx, op in enumerate(ops):
        if isinstance(op, schemas.ObjectCreateOp):
            continue

        dependencies: set[int] = set()
        for ref in _collect_refs(op):
            if not isinstance(ref, str) or not ref.startswith("tmp:"):
                continue
            create_idx = create_index_by_ref.get(ref)
            if create_idx is None or create_idx == idx:
                continue
            dependencies.add(create_idx)

        for dep_idx in dependencies:
            if idx in edges[dep_idx]:
                continue
            edges[dep_idx].add(idx)
            indegree[idx] += 1

    if all(value == 0 for value in indegree):
        return ops

    ready = [idx for idx, value in enumerate(indegree) if value == 0]
    ordered_indices: list[int] = []
    head = 0

    while head < len(ready):
        node = ready[head]
        head += 1
        ordered_indices.append(node)
        for child in sorted(edges[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(ordered_indices) != len(ops):
        return ops

    return [ops[idx] for idx in ordered_indices]


def _validate_patch_ops(ops_payload: Any) -> PatchValidationResult:
    if ops_payload is None:
        ops_payload = []
    if not isinstance(ops_payload, list):
        return PatchValidationResult(
            status="reject",
            reasons=["proposed_updates must be an array"],
            parsed_ops=[],
        )
    if len(ops_payload) > MAX_PATCH_OPS:
        return PatchValidationResult(
            status="reject",
            reasons=[f"too many ops: {len(ops_payload)} > {MAX_PATCH_OPS}"],
            parsed_ops=[],
        )

    try:
        parsed_ops = PATCH_OP_LIST_ADAPTER.validate_python(ops_payload)
    except ValidationError as exc:
        return PatchValidationResult(
            status="reject",
            reasons=[f"PatchOp validation failed: {exc.errors()[0]['msg']}"],
            parsed_ops=[],
        )

    seen_create_refs: set[str] = set()
    for idx, op in enumerate(parsed_ops):
        if not isinstance(op, schemas.ObjectCreateOp):
            continue
        if op.ref in seen_create_refs:
            return PatchValidationResult(
                status="reject",
                reasons=[f"duplicate object.create ref at index {idx}: {op.ref}"],
                parsed_ops=[],
            )
        seen_create_refs.add(op.ref)

    parsed_ops = _toposort_patch_ops(parsed_ops)

    created_refs: set[str] = set()
    uncertain_reasons: list[str] = []
    for idx, op in enumerate(parsed_ops):
        if isinstance(op, schemas.ObjectCreateOp):
            created_refs.add(op.ref)

        if isinstance(op, schemas.LinkCreateOp) and op.type == LOCATED_IN_LINK_TYPE:
            return PatchValidationResult(
                status="reject",
                reasons=["link.create with type='located_in' is forbidden; use player.move"],
                parsed_ops=[],
            )
        if isinstance(op, schemas.LinkCloseOp):
            if op.type == LOCATED_IN_LINK_TYPE:
                return PatchValidationResult(
                    status="reject",
                    reasons=["link.close with type='located_in' is forbidden; use player.move"],
                    parsed_ops=[],
                )
            if op.type == TRACKING_QUEST_LINK_TYPE:
                return PatchValidationResult(
                    status="reject",
                    reasons=[f"link.close with type='{TRACKING_QUEST_LINK_TYPE}' is forbidden; close via quest terminal status"],
                    parsed_ops=[],
                )

        for ref in _collect_refs(op):
            if isinstance(ref, str) and ref.startswith("tmp:") and ref not in created_refs:
                uncertain_reasons.append(
                    f"unresolved temp ref before creation at op {idx}: {ref}"
                )

    if uncertain_reasons:
        return PatchValidationResult(
            status="uncertain",
            reasons=uncertain_reasons,
            parsed_ops=parsed_ops,
        )
    return PatchValidationResult(status="ok", reasons=[], parsed_ops=parsed_ops)


def _normalize_quest_status(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip().casefold()
        return text or None
    text = str(raw_value).strip().casefold()
    return text or None


def _collect_proposed_tracking_quest_targets(
    parsed_ops: list[schemas.PatchOp],
    *,
    player_object_id: uuid.UUID,
) -> set[uuid.UUID]:
    proposed_targets: set[uuid.UUID] = set()
    for op in parsed_ops:
        if not isinstance(op, schemas.LinkCreateOp):
            continue
        if op.type != TRACKING_QUEST_LINK_TYPE:
            continue
        from_ref = op.from_ref
        from_is_player = from_ref == SESSION_PLAYER_REF or (
            isinstance(from_ref, uuid.UUID) and from_ref == player_object_id
        )
        if not from_is_player:
            continue
        if isinstance(op.to, uuid.UUID):
            proposed_targets.add(op.to)
    return proposed_targets


def _find_quest_reopen_tracking_link_reasons(
    db: Session,
    session_id: uuid.UUID,
    parsed_ops: list[schemas.PatchOp],
) -> list[str]:
    if db is None or not hasattr(db, "execute"):
        return []
    if not parsed_ops:
        return []

    try:
        player_object_id = _get_session_player_object_id(db, session_id)
    except HTTPException:
        return []

    proposed_targets = _collect_proposed_tracking_quest_targets(
        parsed_ops,
        player_object_id=player_object_id,
    )
    reasons: list[str] = []
    for idx, op in enumerate(parsed_ops):
        if not isinstance(op, schemas.ObjectUpdateOp):
            continue
        if not isinstance(op.object, uuid.UUID):
            continue
        patch_data = dict(op.patch or {})
        if "status" not in patch_data:
            continue

        new_status = _normalize_quest_status(patch_data.get("status"))
        if new_status is None or new_status in QUEST_TERMINAL_STATUSES:
            continue

        row = db.execute(
            select(models.ObjectModel.type, models.ObjectModel.data)
            .where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.object_id == op.object,
            )
            .limit(1)
        ).first()
        if row is None:
            continue
        object_type, object_data = row
        if str(object_type or "").strip() != "quest":
            continue
        old_data = dict(object_data or {}) if isinstance(object_data, dict) else {}
        old_status = _normalize_quest_status(old_data.get("status"))
        if old_status not in QUEST_TERMINAL_STATUSES:
            continue

        has_active_tracking_link = db.execute(
            select(models.LinkModel.link_id)
            .where(
                models.LinkModel.session_id == session_id,
                models.LinkModel.from_object_id == player_object_id,
                models.LinkModel.to_object_id == op.object,
                models.LinkModel.type == TRACKING_QUEST_LINK_TYPE,
                models.LinkModel.valid_to_turn.is_(None),
            )
            .limit(1)
        ).scalar_one_or_none()
        if has_active_tracking_link is not None:
            continue
        if op.object in proposed_targets:
            continue
        reasons.append(
            "quest_reopen_requires_tracking_link: "
            f"op={idx} quest={op.object} needs link.create type='{TRACKING_QUEST_LINK_TYPE}'"
        )

    return reasons


def _parse_narrator_response(raw_payload: Any) -> schemas.NarratorResponse:
    return schemas.NarratorResponse.model_validate(raw_payload)


def _parse_world_intent_response(raw_payload: Any) -> schemas.WorldIntentResponseV2:
    return schemas.WorldIntentResponseV2.model_validate(raw_payload)


def _parse_post_apply_narration_response(raw_payload: Any) -> schemas.PostApplyNarrationResponse:
    return schemas.PostApplyNarrationResponse.model_validate(raw_payload)


def _format_anchor_objects_for_prompt(raw_anchors: Any) -> str:
    if not isinstance(raw_anchors, list):
        return ""
    formatted: list[str] = []
    for item in raw_anchors:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        object_id = str(item.get("object_id") or "").strip()
        if name and object_id:
            formatted.append(f"{name} ({object_id})")
        elif name:
            formatted.append(name)
        elif object_id:
            formatted.append(object_id)
    return ", ".join(formatted)


def _build_consequence_prompt_sections(context_pack: dict[str, Any]) -> str:
    sections: list[str] = []
    raw_new_turn = context_pack.get("new_turn")
    if isinstance(raw_new_turn, int):
        current_turn = raw_new_turn
    elif isinstance(raw_new_turn, str) and raw_new_turn.strip().isdigit():
        current_turn = int(raw_new_turn.strip())
    else:
        current_turn = 0

    sections.append(
        "\n".join(
            [
                "CONSEQUENCE ENGINE CONTEXT:",
                f"Current turn: {current_turn}",
                "earliest_turn/latest_turn are absolute turn indices, not relative offsets.",
                "For a delay of N turns, set earliest_turn = context.new_turn + N.",
                (
                    "Keep consequence windows bounded: "
                    f"latest_turn <= earliest_turn + {MAX_CONSEQUENCE_WINDOW_SPAN}."
                ),
            ]
        )
    )

    raw_latent = context_pack.get("latent_consequences")
    if isinstance(raw_latent, list) and raw_latent:
        lines = ["LATENT CONSEQUENCES (pressure hints, not obligations):"]
        for raw_item in raw_latent:
            if not isinstance(raw_item, dict):
                continue
            cid = str(raw_item.get("id") or "").strip()
            text = str(raw_item.get("text") or "").strip()
            priority = str(raw_item.get("priority") or "med").strip().lower() or "med"
            latest_turn = raw_item.get("latest_turn")
            if not cid or not text:
                continue
            lines.append(
                f'- [{cid}] "{text}" (priority: {priority}, expires: turn {latest_turn})'
            )
            anchors = _format_anchor_objects_for_prompt(raw_item.get("anchor_objects"))
            if anchors:
                lines.append(f"  anchors: {anchors}")
        lines.extend(
            [
                "Realize zero, one, or more - your call. Zero is valid.",
                "If realized, include id in resolved_consequence_ids.",
                "You may propose new consequence_seeds from this turn's events.",
            ]
        )
        sections.append("\n".join(lines))

    raw_signals = context_pack.get("structural_signals")
    if isinstance(raw_signals, list) and raw_signals:
        lines = ["STRUCTURAL SIGNALS (raw graph facts, interpret yourself):"]
        for raw_item in raw_signals:
            if not isinstance(raw_item, dict):
                continue
            trigger = str(raw_item.get("trigger") or "").strip()
            object_name = str(raw_item.get("object_name") or raw_item.get("object_id") or "").strip()
            turn = raw_item.get("turn")
            if not trigger:
                continue
            lines.append(f"- {trigger}: {object_name} (turn {turn})")
        lines.append(
            "These are facts, not instructions. Decide if they warrant new consequence_seeds."
        )
        sections.append("\n".join(lines))

    return "\n\n" + "\n\n".join(sections)


def _pick_world_constitution_text(context_pack: dict[str, Any]) -> str:
    for key in ("world_constitution_for_system", "world_prompt_for_system", "world_prompt"):
        text = _truncate_text(
            str(context_pack.get(key) or "").strip(),
            MAX_WORLD_CONSTITUTION_CHARS,
        )
        if text:
            return text
    return ""


def _pick_narrative_spine_text_for_system(context_pack: dict[str, Any]) -> str:
    raw_spine = context_pack.get("narrative_spine_for_system")
    if isinstance(raw_spine, dict):
        payload = {
            "player_commitments": raw_spine.get("player_commitments", []),
            "world_changes": raw_spine.get("world_changes", []),
            "key_npc_statuses": raw_spine.get("key_npc_statuses", []),
        }
        rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        return _truncate_text(rendered, 1200)
    if isinstance(raw_spine, str):
        text = raw_spine.strip()
        if text:
            return _truncate_text(text, 1200)

    summaries = context_pack.get("session_summaries")
    if isinstance(summaries, list):
        for summary in summaries:
            if isinstance(summary, dict):
                rendered = json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
                if rendered:
                    return _truncate_text(rendered, 1200)
            elif isinstance(summary, str):
                text = summary.strip()
                if text:
                    return _truncate_text(text, 1200)
    return ""


def _build_narrator_user_payload(context_pack: dict[str, Any]) -> dict[str, Any]:
    payload = dict(context_pack)
    payload.pop("world_prompt_for_system", None)
    payload.pop("world_constitution_for_system", None)
    payload.pop("narrative_spine_for_system", None)
    payload.pop("economy_state", None)
    economy_brief = _visible_economy_brief(context_pack)
    if economy_brief is not None:
        payload["economy_brief"] = economy_brief
    else:
        payload.pop("economy_brief", None)
    if _pick_narrative_spine_text_for_system(context_pack):
        payload.pop("session_summaries", None)

    for source_key, compact_key in _NARRATOR_CONTEXT_KEY_ALIASES.items():
        if source_key not in payload:
            continue
        payload[compact_key] = payload.pop(source_key)
    return payload


def _visible_economy_brief(context_pack: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(context_pack, dict):
        return None
    raw_brief = context_pack.get("economy_brief")
    if not isinstance(raw_brief, dict):
        return None
    if raw_brief.get("included") is False:
        return None
    lines = [
        str(line).strip()
        for line in list(raw_brief.get("lines") or [])
        if str(line).strip()
    ]
    if not lines:
        return None
    payload = {
        "lines": lines[:8],
        "reasons": [
            str(reason).strip()
            for reason in list(raw_brief.get("reasons") or [])
            if str(reason).strip()
        ][:5],
    }
    delta_summary = str(raw_brief.get("delta_summary") or "").strip()
    if delta_summary:
        payload["delta_summary"] = delta_summary
    return payload


def _travel_outcome_summary(context_pack: dict[str, Any]) -> str:
    travel_outcome = dict(context_pack.get("travel_outcome") or {})
    if not travel_outcome:
        return ""
    mode = str(travel_outcome.get("outcome_mode") or "").strip().lower()
    target_name = str(
        travel_outcome.get("resolved_target_name")
        or travel_outcome.get("expanded_node_name")
        or travel_outcome.get("final_node_name")
        or ""
    ).strip()
    final_name = str(travel_outcome.get("final_node_name") or "").strip()
    if mode == "arrive":
        return f"Travel resolved by server: arrived at {final_name or target_name}."
    if mode == "checkpoint":
        return f"Travel resolved by server: reached checkpoint {final_name or target_name} while continuing toward {target_name or final_name}."
    if mode == "interrupted":
        reason = str(travel_outcome.get("interruption_reason") or "the route did not complete").strip()
        return f"Travel resolved by server: movement was interrupted at {final_name or target_name}. Reason: {reason}."
    if mode == "impossible":
        reason = str(travel_outcome.get("interruption_reason") or "the destination is not reachable").strip()
        return f"Travel resolved by server: destination could not be reached. Reason: {reason}."
    if mode == "ambiguous":
        return "Travel resolved by server: destination is ambiguous and requires clarification."
    return ""


def _travel_scene_entities(context_pack: dict[str, Any]) -> list[schemas.SceneEntity]:
    scene_entities: list[schemas.SceneEntity] = []
    current_location = dict(context_pack.get("current_location") or {})
    node_id = current_location.get("node_id")
    node_name = str(current_location.get("name") or "").strip()
    if node_id and node_name:
        scene_entities.append(
            schemas.SceneEntity(
                ref=node_id,
                name=node_name,
                entity_type="zone",
                role="location",
                referent_candidate=True,
                salience=1.0,
            )
        )
    present_entities = [
        item
        for item in list(context_pack.get("present_entities") or [])
        if isinstance(item, dict)
    ]
    for item in present_entities[:5]:
        name = str(item.get("name") or "").strip()
        object_id = str(item.get("object_id") or "").strip()
        if not name:
            continue
        entity_type = str(item.get("object_type") or "other").strip().lower()
        if entity_type not in {"player", "npc", "zone", "item", "quest", "faction"}:
            entity_type = "other"
        scene_entities.append(
            schemas.SceneEntity(
                ref=object_id or None,
                name=name,
                entity_type=entity_type,  # type: ignore[arg-type]
                role="observer" if entity_type != "player" else "focus",
                referent_candidate=entity_type in {"npc", "item", "player"},
                salience=0.7 if entity_type == "player" else 0.55,
            )
        )
    return scene_entities


def _fallback_travel_narration(context_pack: dict[str, Any]) -> str:
    travel_outcome = dict(context_pack.get("travel_outcome") or {})
    summary = _travel_outcome_summary(context_pack)
    if summary:
        return summary
    final_name = str(travel_outcome.get("final_node_name") or travel_outcome.get("resolved_target_name") or "").strip()
    if final_name:
        return f"The travel attempt resolves at {final_name}."
    return "The travel attempt resolves without additional world changes."


def _travel_outcome_uuid(value: Any) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except (TypeError, ValueError, AttributeError):
        return None


def _resolve_travel_turn_plan(
    *,
    context_pack: dict[str, Any],
    session_id: uuid.UUID | None,
    in_game_day: int,
    in_game_minute: int,
) -> TurnPlanResult:
    travel_outcome = dict(context_pack.get("travel_outcome") or {})
    outcome_mode = str(travel_outcome.get("outcome_mode") or "").strip().lower()
    llm_usage: dict[str, Any] = {"narrator": None}

    if outcome_mode == "ambiguous":
        candidate_targets = [
            item
            for item in list(travel_outcome.get("ambiguity_targets") or [])
            if isinstance(item, dict)
        ]
        options = [
            schemas.NarratorChoice(
                id=str(idx + 1),
                text=f"Travel to {str(item.get('name') or '').strip()}",
                intent="travel_clarify",
                risk="low",
            )
            for idx, item in enumerate(candidate_targets[:4])
            if str(item.get("name") or "").strip()
        ]
        names = ", ".join(str(item.get("name") or "").strip() for item in candidate_targets if str(item.get("name") or "").strip())
        narration = (
            f"Your destination is ambiguous. Clarify which place you mean: {names}."
            if names
            else "Your destination is ambiguous. Clarify where you want to go."
        )
    else:
        try:
            narration = _call_narrator_text_only(context_pack=context_pack, session_id=session_id)
        except Exception:  # noqa: BLE001
            logger.warning("Travel narrator text-only failed for session=%s", session_id, exc_info=True)
            narration = _fallback_travel_narration(context_pack)
        options = []

    path_ids = tuple(
        node_id
        for node_id in (
            _travel_outcome_uuid(item)
            for item in list(travel_outcome.get("path") or [])
        )
        if node_id is not None
    )
    semantic_resolution = geography_policy_domain.TravelResolution(
        is_travel_intent=True,
        outcome_mode=outcome_mode if outcome_mode in {"arrive", "checkpoint", "interrupted", "impossible", "ambiguous"} else "impossible",
        clarification_needed=outcome_mode == "ambiguous",
        resolved_target=_travel_outcome_uuid(travel_outcome.get("resolved_target")),
        final_node_id=_travel_outcome_uuid(travel_outcome.get("final_node_id")),
        path=path_ids,
        consumed_cost=float(travel_outcome.get("consumed_cost") or 0.0),
        remaining_cost=float(travel_outcome.get("remaining_cost") or 0.0),
        cost_unit=str(travel_outcome.get("cost_unit") or "minutes"),  # type: ignore[arg-type]
        access_constraints_in_effect=tuple(
            str(item).strip()
            for item in list(travel_outcome.get("access_constraints_in_effect") or [])
            if str(item).strip()
        ),
        interruption_reason=str(travel_outcome.get("interruption_reason") or "").strip() or None,
    )
    current_zone_scope = (context_pack.get("current_zone") or {}).get("zone_id")

    raw_response = {
        "narration": narration,
        "choices": [choice.model_dump(mode="json") for choice in options],
        "proposed_updates": [],
        "memory_candidates": [],
        "semantic_events": geography_policy_domain.travel_semantic_event_payload(
            resolution=semantic_resolution,
            origin_node_id=path_ids[0] if path_ids else None,
        ),
        "scene_entities": [entity.model_dump(mode="json") for entity in _travel_scene_entities(context_pack)],
        "consequence_seeds": [],
        "resolved_consequence_ids": [],
        "zone_scope": str(current_zone_scope) if current_zone_scope else None,
        "in_game_time": {"day": in_game_day, "minute": in_game_minute},
        "travel_outcome": travel_outcome,
    }
    semantic_events = _normalize_semantic_events(raw_response.get("semantic_events"))
    return TurnPlanResult(
        narration=narration,
        choices=options,
        memory_candidates=[],
        semantic_events=semantic_events,
        scene_entities=_travel_scene_entities(context_pack),
        memory_trace=None,
        consequence_seeds=[],
        consequence_intents=[],
        resolved_consequence_ids=[],
        zone_scope=_travel_outcome_uuid(current_zone_scope),
        parsed_ops=[],
        validator_status="ok",
        validator_reasons=[],
        raw_response=raw_response,
        librarian_used=False,
        llm_usage=llm_usage,
        planner_contract_version=2,
    )


def _build_librarian_context_payload(
    *,
    context_pack: dict[str, Any],
    session_id: uuid.UUID,
    new_turn: int,
    user_input: str,
) -> dict[str, Any]:
    payload = {
        "session_id": str(session_id),
        "new_turn": new_turn,
        "user_input": user_input,
        "current_zone": context_pack.get("current_zone"),
        "world_constitution": _pick_world_constitution_text(context_pack),
    }
    economy_brief = _visible_economy_brief(context_pack)
    if economy_brief is not None:
        payload["economy_brief"] = economy_brief
    return payload


def _spine_in_static_prompt_context(context_pack: dict[str, Any] | None) -> bool:
    if not USE_PROMPT_CACHE_LAYOUT or not isinstance(context_pack, dict):
        return False
    raw_value = context_pack.get("narrative_spine_for_system")
    return raw_value not in (None, "", [], {})


def _inject_compact_spine_context(
    payload: dict[str, Any],
    *,
    context_pack: dict[str, Any] | None,
    spine_already_in_prompt: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    payload = dict(payload)
    spine_text = _pick_narrative_spine_text_for_system(context_pack or {})
    if not spine_text:
        return payload
    payload.pop("session_summaries", None)
    if spine_already_in_prompt or _spine_in_static_prompt_context(context_pack):
        return payload
    payload["narrative_spine"] = spine_text
    return payload


def _build_cacheable_prompt_payload(
    payload: dict[str, Any],
    *,
    context_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not USE_PROMPT_CACHE_LAYOUT:
        return payload

    static_context: dict[str, Any] = {}
    dynamic_context: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _PROMPT_CACHE_STATIC_KEYS:
            static_context[key] = value
        else:
            dynamic_context[key] = value

    if isinstance(context_pack, dict):
        for key in ("world_constitution_for_system", "world_prompt_for_system", "narrative_spine_for_system"):
            value = context_pack.get(key)
            if value in (None, "", [], {}):
                continue
            static_context[key] = value

    return {
        "a_static_context": static_context,
        "z_dynamic_context": dynamic_context,
    }


def _call_narrator(
    *,
    context_pack: dict[str, Any],
    response_mode: Literal["legacy", "world_intent"] = "legacy",
    session_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    world_constitution = _pick_world_constitution_text(context_pack)
    if not world_constitution:
        world_constitution = "No explicit world_prompt provided. Preserve consistency with persisted world state."
    narrative_spine = _pick_narrative_spine_text_for_system(context_pack)

    contract_fields = (
        "narration, choices, proposed_updates, memory_candidates, semantic_events, "
        "scene_entities, memory_trace, zone_scope, in_game_time, consequence_intents, resolved_consequence_ids."
        if response_mode == "legacy"
        else "proposed_updates, memory_candidates, semantic_events, scene_entities, "
        "consequence_intents, resolved_consequence_ids, zone_scope."
    )
    role_text = (
        "You are a strict game narrator."
        if response_mode == "legacy"
        else "You are a strict world-state planner. You output only committed-intent structure, never prose."
    )
    reaction_hint_rule = (
        "- If context has reaction_hints — reflect them in narration and choices.\n"
        if response_mode == "legacy"
        else "- If context has reaction_hints — reflect them in proposed_updates, semantic_events, memory_candidates, and scene_entities.\n"
    )
    system_prompt = (
        "# ROLE\n"
        f"{role_text} "
        "You MUST return ONLY a single valid JSON object with exactly these fields:\n"
        f"{contract_fields}\n\n"
        "# KNOWLEDGE BOUNDARY — ABSOLUTELY STRICT (check FIRST)\n"
        "NEVER give any NPC knowledge that is not explicitly allowed.\n"
        "For EVERY piece of information in ANY NPC dialogue or narration:\n"
        "1. Is it in this NPC's npc_knowledge entry? → YES = OK\n"
        "2. Is it directly observable right now? → YES = OK\n"
        "3. Is there a matching zone_claim they could realistically overhear? → YES = OK\n"
        "If NONE of the above — NPC CANNOT know it. "
        "If in doubt — assume they DO NOT know.\n"
        "If you would violate this — have the NPC speak vaguely, "
        "change the subject, or stay silent on that topic. NEVER invent knowledge.\n\n"
        "# OUTPUT RULES — YOU MUST FOLLOW\n"
        "- proposed_updates MUST follow PatchOp schema exactly.\n"
        "- Never mutate located_in links or emit player.move. Travel and player location changes are server-owned.\n"
        "- Use link.close to end obsolete links (except located_in and tracking_quest).\n"
        f"{reaction_hint_rule}"
        "- If context.travel_outcome is present, it is authoritative. Do not choose destinations, validate travel, or move the player. Narrate only the resolved outcome and resulting scene.\n"
        "- npc_knowledge may include off-zone NPCs; knowledge does NOT mean physical presence in this scene.\n"
        "- relevant_memories are narrator-global context only and are NEVER sufficient NPC evidence by themselves.\n"
        "- zone_npcs short_desc/attitude/relationships are flavor only and are NEVER evidence of factual NPC knowledge.\n"
        "- Temporary refs MUST match '^tmp:[A-Za-z0-9_-]+$'.\n"
        "- Use object.update for ANY state change (combat, mood, quests etc.).\n"
        f"- When player accepts/starts a quest, emit link.create type='{TRACKING_QUEST_LINK_TYPE}' "
        "from='session_player' to the quest object.\n"
        f"- When quest status becomes terminal ({', '.join(QUEST_TERMINAL_STATUSES)}), "
        "emit object.update only; do NOT delete tracking links manually.\n"
        "- Emit memory_candidates as the authoritative structured memory output.\n"
        "- memory_candidates[].layer='fact' is for canon-worthy memory that should consolidate over time.\n"
        "- memory_candidates[].layer='event' is for notable episodic memory that may still surface in recall.\n"
        "- Every memory_candidate must include search_recall_summary for semantic retrieval and narrative_recall_summary for compact narrator context.\n"
        "- Emit player_salience, expectation_salience, and continuity_contract_strength as enum values in {none, low, med, high}.\n"
        "- For canon-worthy memory, always include memory_candidates[].canonical_fact with explicit principal roles (actor_ref, counterparty_ref, object_ref, location_ref, quest_ref) and use context_refs only for extra retrieval context.\n"
        "- canonical_fact must include search_recall_summary, narrative_recall_summary, identity_text and state.\n"
        "- If memory_candidates[].layer='event', set event_role in {scene, supporting} and event_outcome in {asserted, failed_attempt, counterfactual}. scene events must include scene_refs. requires_commit=true is allowed only for event_role='scene' with event_outcome='asserted'. anchors are retrieval context only and must not define scene identity.\n"
        "- canonical_fact.identity_text is the stable semantic identity of the fact. Keep it the same across paraphrases of the same canon fact, and change it when the obligation/detail itself is different.\n"
        "- promise/debt/betrayal canonical_fact MUST include actor_ref and counterparty_ref explicitly.\n"
        "- gift canonical_fact MUST include object_ref for the gifted item and counterparty_ref for the current holder/recipient; if giver/source matters, put it in actor_ref and keep any extra context in context_refs.\n"
        "- ownership/trophy canonical_fact MUST include object_ref for the item and counterparty_ref for the current holder/location; keep extra context in context_refs.\n"
        "- relationship canonical_fact MUST include actor_ref, counterparty_ref, and relationship_type; keep any extra context in context_refs.\n"
        "- home_detail, recurring_prop, and location_fact may include location_ref only when callback validity depends on current placement. Do not smuggle placement through unordered context_refs or anchors.\n"
        "- Use durability plus emotional/obligation/sentimental/routine weights to express why the memory should persist.\n"
        "- Treat emotionally important scenes, recurring domestic details, keepsakes, and promises/debts as long-term memory even if they do not change combat/state numbers.\n"
        "- Canon-worthy classes include ownership, home decor, gifts, trophies, debts, promises, betrayals, injuries, quest milestones, recurring props/places, emotionally important scenes, and durable relationship changes.\n"
        "- For gifts, trophies, decor, promises, debts, and recurring home details, include item/zone/NPC/player anchors as retrieval context instead of leaving them as free-floating text.\n"
        "- Never leave canon-worthy facts only in narration.\n"
        "- Emit semantic_events for scene-level semantics that code should validate instead of reparsing language.\n"
        "- Every semantic_event MUST include requires_patch=true or false explicitly; code will not infer this from wording.\n"
        "- Always emit semantic_events for any death/kill claim. Use polarity in {asserted, negated, attempted, hypothetical, uncertain} only as narrative metadata.\n"
        f"- Only emit semantic_events[].type='move' with subject='{SESSION_PLAYER_REF}' when reflecting a server-resolved context.travel_outcome, and set requires_patch=false.\n"
        "- For death events, include subject ref or subject_hint and set requires_patch=true only when a committed death mutation/event must exist.\n"
        "- Emit scene_entities as the 1-6 salient current-scene entities for next-turn reference resolution.\n"
        "- If memory rows from context are used materially, emit memory_trace.used_relevant_ids / used_callback_ids / used_bundle_ids using only the prompt-local ids supplied in context.\n"
        "- Every scene_entity MUST set referent_candidate=true or false explicitly; code will not infer this from role or name.\n"
        "- hard_memory are non-negotiable canon reminders.\n"
        "- entity_histories provide compact continuity for active anchors.\n"
        "- callback_memories are optional resurfacing cues. Use them sparingly and never surface soft domestic callbacks when scene_mode is high_tension.\n\n"
        "- Treat uploaded lore as directional, not exhaustive. You may extend the world locally only when the addition fits the compiled session world model and explicit canon.\n"
        "- Absence from lore is not by itself a prohibition. Incompatibility with the compiled session envelope is.\n"
        "- Never introduce technology, institutions, supernatural systems, travel modes, or activities that fall outside the compiled session envelope unless the constitution lists them as explicit exceptions.\n\n"
        "- If context.has_world_constitution is true, do NOT emit object.create for world_constitution UNLESS explicitly instructed by consequence mode rules below.\n\n"
        "# CONSEQUENCES\n"
        "consequence_intents: earliest_turn and latest_turn are ABSOLUTE turn indices.\n"
        f"consequence_intents: keep windows bounded (latest_turn <= earliest_turn + {MAX_CONSEQUENCE_WINDOW_SPAN}).\n"
        "resolved_consequence_ids: only IDs that were actually addressed this turn.\n\n"
        "# WORLD CONSTITUTION (authoritative, never contradict)\n"
        f"{world_constitution}\n\n"
        "# ETERNAL CORE NARRATIVE SPINE (always preserve these facts)\n"
        f"{narrative_spine or '{}'}\n\n"
        "# FINAL SELF-CHECK (do this BEFORE outputting JSON)\n"
        "Before returning JSON answer these questions:\n"
        "1. Is the output pure JSON and nothing else?\n"
        "2. Did I respect Knowledge Boundary for every NPC?\n"
        "3. Did I follow all technical rules?\n"
        "4. Does any NPC have knowledge they should not have? Fix it.\n"
        "5. If player challenges NPC knowledge ('how do you know', 'where did you learn that'), "
        "did I verify the challenged fact against npc_knowledge and make NPC retract or admit uncertainty if unverified?\n"
        "If ANY answer is NO — fix it before outputting.\n"
        "Only after all YES — output the JSON.\n"
    )
    system_prompt = resolve_system_prompt("narrator_base", system_prompt)
    user_payload = _build_narrator_user_payload(context_pack)
    user_payload = _inject_compact_spine_context(
        user_payload,
        context_pack=context_pack,
        spine_already_in_prompt=True,
    )
    user_payload = _build_cacheable_prompt_payload(
        user_payload,
        context_pack=context_pack,
    )
    user_prompt = _normalize_json_preview_by_tokens(user_payload, max(TURN_CONTEXT_MAX_TOKENS, 1))
    request_type = "narrator_json" if response_mode == "legacy" else "world_intent_v2"
    with telemetry_context(request_type=request_type):
        return openrouter_chat.generate_json(
            model=OPENROUTER_NARRATOR_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_id=str(session_id) if session_id else None,
        )


def _call_librarian(
    *,
    state_summary: str,
    reasons: list[str],
    narrator_json: dict[str, Any],
    context_for_librarian: dict[str, Any],
    response_mode: Literal["legacy", "world_intent"] = "legacy",
    session_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    if response_mode == "legacy":
        system_prompt = (
            "You are a strict validator-corrector. Return only JSON using the narrator contract, including memory_candidates, semantic_events, scene_entities and optional memory_trace. "
            "Fix proposed_updates so they match PatchOp exactly. "
            "Preserve memory_candidates and their canonical_fact fields, and if a canon fact lacks a matching committed update, repair proposed_updates instead of dropping canon unless the fact is clearly unsupported. "
            "Preserve semantic_events and repair them when they disagree with narration or proposed_updates. "
            "Every semantic_event must keep an explicit requires_patch boolean. "
            "Preserve scene_entities as the salient entities needed for next-turn references. "
            "Every scene_entity must keep an explicit referent_candidate boolean. "
            "Treat uploaded lore as directional, not exhaustive: preserve explicit canon and allow only repairs that remain compatible with the compiled session world model. "
            "Do not introduce technology, institutions, supernatural systems, travel modes, or activities that violate the compiled session envelope unless they are explicit exceptions. "
            "Never emit player.move or mutate located_in links. Travel is server-owned. "
            "Use link.close to end obsolete links (except located_in and tracking_quest). "
            f"Ensure quest accept/reopen updates include link.create type='{TRACKING_QUEST_LINK_TYPE}' from player to quest. "
            "If quest update lacks required tracking link, rewrite ops accordingly. "
            "Preserve and validate consequence_intents and resolved_consequence_ids when present. "
            "Do not invent heard/asserted links or NPC knowledge unsupported by context. "
            "Treat zone_npcs short_desc/attitude/relationships as non-evidentiary flavor. "
            "Prefer dropping risky ops over speculative repairs."
        )
    else:
        system_prompt = (
            "You are a strict validator-corrector for the WorldIntentResponseV2 contract. "
            "Return only JSON with proposed_updates, memory_candidates, semantic_events, scene_entities, consequence_intents, resolved_consequence_ids, and zone_scope. "
            "Fix proposed_updates so they match PatchOp exactly. "
            "Preserve memory_candidates and their canonical_fact fields, and if a canon fact lacks a matching committed update, repair proposed_updates instead of dropping canon unless the fact is clearly unsupported. "
            "Preserve semantic_events and keep every semantic_event.requires_patch explicit. "
            "Preserve scene_entities as the salient entities needed for next-turn references, and keep every scene_entity.referent_candidate explicit. "
            "Preserve and validate consequence_intents and resolved_consequence_ids when present. "
            "Treat uploaded lore as directional, not exhaustive: preserve explicit canon and allow only repairs that remain compatible with the compiled session world model. "
            "Do not introduce technology, institutions, supernatural systems, travel modes, or activities that violate the compiled session envelope unless they are explicit exceptions. "
            "Never emit player.move or mutate located_in links. Travel is server-owned. "
            "Use link.close to end obsolete links (except located_in and tracking_quest). "
            f"Ensure quest accept/reopen updates include link.create type='{TRACKING_QUEST_LINK_TYPE}' from player to quest. "
            "If quest update lacks required tracking link, rewrite ops accordingly. "
            "Do not invent heard/asserted links or NPC knowledge unsupported by context. "
            "Treat zone_npcs short_desc/attitude/relationships as non-evidentiary flavor. "
            "Prefer dropping risky ops over speculative repairs."
        )
    
    world_constitution = context_for_librarian.get("world_constitution")
    if world_constitution:
        system_prompt += (
            "\n\nWorld Constitution (authoritative setting rules):\n"
            f"{world_constitution}"
        )
    system_prompt = resolve_system_prompt("librarian_validator", system_prompt)
        
    payload = {
        "state_summary": state_summary,
        "reasons": reasons,
        "narrator_json": narrator_json,
        "context_for_librarian": context_for_librarian,
    }
    payload = _build_cacheable_prompt_payload(
        payload,
        context_pack=context_for_librarian if isinstance(context_for_librarian, dict) else None,
    )
    request_type = "librarian_validator" if response_mode == "legacy" else "world_intent_validator"
    with telemetry_context(request_type=request_type):
        return openrouter_chat.generate_json(
            model=OPENROUTER_LIBRARIAN_MODEL,
            system_prompt=system_prompt,
            user_prompt=_normalize_json_preview_by_tokens(payload, max(TURN_CONTEXT_MAX_TOKENS, 1)),
            session_id=str(session_id) if session_id else None,
        )


def _call_narrator_text_only(
    *,
    context_pack: dict[str, Any],
    session_id: uuid.UUID | None = None,
) -> str:
    """Call xAI Narrator for narration text only (no structured JSON)."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    world_constitution = _pick_world_constitution_text(context_pack)
    if not world_constitution:
        world_constitution = "No explicit world_prompt provided. Preserve consistency with persisted world state."
    narrative_spine = _pick_narrative_spine_text_for_system(context_pack)

    system_prompt = (
        "You are a game narrator for a text RPG. "
        "Write an immersive scene narration in response to the player action. "
        "Include dialogue, atmosphere and consequences of the action. "
        "If context contains committed_turn, it is the authoritative committed result of the just-applied turn. "
        "Narrate only facts that are consistent with committed_turn and the current persisted state. "
        "If context contains reaction_hints, reflect them in the narrative. "
        "If context contains travel_outcome, it is authoritative: do not choose destinations, validate travel, or move the player. Narrate only the resolved outcome and resulting scene. "
        "KNOWLEDGE BOUNDARY - STRICTLY ENFORCED: The context contains npc_knowledge listing what each NPC has "
        "verified knowledge of (from heard/asserted graph links). Before writing any NPC dialogue/action that "
        "implies knowledge of a fact, check their npc_knowledge entry. If the fact is not verified there, the NPC "
        "cannot know it. They may suspect/guess/repeat rumors only if a matching claim exists in zone_claims that "
        "they could plausibly overhear. NPCs with empty npc_knowledge know only directly observable scene facts. "
        "npc_knowledge may include off-zone NPCs and does not imply physical presence in this scene. "
        "relevant_memories are global narrator context only and cannot by themselves justify NPC knowledge. "
        "hard_memory contains non-negotiable canon facts. "
        "entity_histories contain compact continuity for active anchors. "
        "callback_memories are optional resurfacing cues and should be used only when they are genuinely apt; "
        "if scene_mode is high_tension, do not surface soft domestic callbacks. "
        "Treat uploaded lore as directional, not exhaustive: you may extend the world only when the addition fits the compiled session world model and explicit canon. "
        "Do not introduce technology, institutions, supernatural systems, travel modes, or activities that fall outside the compiled session envelope unless they are explicit exceptions. "
        "zone_npcs short_desc/attitude/relationships are flavor only and cannot justify factual NPC knowledge. "
        "For each NPC line, apply this order: (1) in npc_knowledge? (2) overhearable zone_claim? "
        "(3) directly observable now? If none, do not attribute that knowledge. "
        "Do NOT output JSON. Do NOT output patch operations. "
        "Output ONLY the narration text, nothing else.\n\n"
        "Eternal Core Narrative Spine (always preserve these facts):\n"
        f"{narrative_spine or '{}'}\n\n"
        "World Constitution (authoritative setting rules):\n"
        f"{world_constitution}"
    )
    system_prompt = resolve_system_prompt("narrator_text_only", system_prompt)
    user_payload = _build_narrator_user_payload(context_pack)
    user_payload = _inject_compact_spine_context(
        user_payload,
        context_pack=context_pack,
        spine_already_in_prompt=True,
    )
    user_payload = _build_cacheable_prompt_payload(
        user_payload,
        context_pack=context_pack,
    )
    user_prompt = _normalize_json_preview_by_tokens(user_payload, max(TURN_CONTEXT_MAX_TOKENS, 1))
    with telemetry_context(request_type="narrator_text_only"):
        raw = openrouter_chat.generate_text(
            model=OPENROUTER_NARRATOR_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_id=str(session_id) if session_id else None,
        )

    text = str(raw).strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            extracted = str(
                payload.get("narration")
                or payload.get("text")
                or payload.get("message")
                or payload.get("content")
                or ""
            ).strip()
            if extracted:
                return extracted
    return text or FALLBACK_NO_RESPONSE


def _call_post_apply_narrator(
    *,
    context_pack: dict[str, Any],
    session_id: uuid.UUID | None = None,
) -> schemas.PostApplyNarrationResponse:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    world_constitution = _pick_world_constitution_text(context_pack)
    if not world_constitution:
        world_constitution = "No explicit world_prompt provided. Preserve consistency with persisted world state."
    narrative_spine = _pick_narrative_spine_text_for_system(context_pack)

    system_prompt = (
        "You are a post-commit narrator for a text RPG. "
        "Return ONLY valid JSON with fields narration and choices. "
        "If context contains committed_turn, it is the authoritative committed result of the just-applied turn. "
        "Narrate only facts that are consistent with committed_turn and the current persisted state. "
        "Choices must be grounded in the committed state you were given, not in hypothetical branches. "
        "If context contains reaction_hints, reflect them in the narration and choices. "
        "If context contains travel_outcome, it is authoritative: do not choose destinations, validate travel, or move the player. "
        "KNOWLEDGE BOUNDARY - STRICTLY ENFORCED: The context contains npc_knowledge listing what each NPC has "
        "verified knowledge of (from heard/asserted graph links). Before writing any NPC dialogue/action that "
        "implies knowledge of a fact, check their npc_knowledge entry. If the fact is not verified there, the NPC "
        "cannot know it. They may suspect/guess/repeat rumors only if a matching claim exists in zone_claims that "
        "they could plausibly overhear. NPCs with empty npc_knowledge know only directly observable scene facts. "
        "relevant_memories are global narrator context only and cannot by themselves justify NPC knowledge. "
        "Do NOT output patch operations or any fields other than narration and choices.\n\n"
        "Eternal Core Narrative Spine (always preserve these facts):\n"
        f"{narrative_spine or '{}'}\n\n"
        "World Constitution (authoritative setting rules):\n"
        f"{world_constitution}"
    )
    system_prompt = resolve_system_prompt("narrator_post_apply", system_prompt)
    user_payload = _build_narrator_user_payload(context_pack)
    user_payload = _inject_compact_spine_context(
        user_payload,
        context_pack=context_pack,
        spine_already_in_prompt=True,
    )
    user_payload = _build_cacheable_prompt_payload(
        user_payload,
        context_pack=context_pack,
    )
    user_prompt = _normalize_json_preview_by_tokens(user_payload, max(TURN_CONTEXT_MAX_TOKENS, 1))
    with telemetry_context(request_type="narrator_post_apply"):
        raw_payload = openrouter_chat.generate_json(
            model=OPENROUTER_NARRATOR_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_id=str(session_id) if session_id else None,
        )
    coerced_payload = _coerce_narrator_payload(raw_payload)
    return _parse_post_apply_narration_response(
        {
            "narration": coerced_payload.get("narration") or FALLBACK_NO_RESPONSE,
            "choices": coerced_payload.get("choices") or [],
        }
    )


_DEEPSEEK_PATCH_GENERATOR_SYSTEM = (
    "You are a structured data generator for a text RPG game engine. "
    "Given a narration scene and the current game context, produce ONLY a JSON object "
    "with these fields:\n"
    "- proposed_updates: array of PatchOp objects. Allowed ops: "
    "object.create, object.update, link.create, link.close, event.create. "
    "Use link.create/link.close field name 'from'. Never emit player.move or mutate located_in links.\n"
    "- object.create.type must be one of {player, npc, zone, item, quest, claim, faction, world_constitution}. "
    "Every object.create MUST include ref, type, name, and data. "
    "For type='world_constitution', set name='World Constitution'. "
    "Do not emit unsupported object types like fixture or location; encode persistent props/fixtures as item objects when they need canonical refs.\n"
    "- memory_candidates: array of {layer, kind, search_recall_summary, narrative_recall_summary, priority, actor_ref, counterparty_ref, object_ref, location_ref, quest_ref, context_refs, anchors, scene_refs, event_role, event_outcome, player_salience, expectation_salience, continuity_contract_strength, knowledge_scope, callback_strength, durability, emotional_weight, obligation_weight, sentimental_weight, routine_weight, requires_commit, canonical_fact?}. "
    "Use this as the authoritative structured memory output. "
    "layer='fact' is for canon-worthy durable memory; layer='event' is for episodic but recallable memory. "
    "If memory must become canon, include canonical_fact and set requires_commit=true when committed world state must reflect it.\n"
    "requires_commit event candidates must include scene_refs, event_role='scene', and event_outcome='asserted' so the same canon fact can still retain distinct episodic scenes. failed attempts and counterfactuals are event-only and must not create canon by themselves. anchors are retrieval context only.\n"
    "For event-layer memory that does not fit a durable canon kind, use kind='other'.\n"
    "canonical_fact must include search_recall_summary, narrative_recall_summary and identity_text for stable semantic identity where wording may vary or multiple facts can share the same actors.\n"
    "canonical_fact.identity_text is the stable semantic identity of the fact. Keep it invariant across paraphrases of the same fact and distinct for different obligations/details.\n"
    "Emit player_salience, expectation_salience, and continuity_contract_strength as enum values in {none, low, med, high}.\n"
    "promise/debt/betrayal canonical_fact MUST include actor_ref and counterparty_ref explicitly; keep any extra retrieval context in context_refs. gift canonical_fact MUST include object_ref for the gifted item and counterparty_ref for the current holder/recipient; if giver/source matters, put it in actor_ref and keep extra retrieval context in context_refs. ownership/trophy canonical_fact MUST include object_ref for the item and counterparty_ref for the current holder/location; keep extra retrieval context in context_refs. relationship canonical_fact MUST include actor_ref, counterparty_ref, and relationship_type; keep extra retrieval context in context_refs. home_detail/recurring_prop/location_fact may include location_ref only when placement is authoritative; do not encode placement through unordered context_refs or anchors.\n"
    "- semantic_events: array of {type, polarity, requires_patch, subject, object, source, target, subject_hint, object_hint, source_hint, target_hint, note}. "
    "Use this for scene semantics that code should validate without reparsing narration. "
    "Always emit move and death events when narration asserts, negates, or only attempts them. "
    "requires_patch must be explicit true or false for every event.\n"
    "- scene_entities: array of {ref, name, entity_type, role, referent_candidate, salience}. "
    "scene_entities.entity_type must be one of {player, npc, zone, item, quest, faction, other}; use zone for locations and item/other for fixtures or subareas. "
    "scene_entities.role must be one of {focus, speaker, interlocutor, target, observer, location, mentioned, other}; use mentioned when no exact role fits.\n"
    "Include only the 1-6 most salient current-scene entities needed for next-turn references.\n"
    "- zone_scope: UUID string of the current zone, or null.\n"
    "- in_game_time: {day: int|null, minute: int|null}.\n"
    "- turn_weight: number in [0.0, 1.0] for how important this turn is for future context.\n"
    "- choices: array of {id, text, intent, risk} for player options.\n\n"
    "RULES:\n"
    "- Contract: narration is the source of truth. Every factual world-state change mentioned in narration "
    "MUST have a corresponding PatchOp in proposed_updates.\n"
    "- Treat uploaded lore as directional, not exhaustive: you may extend the world only when the addition fits the compiled session world model and explicit canon.\n"
    "- Do not introduce technology, institutions, supernatural systems, travel modes, or activities that fall outside the compiled session envelope unless they are explicit exceptions.\n"
    "- Every canon memory emitted via memory_candidates[].canonical_fact must have a matching committed PatchOp touching the same principal anchors.\n"
    "- Use memory_candidates[].canonical_fact for canon-worthy classes: ownership, home decor, gifts, trophies, debts, promises, betrayals, injuries, quest milestones, recurring props/places, emotionally important scenes, and durable relationship changes.\n"
    "- For gifts, trophies, decor, debts, promises, emotional scenes, and recurring home details, include concrete principal refs plus contextual anchors/context_refs so the memory becomes structured canon instead of floating narration.\n"
    "- Do NOT emit trivial atmospheric details as canonical_fact.\n"
    f"- Only emit semantic_events type='move' with subject='{SESSION_PLAYER_REF}' when reflecting a server-resolved context.travel_outcome, and set requires_patch=false.\n"
    "- If narration contains death/kill semantics, emit semantic_events type='death' with polarity, requires_patch, and subject ref/hint.\n"
    "- Use polarity='negated' or 'attempted' for near-miss or explicitly denied events instead of pretending they happened, but still set requires_patch explicitly.\n"
    "- scene_entities must set referent_candidate=true only for entities that should resolve next-turn pronouns; do not rely on code inferring this from role or type.\n"
    "- If narration says someone was killed/died, include object.update or event.create that persists it.\n"
    "- If narration says someone took an item, include object.update/link.create/object.create to persist it.\n"
    "- Never change player location through proposed_updates. Travel resolution is server-owned.\n"
    "- After each turn, emit object.update ops to maintain data.ctx_weight (0.0..1.0) for touched entities.\n"
    "- Increase ctx_weight for NPCs/items/quests directly involved this turn.\n"
    "- Decrease ctx_weight for entities not mentioned in recent turns when appropriate.\n"
    "- The player's current zone should have ctx_weight=1.0.\n"
    "- Temporary refs for object.create must match '^tmp:[A-Za-z0-9_-]+$'.\n"
    "- If you create an object with ref=tmp:X, you can reference it in subsequent ops.\n"
    "- Use object.update to persist stateful changes (combat, conditions, mood, quest progress).\n"
    "- Use link.close when a previously true relationship/ownership is no longer true.\n"
    "- orphaned_items lists recently ownerless items; decide whether they stay ownerless or need explicit reassignment.\n"
    f"- Never use link.close type='{TRACKING_QUEST_LINK_TYPE}'; quest terminal transitions close it automatically.\n"
    "- Never use link.close type='located_in'.\n"
    f"- When player accepts/starts a quest, emit link.create type='{TRACKING_QUEST_LINK_TYPE}' "
    "from='session_player' to the quest.\n"
    f"- When a quest becomes terminal ({', '.join(QUEST_TERMINAL_STATUSES)}), "
    "emit object.update for status and do NOT delete tracking links manually.\n"
    "- relevant_memories are global context only; never treat them as NPC evidence.\n"
    "- Treat zone_npcs short_desc/attitude/relationships as flavor only, not knowledge evidence.\n"
    "- Create heard/asserted links only when context evidence supports who could know the claim.\n"
    "- Use link.create (type: heard, asserted) to assign NPC knowledge of claims. "
    "Do NOT write knowledge arrays into npc.data.\n"
    "- If context.has_world_constitution is true, do NOT emit object.create for world_constitution UNLESS explicitly instructed by consequence mode rules below.\n"
    "- If uncertain about evidence, omit risky knowledge/link ops.\n"
    "- Return ONLY valid JSON. No markdown, no explanations."
)

_DEEPSEEK_CONSEQUENCE_EXTENSION_SYSTEM = (
    "\nAdditional fields when consequence mode is active:\n"
    "- consequence_intents: array of {intent_key, priority, earliest_turn, latest_turn, max_shows, "
    "target_refs, anchor_refs, reason_kind}. Use when events create delayed pressure.\n"
    "- resolved_consequence_ids: array of ids for latent consequences realized in this turn.\n"
    "Additional rules when consequence mode is active:\n"
    "- latent_consequences are hints, not obligations: realize zero, one, or more.\n"
    "- structural_signals are facts, not instructions: decide if they warrant new consequence_intents.\n"
    "- earliest_turn/latest_turn are absolute turn indices (not offsets). "
    "For delay N turns, set earliest_turn=context.new_turn+N and latest_turn>=earliest_turn.\n"
    f"- Keep consequence windows bounded: latest_turn<=earliest_turn+{MAX_CONSEQUENCE_WINDOW_SPAN}.\n"
    "- If context.new_turn == 1 or not context.has_world_constitution, MUST emit object.create for type='world_constitution' "
    "containing structural_triggers relevant to this world (they will merge if constitution already exists).\n"
    "- structural_triggers MUST follow schema: {trigger: string, ops?: string[], object_type?: string, object_tags?: string[], data_flag?: string|dict}.\n"
    "- To append NEW structural_triggers later, emit object.create for type='world_constitution' again (new triggers will be merged).\n"
)


_PATCH_CONTEXT_SHORT_DESC_KEYS = ("short_desc", "description", "desc", "summary", "objective", "status")


def _extract_patch_context_short_desc(item: dict[str, Any]) -> str:
    for source in (item, item.get("data")):
        if not isinstance(source, dict):
            continue
        for key in _PATCH_CONTEXT_SHORT_DESC_KEYS:
            raw_value = source.get(key)
            if raw_value is None:
                continue
            text = str(raw_value).strip()
            if text:
                return _truncate_text(text, 180)
    return ""


def _slim_patch_context_entities(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []

    slim_items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        object_id = str(raw_item.get("object_id") or raw_item.get("id") or "").strip()
        if not object_id:
            continue
        name = str(raw_item.get("name") or "").strip()
        short_desc = _extract_patch_context_short_desc(raw_item)
        compact: dict[str, Any] = {"object_id": object_id, "name": name}
        if short_desc:
            compact["short_desc"] = short_desc
            
        data = raw_item.get("data")
        if isinstance(data, dict):
            for k in ("objectives", "progress", "status", "stage"):
                if k in data:
                    compact[k] = data[k]

        raw_weight = raw_item.get("ctx_weight")
        if not isinstance(raw_weight, (int, float)) and isinstance(raw_item.get("data"), dict):
            raw_weight = raw_item["data"].get("ctx_weight")
        if isinstance(raw_weight, (int, float)) and not isinstance(raw_weight, bool):
            compact["ctx_weight"] = round(min(max(float(raw_weight), 0.0), 1.0), 6)
        slim_items.append(compact)
    return [dict(item) for item in slim_items]


def _slim_archived_quest_recall(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []

    slim_items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        object_id = str(raw_item.get("object_id") or raw_item.get("id") or "").strip()
        if not object_id:
            continue
        compact: dict[str, Any] = {
            "object_id": object_id,
            "name": str(raw_item.get("name") or "").strip(),
            "archived": True,
        }
        raw_data = raw_item.get("data")
        if isinstance(raw_data, dict):
            status_text = str(raw_data.get("status") or "").strip()
            if status_text:
                compact["status"] = _truncate_text(status_text, 40)
            short_desc_text = str(raw_data.get("short_desc") or "").strip()
            if short_desc_text:
                compact["short_desc"] = _truncate_text(short_desc_text, 180)
        raw_similarity = raw_item.get("similarity")
        if isinstance(raw_similarity, (int, float)) and not isinstance(raw_similarity, bool):
            compact["similarity"] = round(float(raw_similarity), 6)
        slim_items.append(compact)
    return [dict(item) for item in slim_items]


def _slim_orphaned_items(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []

    slim_items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        object_id = str(raw_item.get("object_id") or raw_item.get("id") or "").strip()
        if not object_id:
            continue
        compact: dict[str, Any] = {
            "object_id": object_id,
            "name": str(raw_item.get("name") or "").strip(),
        }
        short_desc = _extract_patch_context_short_desc(raw_item)
        if short_desc:
            compact["short_desc"] = short_desc
        last_owner_id = str(raw_item.get("last_owner_id") or "").strip()
        if last_owner_id:
            compact["last_owner_id"] = last_owner_id
        last_owner_name = str(raw_item.get("last_owner_name") or "").strip()
        if last_owner_name:
            compact["last_owner_name"] = _truncate_text(last_owner_name, 80)
        owner_lost_turn = raw_item.get("owner_lost_turn")
        if isinstance(owner_lost_turn, int) and not isinstance(owner_lost_turn, bool):
            compact["owner_lost_turn"] = owner_lost_turn
        slim_items.append(compact)
    return [dict(item) for item in slim_items]


def _call_deepseek_patch_generator(
    *,
    narration: str,
    context_pack: dict[str, Any],
    session_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Call DeepSeek to generate structured patch ops from narration + context."""
    payload = {
        "narration": _truncate_text(narration, 2000),
        "user_input": str(context_pack.get("user_input") or ""),
        "current_zone": context_pack.get("current_zone"),
        "has_world_constitution": context_pack.get("has_world_constitution"),
        "player": context_pack.get("player"),
        "player_inventory": context_pack.get("player_inventory"),
        "zone_npcs": _slim_patch_context_entities(context_pack.get("zone_npcs")),
        "relevant_quests": _slim_patch_context_entities(context_pack.get("relevant_quests")),
        "archived_quest_recall": _slim_archived_quest_recall(
            context_pack.get("archived_quest_recall")
        ),
        "relevant_items": _slim_patch_context_entities(context_pack.get("relevant_items")),
        "orphaned_items": _slim_orphaned_items(context_pack.get("orphaned_items")),
        "relevant_factions": _slim_patch_context_entities(context_pack.get("relevant_factions")),
        "session_summaries": context_pack.get("session_summaries", [])[:3],
        "recent_turns": context_pack.get("recent_turns", [])[-5:],
        "scene_mode": context_pack.get("scene_mode"),
        # Callback memories only guide narration resurfacing; the patch generator
        # should derive structure from narration plus hard canon anchors.
        "hard_memory": context_pack.get("hard_memory", []),
        "entity_histories": context_pack.get("entity_histories", []),
    }
    payload = _inject_compact_spine_context(
        payload,
        context_pack=context_pack,
        spine_already_in_prompt=False,
    )
    
    world_constitution = _pick_world_constitution_text(context_pack)
    if not world_constitution:
        world_constitution = "No explicit world_prompt provided. Preserve consistency with persisted world state."

    system_prompt = _DEEPSEEK_PATCH_GENERATOR_SYSTEM
    if USE_CONSEQUENCES:
        payload["new_turn"] = context_pack.get("new_turn")
        payload["latent_consequences"] = context_pack.get("latent_consequences", [])
        payload["structural_signals"] = context_pack.get("structural_signals", [])
        consequence_ext = resolve_system_prompt(
            "deepseek_consequence_extension",
            _DEEPSEEK_CONSEQUENCE_EXTENSION_SYSTEM,
        )
        system_prompt = _DEEPSEEK_PATCH_GENERATOR_SYSTEM + consequence_ext
    payload = _build_cacheable_prompt_payload(
        payload,
        context_pack=context_pack,
    )

    system_prompt += (
        "\n\nWorld Constitution (authoritative setting rules):\n"
        f"{world_constitution}"
    )
    system_prompt = resolve_system_prompt("deepseek_patch_generator", system_prompt)

    with telemetry_context(request_type="deepseek_patch_generator"):
        return openrouter_chat.generate_json(
            model=OPENROUTER_CHAT_MODEL,
            system_prompt=system_prompt,
            user_prompt=_normalize_json_preview_by_tokens(payload, max(TURN_CONTEXT_MAX_TOKENS, 1)),
            session_id=str(session_id) if session_id else None,
            max_tokens=1600,
        )


def _build_ops_summary_for_desync(ops: list[schemas.PatchOp], *, limit: int = 12) -> str:
    parts: list[str] = []
    for op in ops[:limit]:
        label = op.op
        if isinstance(op, schemas.ObjectUpdateOp):
            patch_keys = ", ".join(sorted((op.patch or {}).keys())[:4])
            label = f"object.update(patch=[{patch_keys}])"
        elif isinstance(op, schemas.ObjectCreateOp):
            label = f"object.create(type={op.type}, name={op.name})"
        elif isinstance(op, (schemas.LinkCreateOp, schemas.LinkCloseOp)):
            label = f"{op.op}(type={op.type})"
        elif isinstance(op, schemas.PlayerMoveOp):
            label = "player.move"
        elif isinstance(op, schemas.EventCreateOp):
            label = f"event.create(type={op.type})"
        parts.append(label)
    return "; ".join(parts)


def _iter_patch_scalars(value: Any) -> list[tuple[str, Any]]:
    if not isinstance(value, dict):
        return []

    collected: list[tuple[str, Any]] = []
    stack: list[dict[str, Any]] = [value]
    while stack:
        current = stack.pop()
        for key, nested in current.items():
            collected.append((str(key), nested))
            if isinstance(nested, dict):
                stack.append(nested)
    return collected


def _desync_tokens(text: str) -> list[str]:
    return _DESYNC_WORD_RE.findall(str(text or ""))


def _desync_token_forms(token: str) -> set[str]:
    normalized = str(token or "").strip().casefold()
    if len(normalized) < 2:
        return set()

    forms = {normalized}
    if re.search(r"[А-Яа-яЁё]", normalized):
        for suffix in _DESYNC_RU_INFLECTION_SUFFIXES:
            if not normalized.endswith(suffix):
                continue
            stem = normalized[: -len(suffix)]
            if len(stem) >= 3:
                forms.add(stem)
            break
    elif normalized.isascii() and normalized.isalpha():
        if normalized.endswith("ies") and len(normalized) > 4:
            forms.add(normalized[:-3] + "y")
        elif normalized.endswith("es") and len(normalized) > 4:
            forms.add(normalized[:-2])
        elif normalized.endswith("s") and len(normalized) > 3:
            forms.add(normalized[:-1])
    return forms


def _desync_semantic_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _DESYNC_WORD_RE.findall(str(text or "")):
        folded = token.casefold()
        if len(folded) < 2 or folded in _DESYNC_SEMANTIC_STOPWORDS:
            continue
        terms.update(_desync_token_forms(token))
    return terms


def _build_desync_name_map(
    proposed_updates: list[schemas.PatchOp],
    *,
    context_pack: dict[str, Any] | None = None,
) -> dict[str, str]:
    name_map: dict[str, str] = {}

    def _register(raw_id: Any, raw_name: Any) -> None:
        object_id = str(raw_id or "").strip()
        name = str(raw_name or "").strip()
        if object_id and name:
            name_map[object_id] = name

    current_zone = context_pack.get("current_zone") if isinstance(context_pack, dict) else None
    if isinstance(current_zone, dict):
        _register(current_zone.get("zone_id"), current_zone.get("zone_name"))

    if isinstance(context_pack, dict):
        player_payload = context_pack.get("player")
        if isinstance(player_payload, dict):
            player_data = player_payload.get("data") if isinstance(player_payload.get("data"), dict) else {}
            _register(player_payload.get("object_id"), player_payload.get("name") or player_data.get("name"))
        for key in _DESYNC_CONTEXT_NAME_KEYS:
            raw_items = context_pack.get(key)
            if not isinstance(raw_items, list):
                continue
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                _register(item.get("object_id"), item.get("name"))

    for op in proposed_updates:
        if isinstance(op, schemas.ObjectCreateOp):
            _register(op.ref, op.name)
    return name_map


def _ref_semantic_terms(ref: Any, *, name_map: dict[str, str]) -> set[str]:
    ref_text = str(ref or "").strip()
    if not ref_text:
        return set()
    if ref_text == SESSION_PLAYER_REF:
        return {"player", "hero", "игрок", "герой", "ты", "you"}
    mapped_name = name_map.get(ref_text)
    if mapped_name:
        return _desync_semantic_terms(mapped_name)
    if ref_text.startswith("tmp:"):
        return _desync_semantic_terms(ref_text[4:].replace("_", " ").replace("-", " "))
    return set()


def _normalize_semantic_events(events: Any) -> list[schemas.SemanticEvent]:
    if not isinstance(events, list):
        return []
    normalized: list[schemas.SemanticEvent] = []
    for item in events:
        if isinstance(item, schemas.SemanticEvent):
            normalized.append(item)
            continue
        if not isinstance(item, dict):
            continue
        try:
            normalized.append(schemas.SemanticEvent.model_validate(item))
        except ValidationError:
            continue
    return normalized


def _semantic_ref_text(ref: Any) -> str:
    return str(ref or "").strip()


def _semantic_event_anchor_terms(
    event: schemas.SemanticEvent,
    *,
    ref_field: str,
    hint_field: str,
    name_map: dict[str, str],
) -> set[str]:
    terms = _ref_semantic_terms(getattr(event, ref_field), name_map=name_map)
    hint_value = str(getattr(event, hint_field) or "").strip()
    if hint_value:
        terms.update(_desync_semantic_terms(hint_value))
    return terms


def _semantic_event_is_player_subject(event: schemas.SemanticEvent, *, name_map: dict[str, str]) -> bool:
    del name_map
    return _semantic_ref_text(event.subject) == SESSION_PLAYER_REF


def _move_op_matches_semantic_event(
    event: schemas.SemanticEvent,
    move_op: schemas.PlayerMoveOp,
    *,
    name_map: dict[str, str],
) -> bool:
    subject_ref = _semantic_ref_text(event.subject)
    if subject_ref and _semantic_ref_text(move_op.player) != subject_ref:
        return False

    op_target_ref = _semantic_ref_text(move_op.to)
    target_ref = _semantic_ref_text(event.target)
    source_ref = _semantic_ref_text(event.source)
    if target_ref:
        if op_target_ref != target_ref:
            return False
    elif source_ref and op_target_ref == source_ref:
        return False

    op_target_terms = _ref_semantic_terms(move_op.to, name_map=name_map)
    target_terms = _semantic_event_anchor_terms(
        event,
        ref_field="target",
        hint_field="target_hint",
        name_map=name_map,
    )
    if target_terms and not (op_target_terms & target_terms):
        return False

    source_terms = _semantic_event_anchor_terms(
        event,
        ref_field="source",
        hint_field="source_hint",
        name_map=name_map,
    )
    if source_terms and not target_ref and not target_terms and not op_target_terms:
        return False
    if source_terms and op_target_terms and (op_target_terms & source_terms) and not target_terms:
        return False
    return True


def _player_move_semantic_events(
    semantic_events: list[schemas.SemanticEvent],
    *,
    name_map: dict[str, str],
) -> list[schemas.SemanticEvent]:
    return [
        event
        for event in semantic_events
        if event.type == "move" and _semantic_event_is_player_subject(event, name_map=name_map)
    ]


def _player_move_contract_reasons(
    semantic_events: list[schemas.SemanticEvent],
    proposed_updates: list[schemas.PatchOp],
    *,
    name_map: dict[str, str],
) -> list[str]:
    player_move_events = _player_move_semantic_events(semantic_events, name_map=name_map)
    required_events = [event for event in player_move_events if event.requires_patch]
    move_ops = [op for op in proposed_updates if isinstance(op, schemas.PlayerMoveOp)]

    reasons: list[str] = []
    if required_events and not move_ops:
        reasons.append("semantic_desync: missing movement op")
        return reasons

    if required_events and any(
        not any(_move_op_matches_semantic_event(event, move_op, name_map=name_map) for move_op in move_ops)
        for event in required_events
    ):
        reasons.append("semantic_desync: missing movement op")

    if move_ops and (
        not required_events
        or any(
            not any(_move_op_matches_semantic_event(event, move_op, name_map=name_map) for event in required_events)
            for move_op in move_ops
        )
    ):
        reasons.append("semantic_contract: player.move missing semantic event")
    return reasons[:2]


def _server_owned_movement_reasons(
    proposed_updates: list[schemas.PatchOp],
    semantic_events: list[schemas.SemanticEvent],
) -> list[str]:
    reasons: list[str] = []
    if any(isinstance(op, schemas.PlayerMoveOp) for op in proposed_updates):
        reasons.append("server_owned_movement: player.move is forbidden in narrator plans")
    if any(event.type == "move" and bool(event.requires_patch) for event in semantic_events):
        reasons.append("server_owned_movement: move semantic events must not require_patch")
    return reasons[:2]


def _death_op_anchor_terms(
    op: schemas.PatchOp,
    *,
    name_map: dict[str, str],
) -> set[str]:
    if isinstance(op, schemas.EventCreateOp) and _death_event_type_matches(op.type):
        terms = _extract_named_event_payload_terms(op.payload or {})
        if terms:
            return terms
        if op.scope is not None:
            return _ref_semantic_terms(op.scope, name_map=name_map)
        return set()
    if isinstance(op, schemas.ObjectUpdateOp) and _patch_persists_death(op):
        return _ref_semantic_terms(op.object, name_map=name_map)
    return set()


def _death_op_ref(op: schemas.PatchOp) -> str:
    if isinstance(op, schemas.EventCreateOp):
        return _semantic_ref_text(op.scope)
    if isinstance(op, schemas.ObjectUpdateOp):
        return _semantic_ref_text(op.object)
    return ""


def _death_op_matches_semantic_event(
    event: schemas.SemanticEvent,
    op: schemas.PatchOp,
    *,
    name_map: dict[str, str],
) -> bool:
    subject_ref = _semantic_ref_text(event.subject)
    op_ref = _death_op_ref(op)
    if subject_ref:
        return op_ref == subject_ref

    subject_terms = _semantic_event_anchor_terms(
        event,
        ref_field="subject",
        hint_field="subject_hint",
        name_map=name_map,
    )
    if not subject_terms:
        return False
    return bool(_death_op_anchor_terms(op, name_map=name_map) & subject_terms)


def _death_contract_reasons(
    semantic_events: list[schemas.SemanticEvent],
    proposed_updates: list[schemas.PatchOp],
    *,
    name_map: dict[str, str],
) -> list[str]:
    required_events = [
        event
        for event in semantic_events
        if event.type == "death" and event.requires_patch
    ]
    candidate_ops = [
        op
        for op in proposed_updates
        if (isinstance(op, schemas.EventCreateOp) and _death_event_type_matches(op.type))
        or (isinstance(op, schemas.ObjectUpdateOp) and _patch_persists_death(op))
    ]
    reasons: list[str] = []
    if required_events and not candidate_ops:
        reasons.append("semantic_desync: missing death state mutation")
        return reasons

    if required_events and any(
        not any(_death_op_matches_semantic_event(event, op, name_map=name_map) for op in candidate_ops)
        for event in required_events
    ):
        reasons.append("semantic_desync: missing death state mutation")

    if candidate_ops and (
        not required_events
        or any(
            not any(_death_op_matches_semantic_event(event, op, name_map=name_map) for event in required_events)
            for op in candidate_ops
        )
    ):
        reasons.append("semantic_contract: death mutation missing semantic event")
    return reasons[:2]


def _patch_persists_death(op: schemas.ObjectUpdateOp) -> bool:
    for key, raw_value in _iter_patch_scalars(getattr(op, "patch", {})):
        normalized_key = key.strip().casefold()
        if normalized_key in {"hp", "health"}:
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = None
            if numeric_value is not None and numeric_value <= 0:
                return True
        elif normalized_key in {"alive", "is_alive"} and raw_value is False:
            return True
        elif normalized_key in {"dead", "is_dead"} and raw_value is True:
            return True
        elif normalized_key in {"status", "state", "condition"}:
            normalized_value = str(raw_value or "").strip().casefold()
            if normalized_value in _DESYNC_DEATH_STATUS_VALUES:
                return True
    return False


def _death_event_type_matches(event_type: str) -> bool:
    normalized = str(event_type or "").strip().casefold()
    return any(token in normalized for token in ("death", "died", "killed", "slain", "dead", "умер", "погиб", "убит"))


def _extract_named_event_payload_terms(payload: dict[str, Any]) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    for key in _DESYNC_EVENT_NAME_KEYS:
        terms = _desync_semantic_terms(str(payload.get(key) or ""))
        if terms:
            return terms
    return set()


def _find_narration_semantic_desync_reasons(
    narration: str,
    proposed_updates: list[schemas.PatchOp],
    *,
    semantic_events: Any = None,
    context_pack: dict[str, Any] | None = None,
) -> list[str]:
    text = str(narration or "").strip()
    if not text:
        return []

    normalized_events = _normalize_semantic_events(semantic_events)
    name_map = _build_desync_name_map(proposed_updates, context_pack=context_pack)
    reasons = _player_move_contract_reasons(
        normalized_events,
        proposed_updates,
        name_map=name_map,
    )
    reasons.extend(
        reason
        for reason in _death_contract_reasons(
            normalized_events,
            proposed_updates,
            name_map=name_map,
        )
        if reason not in reasons
    )
    if not reasons and not proposed_updates and len(text) > 80:
        return ["semantic_desync: non-trivial narration with zero proposed_updates"]
    return reasons[:4]


def _build_plan_from_debug_patch(
    payload: schemas.TurnIn,
    *,
    allow_debug_patch: bool,
    in_game_day: int,
    in_game_minute: int,
) -> TurnPlanResult:
    if payload.debug_patch is None:
        raise RuntimeError("Debug patch payload is missing")
    if not allow_debug_patch:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Debug patch is disabled")

    ai_json = dict(payload.debug_patch.ai_json or {})
    raw_response: dict[str, Any] = {
        "narration": payload.debug_patch.ai_text,
        "choices": ai_json.get("choices", []),
        "proposed_updates": _serialize_patch_ops(payload.debug_patch.ops),
        "memory_candidates": ai_json.get("memory_candidates", []),
        "semantic_events": ai_json.get("semantic_events", []),
        "scene_entities": ai_json.get("scene_entities", []),
        "consequence_seeds": ai_json.get("consequence_seeds", []),
        "consequence_intents": ai_json.get("consequence_intents", []),
        "resolved_consequence_ids": ai_json.get("resolved_consequence_ids", []),
        "zone_scope": ai_json.get("zone_scope"),
        "in_game_time": {"day": in_game_day, "minute": in_game_minute},
        "turn_weight": _coerce_turn_weight(ai_json.get("turn_weight")),
        "planner_contract_version": int(ai_json.get("planner_contract_version") or 2),
    }

    try:
        parsed = _parse_narrator_response(raw_response)
    except ValidationError as exc:
        return TurnPlanResult(
            narration="Debug patch JSON is invalid",
            choices=[],
            memory_candidates=[],
            consequence_seeds=[],
            consequence_intents=[],
            resolved_consequence_ids=[],
            zone_scope=None,
            parsed_ops=[],
            validator_status="reject",
            validator_reasons=[f"debug_patch invalid: {exc.errors()[0]['msg']}"],
            raw_response=raw_response,
            librarian_used=False,
            semantic_events=[],
            scene_entities=[],
            memory_trace=None,
            llm_usage={},
        )

    validation = _validate_patch_ops(raw_response.get("proposed_updates"))
    return TurnPlanResult(
        narration=parsed.narration,
        choices=parsed.choices,
        memory_candidates=parsed.memory_candidates,
        semantic_events=parsed.semantic_events,
        scene_entities=parsed.scene_entities,
        memory_trace=parsed.memory_trace,
        consequence_seeds=parsed.consequence_seeds,
        consequence_intents=[
            schemas.ConsequenceIntent.model_validate(item)
            for item in list(raw_response.get("consequence_intents") or [])
            if isinstance(item, dict)
        ],
        resolved_consequence_ids=parsed.resolved_consequence_ids,
        zone_scope=parsed.zone_scope,
        parsed_ops=validation.parsed_ops,
        validator_status=validation.status,
        validator_reasons=validation.reasons,
        raw_response=raw_response,
        librarian_used=False,
        llm_usage={},
        planner_contract_version=int(raw_response.get("planner_contract_version") or 2),
    )


def _resolve_turn_plan_legacy(
    db: Session,
    session_id: uuid.UUID,
    *,
    payload: schemas.TurnIn,
    context_pack: dict[str, Any],
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
) -> TurnPlanResult:
    """Original single-call path: Narrator generates narration + structured patches together."""
    librarian_used = False
    llm_usage: dict[str, Any] = {"narrator": None, "librarian": None}

    t0 = time.monotonic()
    try:
        narrator_raw_raw = _call_narrator(context_pack=context_pack, session_id=session_id)
        narrator_latency = time.monotonic() - t0
        logger.info("_call_narrator latency=%.3fs session=%s turn=%s", narrator_latency, session_id, new_turn)
        narrator_usage = _extract_xai_usage(narrator_raw_raw)
        if narrator_usage is not None:
            llm_usage["narrator"] = narrator_usage
        narrator_raw = _coerce_narrator_payload(narrator_raw_raw)
        try:
            narrator = _parse_narrator_response(narrator_raw)
        except ValidationError as parse_exc:
            logger.warning(
                "Narrator payload failed schema validation for session_id=%s turn_index=%s; trying librarian repair",
                session_id,
                new_turn,
            )
            librarian_raw = _call_librarian(
                state_summary=context_pack.get("state_summary", ""),
                reasons=[f"narrator_schema_error: {parse_exc.errors()[0]['msg']}"],
                narrator_json=narrator_raw_raw if isinstance(narrator_raw_raw, dict) else {"raw": str(narrator_raw_raw)},
                context_for_librarian=_build_librarian_context_payload(
                    context_pack=context_pack,
                    session_id=session_id,
                    new_turn=new_turn,
                    user_input=payload.user_input,
                ),
                session_id=session_id,
            )
            librarian_usage = _extract_xai_usage(librarian_raw)
            if librarian_usage is not None:
                llm_usage["librarian"] = librarian_usage
            librarian_used = True
            narrator_raw = _coerce_narrator_payload(librarian_raw)
            narrator = _parse_narrator_response(narrator_raw)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, RuntimeError) and "OPENROUTER_API_KEY" in str(exc):
            logger.warning(
                "Narrator skipped for session_id=%s turn_index=%s: xAI key is missing",
                session_id,
                new_turn,
            )
        else:
            logger.exception("Narrator call failed for session_id=%s turn_index=%s", session_id, new_turn)
        fallback_narration = FALLBACK_AI_UNAVAILABLE
        fallback: dict[str, Any] = {
            "narration": fallback_narration,
            "choices": [],
            "proposed_updates": [],
            "memory_candidates": [],
            "semantic_events": [],
            "scene_entities": [],
            "consequence_seeds": [],
            "resolved_consequence_ids": [],
            "zone_scope": str(context_pack.get("current_zone", {}).get("zone_id"))
            if context_pack.get("current_zone", {}).get("zone_id")
            else None,
            "in_game_time": {"day": in_game_day, "minute": in_game_minute},
        }
        return TurnPlanResult(
            narration=fallback_narration,
            choices=[],
            memory_candidates=[],
            consequence_seeds=[],
            consequence_intents=[],
            resolved_consequence_ids=[],
            zone_scope=None,
            parsed_ops=[],
            validator_status="reject",
            validator_reasons=[f"narrator_error: {type(exc).__name__}"],
            raw_response=fallback,
            librarian_used=False,
            semantic_events=[],
            scene_entities=[],
            memory_trace=None,
            llm_usage=llm_usage,
        )

    validation = _validate_patch_ops(narrator_raw.get("proposed_updates"))
    chosen_raw = narrator_raw
    chosen = narrator

    if validation.status != "reject":
        movement_reasons = _server_owned_movement_reasons(
            validation.parsed_ops,
            chosen.semantic_events,
        )
        if movement_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + movement_reasons,
                parsed_ops=validation.parsed_ops,
            )

    if validation.status != "reject":
        semantic_reasons = _find_narration_semantic_desync_reasons(
            chosen.narration,
            validation.parsed_ops,
            semantic_events=chosen.semantic_events,
            context_pack=context_pack,
        )
        if semantic_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + semantic_reasons,
                parsed_ops=validation.parsed_ops,
            )
    if validation.status != "reject":
        quest_reopen_reasons = _find_quest_reopen_tracking_link_reasons(
            db,
            session_id,
            validation.parsed_ops,
        )
        if quest_reopen_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + quest_reopen_reasons,
                parsed_ops=validation.parsed_ops,
            )
    if validation.status != "reject":
        durable_fact_reasons = _find_durable_fact_mismatch_reasons(
            _memory_candidates_to_durable_facts(chosen.memory_candidates),
            validation.parsed_ops,
        )
        if durable_fact_reasons:
            record_canon_repair("mismatch")
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + durable_fact_reasons,
                parsed_ops=validation.parsed_ops,
            )

    if validation.status == "uncertain" and not librarian_used:
        try:
            _rollback_read_only_autobegin_transaction(db)
            librarian_raw_raw = _call_librarian(
                state_summary=context_pack.get("state_summary", ""),
                reasons=validation.reasons,
                narrator_json=narrator_raw,
                context_for_librarian=_build_librarian_context_payload(
                    context_pack=context_pack,
                    session_id=session_id,
                    new_turn=new_turn,
                    user_input=payload.user_input,
                ),
                session_id=session_id,
            )
            librarian_usage = _extract_xai_usage(librarian_raw_raw)
            if librarian_usage is not None:
                llm_usage["librarian"] = librarian_usage
            librarian_raw = _coerce_narrator_payload(librarian_raw_raw)
            candidate = _parse_narrator_response(librarian_raw)
            candidate_validation = _validate_patch_ops(librarian_raw.get("proposed_updates"))
            if candidate_validation.status != "reject":
                movement_reasons = _server_owned_movement_reasons(
                    candidate_validation.parsed_ops,
                    candidate.semantic_events,
                )
                if movement_reasons:
                    candidate_validation = PatchValidationResult(
                        status="uncertain",
                        reasons=candidate_validation.reasons + movement_reasons,
                        parsed_ops=candidate_validation.parsed_ops,
                    )
            librarian_used = True
            if candidate_validation.status == "ok":
                chosen_raw = librarian_raw
                chosen = candidate
                validation = candidate_validation
            else:
                validation = PatchValidationResult(
                    status="reject",
                    reasons=candidate_validation.reasons
                    + ["librarian result is not confidently applicable"],
                    parsed_ops=[],
                )
                chosen_raw = librarian_raw
                chosen = candidate
        except Exception as exc:  # noqa: BLE001
            logger.exception("Librarian call failed for session_id=%s turn_index=%s", session_id, new_turn)
            validation = PatchValidationResult(
                status="reject",
                reasons=validation.reasons + [f"librarian_error: {type(exc).__name__}"],
                parsed_ops=[],
            )
            librarian_used = True

    return TurnPlanResult(
        narration=chosen.narration,
        choices=chosen.choices,
        memory_candidates=chosen.memory_candidates,
        semantic_events=chosen.semantic_events,
        scene_entities=chosen.scene_entities,
        memory_trace=chosen.memory_trace,
        consequence_seeds=chosen.consequence_seeds,
        consequence_intents=[
            schemas.ConsequenceIntent.model_validate(item)
            for item in list(chosen_raw.get("consequence_intents") or [])
            if isinstance(item, dict)
        ],
        resolved_consequence_ids=chosen.resolved_consequence_ids,
        zone_scope=chosen.zone_scope,
        parsed_ops=validation.parsed_ops if validation.status == "ok" else [],
        validator_status=validation.status,
        validator_reasons=validation.reasons,
        raw_response=chosen_raw,
        librarian_used=librarian_used,
        llm_usage=llm_usage,
        planner_contract_version=int(chosen_raw.get("planner_contract_version") or 2),
    )


def _resolve_turn_plan_state_first(
    db: Session,
    session_id: uuid.UUID,
    *,
    user_input: str,
    context_pack: dict[str, Any],
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
) -> TurnPlanResult:
    librarian_used = False
    llm_usage: dict[str, Any] = {"narrator": None, "librarian": None}

    t0 = time.monotonic()
    try:
        planner_raw_raw = _call_narrator(
            context_pack=context_pack,
            response_mode="world_intent",
            session_id=session_id,
        )
        planner_latency = time.monotonic() - t0
        logger.info("_call_world_intent_v2 latency=%.3fs session=%s turn=%s", planner_latency, session_id, new_turn)
        narrator_usage = _extract_xai_usage(planner_raw_raw)
        if narrator_usage is not None:
            llm_usage["narrator"] = narrator_usage
        planner_raw = _coerce_world_intent_payload(planner_raw_raw)
        planner = _parse_world_intent_response(planner_raw)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, RuntimeError) and "OPENROUTER_API_KEY" in str(exc):
            logger.warning(
                "World intent planner skipped for session_id=%s turn_index=%s: xAI key is missing",
                session_id,
                new_turn,
            )
        else:
            logger.exception("World intent planner failed for session_id=%s turn_index=%s", session_id, new_turn)
        fallback: dict[str, Any] = {
            "proposed_updates": [],
            "memory_candidates": [],
            "semantic_events": [],
            "scene_entities": [],
            "consequence_intents": [],
            "resolved_consequence_ids": [],
            "zone_scope": str(context_pack.get("current_zone", {}).get("zone_id"))
            if context_pack.get("current_zone", {}).get("zone_id")
            else None,
            "planner_contract_version": 2,
        }
        return TurnPlanResult(
            narration="",
            choices=[],
            memory_candidates=[],
            semantic_events=[],
            scene_entities=[],
            consequence_seeds=[],
            consequence_intents=[],
            resolved_consequence_ids=[],
            zone_scope=None,
            parsed_ops=[],
            validator_status="reject",
            validator_reasons=[f"world_intent_error: {type(exc).__name__}"],
            raw_response=fallback,
            librarian_used=False,
            memory_trace=None,
            llm_usage=llm_usage,
            planner_contract_version=2,
        )

    validation = _validate_patch_ops(planner_raw.get("proposed_updates"))
    chosen_raw = planner_raw
    chosen = planner

    if validation.status != "reject":
        movement_reasons = _server_owned_movement_reasons(
            validation.parsed_ops,
            chosen.semantic_events,
        )
        if movement_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + movement_reasons,
                parsed_ops=validation.parsed_ops,
            )

    if validation.status != "reject":
        quest_reopen_reasons = _find_quest_reopen_tracking_link_reasons(
            db,
            session_id,
            validation.parsed_ops,
        )
        if quest_reopen_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + quest_reopen_reasons,
                parsed_ops=validation.parsed_ops,
            )
    if validation.status != "reject":
        durable_fact_reasons = _find_durable_fact_mismatch_reasons(
            _memory_candidates_to_durable_facts(chosen.memory_candidates),
            validation.parsed_ops,
        )
        if durable_fact_reasons:
            record_canon_repair("mismatch")
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + durable_fact_reasons,
                parsed_ops=validation.parsed_ops,
            )

    if validation.status == "uncertain" and not librarian_used:
        try:
            _rollback_read_only_autobegin_transaction(db)
            librarian_raw_raw = _call_librarian(
                state_summary=context_pack.get("state_summary", ""),
                reasons=validation.reasons,
                narrator_json=planner_raw,
                context_for_librarian=_build_librarian_context_payload(
                    context_pack=context_pack,
                    session_id=session_id,
                    new_turn=new_turn,
                    user_input=user_input,
                ),
                response_mode="world_intent",
                session_id=session_id,
            )
            librarian_usage = _extract_xai_usage(librarian_raw_raw)
            if librarian_usage is not None:
                llm_usage["librarian"] = librarian_usage
            librarian_raw = _coerce_world_intent_payload(librarian_raw_raw)
            candidate = _parse_world_intent_response(librarian_raw)
            candidate_validation = _validate_patch_ops(librarian_raw.get("proposed_updates"))
            if candidate_validation.status != "reject":
                movement_reasons = _server_owned_movement_reasons(
                    candidate_validation.parsed_ops,
                    candidate.semantic_events,
                )
                if movement_reasons:
                    candidate_validation = PatchValidationResult(
                        status="uncertain",
                        reasons=candidate_validation.reasons + movement_reasons,
                        parsed_ops=candidate_validation.parsed_ops,
                    )
            librarian_used = True
            if candidate_validation.status == "ok":
                chosen_raw = librarian_raw
                chosen = candidate
                validation = candidate_validation
            else:
                validation = PatchValidationResult(
                    status="reject",
                    reasons=candidate_validation.reasons
                    + ["librarian result is not confidently applicable"],
                    parsed_ops=[],
                )
                chosen_raw = librarian_raw
                chosen = candidate
        except Exception as exc:  # noqa: BLE001
            logger.exception("World intent librarian failed for session_id=%s turn_index=%s", session_id, new_turn)
            validation = PatchValidationResult(
                status="reject",
                reasons=validation.reasons + [f"librarian_error: {type(exc).__name__}"],
                parsed_ops=[],
            )
            librarian_used = True

    return TurnPlanResult(
        narration="",
        choices=[],
        memory_candidates=chosen.memory_candidates,
        semantic_events=chosen.semantic_events,
        scene_entities=chosen.scene_entities,
        memory_trace=None,
        consequence_seeds=[],
        consequence_intents=list(chosen.consequence_intents),
        resolved_consequence_ids=chosen.resolved_consequence_ids,
        zone_scope=chosen.zone_scope,
        parsed_ops=validation.parsed_ops if validation.status == "ok" else [],
        validator_status=validation.status,
        validator_reasons=validation.reasons,
        raw_response=chosen_raw,
        librarian_used=librarian_used,
        llm_usage=llm_usage,
        planner_contract_version=2,
    )


def _resolve_turn_plan_split(
    db: Session,
    session_id: uuid.UUID,
    *,
    payload: schemas.TurnIn,
    context_pack: dict[str, Any],
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
) -> TurnPlanResult:
    """Split path: xAI produces narration text, DeepSeek generates structured patches."""
    llm_usage: dict[str, Any] = {"narrator": None, "deepseek_patch": None, "librarian": None}

    # --- Step 1: Narrator text-only ---
    t0 = time.monotonic()
    try:
        narration_text = _call_narrator_text_only(context_pack=context_pack, session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        narrator_latency = time.monotonic() - t0
        logger.warning(
            "Narrator text-only failed (%.3fs) for session=%s turn=%s: %s",
            narrator_latency, session_id, new_turn, exc,
        )
        fallback_narration = FALLBACK_AI_UNAVAILABLE
        fallback: dict[str, Any] = {
            "narration": fallback_narration,
            "choices": [],
            "proposed_updates": [],
            "memory_candidates": [],
            "semantic_events": [],
            "scene_entities": [],
            "consequence_seeds": [],
            "resolved_consequence_ids": [],
            "zone_scope": str(context_pack.get("current_zone", {}).get("zone_id"))
            if context_pack.get("current_zone", {}).get("zone_id")
            else None,
            "in_game_time": {"day": in_game_day, "minute": in_game_minute},
        }
        return TurnPlanResult(
            narration=fallback_narration,
            choices=[],
            memory_candidates=[],
            consequence_seeds=[],
            consequence_intents=[],
            resolved_consequence_ids=[],
            zone_scope=None,
            parsed_ops=[],
            validator_status="reject",
            validator_reasons=[f"narrator_text_error: {type(exc).__name__}"],
            raw_response=fallback,
            librarian_used=False,
            semantic_events=[],
            scene_entities=[],
            memory_trace=None,
            llm_usage=llm_usage,
        )
    narrator_latency = time.monotonic() - t0
    logger.info("_call_narrator_text_only latency=%.3fs session=%s turn=%s", narrator_latency, session_id, new_turn)

    # --- Step 2: DeepSeek patch generation ---
    t1 = time.monotonic()
    try:
        patch_raw_raw = _call_deepseek_patch_generator(
            narration=narration_text,
            context_pack=context_pack,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        patch_latency = time.monotonic() - t1
        logger.warning(
            "DeepSeek patch generator failed (%.3fs) for session=%s turn=%s: %s",
            patch_latency, session_id, new_turn, exc,
        )
        # Return narration without patches
        fallback_raw: dict[str, Any] = {
            "narration": narration_text,
            "choices": [],
            "proposed_updates": [],
            "memory_candidates": [],
            "semantic_events": [],
            "scene_entities": [],
            "consequence_seeds": [],
            "resolved_consequence_ids": [],
            "zone_scope": str(context_pack.get("current_zone", {}).get("zone_id"))
            if context_pack.get("current_zone", {}).get("zone_id")
            else None,
            "in_game_time": {"day": in_game_day, "minute": in_game_minute},
        }
        return TurnPlanResult(
            narration=narration_text,
            choices=[],
            memory_candidates=[],
            consequence_seeds=[],
            consequence_intents=[],
            resolved_consequence_ids=[],
            zone_scope=None,
            parsed_ops=[],
            validator_status="reject",
            validator_reasons=[f"deepseek_patch_error: {type(exc).__name__}"],
            raw_response=fallback_raw,
            librarian_used=False,
            semantic_events=[],
            scene_entities=[],
            memory_trace=None,
            llm_usage=llm_usage,
        )
    patch_latency = time.monotonic() - t1
    logger.info("_call_deepseek_patch_generator latency=%.3fs session=%s turn=%s", patch_latency, session_id, new_turn)
    deepseek_patch_usage = _extract_openrouter_usage(patch_raw_raw)
    llm_usage["deepseek_patch"] = {"latency_s": round(patch_latency, 3)}
    if deepseek_patch_usage is not None:
        llm_usage["deepseek_patch"].update(deepseek_patch_usage)

    # Inject narration into the patch payload (DeepSeek may not include it)
    if not isinstance(patch_raw_raw, dict):
        patch_raw_raw = {}
    else:
        patch_raw_raw = dict(patch_raw_raw)
        patch_raw_raw.pop("_openrouter_usage", None)
    patch_raw_raw["narration"] = narration_text

    # Coerce and validate
    coerced = _coerce_narrator_payload(patch_raw_raw)
    # Override narration with the original xAI text (authoritative)
    coerced["narration"] = narration_text

    try:
        parsed = _parse_narrator_response(coerced)
    except ValidationError as parse_exc:
        logger.warning(
            "DeepSeek patch payload failed schema validation for session=%s turn=%s; trying librarian",
            session_id, new_turn,
        )
        # Try Librarian repair
        try:
            librarian_raw_raw = _call_librarian(
                state_summary=context_pack.get("state_summary", ""),
                reasons=[f"deepseek_schema_error: {parse_exc.errors()[0]['msg']}"],
                narrator_json=patch_raw_raw,
                context_for_librarian=_build_librarian_context_payload(
                    context_pack=context_pack,
                    session_id=session_id,
                    new_turn=new_turn,
                    user_input=payload.user_input,
                ),
                session_id=session_id,
            )
            librarian_usage = _extract_xai_usage(librarian_raw_raw)
            if librarian_usage is not None:
                llm_usage["librarian"] = librarian_usage
            coerced = _coerce_narrator_payload(librarian_raw_raw)
            coerced["narration"] = narration_text
            parsed = _parse_narrator_response(coerced)
        except Exception as lib_exc:  # noqa: BLE001
            logger.exception("Librarian repair failed for session=%s turn=%s", session_id, new_turn)
            return TurnPlanResult(
                narration=narration_text,
                choices=[],
                memory_candidates=[],
                consequence_seeds=[],
                consequence_intents=[],
                resolved_consequence_ids=[],
                zone_scope=None,
                parsed_ops=[],
                validator_status="reject",
                validator_reasons=[
                    f"deepseek_schema_error: {parse_exc.errors()[0]['msg']}",
                    f"librarian_repair_error: {type(lib_exc).__name__}",
                ],
                raw_response=coerced,
                librarian_used=True,
                semantic_events=[],
                scene_entities=[],
                memory_trace=None,
                llm_usage=llm_usage,
            )

    validation = _validate_patch_ops(coerced.get("proposed_updates"))
    if validation.status != "reject":
        movement_reasons = _server_owned_movement_reasons(
            validation.parsed_ops,
            parsed.semantic_events,
        )
        if movement_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + movement_reasons,
                parsed_ops=validation.parsed_ops,
            )
    if validation.status != "reject":
        semantic_reasons = _find_narration_semantic_desync_reasons(
            narration_text,
            validation.parsed_ops,
            semantic_events=parsed.semantic_events,
            context_pack=context_pack,
        )
        if semantic_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + semantic_reasons,
                parsed_ops=validation.parsed_ops,
            )
    if validation.status != "reject":
        quest_reopen_reasons = _find_quest_reopen_tracking_link_reasons(
            db,
            session_id,
            validation.parsed_ops,
        )
        if quest_reopen_reasons:
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + quest_reopen_reasons,
                parsed_ops=validation.parsed_ops,
            )
    if validation.status != "reject":
        durable_fact_reasons = _find_durable_fact_mismatch_reasons(
            _memory_candidates_to_durable_facts(parsed.memory_candidates),
            validation.parsed_ops,
        )
        if durable_fact_reasons:
            record_canon_repair("mismatch")
            validation = PatchValidationResult(
                status="uncertain",
                reasons=validation.reasons + durable_fact_reasons,
                parsed_ops=validation.parsed_ops,
            )
    librarian_used = False

    if validation.status == "uncertain":
        try:
            t2 = time.monotonic()
            _rollback_read_only_autobegin_transaction(db)
            librarian_raw_raw = _call_librarian(
                state_summary=context_pack.get("state_summary", ""),
                reasons=validation.reasons,
                narrator_json=coerced,
                context_for_librarian=_build_librarian_context_payload(
                    context_pack=context_pack,
                    session_id=session_id,
                    new_turn=new_turn,
                    user_input=payload.user_input,
                ),
                session_id=session_id,
            )
            librarian_latency = time.monotonic() - t2
            logger.info("_call_librarian latency=%.3fs session=%s turn=%s", librarian_latency, session_id, new_turn)
            librarian_usage = _extract_xai_usage(librarian_raw_raw)
            if librarian_usage is not None:
                llm_usage["librarian"] = librarian_usage
            librarian_raw = _coerce_narrator_payload(librarian_raw_raw)
            librarian_raw["narration"] = narration_text
            candidate = _parse_narrator_response(librarian_raw)
            candidate_validation = _validate_patch_ops(librarian_raw.get("proposed_updates"))
            if candidate_validation.status != "reject":
                movement_reasons = _server_owned_movement_reasons(
                    candidate_validation.parsed_ops,
                    candidate.semantic_events,
                )
                if movement_reasons:
                    candidate_validation = PatchValidationResult(
                        status="uncertain",
                        reasons=candidate_validation.reasons + movement_reasons,
                        parsed_ops=candidate_validation.parsed_ops,
                    )
            librarian_used = True
            if candidate_validation.status == "ok":
                coerced = librarian_raw
                parsed = candidate
                validation = candidate_validation
            else:
                validation = PatchValidationResult(
                    status="reject",
                    reasons=candidate_validation.reasons
                    + ["librarian result is not confidently applicable"],
                    parsed_ops=[],
                )
                coerced = librarian_raw
                parsed = candidate
        except Exception as exc:  # noqa: BLE001
            logger.exception("Librarian call failed for session=%s turn=%s", session_id, new_turn)
            validation = PatchValidationResult(
                status="reject",
                reasons=validation.reasons + [f"librarian_error: {type(exc).__name__}"],
                parsed_ops=[],
            )
            librarian_used = True

    total_latency = time.monotonic() - t0
    logger.info("_resolve_turn_plan_split total=%.3fs session=%s turn=%s", total_latency, session_id, new_turn)

    return TurnPlanResult(
        narration=parsed.narration,
        choices=parsed.choices,
        memory_candidates=parsed.memory_candidates,
        semantic_events=parsed.semantic_events,
        scene_entities=parsed.scene_entities,
        memory_trace=parsed.memory_trace,
        consequence_seeds=parsed.consequence_seeds,
        consequence_intents=[
            schemas.ConsequenceIntent.model_validate(item)
            for item in list(coerced.get("consequence_intents") or [])
            if isinstance(item, dict)
        ],
        resolved_consequence_ids=parsed.resolved_consequence_ids,
        zone_scope=parsed.zone_scope,
        parsed_ops=validation.parsed_ops if validation.status == "ok" else [],
        validator_status=validation.status,
        validator_reasons=validation.reasons,
        raw_response=coerced,
        librarian_used=librarian_used,
        llm_usage=llm_usage,
        planner_contract_version=int(coerced.get("planner_contract_version") or 2),
    )


def _resolve_turn_plan_from_request(
    db: Session,
    session_id: uuid.UUID,
    *,
    user_input: str,
    debug_patch: schemas.DebugPatchIn | None,
    allow_debug_patch: bool,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
    context_pack: dict[str, Any] | None = None,
) -> TurnPlanResult:
    if debug_patch is not None:
        payload = schemas.TurnIn.model_construct(user_input=user_input, debug_patch=debug_patch)
        return _build_plan_from_debug_patch(
            payload,
            allow_debug_patch=allow_debug_patch,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )

    if context_pack is None:
        context_pack = _build_turn_context_pack(db, session_id, new_turn=new_turn, user_input=user_input)
    _rollback_read_only_autobegin_transaction(db)

    if USE_STATE_FIRST_PIPELINE:
        return _resolve_turn_plan_state_first(
            db,
            session_id,
            user_input=user_input,
            context_pack=context_pack,
            new_turn=new_turn,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )

    if USE_SPLIT_NARRATOR_PATCHES:
        payload = schemas.TurnIn.model_construct(user_input=user_input, debug_patch=None)
        try:
            split_result = _resolve_turn_plan_split(
                db,
                session_id,
                payload=payload,
                context_pack=context_pack,
                new_turn=new_turn,
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Split planner crashed for session=%s turn=%s; falling back to legacy planner",
                session_id,
                new_turn,
            )
        else:
            fallback_reason_prefixes = ("narrator_text_error:",)
            should_fallback = (
                split_result.validator_status == "reject"
                and any(
                    str(reason).startswith(fallback_reason_prefixes)
                    for reason in split_result.validator_reasons
                )
            )
            if not should_fallback:
                return split_result
            logger.warning(
                "Split planner rejected due provider/runtime error for session=%s turn=%s; "
                "falling back to legacy planner",
                session_id,
                new_turn,
            )

    payload = schemas.TurnIn.model_construct(user_input=user_input, debug_patch=None)
    return _resolve_turn_plan_legacy(
        db,
        session_id,
        payload=payload,
        context_pack=context_pack,
        new_turn=new_turn,
        in_game_day=in_game_day,
        in_game_minute=in_game_minute,
    )


def _resolve_turn_plan(
    db: Session,
    session_id: uuid.UUID,
    *,
    payload: schemas.TurnIn,
    allow_debug_patch: bool,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
    context_pack: dict[str, Any] | None = None,
) -> TurnPlanResult:
    return _resolve_turn_plan_from_request(
        db,
        session_id,
        user_input=payload.user_input,
        debug_patch=payload.debug_patch,
        context_pack=context_pack,
        allow_debug_patch=allow_debug_patch,
        new_turn=new_turn,
        in_game_day=in_game_day,
        in_game_minute=in_game_minute,
    )



__all__ = [
    "PatchValidationResult",
    "TurnPlanResult",
    "PATCH_OP_LIST_ADAPTER",
    "_collect_refs",
    "_toposort_patch_ops",
    "_validate_patch_ops",
    "_parse_narrator_response",
    "_parse_world_intent_response",
    "_call_narrator",
    "_call_post_apply_narrator",
    "_call_narrator_text_only",
    "_call_deepseek_patch_generator",
    "_call_librarian",
    "_build_plan_from_debug_patch",
    "_resolve_travel_turn_plan",
    "_resolve_turn_plan",
    "_resolve_turn_plan_from_request",
    "_resolve_turn_plan_legacy",
    "_resolve_turn_plan_state_first",
    "_resolve_turn_plan_split",
]
