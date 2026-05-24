from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import re
import threading
import uuid
from typing import Any

from sqlalchemy import Float, Integer, and_, case, cast, delete as sa_delete, func, literal, or_, select
from sqlalchemy.orm import Session, aliased

from . import crud_continuity as _continuity
from . import models, schemas
from .constants import (
    LOCATED_IN_LINK_TYPE,
    NPC_SOCIAL_LINK_TYPES,
    ORPHANED_ITEMS_LOOKBACK_TURNS,
    QUEST_TERMINAL_STATUSES,
    REACTION_CONFLICT_LINK_TYPES,
    REACTION_SUPPORT_LINK_TYPES,
    TRACKING_QUEST_LINK_TYPE,
)
from .db import (
    CTX_WEIGHT_DECAY_LAMBDA,
    EMBED_SNIPPET_MAX_CHARS,
    ELASTIC_MIN_RELEVANCE_THRESHOLD,
    OPENROUTER_CHAT_MODEL,
    TURN_CONTEXT_MAX_TOKENS,
    TURN_CONTEXT_TOKEN_RESERVE,
    TURN_CONTEXT_SEMANTIC_TURNS_LIMIT,
    TURN_CONTEXT_TURNS_LIMIT,
    USE_CONSEQUENCES,
    USE_CONTEXT_COMPRESSOR,
    USE_CTX_WEIGHT_DECAY,
    USE_EMBEDDINGS,
    USE_ELASTIC_ENTROPY_THRESHOLD,
    USE_QUERY_REFORMULATOR,
    USE_REACTION_ENRICHER,
    USE_UNIFIED_CONTEXT_SCORING,
    USE_WORLD_PROMPT_SUMMARIZER,
    WORLD_PROMPT_CHUNK_MAX_CHARS,
    WORLD_PROMPT_FALLBACK_MAX_CHARS,
    WORLD_PROMPT_TOP_K,
)
from .crud_consequences import select_due_consequences
from .crud_embeddings_ops import (
    MEMORY_BUNDLE_NAMESPACE,
    MEMORY_BUNDLE_OBJECT_TYPE,
    MEMORY_EVENT_NAMESPACE,
    MEMORY_EVENT_OBJECT_TYPE,
    MEMORY_FACT_NAMESPACE,
    MEMORY_FACT_OBJECT_TYPE,
    _coerce_importance,
    _extract_claim_text,
    _extract_link_context_text,
    _list_active_link_context_snippets,
    _maybe_embed_texts,
    _normalize_memory_text,
    _upsert_object_embedding,
)
from .crud_profiles import _build_player_profile_text
from .crud_shared import (
    _count_json_tokens,
    _count_text_tokens,
    _get_object,
    _get_player_current_zone_id,
    _normalize_json_preview,
    _rollback_read_only_autobegin_transaction,
    _require_session,
    _safe_int,
    _sanitize_object_data_for_context,
    _truncate_text,
    _truncate_text_to_tokens,
)
from .domain import player_commands as player_command_domain
from .domain.memory_policy import (
    annotate_rows_with_narrative_chains,
    build_ranked_memory_context_row as policy_build_ranked_memory_context_row,
    derive_actor_memory_views,
    derive_conflict_edge_payloads,
    derive_memory_policy_state,
    derive_narrative_chains,
    derive_narrative_graph_edges,
    derive_operational_alert_guardrails,
    derive_session_memory_profile,
    derive_session_narrative_mode,
    derive_story_obligation_payloads,
    HIGH_TENSION_BUNDLE_PRESSURE_EXEMPT_KINDS,
    memory_policy_state_payload,
    memory_row_merge_tuple as policy_memory_row_merge_tuple,
    narrative_chain_context_for_payload,
    narrative_chain_index,
    normalize_session_memory_profile,
    resolve_turn_intent,
    retrieval_lane_budgets,
    saturation_limits,
    select_memory_retrieval_rows,
    SESSION_MEMORY_PROFILE_WINDOW_TURNS,
    tuning_weight_config_payload,
)
from .llm import openrouter_chat
from .llm_telemetry import telemetry_context
from .lore_ux import (
    LORE_UX_POLICY,
    build_compiled_world_model_preview,
    build_runtime_lore_brief,
    claims_from_world_constitution_data,
)
from .strings import FALLBACK_NO_RESPONSE, KNOWLEDGE_CHALLENGE_PATTERNS, LABEL_CHOICES

MAX_CONTEXT_TEXT_PER_ROW = 800
ZONE_RECENT_CLAIMS_LIMIT = 6
MAX_REACTION_HINTS = 4
MEMORY_BUNDLE_MAX_ITEMS = 3
RELEVANCE_RECENT_AI_MAX_CHARS = 150
MAX_WORLD_CONSTITUTION_CHARS = 4000
WORLD_PROMPT_CHUNK_OBJECT_TYPE = "__world_prompt_chunk"
WORLD_PROMPT_CHUNK_EMBED_NAMESPACE = "world_prompt_chunk"
MAX_WORLD_PROMPT_CHUNKS = 80
MEMORY_REVIEW_OBJECT_TYPE = "__memory_review_report"
STORY_OBLIGATION_OBJECT_TYPE = "__story_obligation"
EVENT_PAYLOAD_TEXT_KEYS = (
    "text",
    "message",
    "narration",
    "summary",
    "note",
    "details",
    "description",
    "reason",
)
CHRONICLE_OUTPUT_NAMESPACE = "chronicle_output"
CHRONICLE_INPUT_NAMESPACE = "chronicle_input"
SESSION_SUMMARY_OBJECT_TYPE = "__session_summary"
SESSION_SUMMARY_LIVE_TURNS = 8
NARRATIVE_SPINE_OBJECT_TYPE = "__narrative_spine"
NARRATIVE_SPINE_MAX_CHARS = 800
NARRATIVE_SPINE_SOURCE_TURNS = SESSION_SUMMARY_LIVE_TURNS
NARRATIVE_SPINE_FIELDS = (
    "player_commitments",
    "world_changes",
    "key_npc_statuses",
)
NARRATIVE_SPINE_MAX_ITEMS_PER_FIELD = 12
NARRATIVE_SPINE_ITEM_MAX_CHARS = 240
_SPINE_UPDATER_MAX_TOKENS = math.ceil(
    NARRATIVE_SPINE_MAX_ITEMS_PER_FIELD
    * len(NARRATIVE_SPINE_FIELDS)
    * NARRATIVE_SPINE_ITEM_MAX_CHARS
    / 3.5
)
RELEVANCE_QUERY_EMBED_INSTRUCTION = (
    "Retrieve relevant game history and world facts for the current player action"
)
WORLD_PROMPT_EMBED_INSTRUCTION = "Represent this world lore and setting rules for retrieval"
LINK_CONTEXT_NAMESPACE = "link_context"
ARCHIVED_QUEST_RECALL_MAX_DISTANCE = 0.24
ARCHIVED_QUEST_RECALL_LIMIT = 2
KNOWLEDGE_CHALLENGE_HINT = (
    "PLAYER CHALLENGES NPC KNOWLEDGE: verify against npc_knowledge BEFORE responding. "
    "If challenged fact is NOT in npc_knowledge, NPC must retract, admit uncertainty, or deflect. "
    "NEVER fabricate justification."
)
CTX_WEIGHT_KEY = "ctx_weight"
CTX_LAST_TOUCHED_TURN_KEY = "ctx_last_touched_turn"
TURN_WEIGHT_KEY = "turn_weight"
_JSON_FLOAT_PATTERN = r"^-?(?:\d+(?:\.\d+)?|\.\d+)$"
_JSON_INT_PATTERN = r"^-?\d+$"

_UNIFIED_CONTEXT_DIVERSITY_MINIMA: tuple[tuple[str, int], ...] = (
    ("prev_turn", 1),
    ("npc_knowledge", 1),
    ("zone_npc", 1),
    ("relevant_quest", 1),
    ("relevant_memory", 1),
)
_UNIFIED_MEMORY_CLASS_MINIMA: tuple[tuple[str, int], ...] = (
    ("semantic", 1),
    ("episodic", 1),
)
_EMBEDDING_CANDIDATE_TYPES: tuple[tuple[str, str | None, str], ...] = (
    ("npc", "npc_profile", "relevant_npc"),
    ("item", "item_profile", "relevant_item"),
    ("faction", "faction_profile", "relevant_faction"),
    ("claim", "claim_text", "relevant_claim"),
    ("quest", "quest_profile", "archived_quest"),
)

logger = logging.getLogger(__name__)
_QUERY_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-_'][0-9A-Za-zА-Яа-яЁё]+)*")

def _split_world_prompt_chunks(world_prompt: str, max_chars: int) -> list[str]:
    prompt = (world_prompt or "").strip()
    if not prompt or max_chars <= 0:
        return []

    paragraphs = [p.strip() for p in prompt.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [prompt]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks[:MAX_WORLD_PROMPT_CHUNKS]


def _get_latest_world_constitution_data(
    db: Session,
    session_id: uuid.UUID,
) -> tuple[bool, dict[str, Any]]:
    row = db.execute(
        select(models.ObjectModel.data)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "world_constitution",
        )
        .order_by(models.ObjectModel.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if not isinstance(row, dict):
        return False, {}
    return True, dict(row)


def _render_lore_profile_for_system(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    claims = claims_from_world_constitution_data(data)
    runtime_brief = build_runtime_lore_brief(claims, policy=LORE_UX_POLICY)
    if runtime_brief:
        return _truncate_text(runtime_brief, MAX_WORLD_CONSTITUTION_CHARS)

    lore_profile = data.get("lore_profile")
    if not isinstance(lore_profile, dict):
        return ""

    lines: list[str] = []
    for key in sorted(str(raw_key).strip() for raw_key in lore_profile.keys()):
        if not key:
            continue
        value = lore_profile.get(key)
        value_text = _truncate_text(str(value or "").strip(), 260)
        if not value_text:
            continue
        lines.append(f"- {key}: {value_text}")
    if not lines:
        return ""
    return _truncate_text("\n".join(lines), MAX_WORLD_CONSTITUTION_CHARS)


def _render_compiled_world_model_for_system(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    compiled = data.get("compiled_world_model")
    if not isinstance(compiled, dict):
        return ""
    preview = build_compiled_world_model_preview(compiled, policy=LORE_UX_POLICY)
    if not preview:
        return ""

    lines: list[str] = []
    expansion_policy = _truncate_text(str(preview.get("expansion_policy") or "").strip(), 80)
    if expansion_policy:
        lines.append(f"- expansion_policy: {expansion_policy}")

    core_envelope = dict(preview.get("core_envelope") or {})
    for field_name in (
        "tech_level",
        "magic_level",
        "mobility_profile",
        "conflict_profile",
        "social_structure",
        "genre_signals",
        "tone_signals",
    ):
        raw_value = core_envelope.get(field_name)
        if isinstance(raw_value, list):
            rendered = ", ".join(
                _truncate_text(str(item).strip(), 60)
                for item in raw_value
                if str(item).strip()
            )
        else:
            rendered = _truncate_text(str(raw_value or "").strip(), 180)
        if rendered:
            lines.append(f"- {field_name}: {rendered}")

    campaign_frame = dict(preview.get("campaign_frame") or {})
    for field_name in ("power_fantasy", "tone_profile", "activity_bias"):
        raw_value = campaign_frame.get(field_name)
        if isinstance(raw_value, list):
            rendered = ", ".join(
                _truncate_text(str(item).strip(), 60)
                for item in raw_value
                if str(item).strip()
            )
        else:
            rendered = _truncate_text(str(raw_value or "").strip(), 180)
        if rendered:
            lines.append(f"- campaign_frame.{field_name}: {rendered}")

    for field_name in ("affordances", "exceptions", "forbidden_or_rare_elements"):
        raw_value = preview.get(field_name)
        if not isinstance(raw_value, list):
            continue
        rendered = ", ".join(
            _truncate_text(str(item).strip(), 60)
            for item in raw_value
            if str(item).strip()
        )
        if rendered:
            lines.append(f"- {field_name}: {rendered}")

    custom_axes = dict(preview.get("custom_axes") or {})
    for raw_key, raw_value in sorted(custom_axes.items()):
        key = _truncate_text(str(raw_key or "").strip(), 40)
        value = _truncate_text(str(raw_value or "").strip(), 140)
        if key and value:
            lines.append(f"- custom.{key}: {value}")

    if not lines:
        return ""
    return _truncate_text("\n".join(lines), MAX_WORLD_CONSTITUTION_CHARS)


def _render_player_corrections_for_system(state_payload: dict[str, Any]) -> str:
    if not isinstance(state_payload, dict):
        return ""
    rendered = player_command_domain.build_player_corrections_constitution_brief(state_payload)
    if not rendered:
        return ""
    return _truncate_text(rendered, MAX_WORLD_CONSTITUTION_CHARS)


def _merge_constitution_sources(
    world_prompt_text: str,
    lore_profile_text: str,
    compiled_world_model_text: str = "",
    player_corrections_text: str = "",
) -> str:
    world_text = _truncate_text(str(world_prompt_text or "").strip(), MAX_WORLD_CONSTITUTION_CHARS)
    lore_text = _truncate_text(str(lore_profile_text or "").strip(), MAX_WORLD_CONSTITUTION_CHARS)
    compiled_text = _truncate_text(str(compiled_world_model_text or "").strip(), MAX_WORLD_CONSTITUTION_CHARS)
    corrections_text = _truncate_text(str(player_corrections_text or "").strip(), MAX_WORLD_CONSTITUTION_CHARS)
    knowledge_sections: list[str] = []
    if lore_text:
        knowledge_sections.append(f"FINALIZED LORE PROFILE (MECHANICS):\n{lore_text}")
    if compiled_text:
        knowledge_sections.append(
            "COMPILED SESSION WORLD MODEL (EXPANSION ENVELOPE):\n"
            f"{compiled_text}"
        )
    if corrections_text:
        knowledge_sections.append(
            "PLAYER COMMAND CONTROL PLANE (SESSION-LOCAL CORRECTIONS):\n"
            f"{corrections_text}"
        )
    knowledge_text = "\n\n".join(knowledge_sections)

    if not world_text and not knowledge_text:
        return ""
    if world_text and not knowledge_text:
        return world_text
    if knowledge_text and not world_text:
        return knowledge_text

    min_knowledge_budget = min(max(MAX_WORLD_CONSTITUTION_CHARS // 3, 900), MAX_WORLD_CONSTITUTION_CHARS - 400)
    world_budget = max(MAX_WORLD_CONSTITUTION_CHARS - min_knowledge_budget, 200)
    world_part = _truncate_text(world_text, world_budget)
    merged_prefix = "WORLD PROMPT (BASE SETTING):\n" f"{world_part}\n\n"
    knowledge_budget = max(MAX_WORLD_CONSTITUTION_CHARS - len(merged_prefix), 200)
    knowledge_part = _truncate_text(knowledge_text, knowledge_budget)
    return _truncate_text(f"{merged_prefix}{knowledge_part}", MAX_WORLD_CONSTITUTION_CHARS)


def _merge_player_command_reaction_hints(
    base_hints: list[str] | None,
    *,
    state_payload: dict[str, Any],
    max_hints: int = MAX_REACTION_HINTS,
) -> list[str]:
    return player_command_domain.merge_reaction_hints_with_guardrails(
        base_hints,
        state_payload=state_payload,
        max_hints=max_hints,
        policy=player_command_domain.PLAYER_COMMAND_POLICY,
    )


def _ensure_world_prompt_chunks_indexed(
    db: Session,
    session_id: uuid.UUID,
    world_prompt: str,
) -> str | None:
    prompt = (world_prompt or "").strip()
    if not prompt or not USE_EMBEDDINGS:
        return None

    def _source_hash_for(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _has_index(source_hash: str) -> bool:
        return (
            db.execute(
                select(models.ObjectModel.object_id)
                .where(
                    models.ObjectModel.session_id == session_id,
                    models.ObjectModel.type == WORLD_PROMPT_CHUNK_OBJECT_TYPE,
                    models.ObjectModel.data["source_hash"].astext == source_hash,
                )
                .limit(1)
            ).first()
            is not None
        )

    def _write_chunks(*, source_hash: str, chunks: list[str], vectors: list[list[float]]) -> None:
        db.execute(
            sa_delete(models.ObjectModel).where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == WORLD_PROMPT_CHUNK_OBJECT_TYPE,
            )
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, vectors)):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            object_row = models.ObjectModel(
                session_id=session_id,
                type=WORLD_PROMPT_CHUNK_OBJECT_TYPE,
                name=f"world_prompt_chunk_{idx}",
                data={
                    "source_hash": source_hash,
                    "chunk_index": idx,
                    "text": chunk,
                    "status": "active",
                },
            )
            db.add(object_row)
            db.flush()
            _upsert_object_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                namespace=WORLD_PROMPT_CHUNK_EMBED_NAMESPACE,
                text_hash=chunk_hash,
                embedding=embedding,
            )

    if db.in_transaction():
        source_hash = _source_hash_for(prompt)
        if _has_index(source_hash):
            return source_hash

        chunks = _split_world_prompt_chunks(prompt, WORLD_PROMPT_CHUNK_MAX_CHARS)
        if not chunks:
            return None

        vectors = _maybe_embed_texts(
            chunks,
            instruction=WORLD_PROMPT_EMBED_INSTRUCTION,
        )
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"World prompt embedding size mismatch: got {len(vectors)}, expected {len(chunks)}"
            )

        session_row = _require_session(db, session_id, for_update=True)
        current_prompt = str(getattr(session_row, "world_prompt", prompt) or "").strip()
        if current_prompt != prompt:
            raise RuntimeError("world_prompt changed during indexing within active transaction")
        if _has_index(source_hash):
            return source_hash
        _write_chunks(source_hash=source_hash, chunks=chunks, vectors=vectors)
        return source_hash

    while True:
        with db.begin():
            session_row = _require_session(db, session_id, for_update=True)
            current_prompt = str(getattr(session_row, "world_prompt", prompt) or "").strip()
            if not current_prompt:
                return None
            source_hash = _source_hash_for(current_prompt)
            if _has_index(source_hash):
                return source_hash

        chunks = _split_world_prompt_chunks(current_prompt, WORLD_PROMPT_CHUNK_MAX_CHARS)
        if not chunks:
            return None

        vectors = _maybe_embed_texts(
            chunks,
            instruction=WORLD_PROMPT_EMBED_INSTRUCTION,
        )
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"World prompt embedding size mismatch: got {len(vectors)}, expected {len(chunks)}"
            )

        with db.begin():
            session_row = _require_session(db, session_id, for_update=True)
            validated_prompt = str(getattr(session_row, "world_prompt", prompt) or "").strip()
            if not validated_prompt:
                return None
            if validated_prompt != current_prompt:
                continue
            if _has_index(source_hash):
                return source_hash
            _write_chunks(source_hash=source_hash, chunks=chunks, vectors=vectors)
            return source_hash


def _list_relevant_world_prompt_chunks(
    db: Session,
    session_id: uuid.UUID,
    source_hash: str | None,
    query_embedding: list[float] | None,
    *,
    limit: int,
) -> list[str]:
    if query_embedding is None or not source_hash:
        return []

    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
    rows = db.execute(
        select(models.ObjectModel, distance_expr.label("distance"))
        .join(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
                models.ObjectEmbeddingModel.namespace == WORLD_PROMPT_CHUNK_EMBED_NAMESPACE,
            ),
        )
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == WORLD_PROMPT_CHUNK_OBJECT_TYPE,
            models.ObjectModel.data["source_hash"].astext == source_hash,
        )
        .order_by(distance_expr.asc())
        .limit(max(limit, 1))
    ).all()

    chunks: list[str] = []
    for object_row, _distance in rows:
        text_value = str((object_row.data or {}).get("text", "")).strip()
        if text_value:
            chunks.append(text_value)
    return chunks


def _serialize_patch_ops(ops: list[schemas.PatchOp]) -> list[dict[str, Any]]:
    return [op.model_dump(mode="json", by_alias=True) for op in ops]


def _render_turn_ai_text(narration: str, choices: list[schemas.NarratorChoice]) -> str:
    narration_text = narration.strip() or FALLBACK_NO_RESPONSE
    if not choices:
        return narration_text

    lines = [narration_text, "", LABEL_CHOICES]
    for choice in choices[:8]:
        lines.append(f"- {choice.id}: {choice.text}")
    return "\n".join(lines)


def _build_embedding_snippet(
    *,
    user_input: str | None = None,
    narration: str,
    choices: list[schemas.NarratorChoice],
    applied_ops: list[dict[str, Any]],
    event_summaries: list[str] | None = None,
) -> str:
    lines: list[str] = []
    user_text = (user_input or "").strip()
    if user_text:
        lines.append(f"Player: {user_text}")

    narration_text = narration.strip()
    if narration_text:
        lines.append(f"Narrator: {narration_text}")

    if choices:
        first_choice = choices[0]
        lines.append(f"Choice: {first_choice.id} | {first_choice.text}")

    if applied_ops:
        op_names = ", ".join(op.get("op", "?") for op in applied_ops[:8])
        lines.append(f"Applied ops: {op_names}")

    for event_summary in event_summaries or []:
        summary = str(event_summary or "").strip()
        if summary:
            lines.append(summary)

    return _truncate_text("\n".join(lines).strip(), EMBED_SNIPPET_MAX_CHARS)


def _build_input_embedding_snippet(user_input: str) -> str:
    text = str(user_input or "").strip()
    if not text:
        return ""
    return _truncate_text(f"Player: {text}", EMBED_SNIPPET_MAX_CHARS)


def _build_event_embedding_line(event_type: str, payload: dict[str, Any]) -> str:
    payload_dict = payload if isinstance(payload, dict) else {}
    text_parts: list[str] = []
    for key in EVENT_PAYLOAD_TEXT_KEYS:
        raw = payload_dict.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            text_parts.append(f"{key}: {text}")

    if not text_parts:
        preview = _normalize_json_preview(payload_dict, 180)
        if preview and preview != "{}":
            text_parts.append(preview)

    if not text_parts:
        return ""

    body = "; ".join(text_parts[:3])
    return _truncate_text(f"Event {event_type}: {body}", 260)


def _list_turn_event_embedding_lines(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
    *,
    limit: int = 12,
) -> list[str]:
    rows = db.execute(
        select(models.EventModel)
        .where(
            models.EventModel.session_id == session_id,
            models.EventModel.turn_index == turn_index,
        )
        .order_by(models.EventModel.created_at.asc())
        .limit(max(limit, 1))
    ).scalars().all()

    lines: list[str] = []
    for row in rows:
        line = _build_event_embedding_line(row.type, dict(row.payload or {}))
        if line and line not in lines:
            lines.append(line)
    return lines


def _get_recent_ai_text_for_relevance(
    db: Session,
    session_id: uuid.UUID,
    *,
    max_chars: int = RELEVANCE_RECENT_AI_MAX_CHARS,
    max_turn_index: int | None = None,
) -> str | None:
    query = (
        select(models.TurnModel.ai_text)
        .where(
            models.TurnModel.session_id == session_id,
            models.TurnModel.ai_text.isnot(None),
        )
        .order_by(models.TurnModel.turn_index.desc())
        .limit(1)
    )
    if max_turn_index is not None:
        query = query.where(models.TurnModel.turn_index <= max(max_turn_index, 0))
    row = db.execute(query).scalar_one_or_none()
    if not isinstance(row, str):
        return None
    text = row.strip()
    if not text:
        return None
    return _truncate_text(text, max(max_chars, 1))


def _normalize_recent_scene_entities(raw_value: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_value, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_item in raw_value:
        if isinstance(raw_item, schemas.SceneEntity):
            entity = raw_item
        elif isinstance(raw_item, dict):
            try:
                entity = schemas.SceneEntity.model_validate(raw_item)
            except Exception:  # noqa: BLE001
                continue
        else:
            continue

        name = _normalize_query_text(entity.name)
        if not name:
            continue
        ref_text = str(entity.ref).strip() if entity.ref is not None else ""
        key = (ref_text, name.casefold())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "ref": ref_text or None,
                "name": name,
                "entity_type": entity.entity_type,
                "role": entity.role,
                "referent_candidate": bool(entity.referent_candidate),
                "salience": round(float(entity.salience), 6),
            }
        )

    normalized.sort(
        key=lambda item: (
            not bool(item.get("referent_candidate")),
            -float(item.get("salience") or 0.0),
            str(item.get("name") or "").casefold(),
        )
    )
    return normalized[:RECENT_SCENE_ENTITY_QUERY_LIMIT]


def _get_recent_scene_entities_for_relevance(
    db: Session,
    session_id: uuid.UUID,
    *,
    max_turn_index: int | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    query = (
        select(models.TurnModel.ai_json)
        .where(
            models.TurnModel.session_id == session_id,
            models.TurnModel.ai_json.isnot(None),
        )
        .order_by(models.TurnModel.turn_index.desc())
        .limit(max(limit, 1))
    )
    if max_turn_index is not None:
        query = query.where(models.TurnModel.turn_index <= max(max_turn_index, 0))

    rows = db.execute(query).scalars().all()
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = _normalize_recent_scene_entities(row.get("scene_entities"))
        if normalized:
            return normalized
    return []


def _build_relevance_query_text(
    user_input: str,
    *,
    zone_name: str | None = None,
    recent_ai_text: str | None = None,
    recent_scene_entities: list[dict[str, Any]] | None = None,
) -> str:
    base_input = user_input.strip()
    if not base_input:
        return ""
    parts = [base_input]
    if zone_name:
        zone_value = zone_name.strip()
        if zone_value:
            parts.insert(0, f"[zone] {zone_value}")
    recent_entity_names = [
        str(item.get("name") or "").strip()
        for item in (recent_scene_entities or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if recent_entity_names:
        parts.append(f"[recent_scene] {', '.join(recent_entity_names[:RECENT_SCENE_ENTITY_QUERY_LIMIT])}")
    elif recent_ai_text:
        recent_value = recent_ai_text.strip()
        if recent_value:
            parts.append(f"[recent_ai] {recent_value}")
    return "\n".join(parts)


_REACTION_ENRICHER_SYSTEM = (
    "You enrich RPG narrator reaction hints for nearby NPCs. "
    "Based on user action, NPC relations, and recent local claims, produce concise actionable hints. "
    "Focus on believable motives, fears, loyalties, and social consequences. "
    'Return JSON only: {"reaction_hints": ["..."]}. '
    "No lore invention outside given context. Max 4 hints, each <= 180 chars."
)
_WORLD_PROMPT_SUMMARIZER_SYSTEM = (
    "You compress RPG world-rule chunks into one coherent system prompt block. "
    "Preserve hard constraints, canon facts, and safety-critical rules. "
    "Remove duplicates, resolve wording conflicts conservatively, no inventions. "
    "Output plain text only, max 450 words."
)
_NARRATIVE_SPINE_UPDATER_SYSTEM = (
    "You maintain a strict JSON narrative spine for a long-running text RPG session. "
    "Merge prior spine and recent turns into durable continuity facts. "
    "Return JSON only with exactly these keys: "
    "player_commitments, world_changes, key_npc_statuses. "
    "Each key must contain an array of short strings with concrete facts. "
    "No markdown, no explanations, no extra keys."
)
_FIELD_BUDGETS: list[tuple[str, int, int]] = [
    ("session_summaries", 260, 1),
    ("hard_memory", 160, 0),
    ("entity_histories", 260, 0),
    ("callback_memories", 130, 0),
    ("recent_turns", 640, 2),
    ("npc_knowledge", 380, 1),
    ("reaction_hints", 130, 1),
    ("zone_npcs", 190, 1),
    ("relevant_memories", 160, 1),
    ("memory_bundles", 140, 0),
    ("zone_claims", 110, 1),
    ("relevant_claims", 100, 1),
    ("relevant_npcs", 110, 0),
    ("latent_consequences", 100, 1),
    ("relevant_items", 120, 0),
    ("orphaned_items", 80, 0),
    ("relevant_quests", 260, 1),
    ("archived_quest_recall", 80, 0),
    ("relevant_factions", 120, 0),
    ("relevant_links", 65, 0),
    ("player_inventory", 100, 0),
    ("player_location_history", 65, 0),
]
_FIXED_TRIM_ORDER: list[tuple[str, int]] = [
    ("world_prompt_for_system", 100),
    ("state_summary", 130),
    ("world_constitution_for_system", 380),
]
FIXED_TRIM_BUFFER_TOKENS = 6
FIXED_TRIM_MAX_CORRECTION_PASSES = 3
EXACT_NAMES_MAX_ITEMS = 8
EXACT_NAMES_MAX_CHARS = 64
EXACT_NAMES_MIN_CHARS = 3
INTENT_TAGS_MAX_ITEMS = 4
RECENT_SCENE_ENTITY_QUERY_LIMIT = 4
ALLOWED_INTENT_TAGS = frozenset(
    {
        "aggressive",
        "threat",
        "social",
        "investigate",
        "trade",
        "stealth",
        "escape",
        "other",
    }
)
QUERY_REFORMULATOR_MAX_TOKENS = 220
_QUERY_REFORMULATOR_SYSTEM = (
    "You reformulate retrieval queries for a long-running RPG session. "
    'Return JSON only: {"query_text":"...", "exact_names":["..."], "intent_tags":["..."]}. '
    "Ground every exact_names entry in the user input or provided context. "
    "Use recent_scene_entities as the authoritative source for pronoun resolution when available. "
    "Use recent_ai_text only as a soft hint when scene_entities are absent or incomplete. "
    "Do not invent entity names, factions, places, or facts. "
    "query_text must be a concise retrieval query, no markup, max 80 words. "
    "exact_names should contain 0-8 grounded entity names useful for retrieval. "
    "intent_tags must use only the allowed values provided in the payload."
)
TURN_INTENT_CLASSIFIER_MAX_TOKENS = 60
_TURN_INTENT_CLASSIFIER_SYSTEM = (
    "You classify the player's immediate turn intent for RPG context assembly. "
    'Return JSON only: {"intent_tags":["..."]}. '
    "Use only the allowed intent tags provided in the payload. "
    "Infer intent from the user input and local scene context. "
    "Do not invent facts, names, or actions not supported by the payload."
)


def _strip_empty_data(obj: Any) -> Any:
    """Remove empty data dicts from entity objects to save tokens."""
    if isinstance(obj, dict) and isinstance(obj.get("data"), dict) and not obj["data"]:
        return {k: v for k, v in obj.items() if k != "data"}
    return obj


def _coerce_unit_weight(raw_value: Any) -> float | None:
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


def _json_text_float_expr(text_expr: Any) -> Any:
    return case(
        (text_expr.op("~")(_JSON_FLOAT_PATTERN), cast(text_expr, Float)),
        else_=None,
    )


def _json_text_int_expr(text_expr: Any) -> Any:
    return case(
        (text_expr.op("~")(_JSON_INT_PATTERN), cast(text_expr, Integer)),
        else_=None,
    )


def _object_ctx_weight_expr(*, current_turn: int | None = None) -> Any:
    raw_weight_expr = _json_text_float_expr(models.ObjectModel.data[CTX_WEIGHT_KEY].astext)
    if not USE_CTX_WEIGHT_DECAY or current_turn is None:
        return raw_weight_expr

    effective_turn = max(int(current_turn), 0)
    touched_turn_expr = _json_text_int_expr(models.ObjectModel.data[CTX_LAST_TOUCHED_TURN_KEY].astext)
    fallback_touched_expr = func.coalesce(touched_turn_expr, literal(effective_turn))
    turns_since_touch_expr = func.greatest(literal(effective_turn) - fallback_touched_expr, literal(0))
    decay_base = max(1.0 - float(CTX_WEIGHT_DECAY_LAMBDA), 0.0)
    decay_factor_expr = func.power(literal(decay_base), cast(turns_since_touch_expr, Float))
    return case(
        (raw_weight_expr.is_(None), None),
        else_=raw_weight_expr * decay_factor_expr,
    )


def _memory_importance_expr() -> Any:
    return func.coalesce(_json_text_float_expr(models.ObjectModel.data["importance"].astext), literal(0.0))


def _memory_source_turn_expr() -> Any:
    source_turn_expr = _json_text_int_expr(models.ObjectModel.data["source_turn"].astext)
    last_seen_turn_expr = _json_text_int_expr(models.ObjectModel.data["last_seen_turn"].astext)
    return func.coalesce(source_turn_expr, last_seen_turn_expr, literal(0))


def _memory_priority_rank_expr() -> Any:
    priority_expr = func.lower(models.ObjectModel.data["priority"].astext)
    return case(
        (priority_expr == "high", 2),
        (priority_expr == "med", 1),
        (priority_expr == "low", 0),
        else_=0,
    )


def _memory_committed_source_ops_filter() -> Any:
    source_ops_expr = _json_text_int_expr(models.ObjectModel.data["source_ops_count"].astext)
    return func.coalesce(source_ops_expr, literal(1)) != 0


def _turn_weight_expr() -> Any:
    return _json_text_float_expr(models.TurnModel.ai_json[TURN_WEIGHT_KEY].astext)


def _extract_ctx_weight(data: dict[str, Any] | None) -> float | None:
    if not isinstance(data, dict):
        return None
    return _coerce_unit_weight(data.get(CTX_WEIGHT_KEY))


def _extract_effective_ctx_weight(
    data: dict[str, Any] | None,
    *,
    current_turn: int | None = None,
) -> float | None:
    stored = _extract_ctx_weight(data)
    if stored is None:
        return None
    if not USE_CTX_WEIGHT_DECAY or current_turn is None:
        return stored

    if not isinstance(data, dict):
        return stored
    effective_turn = max(int(current_turn), 0)
    touched_turn_raw = _safe_int(data.get(CTX_LAST_TOUCHED_TURN_KEY))
    touched_turn = effective_turn if touched_turn_raw is None else min(touched_turn_raw, effective_turn)
    turns_since_touch = max(effective_turn - touched_turn, 0)
    decay_base = max(1.0 - float(CTX_WEIGHT_DECAY_LAMBDA), 0.0)
    decayed = stored * math.pow(decay_base, turns_since_touch)
    return round(min(max(decayed, 0.0), 1.0), 6)


def _extract_turn_weight(ai_json: Any) -> float | None:
    if not isinstance(ai_json, dict):
        return None
    return _coerce_unit_weight(ai_json.get(TURN_WEIGHT_KEY))


def _unpack_scored_object_row(row: Any) -> tuple[Any, Any, Any, Any]:
    if isinstance(row, (list, tuple)):
        if len(row) >= 4:
            return row[0], row[1], row[2], row[3]
        if len(row) == 3:
            return row[0], row[1], row[2], None
        if len(row) == 2:
            return row[0], row[1], None, None
        if len(row) == 1:
            return row[0], None, None, None
    return row, None, None, None


# ---------------------------------------------------------------------------
# LRU caches for expensive LLM helpers
# ---------------------------------------------------------------------------
_LRU_CACHE_LOCK = threading.Lock()

_WORLD_PROMPT_SUMMARY_CACHE: dict[str, str] = {}  # sha256(chunks) -> summary
_WORLD_PROMPT_SUMMARY_CACHE_MAX = 8


def _lru_get(cache: dict, key: Any) -> Any:
    """Thread-safe cache lookup."""
    with _LRU_CACHE_LOCK:
        return cache.get(key)


def _lru_put(cache: dict, key: Any, value: Any, max_size: int) -> None:
    """Insert into a dict-based LRU cache, evicting oldest entry if over capacity."""
    with _LRU_CACHE_LOCK:
        cache[key] = value
        while len(cache) > max_size:
            oldest = next(iter(cache))
            del cache[oldest]


_AMBIGUOUS_TOKENS = frozenset(
    {
        "he",
        "her",
        "here",
        "him",
        "it",
        "she",
        "that",
        "there",
        "them",
        "they",
        "this",
        "those",
        "его",
        "ее",
        "её",
        "ему",
        "ей",
        "здесь",
        "их",
        "им",
        "него",
        "нее",
        "неё",
        "них",
        "он",
        "она",
        "они",
        "там",
        "тот",
        "эта",
        "это",
        "этот",
        "эти",
    }
)


def _normalize_query_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _query_tokens(text: str) -> list[str]:
    return [token.casefold() for token in _QUERY_WORD_RE.findall(str(text or ""))]


def _contains_ambiguous_reference(text: str) -> bool:
    return bool(set(_query_tokens(text)) & _AMBIGUOUS_TOKENS)


def _normalize_query_reformulation_payload(
    raw_payload: Any,
    *,
    fallback_query: str,
) -> dict[str, Any]:
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    query_text = _normalize_query_text(payload.get("query_text") or payload.get("query") or "")
    if not query_text:
        query_text = fallback_query
    words = query_text.split()
    if len(words) > 80:
        query_text = " ".join(words[:80]).strip()
    exact_names = _normalize_exact_names(payload.get("exact_names") or payload.get("entities"))
    intent_tags = _normalize_intent_tags(payload.get("intent_tags") or payload.get("intents"))
    return {
        "query_text": query_text,
        "exact_names": exact_names,
        "intent_tags": intent_tags or ["other"],
    }


def _call_query_reformulator(
    user_input: str,
    *,
    zone_name: str | None,
    recent_ai_text: str | None,
    recent_scene_entities: list[dict[str, Any]] | None,
    session_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "user_input": _truncate_text(_normalize_query_text(user_input), 280),
        "zone_name": _truncate_text(_normalize_query_text(zone_name or ""), 120) or None,
        "recent_ai_text": _truncate_text(_normalize_query_text(recent_ai_text or ""), 220) or None,
        "recent_scene_entities": _normalize_recent_scene_entities(recent_scene_entities)[:RECENT_SCENE_ENTITY_QUERY_LIMIT],
        "allowed_intent_tags": sorted(ALLOWED_INTENT_TAGS),
    }
    with telemetry_context(request_type="query_reformulator"):
        return openrouter_chat.generate_json(
            model=OPENROUTER_CHAT_MODEL,
            system_prompt=_QUERY_REFORMULATOR_SYSTEM,
            user_prompt=_normalize_json_preview(payload, 2200),
            session_id=session_id,
            max_tokens=QUERY_REFORMULATOR_MAX_TOKENS,
        )


def _call_turn_intent_classifier(
    user_input: str,
    *,
    zone_name: str | None,
    recent_ai_text: str | None,
    zone_npcs: list[dict[str, Any]],
    zone_claims: list[dict[str, Any]],
    session_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "user_input": _truncate_text(_normalize_query_text(user_input), 220),
        "zone_name": _truncate_text(_normalize_query_text(zone_name or ""), 120) or None,
        "recent_ai_text": _truncate_text(_normalize_query_text(recent_ai_text or ""), 180) or None,
        "zone_npcs": [
            {
                "name": _truncate_text(str(npc.get("name") or "").strip(), 80),
                "attitude": _truncate_text(str(npc.get("attitude") or "").strip(), 40),
            }
            for npc in zone_npcs[:4]
            if isinstance(npc, dict) and str(npc.get("name") or "").strip()
        ],
        "zone_claims": [
            {
                "text": _truncate_text(str(claim.get("text") or "").strip(), 120),
                "speaker_name": _truncate_text(str(claim.get("speaker_name") or "").strip(), 60),
            }
            for claim in zone_claims[:3]
            if isinstance(claim, dict) and str(claim.get("text") or "").strip()
        ],
        "allowed_intent_tags": sorted(ALLOWED_INTENT_TAGS),
    }
    with telemetry_context(request_type="turn_intent_classifier"):
        return openrouter_chat.generate_json(
            model=OPENROUTER_CHAT_MODEL,
            system_prompt=_TURN_INTENT_CLASSIFIER_SYSTEM,
            user_prompt=_normalize_json_preview(payload, 1800),
            session_id=session_id,
            max_tokens=TURN_INTENT_CLASSIFIER_MAX_TOKENS,
        )


def _reformulate_query(
    user_input: str,
    *,
    zone_name: str | None,
    recent_ai_text: str | None,
    recent_scene_entities: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    fallback_query = _build_relevance_query_text(
        user_input,
        zone_name=zone_name,
        recent_ai_text=recent_ai_text,
        recent_scene_entities=recent_scene_entities,
    )
    try:
        reformulated = _call_query_reformulator(
            user_input,
            zone_name=zone_name,
            recent_ai_text=recent_ai_text,
            recent_scene_entities=recent_scene_entities,
            session_id=session_id,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Query reformulator failed; using fallback query text", exc_info=True)
        return {
            "query_text": fallback_query,
            "exact_names": [],
            "intent_tags": ["other"],
        }
    return _normalize_query_reformulation_payload(
        reformulated,
        fallback_query=fallback_query,
    )


def _should_reformulate(user_input: str) -> bool:
    """Trigger reformulation for long inputs or those with ambiguous pronouns."""
    words = user_input.strip().split()
    if len(words) >= 6:
        return True
    return _contains_ambiguous_reference(user_input)


def _should_classify_turn_intent(
    *,
    user_input: str,
    zone_npcs: list[dict[str, Any]],
    zone_claims: list[dict[str, Any]],
    existing_intent_tags: list[str] | None,
) -> bool:
    if not user_input.strip():
        return False
    if existing_intent_tags:
        return False
    return bool(zone_npcs or zone_claims)


def _normalize_exact_names(raw_value: Any) -> list[str]:
    if not isinstance(raw_value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_value:
        text = _truncate_text(" ".join(str(raw_item or "").split()).strip(), EXACT_NAMES_MAX_CHARS)
        if len(text) < EXACT_NAMES_MIN_CHARS:
            continue
        if text.isdigit():
            continue
        folded = text.casefold()
        if len(folded) < EXACT_NAMES_MIN_CHARS:
            continue
        if folded in seen:
            continue
        seen.add(folded)
        normalized.append(folded)
        if len(normalized) >= EXACT_NAMES_MAX_ITEMS:
            break
    return normalized


def _normalize_intent_tags(raw_value: Any) -> list[str]:
    if not isinstance(raw_value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_value:
        tag = " ".join(str(raw_item or "").split()).strip().casefold()
        if not tag or tag not in ALLOWED_INTENT_TAGS:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
        if len(normalized) >= INTENT_TAGS_MAX_ITEMS:
            break
    return normalized


def _get_latest_narrative_spine_row(
    db: Session,
    session_id: uuid.UUID,
) -> models.ObjectModel | None:
    rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == NARRATIVE_SPINE_OBJECT_TYPE,
        )
        .order_by(models.ObjectModel.created_at.desc())
    ).scalars().all()
    if not rows:
        return None

    return rows[0]


def _prune_stale_narrative_spine_rows(
    db: Session,
    rows: list[models.ObjectModel],
) -> None:
    if len(rows) <= 1:
        return
    for stale_row in rows[1:]:
        db.delete(stale_row)


def _normalize_narrative_spine_items(raw_items: Any) -> list[str]:
    if not isinstance(raw_items, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item_text = _truncate_text(" ".join(str(raw_item or "").split()).strip(), NARRATIVE_SPINE_ITEM_MAX_CHARS)
        if not item_text:
            continue
        dedupe_key = item_text.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(item_text)
        if len(normalized) >= NARRATIVE_SPINE_MAX_ITEMS_PER_FIELD:
            break
    return normalized


def _normalize_narrative_spine_payload(raw_payload: Any) -> dict[str, list[str]]:
    payload_dict = raw_payload if isinstance(raw_payload, dict) else {}
    normalized = {
        "player_commitments": _normalize_narrative_spine_items(payload_dict.get("player_commitments")),
        "world_changes": _normalize_narrative_spine_items(payload_dict.get("world_changes")),
        "key_npc_statuses": _normalize_narrative_spine_items(payload_dict.get("key_npc_statuses")),
    }
    return normalized


def _render_narrative_spine_json(
    payload: dict[str, list[str]],
    *,
    max_chars: int = NARRATIVE_SPINE_MAX_CHARS,
) -> str:
    capped_limit = max(max_chars, 1)
    working = {
        "player_commitments": list(payload.get("player_commitments", [])),
        "world_changes": list(payload.get("world_changes", [])),
        "key_npc_statuses": list(payload.get("key_npc_statuses", [])),
    }
    fallback: dict[str, list[str]] = {
        "player_commitments": [],
        "world_changes": [],
        "key_npc_statuses": [],
    }
    while True:
        rendered = json.dumps(working, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        if len(rendered) <= capped_limit:
            return rendered
        removed = False
        for field_name in reversed(NARRATIVE_SPINE_FIELDS):
            field_items = working.get(field_name)
            if isinstance(field_items, list) and field_items:
                field_items.pop()
                removed = True
                break
        if not removed:
            fallback_json = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
            return _truncate_text(fallback_json, capped_limit)


def _extract_narrative_spine_payload_from_data(data: dict[str, Any]) -> dict[str, list[str]]:
    normalized = _normalize_narrative_spine_payload(data.get("spine"))
    if any(normalized[field] for field in NARRATIVE_SPINE_FIELDS):
        return normalized
    legacy_text = _truncate_text(" ".join(str(data.get("text") or "").split()).strip(), NARRATIVE_SPINE_ITEM_MAX_CHARS)
    if not legacy_text:
        return normalized
    normalized["world_changes"] = [legacy_text]
    return normalized


def _parse_narrative_spine_summary_text(raw_value: Any) -> dict[str, list[str]] | None:
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    normalized = _normalize_narrative_spine_payload(parsed)
    if not any(normalized[field] for field in NARRATIVE_SPINE_FIELDS):
        return None
    return normalized


def _list_recent_turn_payload_for_spine(
    db: Session,
    session_id: uuid.UUID,
    *,
    start_turn: int,
    end_turn: int,
) -> list[dict[str, Any]]:
    if start_turn < 1 or end_turn < start_turn:
        return []

    rows = db.execute(
        select(models.TurnModel)
        .where(
            models.TurnModel.session_id == session_id,
            models.TurnModel.turn_index >= start_turn,
            models.TurnModel.turn_index <= end_turn,
        )
        .order_by(models.TurnModel.turn_index.asc())
    ).scalars().all()

    payload_rows: list[dict[str, Any]] = []
    for row in rows:
        payload_rows.append(
            {
                "turn_index": row.turn_index,
                "user_input": _truncate_text(str(row.user_input or ""), 180),
                "ai_text": _truncate_text(str(row.ai_text or ""), 220),
            }
        )
    return payload_rows


def _update_narrative_spine(
    db: Session,
    session_id: uuid.UUID,
    *,
    max_turn_index: int,
    live_turns: int,
) -> list[str]:
    current_turn_index = max(_safe_int(max_turn_index) or 0, 0)
    cadence_turns = max(_safe_int(live_turns) or 0, 1)

    read_spine_rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == NARRATIVE_SPINE_OBJECT_TYPE,
        )
        .order_by(models.ObjectModel.created_at.desc())
    ).scalars().all()
    read_spine_row = read_spine_rows[0] if read_spine_rows else None
    read_spine_data = dict(read_spine_row.data or {}) if read_spine_row is not None else {}
    existing_spine_payload = _extract_narrative_spine_payload_from_data(read_spine_data)
    existing_text = _render_narrative_spine_json(existing_spine_payload, max_chars=NARRATIVE_SPINE_MAX_CHARS)
    has_existing_spine = any(existing_spine_payload[field] for field in NARRATIVE_SPINE_FIELDS)
    has_stale_spine_rows = len(read_spine_rows) > 1

    def _return_existing_spine() -> list[str]:
        if has_stale_spine_rows:
            _rollback_read_only_autobegin_transaction(db)
            with db.begin():
                write_spine_rows = db.execute(
                    select(models.ObjectModel)
                    .where(
                        models.ObjectModel.session_id == session_id,
                        models.ObjectModel.type == NARRATIVE_SPINE_OBJECT_TYPE,
                    )
                    .order_by(models.ObjectModel.created_at.desc())
                ).scalars().all()
                _prune_stale_narrative_spine_rows(db, list(write_spine_rows))
                db.flush()
        return [existing_text] if has_existing_spine else []

    if read_spine_row is None and current_turn_index < cadence_turns:
        return []

    updated_at_turn = _safe_int(read_spine_data.get("updated_at_turn"))
    if (
        read_spine_row is not None
        and updated_at_turn is not None
        and (current_turn_index - updated_at_turn) < cadence_turns
    ):
        return _return_existing_spine()
    if (
        read_spine_row is not None
        and updated_at_turn is None
        and current_turn_index < cadence_turns
    ):
        return _return_existing_spine()
    if current_turn_index < 1:
        return _return_existing_spine()

    source_from_turn = max(1, current_turn_index - cadence_turns + 1)
    source_to_turn = current_turn_index
    turn_payload = _list_recent_turn_payload_for_spine(
        db,
        session_id,
        start_turn=source_from_turn,
        end_turn=source_to_turn,
    )
    if not turn_payload:
        return _return_existing_spine()

    payload = {
        "current_spine": existing_spine_payload,
        "recent_turns": turn_payload,
        "required_schema": {
            "player_commitments": ["string"],
            "world_changes": ["string"],
            "key_npc_statuses": ["string"],
        },
        "target_max_chars": NARRATIVE_SPINE_MAX_CHARS,
    }
    _rollback_read_only_autobegin_transaction(db)
    try:
        with telemetry_context(request_type="narrative_spine_updater"):
            updated_payload = openrouter_chat.generate_json(
                model=OPENROUTER_CHAT_MODEL,
                system_prompt=_NARRATIVE_SPINE_UPDATER_SYSTEM,
                user_prompt=_normalize_json_preview(payload, 5000),
                session_id=str(session_id),
                max_tokens=max(_SPINE_UPDATER_MAX_TOKENS, 1),
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Narrative spine update failed for session_id=%s turn_index=%s",
            session_id,
            current_turn_index,
            exc_info=True,
        )
        return _return_existing_spine()

    normalized_spine_payload = _normalize_narrative_spine_payload(updated_payload)
    if not any(normalized_spine_payload[field] for field in NARRATIVE_SPINE_FIELDS):
        return _return_existing_spine()
    spine_text = _render_narrative_spine_json(normalized_spine_payload, max_chars=NARRATIVE_SPINE_MAX_CHARS)

    updated_data = {
        "spine": normalized_spine_payload,
        "text": spine_text,
        "updated_at_turn": current_turn_index,
        "source_from_turn": source_from_turn,
        "source_to_turn": source_to_turn,
        "status": "active",
    }

    with db.begin():
        write_spine_rows = db.execute(
            select(models.ObjectModel)
            .where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == NARRATIVE_SPINE_OBJECT_TYPE,
            )
            .order_by(models.ObjectModel.created_at.desc())
        ).scalars().all()
        _prune_stale_narrative_spine_rows(db, list(write_spine_rows))
        write_spine_row = write_spine_rows[0] if write_spine_rows else None
        write_spine_data = dict(write_spine_row.data or {}) if write_spine_row is not None else {}
        write_updated_at_turn = _safe_int(write_spine_data.get("updated_at_turn"))
        if (
            write_spine_row is not None
            and write_updated_at_turn is not None
            and write_updated_at_turn >= current_turn_index
        ):
            current_payload = _extract_narrative_spine_payload_from_data(write_spine_data)
            current_text = _render_narrative_spine_json(current_payload, max_chars=NARRATIVE_SPINE_MAX_CHARS)
            return [current_text] if any(current_payload[field] for field in NARRATIVE_SPINE_FIELDS) else []
        if write_spine_row is None:
            write_spine_row = models.ObjectModel(
                session_id=session_id,
                type=NARRATIVE_SPINE_OBJECT_TYPE,
                name="narrative_spine",
                data=updated_data,
            )
            db.add(write_spine_row)
        else:
            write_spine_row.name = "narrative_spine"
            write_spine_row.data = updated_data
        db.flush()
    return [spine_text]


def _summarize_world_prompt_chunks(
    chunks: list[str],
    *,
    user_input: str,
    zone_name: str | None,
    session_id: str | None = None,
) -> str:
    """Summarize relevant world-prompt chunks into one coherent block.

    Results are cached by sha256 of sorted chunk content. The cache is
    automatically invalidated when world prompt chunks change.
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return _truncate_text(str(chunks[0] or "").strip(), WORLD_PROMPT_FALLBACK_MAX_CHARS)

    # Cache key is content-based — independent of user_input/zone_name
    # because the summary captures the world constitution, not the query.
    sorted_chunks = sorted(str(c or "").strip() for c in chunks)
    cache_key = hashlib.sha256("\n".join(sorted_chunks).encode()).hexdigest()
    cached = _lru_get(_WORLD_PROMPT_SUMMARY_CACHE, cache_key)
    if cached is not None:
        return cached

    payload = {
        "chunks": [_truncate_text(str(chunk or "").strip(), 900) for chunk in chunks[:8]],
    }
    try:
        with telemetry_context(request_type="world_prompt_summarizer"):
            summary = openrouter_chat.generate_text(
                model=OPENROUTER_CHAT_MODEL,
                system_prompt=_WORLD_PROMPT_SUMMARIZER_SYSTEM,
                user_prompt=_normalize_json_preview(payload, 7000),
                session_id=session_id,
                max_tokens=340,
            )
    except Exception:  # noqa: BLE001
        logger.warning("World prompt summarizer failed, fallback to raw chunks", exc_info=True)
        return "\n\n".join(chunks)

    summary_text = _truncate_text(str(summary or "").strip(), WORLD_PROMPT_FALLBACK_MAX_CHARS)
    if not summary_text:
        logger.warning("World prompt summarizer returned empty output, fallback to raw chunks")
        return "\n\n".join(chunks)

    _lru_put(_WORLD_PROMPT_SUMMARY_CACHE, cache_key, summary_text, _WORLD_PROMPT_SUMMARY_CACHE_MAX)
    return summary_text


def _item_relevance_score(item: Any, *, current_turn: int | None = None) -> float:
    if isinstance(item, dict):
        for key in ("turn_weight", "ctx_weight", "weighted_similarity", "relevance", "similarity", "importance"):
            value = _coerce_unit_weight(item.get(key))
            if value is not None:
                return value
        data = item.get("data")
        if isinstance(data, dict):
            value = _extract_effective_ctx_weight(data, current_turn=current_turn)
            if value is not None:
                return value
        turn_index = _safe_int(item.get("turn_index"))
        if turn_index is not None:
            return min(max(float(turn_index) / 100.0, 0.0), 1.0)
    return 0.0


def _serialized_tokens(payload: Any) -> int:
    return _count_json_tokens(payload)


def _trim_fixed_string_to_budget(
    fixed_pack: dict[str, Any],
    *,
    key: str,
    max_total_tokens: int,
) -> None:
    value = fixed_pack.get(key)
    if not isinstance(value, str):
        return
    current_total = _serialized_tokens(fixed_pack)
    if current_total <= max_total_tokens:
        return

    source_tokens = _count_text_tokens(value)
    excess_tokens = current_total - max_total_tokens
    target_tokens = max(source_tokens - excess_tokens - FIXED_TRIM_BUFFER_TOKENS, 0)
    fixed_pack[key] = _truncate_text_to_tokens(value, target_tokens)

    for _ in range(FIXED_TRIM_MAX_CORRECTION_PASSES):
        current_total = _serialized_tokens(fixed_pack)
        if current_total <= max_total_tokens:
            return
        remaining_excess = current_total - max_total_tokens
        current_value = str(fixed_pack.get(key) or "")
        if not current_value:
            break
        current_tokens = _count_text_tokens(current_value)
        next_target = max(current_tokens - remaining_excess - FIXED_TRIM_BUFFER_TOKENS, 0)
        if next_target >= current_tokens:
            next_target = max(current_tokens - max(remaining_excess, 1), 0)
        fixed_pack[key] = _truncate_text_to_tokens(current_value, next_target)

    if _serialized_tokens(fixed_pack) > max_total_tokens:
        fixed_pack[key] = ""


def _fit_fixed_overhead_budget(
    fixed_pack: dict[str, Any],
    max_total_tokens: int,
) -> dict[str, Any]:
    capped_max_tokens = max(max_total_tokens, 0)
    fitted_pack = dict(fixed_pack)
    for key, preferred_floor in _FIXED_TRIM_ORDER:
        if _serialized_tokens(fitted_pack) <= capped_max_tokens:
            return fitted_pack
        value = fitted_pack.get(key)
        if not isinstance(value, str):
            continue
        if _count_json_tokens(value) > preferred_floor:
            fitted_pack[key] = _truncate_text_to_tokens(value, preferred_floor)

    for key, _ in _FIXED_TRIM_ORDER:
        if _serialized_tokens(fitted_pack) <= capped_max_tokens:
            break
        _trim_fixed_string_to_budget(
            fitted_pack,
            key=key,
            max_total_tokens=capped_max_tokens,
        )
    return fitted_pack


def _apply_elastic_field_budgets(
    context_pack: dict[str, Any],
    *,
    max_total_tokens: int,
) -> dict[str, Any]:
    if not isinstance(context_pack, dict):
        return {}

    capped_max_tokens = max(max_total_tokens, 0)
    budget_fields = [field for field, _, _ in _FIELD_BUDGETS]
    budget_field_set = set(budget_fields)

    fixed_pack = {key: value for key, value in context_pack.items() if key not in budget_field_set}
    trimmed_pack = _fit_fixed_overhead_budget(fixed_pack, capped_max_tokens)
    context_new_turn = _safe_int(context_pack.get("new_turn"))
    current_turn_for_decay = max(context_new_turn - 1, 0) if context_new_turn is not None else None

    overflow = 0
    current_total = _serialized_tokens(trimmed_pack)
    available = max(capped_max_tokens - current_total, 0)

    for field, base_tokens, min_items in _FIELD_BUDGETS:
        field_cap = min(max(base_tokens + overflow, 0), available)
        source_value = context_pack.get(field)

        if not isinstance(source_value, list):
            field_used_tokens = 0
            if field in context_pack:
                baseline_pack = dict(trimmed_pack)
                baseline_pack.pop(field, None)
                baseline_total = _serialized_tokens(baseline_pack)
                candidate_pack = dict(baseline_pack)
                candidate_pack[field] = source_value
                candidate_total = _serialized_tokens(candidate_pack)
                if candidate_total <= capped_max_tokens:
                    trimmed_pack = candidate_pack
                    current_total = candidate_total
                    field_used_tokens = candidate_total - baseline_total
                else:
                    trimmed_pack = baseline_pack
                    current_total = baseline_total
            available = max(capped_max_tokens - current_total, 0)
            overflow = max(field_cap - field_used_tokens, 0)
            continue

        if not source_value:
            overflow = field_cap
            continue

        baseline_pack = dict(trimmed_pack)
        baseline_pack.pop(field, None)
        baseline_total = _serialized_tokens(baseline_pack)
        if baseline_total > capped_max_tokens:
            trimmed_pack = baseline_pack
            current_total = baseline_total
            available = 0
            overflow = 0
            continue

        ranked_items = sorted(
            enumerate(source_value),
            key=lambda pair: (-_item_relevance_score(pair[1], current_turn=current_turn_for_decay), pair[0]),
        )
        kept_items: list[Any] = []
        field_used_tokens = 0
        low_relevance_filtered = 0

        for _, item in ranked_items:
            item_score = _item_relevance_score(item, current_turn=current_turn_for_decay)
            if (
                USE_ELASTIC_ENTROPY_THRESHOLD
                and item_score < ELASTIC_MIN_RELEVANCE_THRESHOLD
                and len(kept_items) >= min_items
            ):
                low_relevance_filtered += 1
                continue
            candidate_items = kept_items + [item]
            candidate_pack = dict(baseline_pack)
            candidate_pack[field] = candidate_items
            candidate_total = _serialized_tokens(candidate_pack)
            candidate_field_tokens = candidate_total - baseline_total
            within_field_cap = candidate_field_tokens <= field_cap or len(kept_items) < min_items
            if candidate_total <= capped_max_tokens and within_field_cap:
                kept_items = candidate_items
                field_used_tokens = candidate_field_tokens

        if USE_ELASTIC_ENTROPY_THRESHOLD and low_relevance_filtered > 0:
            logger.debug(
                "Elastic entropy threshold filtered %s low-relevance items in field '%s' (threshold=%.3f)",
                low_relevance_filtered,
                field,
                ELASTIC_MIN_RELEVANCE_THRESHOLD,
            )

        if kept_items:
            trimmed_pack = dict(baseline_pack)
            trimmed_pack[field] = kept_items
            current_total = baseline_total + field_used_tokens
        else:
            trimmed_pack = baseline_pack
            current_total = baseline_total
            field_used_tokens = 0

        available = max(capped_max_tokens - current_total, 0)
        overflow = max(field_cap - field_used_tokens, 0)

    return trimmed_pack


@dataclass(slots=True)
class ContextItem:
    category: str
    payload: Any
    token_cost: int
    similarity: float
    importance: float
    score: float
    source_id: str
    variant: str


def _active_context_status_filter() -> Any:
    status_expr = models.ObjectModel.data["status"].astext
    return or_(
        status_expr.is_(None),
        func.jsonb_typeof(models.ObjectModel.data["status"]) == "null",
        and_(
            status_expr != "inactive",
            status_expr != "archived",
            status_expr != "stale",
        ),
    )


def _resolve_embedding_candidate_category(
    *,
    object_type: str,
    namespace: str,
) -> str | None:
    if namespace == LINK_CONTEXT_NAMESPACE:
        return "relevant_link"
    for candidate_type, candidate_namespace, category in _EMBEDDING_CANDIDATE_TYPES:
        if candidate_type == object_type and candidate_namespace == namespace:
            return category
    return None


def _collect_embedding_candidates(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    new_turn: int | None = None,
    current_turn: int | None = None,
    zone_id: uuid.UUID | None = None,
    pool_size: int = 80,
) -> list[ContextItem]:
    """Build a shared embedding candidate pool for unified context."""
    resolved_pool_size = max(pool_size, 1)
    effective_turn = max(new_turn if isinstance(new_turn, int) else 0, 0)
    ctx_weight_expr = _object_ctx_weight_expr(current_turn=current_turn)
    type_values = tuple(
        sorted(
            {
                candidate_type
                for candidate_type, _, _ in _EMBEDDING_CANDIDATE_TYPES
                if candidate_type is not None
            }
        )
    )
    namespace_values = tuple(
        sorted(
            {
                candidate_namespace
                for _, candidate_namespace, _ in _EMBEDDING_CANDIDATE_TYPES
                if candidate_namespace is not None
            }
        )
    )
    type_and_namespace_filter = and_(
        models.ObjectModel.type.in_(type_values),
        models.ObjectEmbeddingModel.namespace.in_(namespace_values),
    )
    namespace_filter = or_(
        type_and_namespace_filter,
        models.ObjectEmbeddingModel.namespace == LINK_CONTEXT_NAMESPACE,
    )
    if zone_id is None:
        npc_zone_filter: Any = literal(True)
    else:
        npc_zone_filter = or_(
            models.ObjectModel.type != "npc",
            select(models.LinkModel.link_id)
            .where(
                models.LinkModel.session_id == models.ObjectModel.session_id,
                models.LinkModel.from_object_id == models.ObjectModel.object_id,
                models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                models.LinkModel.valid_to_turn.is_(None),
                models.LinkModel.to_object_id == zone_id,
            )
            .exists(),
        )
    base_query = (
        select(
            models.ObjectModel,
            models.ObjectEmbeddingModel.namespace,
            ctx_weight_expr.label("ctx_weight"),
        )
        .join(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
            ),
        )
        .where(
            models.ObjectModel.session_id == session_id,
            _active_context_status_filter(),
            namespace_filter,
            npc_zone_filter,
        )
    )
    if query_embedding is None:
        rows = db.execute(
            base_query.add_columns(literal(None).label("distance"))
            .order_by(
                ctx_weight_expr.desc().nulls_last(),
                models.ObjectModel.created_at.desc(),
            )
            .limit(resolved_pool_size)
        ).all()
    else:
        distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
        rows = db.execute(
            base_query.add_columns(distance_expr.label("distance"))
            .order_by(
                distance_expr.asc(),
                ctx_weight_expr.desc().nulls_last(),
                models.ObjectModel.created_at.desc(),
            )
            .limit(resolved_pool_size)
        ).all()

    candidates: list[ContextItem] = []
    seen_source_ids: set[str] = set()
    for object_row, namespace_raw, ctx_weight_raw, distance in rows:
        namespace = str(namespace_raw or "").strip()
        object_type = str(object_row.type or "").strip()
        if not namespace or not object_type:
            continue
        category = _resolve_embedding_candidate_category(
            object_type=object_type,
            namespace=namespace,
        )
        if category is None:
            continue

        object_data = dict(object_row.data or {})
        sanitized_data = _sanitize_object_data_for_context(object_data)
        similarity_value: float | None = None
        if distance is not None:
            similarity_value = round(max(1.0 - float(distance), 0.0), 6)
        ctx_weight = _coerce_unit_weight(ctx_weight_raw)
        if ctx_weight is None:
            ctx_weight = _extract_effective_ctx_weight(object_data, current_turn=current_turn)

        payload: dict[str, Any] = {
            "object_id": str(object_row.object_id),
            "name": str(object_row.name or ""),
            "data": sanitized_data,
            "similarity": similarity_value,
            "ctx_weight": ctx_weight,
            "distance": round(float(distance), 6) if distance is not None else None,
        }
        if ctx_weight is not None:
            relevance = ctx_weight
            if similarity_value is not None:
                relevance = round((ctx_weight * 0.65) + (similarity_value * 0.35), 6)
            payload["relevance"] = relevance

        if category == "relevant_claim":
            claim_text = _extract_claim_text(object_data)
            if not claim_text:
                continue
            payload["text"] = claim_text
            payload["confidence"] = object_data.get("confidence")
            payload["about_object_id"] = object_data.get("about_object_id")
            payload["location_id"] = object_data.get("location_id")
        elif category == "relevant_link":
            payload["from_object_id"] = str(object_row.object_id)
            payload["from_name"] = str(object_row.name or "")
        elif category == "archived_quest":
            payload["archived"] = True

        source_id = f"{category}:{payload.get('object_id')}"
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)

        prefilter_score, scored_similarity, scored_importance = _score_context_item_payload(
            category=category,
            payload=payload,
            new_turn=effective_turn,
        )
        candidates.append(
            ContextItem(
                category=category,
                payload=payload,
                token_cost=max(_count_json_tokens(payload), 1),
                similarity=scored_similarity,
                importance=scored_importance,
                score=prefilter_score,
                source_id=source_id,
                variant="default",
            )
        )

    candidates.sort(
        key=lambda item: (item.score, item.similarity, item.importance),
        reverse=True,
    )
    return candidates[:resolved_pool_size]
_UNIFIED_FIELD_CATEGORY: dict[str, str] = {
    "session_summaries": "session_summary",
    "hard_memory": "hard_memory",
    "entity_histories": "entity_history",
    "callback_memories": "callback_memory",
    "recent_turns": "recent_turn",
    "npc_knowledge": "npc_knowledge",
    "reaction_hints": "reaction_hint",
    "zone_npcs": "zone_npc",
    "relevant_memories": "relevant_memory",
    "zone_claims": "zone_claim",
    "relevant_claims": "relevant_claim",
    "relevant_npcs": "relevant_npc",
    "latent_consequences": "latent_consequence",
    "relevant_items": "relevant_item",
    "orphaned_items": "orphaned_item",
    "relevant_quests": "relevant_quest",
    "archived_quest_recall": "archived_quest",
    "relevant_factions": "relevant_faction",
    "relevant_links": "relevant_link",
    "player_inventory": "player_inventory",
    "player_location_history": "player_location_history",
}
_UNIFIED_CATEGORY_TYPE_PRIOR: dict[str, float] = {
    "prev_turn": 1.0,
    "hard_memory": 0.96,
    "entity_history": 0.9,
    "callback_memory": 0.74,
    "recent_turn": 0.82,
    "semantic_turn": 0.8,
    "npc_knowledge": 0.95,
    "zone_npc": 0.9,
    "relevant_quest": 0.9,
    "relevant_memory": 0.86,
    "relevant_claim": 0.74,
    "zone_claim": 0.7,
    "relevant_link": 0.72,
    "one_hop_link": 0.76,
    "session_summary": 0.73,
    "reaction_hint": 0.62,
    "relevant_npc": 0.66,
    "relevant_item": 0.6,
    "orphaned_item": 0.52,
    "relevant_faction": 0.58,
    "archived_quest": 0.6,
    "latent_consequence": 0.69,
    "player_inventory": 0.5,
    "player_location_history": 0.46,
}


def _coerce_priority_importance(priority: Any) -> float:
    normalized = str(priority or "").strip().lower()
    if normalized == "high":
        return 1.0
    if normalized == "med":
        return 0.65
    if normalized == "low":
        return 0.35
    return 0.5


def _derive_memory_class_for_context(memory_type: str, memory_data: dict[str, Any]) -> str:
    existing_class = str(memory_data.get("memory_class") or "").strip().lower()
    if existing_class in {"semantic", "episodic", "decision", "bundle"}:
        return existing_class
    if str(memory_data.get("layer") or "").strip().lower() == "bundle":
        return "bundle"
    if str(memory_type or "").strip().lower() == "decision":
        return "decision"
    priority = str(memory_data.get("priority") or "med").strip().lower()
    importance = _coerce_importance(memory_data.get("importance"))
    seen_count = max(_safe_int(memory_data.get("seen_count")) or 0, 0)
    if priority == "high":
        return "semantic"
    if importance is not None and importance >= 0.72:
        return "semantic"
    if seen_count >= 3:
        return "semantic"
    return "episodic"


def _context_item_similarity(item: Any) -> float:
    if isinstance(item, dict):
        for key in ("weighted_similarity", "similarity", "relevance", "turn_weight"):
            value = _coerce_unit_weight(item.get(key))
            if value is not None:
                return value
    return 0.0


def _context_item_importance(item: Any) -> float:
    if isinstance(item, dict):
        raw_importance = _coerce_importance(item.get("importance"))
        if raw_importance is not None:
            return raw_importance
        ctx_weight = _coerce_unit_weight(item.get("ctx_weight"))
        if ctx_weight is not None:
            return ctx_weight
        data = item.get("data")
        if isinstance(data, dict):
            data_weight = _extract_ctx_weight(data)
            if data_weight is not None:
                return data_weight
        return _coerce_priority_importance(item.get("priority"))
    if isinstance(item, str):
        return 0.45
    return 0.5


def _context_item_recency(item: Any, *, new_turn: int) -> float:
    if isinstance(item, dict):
        for key in ("turn_index", "last_seen_turn", "source_turn", "created_turn"):
            turn_value = _safe_int(item.get(key))
            if turn_value is None:
                continue
            delta = max(new_turn - turn_value, 0)
            return round(1.0 / (1.0 + 0.02 * delta), 6)
    return 0.5


def _context_type_prior(category: str, item: Any) -> float:
    base = _UNIFIED_CATEGORY_TYPE_PRIOR.get(category, 0.5)
    if category == "relevant_memory" and isinstance(item, dict):
        memory_class = str(item.get("memory_class") or "").strip().lower()
        if memory_class == "decision":
            base += 0.12
        elif memory_class == "semantic":
            base += 0.08
        elif memory_class == "episodic":
            base += 0.04
        elif memory_class == "bundle":
            base += 0.06
    return round(min(max(base, 0.0), 1.0), 6)


def _context_struct_bonus(category: str, item: Any) -> float:
    bonus = 0.0
    if isinstance(item, dict):
        if item.get("anchor_match"):
            bonus += 0.22
        if isinstance(item.get("anchor_hits"), list) and item.get("anchor_hits"):
            bonus += 0.2
        if category in {"one_hop_link", "relevant_link"}:
            link_type = str(item.get("type") or "").strip().lower()
            if link_type in {"heard", "asserted"}:
                bonus += 0.2
            if not str(item.get("preview") or "").strip():
                bonus -= 0.08
        if isinstance(item.get("applied_ops"), list) and item.get("applied_ops"):
            bonus += 0.12
        semantic_context = item.get("semantic_context")
        if isinstance(semantic_context, dict) and (
            semantic_context.get("prev") or semantic_context.get("next")
        ):
            bonus += 0.08
    return round(min(max(bonus, 0.0), 1.0), 6)


def _score_context_item_payload(
    *,
    category: str,
    payload: Any,
    new_turn: int,
) -> tuple[float, float, float]:
    similarity = _context_item_similarity(payload)
    recency = _context_item_recency(payload, new_turn=new_turn)
    importance = _context_item_importance(payload)
    type_prior = _context_type_prior(category, payload)
    struct_bonus = _context_struct_bonus(category, payload)
    score = (
        0.35 * similarity
        + 0.15 * recency
        + 0.20 * importance
        + 0.20 * type_prior
        + 0.10 * struct_bonus
    )
    return (
        round(min(max(score, 0.0), 1.0), 6),
        similarity,
        importance,
    )


def _context_item_density(item: ContextItem) -> float:
    cost = max(item.token_cost, 1)
    return item.score / float(cost)


def _choose_best_fitting_item(
    candidates: list[ContextItem],
    *,
    selected_source_ids: set[str],
    remaining_tokens: int,
) -> ContextItem | None:
    fitting = [
        item
        for item in candidates
        if item.source_id not in selected_source_ids and item.token_cost <= remaining_tokens
    ]
    if not fitting:
        return None
    fitting.sort(
        key=lambda item: (_context_item_density(item), item.score, -item.token_cost),
        reverse=True,
    )
    return fitting[0]


def _apply_unified_context_scoring(
    context_pack: dict[str, Any],
    *,
    max_total_tokens: int,
    new_turn: int,
) -> dict[str, Any]:
    if not isinstance(context_pack, dict):
        return {}

    capped_max_tokens = max(max_total_tokens, 0)
    internal_keys = {"_recent_turn_variants", "_one_hop_link_candidates"}
    list_fields = [
        key
        for key, value in context_pack.items()
        if key not in internal_keys and isinstance(value, list)
    ]
    fixed_pack = {
        key: value
        for key, value in context_pack.items()
        if key not in internal_keys and key not in list_fields
    }
    trimmed_pack = _fit_fixed_overhead_budget(fixed_pack, capped_max_tokens)
    current_tokens = _serialized_tokens(trimmed_pack)
    remaining_tokens = max(capped_max_tokens - current_tokens, 0)
    if remaining_tokens <= 0:
        for field in list_fields:
            trimmed_pack[field] = []
        return trimmed_pack

    candidates: list[ContextItem] = []
    for field in list_fields:
        if field == "recent_turns":
            continue
        field_items = context_pack.get(field)
        if not isinstance(field_items, list):
            continue
        category = _UNIFIED_FIELD_CATEGORY.get(field, field)
        for idx, payload in enumerate(field_items):
            score, similarity, importance = _score_context_item_payload(
                category=category,
                payload=payload,
                new_turn=new_turn,
            )
            token_cost = max(_count_json_tokens(payload), 1)
            candidates.append(
                ContextItem(
                    category=category,
                    payload=payload,
                    token_cost=token_cost,
                    similarity=similarity,
                    importance=importance,
                    score=score,
                    source_id=f"{field}:{idx}",
                    variant="default",
                )
            )

    raw_turn_variants = context_pack.get("_recent_turn_variants")
    if isinstance(raw_turn_variants, list):
        for variant_item in raw_turn_variants:
            if not isinstance(variant_item, dict):
                continue
            source_id = str(variant_item.get("source_id") or "").strip()
            if not source_id:
                continue
            category = str(variant_item.get("category") or "recent_turn")
            compact_payload = variant_item.get("compact")
            full_payload = variant_item.get("full")
            if compact_payload is not None:
                compact_score, compact_similarity, compact_importance = _score_context_item_payload(
                    category=category,
                    payload=compact_payload,
                    new_turn=new_turn,
                )
                candidates.append(
                    ContextItem(
                        category=category,
                        payload=compact_payload,
                        token_cost=max(_count_json_tokens(compact_payload), 1),
                        similarity=compact_similarity,
                        importance=compact_importance,
                        score=compact_score,
                        source_id=source_id,
                        variant="compact",
                    )
                )
            if full_payload is not None:
                full_score, full_similarity, full_importance = _score_context_item_payload(
                    category=category,
                    payload=full_payload,
                    new_turn=new_turn,
                )
                candidates.append(
                    ContextItem(
                        category=category,
                        payload=full_payload,
                        token_cost=max(_count_json_tokens(full_payload), 1),
                        similarity=full_similarity,
                        importance=full_importance,
                        score=full_score,
                        source_id=source_id,
                        variant="full",
                    )
                )

    one_hop_links = context_pack.get("_one_hop_link_candidates")
    if isinstance(one_hop_links, list):
        for idx, payload in enumerate(one_hop_links):
            score, similarity, importance = _score_context_item_payload(
                category="one_hop_link",
                payload=payload,
                new_turn=new_turn,
            )
            candidates.append(
                ContextItem(
                    category="one_hop_link",
                    payload=payload,
                    token_cost=max(_count_json_tokens(payload), 1),
                    similarity=similarity,
                    importance=importance,
                    score=score,
                    source_id=f"one_hop_link:{idx}",
                    variant="default",
                )
            )

    selected: list[ContextItem] = []
    selected_source_ids: set[str] = set()

    def _select(item: ContextItem) -> None:
        nonlocal remaining_tokens
        selected.append(item)
        selected_source_ids.add(item.source_id)
        remaining_tokens = max(remaining_tokens - item.token_cost, 0)

    # Required diversity categories.
    for category, min_count in _UNIFIED_CONTEXT_DIVERSITY_MINIMA:
        for _ in range(min_count):
            category_candidates = [item for item in candidates if item.category == category]
            if not category_candidates:
                break
            best = _choose_best_fitting_item(
                category_candidates,
                selected_source_ids=selected_source_ids,
                remaining_tokens=remaining_tokens,
            )
            if best is None:
                break
            _select(best)

    # Memory subclass diversity.
    memory_candidates = [item for item in candidates if item.category == "relevant_memory"]
    if memory_candidates:
        for memory_class, min_count in _UNIFIED_MEMORY_CLASS_MINIMA:
            for _ in range(min_count):
                class_candidates = [
                    item
                    for item in memory_candidates
                    if isinstance(item.payload, dict)
                    and str(item.payload.get("memory_class") or "").strip().lower() == memory_class
                ]
                if not class_candidates:
                    break
                best = _choose_best_fitting_item(
                    class_candidates,
                    selected_source_ids=selected_source_ids,
                    remaining_tokens=remaining_tokens,
                )
                if best is None:
                    break
                _select(best)

    # Fill remaining budget by density.
    remaining_candidates = sorted(
        candidates,
        key=lambda item: (_context_item_density(item), item.score, -item.token_cost),
        reverse=True,
    )
    for candidate in remaining_candidates:
        if candidate.source_id in selected_source_ids:
            continue
        if candidate.token_cost > remaining_tokens:
            continue
        _select(candidate)

    # Upgrade compact turn variants to full when budget allows.
    best_full_by_source: dict[str, ContextItem] = {}
    for candidate in candidates:
        if candidate.variant != "full":
            continue
        current = best_full_by_source.get(candidate.source_id)
        if current is None or candidate.score > current.score:
            best_full_by_source[candidate.source_id] = candidate
    for idx, selected_item in enumerate(list(selected)):
        if selected_item.variant != "compact":
            continue
        full_candidate = best_full_by_source.get(selected_item.source_id)
        if full_candidate is None:
            continue
        token_delta = full_candidate.token_cost - selected_item.token_cost
        if token_delta > remaining_tokens:
            continue
        if full_candidate.score < selected_item.score:
            continue
        selected[idx] = full_candidate
        remaining_tokens = max(remaining_tokens - max(token_delta, 0), 0)

    selected_by_field: dict[str, list[ContextItem]] = {field: [] for field in list_fields}
    for item in selected:
        if item.category in {"prev_turn", "recent_turn", "semantic_turn"}:
            selected_by_field.setdefault("recent_turns", []).append(item)
            continue
        if item.category == "one_hop_link":
            selected_by_field.setdefault("relevant_links", []).append(item)
            continue
        resolved_field = next(
            (field for field, category in _UNIFIED_FIELD_CATEGORY.items() if category == item.category),
            None,
        )
        if resolved_field is None:
            continue
        selected_by_field.setdefault(resolved_field, []).append(item)

    for field in list_fields:
        picked = selected_by_field.get(field, [])
        picked.sort(
            key=lambda item: (
                item.score,
                (_safe_int(item.payload.get("turn_index")) or -1) if isinstance(item.payload, dict) else -1,
            ),
            reverse=True,
        )
        trimmed_pack[field] = [item.payload for item in picked]

    # Guardrail: if rough token accounting overshoots, evict lowest-density picks until within cap.
    while _serialized_tokens(trimmed_pack) > capped_max_tokens and selected:
        selected.sort(key=lambda item: (_context_item_density(item), item.score))
        evicted = selected.pop(0)
        for field, items in selected_by_field.items():
            selected_by_field[field] = [item for item in items if item is not evicted]
        for field in list_fields:
            trimmed_pack[field] = [item.payload for item in selected_by_field.get(field, [])]

    return trimmed_pack


def _embed_query_for_relevance(
    session_id: uuid.UUID,
    user_input: str,
    *,
    zone_name: str | None = None,
    recent_ai_text: str | None = None,
    recent_scene_entities: list[dict[str, Any]] | None = None,
    query_text_override: str | None = None,
) -> list[float] | None:
    if not USE_EMBEDDINGS or not user_input.strip():
        return None

    if query_text_override is not None:
        prompt_text = str(query_text_override).strip()
    elif USE_QUERY_REFORMULATOR and _should_reformulate(user_input):
        reformulation = _reformulate_query(
            user_input,
            zone_name=zone_name,
            recent_ai_text=recent_ai_text,
            recent_scene_entities=recent_scene_entities,
            session_id=str(session_id),
        )
        prompt_text = str(reformulation.get("query_text") or "").strip()
    else:
        prompt_text = _build_relevance_query_text(
            user_input,
            zone_name=zone_name,
            recent_ai_text=recent_ai_text,
            recent_scene_entities=recent_scene_entities,
        )
    if not prompt_text:
        return None

    try:
        query_text = f"Query: {prompt_text}"
        return _maybe_embed_texts(
            [query_text],
            instruction=RELEVANCE_QUERY_EMBED_INSTRUCTION,
        )[0]
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to compute query embedding for session_id=%s",
            session_id,
            exc_info=True,
        )
        return None


def _list_relevant_objects_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    object_type: str,
    namespace: str,
    current_turn: int | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    row_limit = max(limit, 1)
    ctx_weight_expr = _object_ctx_weight_expr(current_turn=current_turn)
    base_filters = (
        models.ObjectModel.session_id == session_id,
        models.ObjectModel.type == object_type,
        or_(
            models.ObjectModel.data["status"].astext.is_(None),
            models.ObjectModel.data["status"].astext != "inactive",
        ),
    )

    if query_embedding is None:
        object_rows = db.execute(
            select(models.ObjectModel)
            .where(*base_filters)
            .order_by(
                ctx_weight_expr.desc().nulls_last(),
                models.ObjectModel.created_at.desc(),
            )
            .limit(row_limit)
        ).scalars().all()

        fallback_relevant: list[dict[str, Any]] = []
        for object_row in object_rows:
            object_data = dict(object_row.data or {})
            ctx_weight = _extract_effective_ctx_weight(object_data, current_turn=current_turn)
            entry: dict[str, Any] = {
                "object_id": str(object_row.object_id),
                "name": object_row.name,
                "data": _sanitize_object_data_for_context(object_data),
                "similarity": None,
            }
            if ctx_weight is not None:
                entry["ctx_weight"] = ctx_weight
                entry["relevance"] = ctx_weight
            fallback_relevant.append(entry)
        return fallback_relevant

    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
    similarity_expr = 1.0 - distance_expr
    relevance_expr = (func.coalesce(ctx_weight_expr, 0.0) * 0.65) + (similarity_expr * 0.35)
    scored_rows = db.execute(
        select(
            models.ObjectModel,
            distance_expr.label("distance"),
            ctx_weight_expr.label("ctx_weight"),
            relevance_expr.label("relevance"),
        )
        .join(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
            ),
        )
        .where(
            *base_filters,
            models.ObjectEmbeddingModel.namespace == namespace,
        )
        .order_by(
            relevance_expr.desc(),
            distance_expr.asc(),
            models.ObjectModel.created_at.desc(),
        )
        .limit(row_limit)
    ).all()

    scored_relevant: list[dict[str, Any]] = []
    for row in scored_rows:
        object_row, distance, ctx_weight_raw, relevance_raw = _unpack_scored_object_row(row)
        similarity: float | None = None
        if distance is not None:
            similarity = round(1.0 - float(distance), 6)
        ctx_weight = _coerce_unit_weight(ctx_weight_raw)
        if ctx_weight is None:
            ctx_weight = _extract_effective_ctx_weight(dict(object_row.data or {}), current_turn=current_turn)
        relevance: float | None = None
        if relevance_raw is not None:
            relevance = round(float(relevance_raw), 6)
        scored_relevant.append(
            {
                "object_id": str(object_row.object_id),
                "name": object_row.name,
                "data": _sanitize_object_data_for_context(dict(object_row.data or {})),
                "similarity": similarity,
                "ctx_weight": ctx_weight,
                "relevance": relevance,
            }
        )
    return scored_relevant


def _list_relevant_npcs_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    zone_id: uuid.UUID | None,
    current_turn: int | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    row_limit = max(limit, 1)
    ctx_weight_expr = _object_ctx_weight_expr(current_turn=current_turn)
    base_filters = (
        models.ObjectModel.session_id == session_id,
        models.ObjectModel.type == "npc",
        or_(
            models.ObjectModel.data["status"].astext.is_(None),
            models.ObjectModel.data["status"].astext != "inactive",
        ),
    )

    if query_embedding is None:
        query = (
            select(models.ObjectModel)
            .where(*base_filters)
            .order_by(
                ctx_weight_expr.desc().nulls_last(),
                models.ObjectModel.created_at.desc(),
            )
            .limit(row_limit)
        )
        if zone_id is not None:
            query = query.join(
                models.LinkModel,
                and_(
                    models.LinkModel.session_id == models.ObjectModel.session_id,
                    models.LinkModel.from_object_id == models.ObjectModel.object_id,
                    models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                    models.LinkModel.valid_to_turn.is_(None),
                    models.LinkModel.to_object_id == zone_id,
                ),
            )
        object_rows = db.execute(query).scalars().all()
        fallback_relevant: list[dict[str, Any]] = []
        for object_row in object_rows:
            object_data = dict(object_row.data or {})
            ctx_weight = _extract_effective_ctx_weight(object_data, current_turn=current_turn)
            entry: dict[str, Any] = {
                "object_id": str(object_row.object_id),
                "name": object_row.name,
                "data": _sanitize_object_data_for_context(object_data),
                "similarity": None,
            }
            if ctx_weight is not None:
                entry["ctx_weight"] = ctx_weight
                entry["relevance"] = ctx_weight
            fallback_relevant.append(entry)
        return fallback_relevant

    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
    similarity_expr = 1.0 - distance_expr
    relevance_expr = (func.coalesce(ctx_weight_expr, 0.0) * 0.65) + (similarity_expr * 0.35)
    query = (
        select(
            models.ObjectModel,
            distance_expr.label("distance"),
            ctx_weight_expr.label("ctx_weight"),
            relevance_expr.label("relevance"),
        )
        .join(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
                models.ObjectEmbeddingModel.namespace == "npc_profile",
            ),
        )
        .where(*base_filters)
        .order_by(
            relevance_expr.desc(),
            distance_expr.asc(),
            models.ObjectModel.created_at.desc(),
        )
        .limit(row_limit)
    )

    if zone_id is not None:
        query = query.join(
            models.LinkModel,
            and_(
                models.LinkModel.session_id == models.ObjectModel.session_id,
                models.LinkModel.from_object_id == models.ObjectModel.object_id,
                models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                models.LinkModel.valid_to_turn.is_(None),
                models.LinkModel.to_object_id == zone_id,
            ),
        )

    scored_rows = db.execute(query).all()
    scored_relevant: list[dict[str, Any]] = []
    for row in scored_rows:
        object_row, distance, ctx_weight_raw, relevance_raw = _unpack_scored_object_row(row)
        similarity: float | None = None
        if distance is not None:
            similarity = round(1.0 - float(distance), 6)
        ctx_weight = _coerce_unit_weight(ctx_weight_raw)
        if ctx_weight is None:
            ctx_weight = _extract_effective_ctx_weight(dict(object_row.data or {}), current_turn=current_turn)
        relevance: float | None = None
        if relevance_raw is not None:
            relevance = round(float(relevance_raw), 6)
        scored_relevant.append(
            {
                "object_id": str(object_row.object_id),
                "name": object_row.name,
                "data": _sanitize_object_data_for_context(dict(object_row.data or {})),
                "similarity": similarity,
                "ctx_weight": ctx_weight,
                "relevance": relevance,
            }
        )
    return scored_relevant


def _list_zone_npcs_with_relationships(
    db: Session,
    session_id: uuid.UUID,
    zone_id: uuid.UUID | None,
    *,
    current_turn: int | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if zone_id is None:
        return []

    npc_rows = db.execute(
        select(models.ObjectModel)
        .join(
            models.LinkModel,
            and_(
                models.LinkModel.session_id == models.ObjectModel.session_id,
                models.LinkModel.from_object_id == models.ObjectModel.object_id,
                models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                models.LinkModel.valid_to_turn.is_(None),
                models.LinkModel.to_object_id == zone_id,
            ),
        )
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "npc",
            or_(
                models.ObjectModel.data["status"].astext.is_(None),
                models.ObjectModel.data["status"].astext != "inactive",
            ),
        )
        .order_by(
            _object_ctx_weight_expr(current_turn=current_turn).desc().nulls_last(),
            models.ObjectModel.created_at.desc(),
        )
        .limit(max(limit, 1))
    ).scalars().all()

    if not npc_rows:
        return []

    npc_ids = [row.object_id for row in npc_rows]
    relationship_rows = db.execute(
        select(
            models.LinkModel.from_object_id,
            models.LinkModel.type,
            models.ObjectModel.object_id,
            models.ObjectModel.name,
        )
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.LinkModel.session_id,
                models.ObjectModel.object_id == models.LinkModel.to_object_id,
            ),
        )
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id.in_(npc_ids),
            models.LinkModel.valid_to_turn.is_(None),
            models.LinkModel.type.in_(tuple(sorted(NPC_SOCIAL_LINK_TYPES))),
        )
        .order_by(models.LinkModel.created_at.asc())
    ).all()

    relationships_by_from: dict[uuid.UUID, list[dict[str, str]]] = {npc_id: [] for npc_id in npc_ids}
    seen_pairs_by_from: dict[uuid.UUID, set[tuple[str, uuid.UUID]]] = {npc_id: set() for npc_id in npc_ids}
    for from_object_id, link_type, target_object_id, target_name in relationship_rows:
        if from_object_id not in relationships_by_from:
            continue
        if target_object_id is None:
            continue
        dedup_key = (str(link_type), target_object_id)
        if dedup_key in seen_pairs_by_from[from_object_id]:
            continue
        seen_pairs_by_from[from_object_id].add(dedup_key)
        relationships_by_from[from_object_id].append(
            {
                "type": str(link_type),
                "target_id": str(target_object_id),
                "target_name": str(target_name or ""),
            }
        )

    result: list[dict[str, Any]] = []
    for npc_row in npc_rows:
        npc_data = dict(npc_row.data or {})
        ctx_weight = _extract_effective_ctx_weight(npc_data, current_turn=current_turn)
        result.append(
            {
                "object_id": str(npc_row.object_id),
                "name": npc_row.name,
                "attitude": (
                    npc_data.get("attitude")
                    or npc_data.get("отношение")
                    or npc_data.get("relation_to_player")
                ),
                "short_desc": _truncate_text(str(npc_data.get("short_desc") or ""), 200),
                "ctx_weight": ctx_weight,
                "relationships": relationships_by_from.get(npc_row.object_id, []),
            }
        )
    return result


def _get_relevant_player_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    player_object_id: uuid.UUID | None,
) -> dict[str, Any] | None:
    if player_object_id is None:
        return None

    player_object = _get_object(db, session_id, player_object_id)
    if player_object is None or player_object.type != "player":
        return None

    player_data = dict(player_object.data or {})
    _rollback_read_only_autobegin_transaction(db)
    profile_text = _build_player_profile_text(player_object.name, player_data)
    similarity: float | None = None

    if query_embedding is not None and USE_EMBEDDINGS:
        distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
        distance = db.execute(
            select(distance_expr.label("distance")).where(
                models.ObjectEmbeddingModel.session_id == session_id,
                models.ObjectEmbeddingModel.object_id == player_object_id,
                models.ObjectEmbeddingModel.namespace == "player_profile",
            )
        ).scalar_one_or_none()
        if distance is not None:
            similarity = round(1.0 - float(distance), 6)

    return {
        "object_id": str(player_object.object_id),
        "name": player_object.name,
        "data": _sanitize_object_data_for_context(player_data),
        "profile": _truncate_text(profile_text, 260),
        "similarity": similarity,
    }


def _get_player_inventory(
    db: Session,
    session_id: uuid.UUID,
    player_object_id: uuid.UUID | None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if player_object_id is None:
        return []

    rows = db.execute(
        select(models.ObjectModel)
        .join(
            models.LinkModel,
            and_(
                models.LinkModel.session_id == models.ObjectModel.session_id,
                models.LinkModel.from_object_id == models.ObjectModel.object_id,
                models.LinkModel.type == "carried_by",
                models.LinkModel.to_object_id == player_object_id,
                models.LinkModel.valid_to_turn.is_(None),
            ),
        )
        .where(models.ObjectModel.session_id == session_id)
        .order_by(models.ObjectModel.created_at.asc())
        .limit(max(limit, 0))
    ).scalars().all()

    return [
        {
            "object_id": str(row.object_id),
            "name": row.name,
            "data": _sanitize_object_data_for_context(dict(row.data or {})),
        }
        for row in rows
    ]


def _list_orphaned_items_for_context(
    db: Session,
    session_id: uuid.UUID,
    *,
    new_turn: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    normalized_turn = max(int(new_turn), 0) if isinstance(new_turn, int) else 0
    if normalized_turn <= 1:
        return []

    row_limit = max(limit, 1)
    lookback_span = max(int(ORPHANED_ITEMS_LOOKBACK_TURNS), 1)
    lookback_end = max(normalized_turn - 1, 0)
    lookback_start = max(lookback_end - (lookback_span - 1), 0)
    carried_by_type = "carried_by"

    active_owner_exists = (
        select(models.LinkModel.link_id)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == models.ObjectModel.object_id,
            models.LinkModel.type == carried_by_type,
            models.LinkModel.valid_to_turn.is_(None),
        )
        .exists()
    )
    recent_closed_owner_exists = (
        select(models.LinkModel.link_id)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == models.ObjectModel.object_id,
            models.LinkModel.type == carried_by_type,
            models.LinkModel.valid_to_turn.is_not(None),
            models.LinkModel.valid_to_turn >= lookback_start,
            models.LinkModel.valid_to_turn <= lookback_end,
        )
        .exists()
    )
    latest_closed_owner_id = (
        select(models.LinkModel.to_object_id)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == models.ObjectModel.object_id,
            models.LinkModel.type == carried_by_type,
            models.LinkModel.valid_to_turn.is_not(None),
            models.LinkModel.valid_to_turn >= lookback_start,
            models.LinkModel.valid_to_turn <= lookback_end,
        )
        .order_by(
            models.LinkModel.valid_to_turn.desc(),
            models.LinkModel.created_at.desc(),
        )
        .limit(1)
        .correlate(models.ObjectModel)
        .scalar_subquery()
    )
    latest_closed_owner_turn = (
        select(models.LinkModel.valid_to_turn)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == models.ObjectModel.object_id,
            models.LinkModel.type == carried_by_type,
            models.LinkModel.valid_to_turn.is_not(None),
            models.LinkModel.valid_to_turn >= lookback_start,
            models.LinkModel.valid_to_turn <= lookback_end,
        )
        .order_by(
            models.LinkModel.valid_to_turn.desc(),
            models.LinkModel.created_at.desc(),
        )
        .limit(1)
        .correlate(models.ObjectModel)
        .scalar_subquery()
    )

    owner_object = aliased(models.ObjectModel)
    rows = db.execute(
        select(
            models.ObjectModel.object_id,
            models.ObjectModel.name,
            models.ObjectModel.data,
            latest_closed_owner_id.label("last_owner_id"),
            owner_object.name.label("last_owner_name"),
            latest_closed_owner_turn.label("owner_lost_turn"),
        )
        .select_from(models.ObjectModel)
        .outerjoin(
            owner_object,
            and_(
                owner_object.session_id == models.ObjectModel.session_id,
                owner_object.object_id == latest_closed_owner_id,
            ),
        )
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "item",
            ~active_owner_exists,
            recent_closed_owner_exists,
        )
        .order_by(
            latest_closed_owner_turn.desc().nulls_last(),
            models.ObjectModel.created_at.desc(),
        )
        .limit(row_limit)
    ).all()

    orphaned_items: list[dict[str, Any]] = []
    for object_id, name, data, last_owner_id, last_owner_name, owner_lost_turn in rows:
        item_data = _sanitize_object_data_for_context(dict(data or {}))
        short_desc = str(item_data.get("short_desc") or "").strip()
        entry: dict[str, Any] = {
            "object_id": str(object_id),
            "name": str(name or ""),
            "owner_lost_turn": int(owner_lost_turn) if isinstance(owner_lost_turn, int) else None,
            "last_owner_id": str(last_owner_id) if isinstance(last_owner_id, uuid.UUID) else None,
            "last_owner_name": str(last_owner_name or "") if last_owner_name is not None else None,
        }
        if short_desc:
            entry["short_desc"] = _truncate_text(short_desc, 180)
        orphaned_items.append(entry)
    return orphaned_items


def _get_player_location_history(
    db: Session,
    session_id: uuid.UUID,
    player_object_id: uuid.UUID | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if player_object_id is None:
        return []

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
            models.LinkModel.from_object_id == player_object_id,
            models.LinkModel.type == LOCATED_IN_LINK_TYPE,
            models.LinkModel.valid_to_turn.is_not(None),
        )
        .order_by(models.LinkModel.valid_to_turn.desc())
        .limit(max(limit, 1))
    ).all()

    return [
        {
            "zone_name": zone_row.name,
            "zone_id": str(link_row.to_object_id),
            "from_turn": link_row.valid_from_turn,
            "to_turn": link_row.valid_to_turn,
        }
        for link_row, zone_row in rows
    ]


def _list_relevant_items_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    current_turn: int | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    return _list_relevant_objects_for_input(
        db=db,
        session_id=session_id,
        query_embedding=query_embedding,
        object_type="item",
        namespace="item_profile",
        current_turn=current_turn,
        limit=limit,
    )


def _list_relevant_factions_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    current_turn: int | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    return _list_relevant_objects_for_input(
        db=db,
        session_id=session_id,
        query_embedding=query_embedding,
        object_type="faction",
        namespace="faction_profile",
        current_turn=current_turn,
        limit=limit,
    )


def _list_relevant_links_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    current_turn: int | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    row_limit = max(limit, 1)
    ctx_weight_expr = _object_ctx_weight_expr(current_turn=current_turn)
    base_filters = (
        models.ObjectModel.session_id == session_id,
        or_(
            models.ObjectModel.data["status"].astext.is_(None),
            models.ObjectModel.data["status"].astext != "inactive",
        ),
    )

    if query_embedding is None:
        object_rows = db.execute(
            select(models.ObjectModel)
            .where(*base_filters)
            .order_by(
                ctx_weight_expr.desc().nulls_last(),
                models.ObjectModel.created_at.desc(),
            )
            .limit(row_limit)
        ).scalars().all()

        fallback_relevant: list[dict[str, Any]] = []
        for object_row in object_rows:
            facts = _list_active_link_context_snippets(
                db=db,
                session_id=session_id,
                from_object_id=object_row.object_id,
                from_name=object_row.name,
            )
            if not facts:
                continue
            ctx_weight = _extract_effective_ctx_weight(dict(object_row.data or {}), current_turn=current_turn)
            fallback_relevant.append(
                {
                    "from_object_id": str(object_row.object_id),
                    "from_name": object_row.name,
                    "facts": facts[:3],
                    "preview": _truncate_text(" | ".join(facts[:2]), 320),
                    "similarity": None,
                    "ctx_weight": ctx_weight,
                    "relevance": ctx_weight,
                }
            )
        return fallback_relevant

    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
    similarity_expr = 1.0 - distance_expr
    relevance_expr = (func.coalesce(ctx_weight_expr, 0.0) * 0.65) + (similarity_expr * 0.35)
    scored_rows = db.execute(
        select(
            models.ObjectModel,
            distance_expr.label("distance"),
            ctx_weight_expr.label("ctx_weight"),
            relevance_expr.label("relevance"),
        )
        .join(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
                models.ObjectEmbeddingModel.namespace == LINK_CONTEXT_NAMESPACE,
            ),
        )
        .where(*base_filters)
        .order_by(
            relevance_expr.desc(),
            distance_expr.asc(),
            models.ObjectModel.created_at.desc(),
        )
        .limit(row_limit)
    ).all()

    scored_relevant: list[dict[str, Any]] = []
    for row in scored_rows:
        object_row, distance, ctx_weight_raw, relevance_raw = _unpack_scored_object_row(row)
        facts = _list_active_link_context_snippets(
            db=db,
            session_id=session_id,
            from_object_id=object_row.object_id,
            from_name=object_row.name,
        )
        if not facts:
            continue
        similarity: float | None = None
        if distance is not None:
            similarity = round(1.0 - float(distance), 6)
        ctx_weight = _coerce_unit_weight(ctx_weight_raw)
        if ctx_weight is None:
            ctx_weight = _extract_effective_ctx_weight(dict(object_row.data or {}), current_turn=current_turn)
        relevance: float | None = None
        if relevance_raw is not None:
            relevance = round(float(relevance_raw), 6)
        preview = _truncate_text(" | ".join(facts[:2]), 320)
        scored_relevant.append(
            {
                "from_object_id": str(object_row.object_id),
                "from_name": object_row.name,
                "facts": facts[:3],
                "preview": preview,
                "similarity": similarity,
                "ctx_weight": ctx_weight,
                "relevance": relevance,
            }
        )
    return scored_relevant


def _list_relevant_quests_for_input(
    db: Session,
    session_id: uuid.UUID,
    *,
    player_object_id: uuid.UUID | None,
    current_turn: int | None = None,
) -> list[dict[str, Any]]:
    if player_object_id is None:
        return []

    status_expr = models.ObjectModel.data["status"].astext
    rows = db.execute(
        select(models.ObjectModel, models.LinkModel.valid_from_turn)
        .join(
            models.LinkModel,
            and_(
                models.LinkModel.session_id == models.ObjectModel.session_id,
                models.LinkModel.to_object_id == models.ObjectModel.object_id,
            ),
        )
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "quest",
            models.LinkModel.from_object_id == player_object_id,
            models.LinkModel.type == TRACKING_QUEST_LINK_TYPE,
            models.LinkModel.valid_to_turn.is_(None),
            or_(
                status_expr.is_(None),
                func.lower(status_expr).notin_(QUEST_TERMINAL_STATUSES),
            ),
        )
        .order_by(
            models.LinkModel.valid_from_turn.desc(),
            models.ObjectModel.created_at.desc(),
        )
    ).all()

    relevant: list[dict[str, Any]] = []
    for object_row, _valid_from_turn in rows:
        object_data = dict(object_row.data or {})
        ctx_weight = _extract_effective_ctx_weight(object_data, current_turn=current_turn)
        entry: dict[str, Any] = {
            "object_id": str(object_row.object_id),
            "name": object_row.name,
            "data": _sanitize_object_data_for_context(object_data),
            "similarity": None,
        }
        if ctx_weight is not None:
            entry["ctx_weight"] = ctx_weight
            entry["relevance"] = ctx_weight
        relevant.append(entry)
    return relevant


def _sanitize_archived_quest_data_for_context(raw_data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_data, dict):
        return {}

    sanitized = _sanitize_object_data_for_context(raw_data)
    result: dict[str, Any] = {}
    status_value = raw_data.get("status")
    if isinstance(status_value, str):
        normalized_status = status_value.strip()
        if normalized_status:
            result["status"] = _truncate_text(normalized_status, 40)

    short_desc_value = sanitized.get("short_desc")
    if isinstance(short_desc_value, str):
        normalized_short_desc = short_desc_value.strip()
        if normalized_short_desc:
            result["short_desc"] = _truncate_text(normalized_short_desc, 220)
    return result


def _list_archived_quest_recall_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    player_object_id: uuid.UUID | None,
    exact_names: list[str],
    limit: int = ARCHIVED_QUEST_RECALL_LIMIT,
    max_distance: float = ARCHIVED_QUEST_RECALL_MAX_DISTANCE,
) -> list[dict[str, Any]]:
    if query_embedding is None or player_object_id is None:
        return []

    resolved_limit = max(limit, 1)
    max_distance_value = max(float(max_distance), 0.0)
    exact_name_values = tuple(
        str(item).strip().casefold()
        for item in exact_names
        if isinstance(item, str) and str(item).strip()
    )

    status_expr = models.ObjectModel.data["status"].astext
    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
    similarity_expr = 1.0 - distance_expr
    exact_name_match_expr = case(
        (func.lower(models.ObjectModel.name).in_(exact_name_values), 1),
        else_=0,
    )
    active_tracking_exists = (
        select(models.LinkModel.link_id)
        .where(
            models.LinkModel.session_id == models.ObjectModel.session_id,
            models.LinkModel.from_object_id == player_object_id,
            models.LinkModel.to_object_id == models.ObjectModel.object_id,
            models.LinkModel.type == TRACKING_QUEST_LINK_TYPE,
            models.LinkModel.valid_to_turn.is_(None),
        )
        .exists()
    )

    rows = db.execute(
        select(
            models.ObjectModel,
            distance_expr.label("distance"),
            similarity_expr.label("similarity"),
            exact_name_match_expr.label("exact_name_match"),
        )
        .join(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
                models.ObjectEmbeddingModel.namespace == "quest_profile",
            ),
        )
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "quest",
            func.lower(status_expr).in_(QUEST_TERMINAL_STATUSES),
            distance_expr <= max_distance_value,
            ~active_tracking_exists,
        )
        .order_by(
            exact_name_match_expr.desc(),
            distance_expr.asc(),
            models.ObjectModel.created_at.desc(),
        )
        .limit(resolved_limit)
    ).all()

    relevant: list[dict[str, Any]] = []
    for object_row, distance, similarity_raw, exact_name_match in rows:
        similarity_value: float | None = None
        if similarity_raw is not None:
            similarity_value = round(float(similarity_raw), 6)
        distance_value: float | None = None
        if distance is not None:
            distance_value = round(float(distance), 6)
        relevant.append(
            {
                "object_id": str(object_row.object_id),
                "name": object_row.name,
                "archived": True,
                "exact_name_match": bool(exact_name_match),
                "distance": distance_value,
                "similarity": similarity_value,
                "data": _sanitize_archived_quest_data_for_context(dict(object_row.data or {})),
            }
        )
    return relevant


def _build_claim_audience_map(
    db: Session,
    session_id: uuid.UUID,
    claim_ids: list[uuid.UUID],
) -> dict[uuid.UUID, dict[str, Any]]:
    if not claim_ids:
        return {}

    link_rows = db.execute(
        select(
            models.LinkModel.to_object_id,
            models.LinkModel.type,
            models.LinkModel.from_object_id,
            models.ObjectModel.name,
        )
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.LinkModel.session_id,
                models.ObjectModel.object_id == models.LinkModel.from_object_id,
            ),
        )
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.to_object_id.in_(claim_ids),
            models.LinkModel.valid_to_turn.is_(None),
            models.LinkModel.type.in_(("asserted", "heard")),
        )
        .order_by(models.LinkModel.created_at.asc())
    ).all()

    audience_by_claim: dict[uuid.UUID, dict[str, Any]] = {
        claim_id: {
            "speaker_id": None,
            "speaker_name": None,
            "listener_ids": [],
            "listener_names": [],
        }
        for claim_id in claim_ids
    }
    seen_listeners_by_claim: dict[uuid.UUID, set[uuid.UUID]] = {claim_id: set() for claim_id in claim_ids}
    for claim_id, link_type, actor_id, actor_name in link_rows:
        if claim_id not in audience_by_claim:
            continue
        if link_type == "asserted":
            if audience_by_claim[claim_id]["speaker_id"] is None:
                audience_by_claim[claim_id]["speaker_id"] = str(actor_id) if actor_id else None
                audience_by_claim[claim_id]["speaker_name"] = str(actor_name or "")
            continue
        if link_type != "heard" or actor_id is None:
            continue
        if actor_id in seen_listeners_by_claim[claim_id]:
            continue
        seen_listeners_by_claim[claim_id].add(actor_id)
        audience_by_claim[claim_id]["listener_ids"].append(str(actor_id))
        audience_by_claim[claim_id]["listener_names"].append(str(actor_name or ""))

    return audience_by_claim


def _list_relevant_claims_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    zone_id: uuid.UUID | None = None,
    current_turn: int | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    row_limit = max(limit, 1)
    fallback_limit = max(row_limit * 3, row_limit)
    ctx_weight_expr = _object_ctx_weight_expr(current_turn=current_turn)
    base_filters = (
        models.ObjectModel.session_id == session_id,
        models.ObjectModel.type == "claim",
    )
    zone_id_text = str(zone_id) if zone_id is not None else None
    null_location_filter = or_(
        models.ObjectModel.data["location_id"].astext.is_(None),
        models.ObjectModel.data["location_id"].astext == "",
        models.ObjectModel.data["location_id"].astext == "null",
    )
    zone_actor_ids: set[str] | None = None

    def _has_zone_audience_overlap(audience: dict[str, Any]) -> bool:
        nonlocal zone_actor_ids
        if zone_id is None:
            return True
        if zone_actor_ids is None:
            zone_actor_ids = _list_active_zone_actor_ids(db, session_id, zone_id)
        speaker_id = str(audience.get("speaker_id") or "").strip()
        listener_ids = [str(raw).strip() for raw in list(audience.get("listener_ids") or []) if str(raw).strip()]
        return bool(
            (speaker_id and speaker_id in zone_actor_ids)
            or any(listener_id in zone_actor_ids for listener_id in listener_ids)
        )

    def _has_explicit_zone_location(object_row: models.ObjectModel) -> bool:
        if zone_id is None:
            return True
        object_data = dict(object_row.data or {})
        return str(object_data.get("location_id") or "").strip() == zone_id_text

    def _build_claim_entry(
        *,
        object_row: models.ObjectModel,
        audience: dict[str, Any],
        similarity: float | None,
        ctx_weight: float | None,
        relevance: float | None,
    ) -> dict[str, Any] | None:
        object_data = dict(object_row.data or {})
        claim_text = _extract_claim_text(object_data)
        if not claim_text:
            return None
        return {
            "object_id": str(object_row.object_id),
            "text": claim_text,
            "confidence": object_data.get("confidence"),
            "about_object_id": object_data.get("about_object_id"),
            "location_id": object_data.get("location_id"),
            "speaker_id": audience.get("speaker_id"),
            "speaker_name": audience.get("speaker_name"),
            "listener_ids": list(audience.get("listener_ids") or []),
            "listener_names": list(audience.get("listener_names") or []),
            "similarity": similarity,
            "ctx_weight": ctx_weight,
            "relevance": relevance,
        }

    if query_embedding is None:
        if zone_id is None:
            explicit_rows = list(
                db.execute(
                    select(models.ObjectModel)
                    .where(*base_filters)
                    .order_by(
                        ctx_weight_expr.desc().nulls_last(),
                        models.ObjectModel.created_at.desc(),
                    )
                    .limit(row_limit)
                ).scalars().all()
            )
            fallback_rows: list[models.ObjectModel] = []
        else:
            explicit_rows = list(
                db.execute(
                    select(models.ObjectModel)
                    .where(
                        *base_filters,
                        models.ObjectModel.data["location_id"].astext == zone_id_text,
                    )
                    .order_by(
                        ctx_weight_expr.desc().nulls_last(),
                        models.ObjectModel.created_at.desc(),
                    )
                    .limit(row_limit)
                ).scalars().all()
            )
            fallback_rows = []
            if len(explicit_rows) < row_limit:
                fallback_rows = list(
                    db.execute(
                        select(models.ObjectModel)
                        .where(*base_filters, null_location_filter)
                        .order_by(
                            ctx_weight_expr.desc().nulls_last(),
                            models.ObjectModel.created_at.desc(),
                        )
                        .limit(fallback_limit)
                    ).scalars().all()
                )

        candidate_rows: list[models.ObjectModel] = list(explicit_rows)
        candidate_rows.extend(row for row in fallback_rows if row.object_id not in {r.object_id for r in explicit_rows})
        audience_by_claim = _build_claim_audience_map(
            db,
            session_id,
            [row.object_id for row in candidate_rows],
        )

        relevant: list[dict[str, Any]] = []
        seen_ids: set[uuid.UUID] = set()
        for object_row in explicit_rows:
            if not _has_explicit_zone_location(object_row):
                continue
            audience = audience_by_claim.get(object_row.object_id, {})
            row_ctx_weight = _extract_effective_ctx_weight(
                dict(object_row.data or {}),
                current_turn=current_turn,
            )
            claim_entry = _build_claim_entry(
                object_row=object_row,
                audience=audience,
                similarity=None,
                ctx_weight=row_ctx_weight,
                relevance=row_ctx_weight,
            )
            if claim_entry is None:
                continue
            relevant.append(claim_entry)
            seen_ids.add(object_row.object_id)
            if len(relevant) >= row_limit:
                return relevant

        for object_row in fallback_rows:
            if len(relevant) >= row_limit:
                break
            if object_row.object_id in seen_ids:
                continue
            audience = audience_by_claim.get(object_row.object_id, {})
            if not _has_zone_audience_overlap(audience):
                continue
            row_ctx_weight = _extract_effective_ctx_weight(
                dict(object_row.data or {}),
                current_turn=current_turn,
            )
            claim_entry = _build_claim_entry(
                object_row=object_row,
                audience=audience,
                similarity=None,
                ctx_weight=row_ctx_weight,
                relevance=row_ctx_weight,
            )
            if claim_entry is None:
                continue
            relevant.append(claim_entry)
            seen_ids.add(object_row.object_id)
        return relevant

    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
    similarity_expr = 1.0 - distance_expr
    relevance_expr = (func.coalesce(ctx_weight_expr, 0.0) * 0.65) + (similarity_expr * 0.35)
    query_base = (
        select(
            models.ObjectModel,
            distance_expr.label("distance"),
            ctx_weight_expr.label("ctx_weight"),
            relevance_expr.label("relevance"),
        )
        .join(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
                models.ObjectEmbeddingModel.namespace == "claim_text",
            ),
        )
        .order_by(
            relevance_expr.desc(),
            distance_expr.asc(),
            models.ObjectModel.created_at.desc(),
        )
    )
    if zone_id is None:
        explicit_scored_rows = list(
            db.execute(
                query_base.where(*base_filters).limit(row_limit)
            ).all()
        )
        fallback_scored_rows: list[Any] = []
    else:
        explicit_scored_rows = list(
            db.execute(
                query_base.where(
                    *base_filters,
                    models.ObjectModel.data["location_id"].astext == zone_id_text,
                ).limit(row_limit)
            ).all()
        )
        fallback_scored_rows = []
        if len(explicit_scored_rows) < row_limit:
            fallback_scored_rows = list(
                db.execute(
                    query_base.where(*base_filters, null_location_filter).limit(fallback_limit)
                ).all()
            )

    explicit_scored_ids: set[uuid.UUID] = set()
    for scored_row in explicit_scored_rows:
        scored_object_row, _distance, _ctx_weight_raw, _relevance_raw = _unpack_scored_object_row(scored_row)
        if scored_object_row is not None:
            explicit_scored_ids.add(scored_object_row.object_id)

    scored_candidate_rows: list[Any] = list(explicit_scored_rows)
    for scored_row in fallback_scored_rows:
        scored_object_row, _distance, _ctx_weight_raw, _relevance_raw = _unpack_scored_object_row(scored_row)
        if scored_object_row is None or scored_object_row.object_id in explicit_scored_ids:
            continue
        scored_candidate_rows.append(scored_row)

    scored_candidate_ids: list[uuid.UUID] = []
    for scored_row in scored_candidate_rows:
        scored_object_row, _distance, _ctx_weight_raw, _relevance_raw = _unpack_scored_object_row(scored_row)
        if scored_object_row is not None:
            scored_candidate_ids.append(scored_object_row.object_id)
    audience_by_claim = _build_claim_audience_map(
        db,
        session_id,
        scored_candidate_ids,
    )

    relevant_scored: list[dict[str, Any]] = []
    seen_scored_ids: set[uuid.UUID] = set()
    for row in explicit_scored_rows:
        object_row, distance, ctx_weight_raw, relevance_raw = _unpack_scored_object_row(row)
        if not _has_explicit_zone_location(object_row):
            continue

        similarity: float | None = None
        if distance is not None:
            similarity = round(1.0 - float(distance), 6)
        ctx_weight = _coerce_unit_weight(ctx_weight_raw)
        if ctx_weight is None:
            ctx_weight = _extract_effective_ctx_weight(dict(object_row.data or {}), current_turn=current_turn)
        relevance: float | None = None
        if relevance_raw is not None:
            relevance = round(float(relevance_raw), 6)
        elif ctx_weight is not None:
            relevance = ctx_weight

        audience = audience_by_claim.get(object_row.object_id, {})
        claim_entry = _build_claim_entry(
            object_row=object_row,
            audience=audience,
            similarity=similarity,
            ctx_weight=ctx_weight,
            relevance=relevance,
        )
        if claim_entry is None:
            continue
        relevant_scored.append(claim_entry)
        seen_scored_ids.add(object_row.object_id)
        if len(relevant_scored) >= row_limit:
            return relevant_scored

    for row in fallback_scored_rows:
        if len(relevant_scored) >= row_limit:
            break
        object_row, distance, ctx_weight_raw, relevance_raw = _unpack_scored_object_row(row)
        if object_row.object_id in seen_scored_ids:
            continue
        audience = audience_by_claim.get(object_row.object_id, {})
        if not _has_zone_audience_overlap(audience):
            continue

        similarity = None
        if distance is not None:
            similarity = round(1.0 - float(distance), 6)
        ctx_weight = _coerce_unit_weight(ctx_weight_raw)
        if ctx_weight is None:
            ctx_weight = _extract_effective_ctx_weight(dict(object_row.data or {}), current_turn=current_turn)
        relevance = None
        if relevance_raw is not None:
            relevance = round(float(relevance_raw), 6)
        elif ctx_weight is not None:
            relevance = ctx_weight

        claim_entry = _build_claim_entry(
            object_row=object_row,
            audience=audience,
            similarity=similarity,
            ctx_weight=ctx_weight,
            relevance=relevance,
        )
        if claim_entry is None:
            continue
        relevant_scored.append(claim_entry)
        seen_scored_ids.add(object_row.object_id)
    return relevant_scored


def _list_zone_recent_claims(
    db: Session,
    session_id: uuid.UUID,
    zone_id: uuid.UUID | None,
    *,
    current_turn: int | None = None,
    limit: int = ZONE_RECENT_CLAIMS_LIMIT,
) -> list[dict[str, Any]]:
    if zone_id is None:
        return []

    resolved_limit = max(limit, 1)
    claim_rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "claim",
            models.ObjectModel.data["location_id"].astext == str(zone_id),
        )
        .order_by(
            _object_ctx_weight_expr(current_turn=current_turn).desc().nulls_last(),
            models.ObjectModel.created_at.desc(),
        )
        .limit(resolved_limit)
    ).scalars().all()

    seen_claim_ids: set[uuid.UUID] = {row.object_id for row in claim_rows}
    legacy_rows: list[models.ObjectModel] = []
    if len(claim_rows) < resolved_limit:
        legacy_rows = list(
            db.execute(
            select(models.ObjectModel)
            .where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == "claim",
                or_(
                    models.ObjectModel.data["location_id"].astext.is_(None),
                    models.ObjectModel.data["location_id"].astext == "",
                    models.ObjectModel.data["location_id"].astext == "null",
                ),
            )
            .order_by(
                _object_ctx_weight_expr(current_turn=current_turn).desc().nulls_last(),
                models.ObjectModel.created_at.desc(),
            )
            .limit(resolved_limit * 3)
            ).scalars().all()
        )

    combined_claim_rows = list(claim_rows)
    combined_claim_rows.extend(row for row in legacy_rows if row.object_id not in seen_claim_ids)
    if not combined_claim_rows:
        return []

    claim_ids = [row.object_id for row in combined_claim_rows]
    claim_meta = _build_claim_audience_map(db, session_id, claim_ids)
    zone_actor_ids = _list_active_zone_actor_ids(
        db,
        session_id,
        zone_id,
    )

    rows: list[dict[str, Any]] = []
    for claim_row in combined_claim_rows:
        claim_data = dict(claim_row.data or {})
        claim_text = _extract_claim_text(claim_data)
        if not claim_text:
            continue
        claim_id = claim_row.object_id
        raw_location_id = str(claim_data.get("location_id") or "").strip()
        has_explicit_zone_location = raw_location_id == str(zone_id)
        is_missing_location = not raw_location_id or raw_location_id.lower() == "null"
        meta = claim_meta.get(claim_id, {})
        if not has_explicit_zone_location:
            if not is_missing_location:
                continue
            speaker_id = str(meta.get("speaker_id") or "").strip()
            listener_ids = [str(raw).strip() for raw in list(meta.get("listener_ids") or []) if str(raw).strip()]
            has_zone_overlap = bool(
                (speaker_id and speaker_id in zone_actor_ids)
                or any(listener_id in zone_actor_ids for listener_id in listener_ids)
            )
            if not has_zone_overlap:
                continue

        ctx_weight = _extract_effective_ctx_weight(claim_data, current_turn=current_turn)
        row_payload: dict[str, Any] = {
            "claim_object_id": str(claim_id),
            "text": _truncate_text(claim_text, 220),
            "confidence": claim_data.get("confidence"),
            "about_object_id": claim_data.get("about_object_id"),
            "location_id": raw_location_id if has_explicit_zone_location else str(zone_id),
            "ctx_weight": ctx_weight,
            "speaker_id": meta.get("speaker_id"),
            "speaker_name": meta.get("speaker_name"),
            "listener_ids": list(meta.get("listener_ids") or []),
            "listener_names": list(meta.get("listener_names") or []),
        }
        if not has_explicit_zone_location:
            row_payload["location_inferred"] = True
        rows.append(row_payload)
        if len(rows) >= resolved_limit:
            break
    return rows


def _merge_npc_knowledge_subjects(
    zone_npcs: list[dict[str, Any]],
    relevant_npcs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for npc in zone_npcs:
        if not isinstance(npc, dict):
            continue
        npc_id = str(npc.get("object_id") or "").strip()
        if not npc_id or npc_id in seen_ids:
            continue
        merged.append(
            {
                "object_id": npc_id,
                "name": str(npc.get("name") or "").strip(),
                "in_current_zone": True,
            }
        )
        seen_ids.add(npc_id)

    for npc in relevant_npcs:
        if not isinstance(npc, dict):
            continue
        npc_id = str(npc.get("object_id") or "").strip()
        if not npc_id or npc_id in seen_ids:
            continue
        merged.append(
            {
                "object_id": npc_id,
                "name": str(npc.get("name") or "").strip(),
                "in_current_zone": False,
            }
        )
        seen_ids.add(npc_id)

    return merged


def _list_active_zone_actor_ids(
    db: Session,
    session_id: uuid.UUID,
    zone_id: uuid.UUID,
) -> set[str]:
    rows = db.execute(
        select(models.LinkModel.from_object_id)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.type == LOCATED_IN_LINK_TYPE,
            models.LinkModel.to_object_id == zone_id,
            models.LinkModel.valid_to_turn.is_(None),
        )
    ).scalars().all()
    actor_ids: set[str] = set()
    for actor_id in rows:
        if actor_id is None:
            continue
        actor_ids.add(str(actor_id))
    return actor_ids


def _list_active_npc_claim_links_for_knowledge(
    db: Session,
    session_id: uuid.UUID,
    *,
    npc_ids: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not npc_ids:
        return {}

    parsed_npc_ids: list[uuid.UUID] = []
    for npc_id in npc_ids:
        try:
            parsed_npc_ids.append(uuid.UUID(str(npc_id)))
        except (TypeError, ValueError, AttributeError):
            continue
    if not parsed_npc_ids:
        return {}

    rows = db.execute(
        select(
            models.LinkModel.from_object_id,
            models.LinkModel.to_object_id,
            models.LinkModel.type,
            models.LinkModel.valid_from_turn,
        )
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.LinkModel.session_id,
                models.ObjectModel.object_id == models.LinkModel.to_object_id,
            ),
        )
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id.in_(parsed_npc_ids),
            models.LinkModel.valid_to_turn.is_(None),
            models.LinkModel.type.in_(("heard", "asserted")),
            models.ObjectModel.type == "claim",
        )
        .order_by(models.LinkModel.valid_from_turn.desc(), models.LinkModel.created_at.desc())
    ).all()

    links_by_npc: dict[str, dict[str, dict[str, Any]]] = {}
    for npc_object_id, claim_object_id, link_type, valid_from_turn in rows:
        if npc_object_id is None or claim_object_id is None:
            continue
        npc_id = str(npc_object_id)
        claim_id = str(claim_object_id)
        npc_links = links_by_npc.setdefault(npc_id, {})
        link_entry = npc_links.setdefault(
            claim_id,
            {
                "types": set(),
                "latest_turn": -1,
            },
        )

        link_type_text = str(link_type or "").strip()
        if link_type_text in {"heard", "asserted"}:
            link_entry["types"].add(link_type_text)

        if isinstance(valid_from_turn, int) and valid_from_turn > int(link_entry.get("latest_turn", -1)):
            link_entry["latest_turn"] = valid_from_turn

    return links_by_npc


def _build_zone_npc_knowledge(
    db: Session,
    session_id: uuid.UUID,
    npc_subjects: list[dict[str, Any]],
    *,
    current_turn: int | None = None,
) -> list[dict[str, Any]]:
    npc_ids: list[str] = []
    seen_npc_ids: set[str] = set()
    for npc in npc_subjects:
        if not isinstance(npc, dict):
            continue
        npc_id = str(npc.get("object_id") or "").strip()
        if not npc_id or npc_id in seen_npc_ids:
            continue
        seen_npc_ids.add(npc_id)
        npc_ids.append(npc_id)

    active_links_by_npc = _list_active_npc_claim_links_for_knowledge(
        db,
        session_id,
        npc_ids=npc_ids,
    )
    all_claim_ids: set[str] = set()
    for npc_links in active_links_by_npc.values():
        all_claim_ids.update(npc_links.keys())

    claim_meta_by_id: dict[str, dict[str, Any]] = {}
    if all_claim_ids:
        parsed_claim_ids: list[uuid.UUID] = []
        for claim_id in sorted(all_claim_ids):
            try:
                parsed_claim_ids.append(uuid.UUID(claim_id))
            except (TypeError, ValueError, AttributeError):
                continue
        if parsed_claim_ids:
            claim_rows = db.execute(
                select(models.ObjectModel.object_id, models.ObjectModel.data)
                .where(
                    models.ObjectModel.session_id == session_id,
                    models.ObjectModel.type == "claim",
                    models.ObjectModel.object_id.in_(parsed_claim_ids),
                )
            ).all()
            for claim_object_id, claim_data_raw in claim_rows:
                claim_data = dict(claim_data_raw or {})
                claim_text = _truncate_text(_extract_claim_text(claim_data), 120)
                if claim_text:
                    ctx_weight = _extract_effective_ctx_weight(
                        claim_data,
                        current_turn=current_turn,
                    )
                    if not isinstance(ctx_weight, (int, float)):
                        ctx_weight = 0.0
                    claim_meta_by_id[str(claim_object_id)] = {
                        "text": claim_text,
                        "ctx_weight": float(ctx_weight),
                    }

    payload: list[dict[str, Any]] = []
    for npc in npc_subjects:
        if not isinstance(npc, dict):
            continue
        npc_id = str(npc.get("object_id") or "").strip()
        if not npc_id:
            continue
        npc_name = str(npc.get("name") or "").strip()
        npc_links = active_links_by_npc.get(npc_id, {})

        def _rank_claim_id(claim_id: str) -> tuple[int, float, str]:
            link_entry = npc_links.get(claim_id, {})
            latest_turn = link_entry.get("latest_turn")
            if not isinstance(latest_turn, int):
                latest_turn = -1
            claim_meta = claim_meta_by_id.get(claim_id, {})
            claim_ctx_weight = claim_meta.get("ctx_weight")
            if not isinstance(claim_ctx_weight, (int, float)):
                claim_ctx_weight = 0.0
            return (-latest_turn, -float(claim_ctx_weight), claim_id)

        filtered_known_claim_ids: list[str] = []
        filtered_asserted_claim_ids: list[str] = []
        for claim_id, link_entry in npc_links.items():
            if claim_id not in claim_meta_by_id:
                continue
            if not isinstance(link_entry, dict):
                continue
            link_types = link_entry.get("types")
            if not isinstance(link_types, set):
                continue
            if {"heard", "asserted"} & link_types:
                filtered_known_claim_ids.append(claim_id)
            if "asserted" in link_types:
                filtered_asserted_claim_ids.append(claim_id)

        filtered_known_claim_ids.sort(key=_rank_claim_id)
        filtered_asserted_claim_ids.sort(key=_rank_claim_id)

        known_claim_texts = [
            str(claim_meta_by_id[claim_id]["text"])
            for claim_id in filtered_known_claim_ids[:6]
            if claim_id in claim_meta_by_id
        ]
        asserted_claim_texts = [
            str(claim_meta_by_id[claim_id]["text"])
            for claim_id in filtered_asserted_claim_ids[:6]
            if claim_id in claim_meta_by_id
        ]
        known_preview = (
            ", ".join([f'claim: "{text}"' for text in known_claim_texts[:3]])
            if known_claim_texts
            else "(none verified)"
        )
        asserted_preview = (
            ", ".join([f'claim: "{text}"' for text in asserted_claim_texts[:2]])
            if asserted_claim_texts
            else "(none verified)"
        )
        display_name = npc_name or npc_id
        payload.append(
            {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "known_claim_texts": known_claim_texts,
                "asserted_claim_texts": asserted_claim_texts,
                "in_current_zone": bool(npc.get("in_current_zone", False)),
                "verified_summary": _truncate_text(
                    f"{display_name} knows: {known_preview}; asserted: {asserted_preview}.",
                    320,
                ),
            }
        )
    return payload


def _build_reaction_hints(
    *,
    user_input: str,
    zone_npcs: list[dict[str, Any]],
    zone_claims: list[dict[str, Any]],
    intent_tags: list[str] | None = None,
    max_hints: int = MAX_REACTION_HINTS,
) -> list[str]:
    if max_hints <= 0:
        return []

    hints: list[str] = []
    seen: set[str] = set()

    def _push_hint(raw_hint: str) -> None:
        if len(hints) >= max_hints:
            return
        hint = " ".join(str(raw_hint or "").split()).strip()
        if not hint or hint in seen:
            return
        seen.add(hint)
        hints.append(hint)

    user_input_norm = str(user_input or "").lower()
    is_knowledge_challenge = any(pattern in user_input_norm for pattern in KNOWLEDGE_CHALLENGE_PATTERNS)
    if is_knowledge_challenge:
        _push_hint(KNOWLEDGE_CHALLENGE_HINT)
    if not zone_npcs and not zone_claims:
        return hints[:max_hints]

    zone_npc_names_by_id: dict[str, str] = {}
    for npc in zone_npcs:
        npc_id = str(npc.get("object_id") or "").strip()
        npc_name = str(npc.get("name") or "").strip()
        if npc_id and npc_name:
            zone_npc_names_by_id[npc_id] = npc_name

    for npc in zone_npcs:
        npc_name = str(npc.get("name") or "").strip()
        if not npc_name:
            continue
        relationships = npc.get("relationships")
        if not isinstance(relationships, list):
            continue
        for relationship in relationships:
            if not isinstance(relationship, dict):
                continue
            relation_type = str(relationship.get("type") or "").strip()
            target_id = str(relationship.get("target_id") or "").strip()
            target_name = str(relationship.get("target_name") or "").strip()
            if not target_name and target_id:
                target_name = zone_npc_names_by_id.get(target_id, "")
            if not relation_type or not target_name:
                continue
            if relation_type in REACTION_SUPPORT_LINK_TYPES:
                _push_hint(
                    f"If {target_name} is harmed, accused, or threatened, "
                    f"{npc_name} should react consistently with their {relation_type} relationship."
                )
            elif relation_type in REACTION_CONFLICT_LINK_TYPES:
                _push_hint(
                    f"{npc_name} has {relation_type} tension with {target_name}; "
                    "reflect this in dialogue and choices."
                )
            if len(hints) >= max_hints:
                return hints

    for claim in zone_claims:
        if len(hints) >= max_hints:
            break
        if not isinstance(claim, dict):
            continue
        claim_text = _truncate_text(str(claim.get("text") or "").strip(), 140)
        if not claim_text:
            continue
        listener_names = claim.get("listener_names")
        if isinstance(listener_names, list):
            non_empty_listener_names = [str(name).strip() for name in listener_names if str(name or "").strip()]
        else:
            non_empty_listener_names = []
        if non_empty_listener_names:
            listeners_preview = ", ".join(non_empty_listener_names[:3])
            _push_hint(f"Recent local claim heard by {listeners_preview}: {claim_text}")
            continue
        speaker_name = str(claim.get("speaker_name") or "").strip()
        if speaker_name:
            _push_hint(f"Recent local claim from {speaker_name}: {claim_text}")

    if _is_aggressive_intent(intent_tags):
        witnesses = [str(npc.get("name") or "").strip() for npc in zone_npcs]
        witness_preview = ", ".join([name for name in witnesses if name][:3])
        if witness_preview:
            _push_hint(
                f"Potentially violent intent detected; nearby witnesses ({witness_preview}) should react visibly."
            )

    has_substance = len(hints) > 1 or (zone_claims and len(zone_claims) > 0)
    if USE_REACTION_ENRICHER and has_substance and zone_npcs:
        enriched_hints = _enrich_reaction_hints(
            user_input=user_input,
            zone_npcs=zone_npcs,
            zone_claims=zone_claims,
            base_hints=hints,
            max_hints=max_hints,
        )
        if enriched_hints:
            if is_knowledge_challenge and KNOWLEDGE_CHALLENGE_HINT not in enriched_hints:
                return [KNOWLEDGE_CHALLENGE_HINT, *enriched_hints][:max_hints]
            return enriched_hints[:max_hints]

    return hints[:max_hints]


def _enrich_reaction_hints(
    *,
    user_input: str,
    zone_npcs: list[dict[str, Any]],
    zone_claims: list[dict[str, Any]],
    base_hints: list[str],
    max_hints: int,
) -> list[str]:
    """LLM-based reaction hint enrichment with safe fallback to base hints."""
    if max_hints <= 0:
        return []

    npc_payload: list[dict[str, Any]] = []
    for npc in zone_npcs[:8]:
        if not isinstance(npc, dict):
            continue
        relationships_raw = npc.get("relationships")
        relationship_items: list[dict[str, str]] = []
        if isinstance(relationships_raw, list):
            for relationship in relationships_raw[:4]:
                if not isinstance(relationship, dict):
                    continue
                relationship_items.append(
                    {
                        "type": _truncate_text(str(relationship.get("type") or "").strip(), 40),
                        "target_name": _truncate_text(
                            str(relationship.get("target_name") or relationship.get("target_id") or "").strip(),
                            80,
                        ),
                    }
                )
        npc_payload.append(
            {
                "name": _truncate_text(str(npc.get("name") or "").strip(), 80),
                "attitude": _truncate_text(str(npc.get("attitude") or "").strip(), 40),
                "short_desc": _truncate_text(str(npc.get("short_desc") or "").strip(), 140),
                "relationships": relationship_items,
            }
        )

    claims_payload: list[dict[str, Any]] = []
    for claim in zone_claims[:5]:
        if not isinstance(claim, dict):
            continue
        claims_payload.append(
            {
                "text": _truncate_text(str(claim.get("text") or "").strip(), 140),
                "speaker_name": _truncate_text(str(claim.get("speaker_name") or "").strip(), 80),
                "listener_names": [
                    _truncate_text(str(name).strip(), 60)
                    for name in (claim.get("listener_names") or [])[:3]
                    if str(name or "").strip()
                ],
            }
        )

    payload = {
        "user_input": _truncate_text(str(user_input or ""), 220),
        "base_hints": [_truncate_text(str(hint or "").strip(), 180) for hint in base_hints[:max_hints]],
        "zone_npcs": npc_payload,
        "zone_claims": claims_payload,
        "max_hints": max_hints,
    }

    try:
        with telemetry_context(request_type="reaction_enricher"):
            result = openrouter_chat.generate_json(
                model=OPENROUTER_CHAT_MODEL,
                system_prompt=_REACTION_ENRICHER_SYSTEM,
                user_prompt=_normalize_json_preview(payload, 3000),
                max_tokens=220,
            )
    except Exception:  # noqa: BLE001
        logger.warning("Reaction enricher failed, using base hints", exc_info=True)
        return base_hints[:max_hints]

    raw_hints = result.get("reaction_hints")
    if not isinstance(raw_hints, list):
        logger.warning("Reaction enricher returned invalid payload without reaction_hints list, using base hints")
        return base_hints[:max_hints]

    enriched: list[str] = []
    seen: set[str] = set()
    for raw in raw_hints:
        hint = _truncate_text(" ".join(str(raw or "").split()).strip(), 180)
        if not hint or hint in seen:
            continue
        seen.add(hint)
        enriched.append(hint)
        if len(enriched) >= max_hints:
            break

    if not enriched:
        logger.warning("Reaction enricher returned empty usable hints, using base hints")
        return base_hints[:max_hints]
    return enriched


def _is_aggressive_intent(intent_tags: list[str] | None) -> bool:
    if not isinstance(intent_tags, list):
        return False
    normalized = _normalize_intent_tags(intent_tags)
    return bool({"aggressive", "threat"} & set(normalized))
def _resolve_exact_name_object_ids_for_context(
    db: Session,
    session_id: uuid.UUID,
    exact_names: list[str] | None,
    *,
    limit: int = 24,
) -> list[str]:
    normalized_names = _normalize_exact_names(exact_names)
    if not normalized_names:
        return []

    rows = db.execute(
        select(models.ObjectModel.object_id)
        .where(
            models.ObjectModel.session_id == session_id,
            func.lower(models.ObjectModel.name).in_(tuple(normalized_names)),
            or_(
                models.ObjectModel.data["status"].astext.is_(None),
                models.ObjectModel.data["status"].astext != "inactive",
            ),
        )
        .order_by(models.ObjectModel.created_at.desc())
        .limit(max(limit, 1))
    ).scalars().all()

    resolved: list[str] = []
    seen: set[str] = set()
    for object_id in rows:
        if object_id is None:
            continue
        key = str(object_id)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(key)
    return resolved


def _list_one_hop_link_candidates_for_context(
    db: Session,
    session_id: uuid.UUID,
    *,
    anchor_object_ids: list[str],
    query_embedding: list[float] | None,
    current_turn: int | None = None,
    limit: int = 24,
) -> list[dict[str, Any]]:
    parsed_anchor_ids: list[uuid.UUID] = []
    for raw_object_id in anchor_object_ids:
        try:
            parsed_anchor_ids.append(uuid.UUID(str(raw_object_id)))
        except (TypeError, ValueError, AttributeError):
            continue
    if not parsed_anchor_ids:
        return []

    from_object = aliased(models.ObjectModel)
    to_object = aliased(models.ObjectModel)
    distance_expr = None
    if query_embedding is not None:
        distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding).label("distance")

    base_query = (
        select(
            models.LinkModel,
            from_object.object_id,
            from_object.name,
            from_object.data,
            to_object.object_id,
            to_object.name,
            to_object.data,
        )
        .join(
            from_object,
            and_(
                from_object.session_id == models.LinkModel.session_id,
                from_object.object_id == models.LinkModel.from_object_id,
            ),
        )
        .join(
            to_object,
            and_(
                to_object.session_id == models.LinkModel.session_id,
                to_object.object_id == models.LinkModel.to_object_id,
            ),
        )
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.valid_to_turn.is_(None),
            or_(
                models.LinkModel.from_object_id.in_(parsed_anchor_ids),
                models.LinkModel.to_object_id.in_(parsed_anchor_ids),
            ),
        )
        .order_by(models.LinkModel.created_at.desc())
        .limit(max(limit * 4, limit))
    )
    if distance_expr is not None:
        base_query = base_query.add_columns(distance_expr).outerjoin(
            models.ObjectEmbeddingModel,
            and_(
                models.ObjectEmbeddingModel.session_id == models.LinkModel.session_id,
                models.ObjectEmbeddingModel.object_id == models.LinkModel.from_object_id,
                models.ObjectEmbeddingModel.namespace == LINK_CONTEXT_NAMESPACE,
            ),
        )

    rows = db.execute(base_query).all()
    anchor_set = {str(object_id) for object_id in parsed_anchor_ids}
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for row in rows:
        if distance_expr is None:
            (
                link_row,
                from_object_id,
                from_name,
                from_data,
                to_object_id,
                to_name,
                to_data,
            ) = row
            distance = None
        else:
            (
                link_row,
                from_object_id,
                from_name,
                from_data,
                to_object_id,
                to_name,
                to_data,
                distance,
            ) = row

        if from_object_id is None or to_object_id is None:
            continue
        dedupe_key = (str(from_object_id), str(to_object_id), str(link_row.type or ""))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        link_data = dict(link_row.data or {})
        raw_text = _extract_link_context_text(link_data)
        preview = ""
        if raw_text:
            preview = _truncate_text(
                f"{str(from_name or '').strip()} {str(link_row.type or '').strip()} {str(to_name or '').strip()}: {raw_text}",
                320,
            )
        similarity = 0.0
        if distance is not None:
            similarity = round(max(1.0 - float(distance), 0.0), 6)
        from_id_text = str(from_object_id)
        to_id_text = str(to_object_id)
        candidates.append(
            {
                "from_object_id": from_id_text,
                "from_name": str(from_name or ""),
                "to_object_id": to_id_text,
                "to_name": str(to_name or ""),
                "type": str(link_row.type or ""),
                "preview": preview,
                "similarity": similarity,
                "anchor_match": from_id_text in anchor_set or to_id_text in anchor_set,
                "ctx_weight": _extract_effective_ctx_weight(
                    dict(from_data or {}),
                    current_turn=current_turn,
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item.get("anchor_match") or 0),
            float(item.get("similarity") or 0.0),
            float(item.get("ctx_weight") or 0.0),
        ),
        reverse=True,
    )
    return candidates[: max(limit, 1)]


def _extract_turn_applied_ops_for_context(ai_json: Any) -> list[dict[str, Any]]:
    if not isinstance(ai_json, dict):
        return []
    raw_ops = ai_json.get("applied_ops")
    if not isinstance(raw_ops, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw_op in raw_ops[:12]:
        if not isinstance(raw_op, dict):
            continue
        payload: dict[str, Any] = {"op": str(raw_op.get("op") or "").strip()}
        for key in ("object", "object_id", "from", "to", "scope", "type"):
            value = str(raw_op.get(key) or "").strip()
            if value:
                payload[key] = value
        normalized.append(payload)
    return normalized


def _build_ranked_memory_context_row(
    memory_payload: dict[str, Any],
    *,
    reference_turn: int | None,
    normalized_anchor_ids: set[str],
    weight_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return policy_build_ranked_memory_context_row(
        memory_payload,
        reference_turn=reference_turn,
        normalized_anchor_ids=normalized_anchor_ids,
        weight_config=weight_config,
    )


def _memory_row_merge_tuple(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return policy_memory_row_merge_tuple(row)


def _derive_session_memory_profile_for_context(
    db: Session,
    *,
    session_id: uuid.UUID,
) -> str:
    session_row = db.get(models.SessionModel, session_id)
    if session_row is None or not isinstance(session_row, models.SessionModel):
        return normalize_session_memory_profile(None)
    session_state = dict(getattr(session_row, "state_json", {}) or {})
    override = session_state.get("memory_profile_override")
    recent_turn_rows = db.execute(
        select(models.TurnModel.turn_index, models.TurnModel.ai_json)
        .where(models.TurnModel.session_id == session_id)
        .order_by(models.TurnModel.turn_index.desc())
        .limit(SESSION_MEMORY_PROFILE_WINDOW_TURNS)
    ).all()
    turn_rows: list[dict[str, Any]] = []
    for row in recent_turn_rows:
        if not isinstance(row, (tuple, list)) or len(row) < 2:
            continue
        turn_index, ai_json = row[0], row[1]
        ai_payload = dict(ai_json or {})
        memory_debug = dict(ai_payload.get("memory_debug") or {})
        surfaced_relevant_rows = [
            dict(item)
            for item in list(memory_debug.get("surfaced_relevant_rows") or [])
            if isinstance(item, dict)
        ]
        turn_rows.append(
            {
                "turn_index": int(turn_index),
                "turn_intent": memory_debug.get("turn_intent") or ai_payload.get("turn_intent"),
                "scene_mode": memory_debug.get("scene_mode") or ai_payload.get("scene_mode"),
                "callback_count": len(list(memory_debug.get("surfaced_callback_rows") or [])),
                "bundle_count": len(list(memory_debug.get("surfaced_bundle_rows") or [])),
                "relevant_count": len(surfaced_relevant_rows),
                "fact_surface_count": sum(
                    1
                    for item in surfaced_relevant_rows
                    if str(item.get("layer") or "").strip().lower() != "event"
                    and str(item.get("memory_class") or "").strip().lower() != "episodic"
                ),
                "event_surface_count": sum(
                    1
                    for item in surfaced_relevant_rows
                    if str(item.get("layer") or "").strip().lower() == "event"
                    or str(item.get("memory_class") or "").strip().lower() == "episodic"
                ),
                "bundle_surface_count": len(list(memory_debug.get("surfaced_bundle_rows") or [])),
                "obligation_count": max(
                    int(memory_debug.get("obligation_count") or len(list(memory_debug.get("story_obligation_keys") or [])) or 0),
                    0,
                ),
            }
        )
    return derive_session_memory_profile(turn_rows, override=override)


def _latest_memory_review_report(
    db: Session,
    *,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    row = db.execute(
        select(models.ObjectModel.data)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == MEMORY_REVIEW_OBJECT_TYPE,
        )
        .order_by(models.ObjectModel.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return dict(row or {})


def _latest_completed_turn_index_for_context(
    db: Session,
    *,
    session_id: uuid.UUID,
) -> int:
    latest_turn = db.execute(
        select(func.max(models.TurnModel.turn_index)).where(
            models.TurnModel.session_id == session_id,
        )
    ).scalar_one_or_none()
    return max(int(latest_turn or 0), 0)


def _review_freshness_for_context(
    review_report: dict[str, Any],
    *,
    latest_completed_turn: int,
) -> dict[str, Any]:
    trace_rows = [
        dict(item)
        for item in list(dict(review_report.get("trace_corpus") or {}).get("rows") or [])
        if isinstance(item, dict)
    ]
    reviewed_through_turn = max(
        int(review_report.get("reviewed_through_turn") or 0),
        max((int(row.get("turn_index") or 0) for row in trace_rows), default=0),
    )
    has_review = bool(review_report)
    lag_turns = max(int(latest_completed_turn) - int(reviewed_through_turn), 0)
    if not has_review:
        mode = "missing_review_fallback"
    elif lag_turns > 0:
        mode = "stale_review_fallback"
    else:
        mode = "fresh_review"
    return {
        "mode": mode,
        "reviewed_through_turn": reviewed_through_turn,
        "latest_completed_turn": int(latest_completed_turn),
        "lag_turns": lag_turns,
        "used_persisted_review": mode == "fresh_review",
        "used_runtime_fallback": mode != "fresh_review",
        "fallback_fields": [],
    }


def _list_relevant_memories_for_input(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    user_input: str | None = None,
    exact_names: list[str] | None = None,
    anchor_object_ids: list[str] | None = None,
    current_turn: int | None = None,
    scene_mode: str | None = None,
    turn_intent: str | None = None,
    limit: int = 5,
    return_meta: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    row_limit = max(limit, 1)
    review_report = _latest_memory_review_report(db, session_id=session_id) if return_meta else {}
    latest_completed_turn = (
        _latest_completed_turn_index_for_context(db, session_id=session_id)
        if return_meta
        else max(int(current_turn or 0) - 1, 0)
    )
    review_freshness = _review_freshness_for_context(
        review_report,
        latest_completed_turn=latest_completed_turn,
    ) if return_meta else {}
    review_is_fresh = (not return_meta) or str(review_freshness.get("mode") or "") == "fresh_review"
    review_obligation_rows = [
        dict(item)
        for item in list(review_report.get("story_obligations") or [])
        if isinstance(item, dict)
    ]
    review_narrative_chain_rows = [
        dict(item)
        for item in list(review_report.get("narrative_chains") or [])
        if isinstance(item, dict)
    ]
    narrative_chain_rows = list(review_narrative_chain_rows if review_is_fresh else [])
    narrative_chain_lookup = narrative_chain_index(narrative_chain_rows)
    actor_view_feedback_by_key = {
        str(item.get("view_key") or "").strip(): dict(item)
        for item in list(review_report.get("actor_memory_views") or [])
        if isinstance(item, dict) and str(item.get("view_key") or "").strip()
    }
    session_row = db.get(models.SessionModel, session_id)
    session_state = (
        dict(getattr(session_row, "state_json", {}) or {})
        if isinstance(session_row, models.SessionModel)
        else {}
    )
    player_object_id = str(session_state.get("player_object_id") or "").strip() or None
    turn_rows_for_mode = list((dict(review_report.get("trace_corpus") or {}).get("rows") or []))
    session_profile_seed = str(review_report.get("session_memory_profile") or "").strip()
    if not session_profile_seed and return_meta:
        session_profile_seed = _derive_session_memory_profile_for_context(
            db,
            session_id=session_id,
        )
    policy_state = derive_memory_policy_state(
        turn_rows=turn_rows_for_mode,
        session_state=session_state,
        session_memory_profile=session_profile_seed or None,
        session_narrative_mode=str(review_report.get("session_narrative_mode") or "").strip() or None,
        row_limit=row_limit,
        scene_mode=scene_mode,
        turn_intent=turn_intent,
        current_memory_health=review_report.get("session_memory_health_score"),
        obligation_rows=review_obligation_rows,
        tuning_report=dict(review_report.get("tuning_report") or {}),
    )
    session_profile = policy_state.session_memory_profile
    session_narrative_mode = policy_state.session_narrative_mode
    session_narrative_mode_override = policy_state.session_narrative_mode_override or ""
    session_narrative_mode_source = policy_state.session_narrative_mode_source
    lane_budgets = dict(policy_state.lane_budgets)
    saturation_cap_values = dict(policy_state.saturation_limits)
    tuning_weights = tuning_weight_config_payload(policy_state.tuning_state)
    tuning_policy_version = policy_state.tuning_state.policy_version
    policy_state_data = memory_policy_state_payload(policy_state)
    operational_alert_rows = [
        dict(item)
        for item in list(review_report.get("operational_alerts") or [])
        if isinstance(item, dict)
    ]
    operational_guardrails = (
        derive_operational_alert_guardrails(
            alerts=operational_alert_rows,
            lane_budgets=lane_budgets,
            saturation_cap_values=saturation_cap_values,
        )
        if review_is_fresh
        else {
            "source": str(review_freshness.get("mode") or "stale_review_fallback"),
            "applied_alert_keys": [],
            "stale_review_alert_keys": [
                str(item.get("alert_key") or "").strip()
                for item in operational_alert_rows
                if str(item.get("alert_key") or "").strip()
            ],
            "adjusted_lane_budgets": dict(lane_budgets),
            "adjusted_saturation_limits": dict(saturation_cap_values),
        }
    )
    lane_budgets = dict(operational_guardrails.get("adjusted_lane_budgets") or lane_budgets)
    saturation_cap_values = dict(
        operational_guardrails.get("adjusted_saturation_limits") or saturation_cap_values
    )
    obligation_quota = int(lane_budgets.get("obligations") or 0)
    fact_quota = int(lane_budgets.get("durable_facts") or 0)
    event_quota = int(lane_budgets.get("episodic_events") or 0)
    bundle_quota = int(lane_budgets.get("narrative_bundles") or 0)
    actor_view_quota = int(lane_budgets.get("actor_views") or 0)
    normalized_anchor_ids = set(str(value).strip() for value in (anchor_object_ids or []) if str(value).strip())
    raw_payloads_by_object_id: dict[str, dict[str, Any]] = {}

    fact_priority_order = (
        _memory_priority_rank_expr().desc(),
        func.coalesce(_json_text_float_expr(models.ObjectModel.data["confidence"].astext), literal(0.0)).desc(),
        func.coalesce(_json_text_int_expr(models.ObjectModel.data["independent_evidence_count"].astext), literal(0)).desc(),
        func.coalesce(_json_text_int_expr(models.ObjectModel.data["last_confirmed_turn"].astext), literal(0)).desc(),
        models.ObjectModel.created_at.desc(),
    )
    event_priority_order = (
        _memory_priority_rank_expr().desc(),
        func.coalesce(_json_text_float_expr(models.ObjectModel.data["durability"].astext), literal(0.0)).desc(),
        func.coalesce(_json_text_int_expr(models.ObjectModel.data["last_seen_turn"].astext), literal(0)).desc(),
        models.ObjectModel.created_at.desc(),
    )
    narrator_visible_scope_filter = or_(
        models.ObjectModel.data["knowledge_scope"].astext.is_(None),
        models.ObjectModel.data["knowledge_scope"].astext == "global",
        models.ObjectModel.data["knowledge_scope"].astext == "public",
    )
    active_memory_filter = or_(
        models.ObjectModel.data["status"].astext.is_(None),
        models.ObjectModel.data["status"].astext == "active",
    )

    recent_cutoff = max(current_turn - 6, 0) if isinstance(current_turn, int) else None
    event_eligibility_filter = (
        or_(
            func.coalesce(_json_text_float_expr(models.ObjectModel.data["durability"].astext), literal(0.0)) >= literal(0.65),
            func.coalesce(_json_text_int_expr(models.ObjectModel.data["source_turn"].astext), literal(0))
            >= literal(recent_cutoff),
            func.coalesce(_json_text_int_expr(models.ObjectModel.data["last_seen_turn"].astext), literal(0))
            >= literal(recent_cutoff),
        )
        if recent_cutoff is not None
        else func.coalesce(_json_text_float_expr(models.ObjectModel.data["durability"].astext), literal(0.0))
        >= literal(0.65)
    )
    obligation_priority_order = (
        case(
            (func.lower(models.ObjectModel.data["lane_priority"].astext) == "critical", 3),
            (func.lower(models.ObjectModel.data["lane_priority"].astext) == "high", 2),
            (func.lower(models.ObjectModel.data["lane_priority"].astext) == "med", 1),
            (func.lower(models.ObjectModel.data["lane_priority"].astext) == "low", 0),
            else_=1,
        ).desc(),
        func.coalesce(_json_text_float_expr(models.ObjectModel.data["obligation_pressure_score"].astext), literal(0.0)).desc(),
        func.coalesce(_json_text_float_expr(models.ObjectModel.data["expectation_debt_score"].astext), literal(0.0)).desc(),
        func.coalesce(_json_text_int_expr(models.ObjectModel.data["updated_turn"].astext), literal(0)).desc(),
        models.ObjectModel.created_at.desc(),
    )

    def _collect_ranked_rows(
        *,
        object_type: str,
        namespace: str | None,
        salience_ordering: tuple[Any, ...],
        row_cap: int,
        eligibility_filter: Any | None = None,
        default_layer: str,
        default_memory_class: str | None = None,
        default_lane: str | None = None,
        use_embeddings: bool = True,
    ) -> list[dict[str, Any]]:
        candidate_buckets: dict[str, dict[str, Any]] = {}

        def _merge_candidate_bucket(
            object_row: models.ObjectModel,
            *,
            distance: float | None,
            anchor_hits: set[str] | None = None,
        ) -> None:
            object_id = str(object_row.object_id)
            bucket = candidate_buckets.get(object_id)
            if bucket is None:
                candidate_buckets[object_id] = {
                    "object_row": object_row,
                    "distance": distance,
                    "anchor_hits": set(anchor_hits or ()),
                }
                return
            existing_distance = bucket.get("distance")
            if distance is not None and (existing_distance is None or float(distance) < float(existing_distance)):
                bucket["distance"] = distance
            if anchor_hits:
                bucket_anchor_hits = bucket.setdefault("anchor_hits", set())
                if isinstance(bucket_anchor_hits, set):
                    bucket_anchor_hits.update(anchor_hits)

        base_filters = [
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == object_type,
            narrator_visible_scope_filter,
            active_memory_filter,
            _memory_committed_source_ops_filter(),
        ]
        if eligibility_filter is not None:
            base_filters.append(eligibility_filter)

        if use_embeddings and namespace and query_embedding is not None:
            distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(query_embedding)
            rows = db.execute(
                select(models.ObjectModel, distance_expr.label("distance"))
                .join(
                    models.ObjectEmbeddingModel,
                    and_(
                        models.ObjectEmbeddingModel.session_id == models.ObjectModel.session_id,
                        models.ObjectEmbeddingModel.object_id == models.ObjectModel.object_id,
                        models.ObjectEmbeddingModel.namespace == namespace,
                    ),
                )
                .where(*base_filters)
                .order_by(distance_expr.asc())
                .limit(max(row_limit * 8, row_limit, 1))
            ).all()
            for row in rows:
                if not isinstance(row, (tuple, list)) or len(row) < 2:
                    continue
                object_row = row[0]
                distance = row[-1]
                _merge_candidate_bucket(object_row, distance=float(distance) if distance is not None else None)

        salient_rows = db.execute(
            select(models.ObjectModel)
            .where(*base_filters)
            .order_by(*salience_ordering)
            .limit(max(row_cap * 12, 24))
        ).scalars().all()
        for object_row in salient_rows:
            _merge_candidate_bucket(object_row, distance=None)

        if normalized_anchor_ids:
            anchor_filters = [
                models.ObjectModel.data.contains({"anchor_object_ids": [anchor_id]})
                for anchor_id in sorted(normalized_anchor_ids)
            ]
            anchor_rows = db.execute(
                select(models.ObjectModel)
                .where(*base_filters, or_(*anchor_filters))
                .order_by(*salience_ordering)
                .limit(max(row_cap * 10, 20))
            ).scalars().all()
            for object_row in anchor_rows:
                object_data = dict(object_row.data or {})
                raw_anchor_values = object_data.get("anchor_object_ids")
                if not isinstance(raw_anchor_values, list):
                    continue
                anchor_hits = {
                    str(anchor_id).strip()
                    for anchor_id in raw_anchor_values
                    if str(anchor_id).strip() in normalized_anchor_ids
                }
                if anchor_hits:
                    _merge_candidate_bucket(object_row, distance=None, anchor_hits=anchor_hits)

        ranked_rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for bucket in candidate_buckets.values():
            object_row = bucket.get("object_row")
            if object_row is None:
                continue
            object_data = dict(object_row.data or {})
            normalized_object_layer = str(object_data.get("layer") or "").strip().lower()
            if default_layer == "obligation" and normalized_object_layer != "obligation" and not str(object_data.get("obligation_key") or "").strip():
                continue
            if default_layer == "bundle" and normalized_object_layer != "bundle" and not (
                str(object_data.get("bundle_key") or "").strip()
                or str(object_data.get("bundle_family") or "").strip()
                or list(object_data.get("bundle_relationships") or [])
                or list(object_data.get("fact_keys") or [])
                or str(object_data.get("core_fact_key") or "").strip()
            ):
                continue
            payload = {
                "row_object_id": str(object_row.object_id),
                "layer": object_data.get("layer") or default_layer,
                "lane": object_data.get("lane") or default_lane,
                "kind": object_data.get("kind"),
                "memory_class": default_memory_class
                or _derive_memory_class_for_context(str(object_data.get("kind") or default_layer), object_data),
                "search_recall_summary": object_data.get("search_recall_summary"),
                "narrative_recall_summary": object_data.get("narrative_recall_summary"),
                "state": object_data.get("state"),
                "knowledge_scope": object_data.get("knowledge_scope"),
                "priority": object_data.get("priority"),
                "anchor_object_ids": list(object_data.get("anchor_object_ids") or []),
                "actor_object_id": object_data.get("actor_object_id"),
                "counterparty_object_id": object_data.get("counterparty_object_id"),
                "fact_object_id": object_data.get("object_id"),
                "principal_object_id": (
                    object_data.get("principal_object_id")
                    or object_data.get("object_id")
                    or object_data.get("actor_object_id")
                    or object_data.get("quest_object_id")
                ),
                "location_object_id": object_data.get("location_object_id"),
                "quest_object_id": object_data.get("quest_object_id"),
                "source_turn": _safe_int(object_data.get("source_turn")),
                "last_seen_turn": _safe_int(object_data.get("last_seen_turn")),
                "last_confirmed_turn": _safe_int(object_data.get("last_confirmed_turn")),
                "last_reconfirmed_turn": _safe_int(object_data.get("last_reconfirmed_turn")),
                "source_ops_count": _safe_int(object_data.get("source_ops_count")),
                "bundle_key": object_data.get("bundle_key"),
                "related_bundle_keys": list(object_data.get("related_bundle_keys") or []),
                "current_relevance_reason": object_data.get("current_relevance_reason"),
                "similarity": round(1.0 - float(bucket.get("distance")), 6) if bucket.get("distance") is not None else None,
                "importance": _coerce_importance(object_data.get("importance")),
                "confidence": _coerce_importance(object_data.get("confidence")),
                "durability": _coerce_importance(object_data.get("durability")),
                "callback_strength": object_data.get("callback_strength"),
                "player_salience_score": _coerce_importance(object_data.get("player_salience_score")),
                "expectation_salience_score": _coerce_importance(object_data.get("expectation_salience_score")),
                "continuity_contract_strength_score": _coerce_importance(object_data.get("continuity_contract_strength_score")),
                "continuity_pressure_score": _coerce_importance(object_data.get("continuity_pressure_score")),
                "expectation_debt_score": _coerce_importance(object_data.get("expectation_debt_score")),
                "obligation_pressure_score": _coerce_importance(object_data.get("obligation_pressure_score")),
                "quality_feedback_score": _coerce_importance(object_data.get("quality_feedback_score")),
                "family_priority_boost": _coerce_importance(object_data.get("family_priority_boost")),
                "family_noise_score": _coerce_importance(object_data.get("family_noise_score")),
                "family_fragmentation_score": _coerce_importance(object_data.get("family_fragmentation_score")),
                "family_miss_pressure": _coerce_importance(object_data.get("family_miss_pressure")),
                "independent_evidence_count": _safe_int(object_data.get("independent_evidence_count")),
                "repetition_count": _safe_int(object_data.get("repetition_count")),
                "compression_mode": object_data.get("compression_mode"),
                "emotional_weight": _coerce_importance(object_data.get("emotional_weight")),
                "obligation_weight": _coerce_importance(object_data.get("obligation_weight")),
                "sentimental_weight": _coerce_importance(object_data.get("sentimental_weight")),
                "routine_weight": _coerce_importance(object_data.get("routine_weight")),
                "certainty": object_data.get("certainty"),
                "severity": object_data.get("severity"),
                "lane_priority": object_data.get("lane_priority"),
                "view_key": object_data.get("view_key"),
                "obligation_key": object_data.get("obligation_key"),
            }
            for passthrough_key in (
                "fact_key",
                "view_key",
                "obligation_key",
                "source_fact_keys",
                "source_event_ids",
                "source_bundle_keys",
                "bundle_relationships",
                "core_fact_key",
                "supporting_fact_keys",
                "recent_event_ids",
                "callback_candidate",
                "narrative_packet_role",
                "packet_pressure_score",
                "why_packet_became_core",
                "actor_view_scope",
                "deadline_turn",
                "resolution_confidence",
                "inactive_reason",
                "started_at_turn",
                "resolved_at_turn",
                "last_salient_turn",
                "dormant_since_turn",
                "persisted_dormancy_state",
                "dormancy_transition",
                "dormancy_transition_turn",
                "dormancy_reason_flags",
                "current_relevance_reason_context",
            ):
                if passthrough_key in object_data:
                    payload[passthrough_key] = object_data.get(passthrough_key)
            if narrative_chain_lookup:
                payload.update(
                    narrative_chain_context_for_payload(
                        payload,
                        chain_index=narrative_chain_lookup,
                    )
                )
            ranked_row = _build_ranked_memory_context_row(
                payload,
                reference_turn=current_turn,
                normalized_anchor_ids=normalized_anchor_ids,
                weight_config=tuning_weights,
            )
            if ranked_row is None:
                continue
            object_id = str(ranked_row.get("object_id") or "")
            if not object_id or object_id in seen_ids:
                continue
            seen_ids.add(object_id)
            raw_payloads_by_object_id[object_id] = payload
            for passthrough_key in (
                "fact_key",
                "view_key",
                "obligation_key",
                "source_fact_keys",
                "source_event_ids",
                "source_bundle_keys",
                "bundle_relationships",
                "core_fact_key",
                "supporting_fact_keys",
                "recent_event_ids",
                "callback_candidate",
                "narrative_packet_role",
                "packet_pressure_score",
                "why_packet_became_core",
                "actor_view_scope",
                "deadline_turn",
                "resolution_confidence",
                "inactive_reason",
                "started_at_turn",
                "resolved_at_turn",
                "last_salient_turn",
                "dormant_since_turn",
                "persisted_dormancy_state",
                "dormancy_transition",
                "dormancy_transition_turn",
                "dormancy_reason_flags",
                "current_relevance_reason_context",
            ):
                if payload.get(passthrough_key) not in (None, "", [], {}):
                    ranked_row[passthrough_key] = payload.get(passthrough_key)
            for passthrough_key in (
                "narrative_chain_keys",
                "narrative_chain_pressure_score",
                "chain_influence_score",
                "obligation_cluster_summary",
            ):
                if payload.get(passthrough_key) not in (None, "", [], {}):
                    ranked_row[passthrough_key] = payload.get(passthrough_key)
            ranked_rows.append(ranked_row)
        ranked_rows.sort(key=_memory_row_merge_tuple, reverse=True)
        return ranked_rows

    obligation_rows = (
        _collect_ranked_rows(
            object_type=STORY_OBLIGATION_OBJECT_TYPE,
            namespace=None,
            salience_ordering=obligation_priority_order,
            row_cap=max(obligation_quota * 4, row_limit, 4),
            default_layer="obligation",
            default_memory_class="obligation",
            default_lane="obligations",
            use_embeddings=False,
        )
        if obligation_quota > 0
        else []
    )
    fact_rows = _collect_ranked_rows(
        object_type=MEMORY_FACT_OBJECT_TYPE,
        namespace=MEMORY_FACT_NAMESPACE,
        salience_ordering=fact_priority_order,
        row_cap=max(fact_quota * 3, row_limit),
        default_layer="fact",
        default_memory_class="semantic",
        default_lane="durable_facts",
    )
    event_rows = _collect_ranked_rows(
        object_type=MEMORY_EVENT_OBJECT_TYPE,
        namespace=MEMORY_EVENT_NAMESPACE,
        salience_ordering=event_priority_order,
        row_cap=max(event_quota * 3, row_limit),
        eligibility_filter=event_eligibility_filter,
        default_layer="event",
        default_memory_class="episodic",
        default_lane="episodic_events",
    )
    bundle_rows = _collect_ranked_rows(
        object_type=MEMORY_BUNDLE_OBJECT_TYPE,
        namespace=MEMORY_BUNDLE_NAMESPACE,
        salience_ordering=(
            func.coalesce(_json_text_float_expr(models.ObjectModel.data["importance"].astext), literal(0.0)).desc(),
            func.coalesce(_json_text_float_expr(models.ObjectModel.data["player_salience_score"].astext), literal(0.0)).desc(),
            func.coalesce(_json_text_int_expr(models.ObjectModel.data["source_turn"].astext), literal(0)).desc(),
            models.ObjectModel.created_at.desc(),
        ),
        row_cap=max(bundle_quota * 3, row_limit),
        default_layer="bundle",
        default_memory_class="bundle",
        default_lane="narrative_bundles",
    )
    raw_fact_payloads = [
        dict(raw_payloads_by_object_id.get(str(row.get("object_id") or "")) or {})
        for row in fact_rows
        if str(row.get("object_id") or "").strip()
    ]
    raw_bundle_payloads = [
        dict(raw_payloads_by_object_id.get(str(row.get("object_id") or "")) or {})
        for row in bundle_rows
        if str(row.get("object_id") or "").strip()
    ]
    raw_event_payloads = [
        dict(raw_payloads_by_object_id.get(str(row.get("object_id") or "")) or {})
        for row in event_rows
        if str(row.get("object_id") or "").strip()
    ]
    raw_obligation_payloads = [
        dict(raw_payloads_by_object_id.get(str(row.get("object_id") or "")) or {})
        for row in obligation_rows
        if str(row.get("object_id") or "").strip()
    ]
    reference_turn = max(int(current_turn or 0), latest_completed_turn)
    if return_meta and not review_is_fresh:
        runtime_base_conflict_edges = derive_conflict_edge_payloads(
            fact_rows=raw_fact_payloads,
            event_rows=raw_event_payloads,
            bundle_rows=raw_bundle_payloads,
            current_turn=reference_turn,
        )
        runtime_obligation_payloads = derive_story_obligation_payloads(
            fact_rows=raw_fact_payloads,
            event_rows=raw_event_payloads,
            bundle_rows=raw_bundle_payloads,
            conflict_edges=runtime_base_conflict_edges,
            current_turn=reference_turn,
        )
        runtime_conflict_edges = derive_conflict_edge_payloads(
            fact_rows=raw_fact_payloads,
            event_rows=raw_event_payloads,
            bundle_rows=raw_bundle_payloads,
            obligation_rows=runtime_obligation_payloads,
            current_turn=reference_turn,
        )
        narrative_chain_rows = derive_narrative_chains(
            narrative_graph_edges=derive_narrative_graph_edges(
                conflict_edges=runtime_conflict_edges,
                bundle_rows=raw_bundle_payloads,
                obligation_rows=runtime_obligation_payloads,
            ),
            obligation_rows=runtime_obligation_payloads,
            bundle_rows=raw_bundle_payloads,
        )
        runtime_obligation_payloads = annotate_rows_with_narrative_chains(
            runtime_obligation_payloads,
            narrative_chains=narrative_chain_rows,
        )
        narrative_chain_lookup = narrative_chain_index(narrative_chain_rows)
        runtime_obligation_rows: list[dict[str, Any]] = []
        for obligation_payload in runtime_obligation_payloads:
            payload = dict(obligation_payload or {})
            payload["row_object_id"] = str(
                payload.get("obligation_key") or payload.get("object_id") or ""
            ).strip()
            if narrative_chain_lookup:
                payload.update(
                    narrative_chain_context_for_payload(
                        payload,
                        chain_index=narrative_chain_lookup,
                    )
                )
            ranked_row = _build_ranked_memory_context_row(
                payload,
                reference_turn=current_turn,
                normalized_anchor_ids=normalized_anchor_ids,
                weight_config=tuning_weights,
            )
            if ranked_row is None:
                continue
            object_id = str(ranked_row.get("object_id") or "").strip()
            if not object_id:
                continue
            raw_payloads_by_object_id[object_id] = payload
            for passthrough_key in (
                "obligation_key",
                "source_fact_keys",
                "source_event_ids",
                "source_bundle_keys",
                "deadline_turn",
                "resolution_confidence",
                "started_at_turn",
                "resolved_at_turn",
                "last_salient_turn",
                "dormant_since_turn",
                "status",
                "updated_turn",
            ):
                if payload.get(passthrough_key) not in (None, "", [], {}):
                    ranked_row[passthrough_key] = payload.get(passthrough_key)
            for passthrough_key in (
                "narrative_chain_keys",
                "narrative_chain_pressure_score",
                "chain_influence_score",
                "obligation_cluster_summary",
            ):
                if payload.get(passthrough_key) not in (None, "", [], {}):
                    ranked_row[passthrough_key] = payload.get(passthrough_key)
            runtime_obligation_rows.append(ranked_row)
        runtime_obligation_rows.sort(key=_memory_row_merge_tuple, reverse=True)
        obligation_rows = runtime_obligation_rows
        raw_obligation_payloads = [dict(payload) for payload in runtime_obligation_payloads]
        policy_state = derive_memory_policy_state(
            turn_rows=turn_rows_for_mode,
            session_state=session_state,
            session_memory_profile=session_profile_seed or None,
            session_narrative_mode=str(review_report.get("session_narrative_mode") or "").strip() or None,
            row_limit=row_limit,
            scene_mode=scene_mode,
            turn_intent=turn_intent,
            current_memory_health=review_report.get("session_memory_health_score"),
            obligation_rows=runtime_obligation_payloads,
            tuning_report=dict(review_report.get("tuning_report") or {}),
        )
        session_profile = policy_state.session_memory_profile
        session_narrative_mode = policy_state.session_narrative_mode
        session_narrative_mode_override = policy_state.session_narrative_mode_override or ""
        session_narrative_mode_source = policy_state.session_narrative_mode_source
        lane_budgets = dict(policy_state.lane_budgets)
        saturation_cap_values = dict(policy_state.saturation_limits)
        tuning_weights = tuning_weight_config_payload(policy_state.tuning_state)
        tuning_policy_version = policy_state.tuning_state.policy_version
        policy_state_data = memory_policy_state_payload(policy_state)
        operational_guardrails = {
            "source": str(review_freshness.get("mode") or "stale_review_fallback"),
            "applied_alert_keys": [],
            "stale_review_alert_keys": [
                str(item.get("alert_key") or "").strip()
                for item in operational_alert_rows
                if str(item.get("alert_key") or "").strip()
            ],
            "adjusted_lane_budgets": dict(lane_budgets),
            "adjusted_saturation_limits": dict(saturation_cap_values),
        }
        lane_budgets = dict(operational_guardrails.get("adjusted_lane_budgets") or lane_budgets)
        saturation_cap_values = dict(
            operational_guardrails.get("adjusted_saturation_limits") or saturation_cap_values
        )
        obligation_quota = int(lane_budgets.get("obligations") or 0)
        fact_quota = int(lane_budgets.get("durable_facts") or 0)
        event_quota = int(lane_budgets.get("episodic_events") or 0)
        bundle_quota = int(lane_budgets.get("narrative_bundles") or 0)
        actor_view_quota = int(lane_budgets.get("actor_views") or 0)
        review_freshness["fallback_fields"] = [
            "story_obligations",
            "actor_views",
            "narrative_chains",
            "operational_guardrails",
        ]
        review_freshness["runtime_obligation_count"] = len(runtime_obligation_payloads)
    actor_view_rows: list[dict[str, Any]] = []
    if actor_view_quota > 0:
        actor_view_payloads = derive_actor_memory_views(
            fact_rows=raw_fact_payloads,
            event_rows=raw_event_payloads,
            bundle_rows=raw_bundle_payloads,
            obligation_rows=raw_obligation_payloads,
            player_object_id=player_object_id,
            anchor_object_ids=sorted(normalized_anchor_ids),
            max_views=max(actor_view_quota * 3, row_limit),
        )
        for actor_payload in actor_view_payloads:
            payload = dict(actor_payload or {})
            feedback_payload = dict(
                actor_view_feedback_by_key.get(str(payload.get("view_key") or "").strip()) or {}
            )
            for feedback_key in (
                "surface_count",
                "used_count",
                "missed_count",
                "miss_then_reconfirmed_count",
                "quality_feedback_score",
                "family_priority_boost",
                "family_noise_score",
                "family_fragmentation_score",
                "family_miss_pressure",
            ):
                if feedback_payload.get(feedback_key) is not None:
                    payload[feedback_key] = feedback_payload.get(feedback_key)
            payload["row_object_id"] = str(payload.get("view_key") or "")
            if narrative_chain_lookup:
                payload.update(
                    narrative_chain_context_for_payload(
                        payload,
                        chain_index=narrative_chain_lookup,
                    )
                )
            ranked_row = _build_ranked_memory_context_row(
                payload,
                reference_turn=current_turn,
                normalized_anchor_ids=normalized_anchor_ids,
                weight_config=tuning_weights,
            )
            if ranked_row is None:
                continue
            object_id = str(ranked_row.get("object_id") or "").strip()
            if not object_id:
                continue
            raw_payloads_by_object_id[object_id] = payload
            for passthrough_key in (
                "view_key",
                "actor_view_scope",
                "source_fact_keys",
                "source_event_ids",
                "source_bundle_keys",
                "source_obligation_keys",
                "certainty_mix",
                "local_disputed_nodes",
                "belief_conflict_score",
                "global_vs_local_gap",
                "narrative_chain_keys",
                "narrative_chain_pressure_score",
                "chain_influence_score",
                "obligation_cluster_summary",
            ):
                if payload.get(passthrough_key) not in (None, "", [], {}):
                    ranked_row[passthrough_key] = payload.get(passthrough_key)
            actor_view_rows.append(ranked_row)
        actor_view_rows.sort(key=_memory_row_merge_tuple, reverse=True)
    selected_rows, selection_meta = select_memory_retrieval_rows(
        row_limit=row_limit,
        lane_budgets=lane_budgets,
        saturation_cap_values=saturation_cap_values,
        obligation_rows=obligation_rows,
        actor_view_rows=actor_view_rows,
        bundle_rows=bundle_rows,
        fact_rows=fact_rows,
        event_rows=event_rows,
        raw_payloads_by_object_id=raw_payloads_by_object_id,
        guardrails=operational_guardrails,
    )
    if not return_meta:
        return selected_rows
    selection_meta["session_profile"] = session_profile
    selection_meta["session_memory_profile_override"] = policy_state.session_memory_profile_override
    selection_meta["session_narrative_mode"] = session_narrative_mode
    selection_meta["session_narrative_mode_override"] = session_narrative_mode_override or None
    selection_meta["session_narrative_mode_source"] = session_narrative_mode_source
    selection_meta["memory_policy_state"] = policy_state_data
    selection_meta["operational_guardrails"] = operational_guardrails
    selection_meta["review_freshness"] = review_freshness
    return selected_rows, selection_meta


def _assign_memory_prompt_ids(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        payload = dict(row)
        payload["prompt_id"] = f"{prefix}{index}"
        assigned.append(payload)
    return assigned


def _assign_relevant_memory_prompt_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    fact_index = 0
    event_index = 0
    obligation_index = 0
    actor_view_index = 0
    for row in rows:
        payload = dict(row)
        normalized_layer = str(payload.get("layer") or "").strip().lower()
        normalized_class = str(payload.get("memory_class") or "").strip().lower()
        if normalized_layer == "obligation" or normalized_class == "obligation":
            obligation_index += 1
            payload["prompt_id"] = f"O{obligation_index}"
        elif normalized_layer == "actor_view" or normalized_class == "actor_view":
            actor_view_index += 1
            payload["prompt_id"] = f"V{actor_view_index}"
        elif normalized_layer == "event" or normalized_class == "episodic":
            event_index += 1
            payload["prompt_id"] = f"E{event_index}"
        else:
            fact_index += 1
            payload["prompt_id"] = f"F{fact_index}"
        assigned.append(payload)
    return assigned


def _list_semantically_relevant_turn_indices(
    db: Session,
    session_id: uuid.UUID,
    query_embedding: list[float] | None,
    *,
    limit: int,
    max_turn_index: int | None = None,
    exclude_turn_indices: set[int] | None = None,
) -> list[int]:
    if query_embedding is None or limit <= 0:
        return []

    excluded = set(exclude_turn_indices or set())
    distance_expr = models.ChronicleChunkModel.embedding.cosine_distance(query_embedding)
    query = (
        select(models.ChronicleChunkModel.turn_index, distance_expr.label("distance"))
        .where(
            models.ChronicleChunkModel.session_id == session_id,
            models.ChronicleChunkModel.namespace.in_(
                (CHRONICLE_OUTPUT_NAMESPACE, CHRONICLE_INPUT_NAMESPACE)
            ),
        )
        .order_by(distance_expr.asc(), models.ChronicleChunkModel.turn_index.desc())
        .limit(max(limit * 4, limit))
    )
    if max_turn_index is not None:
        query = query.where(models.ChronicleChunkModel.turn_index <= max(max_turn_index, 0))
    rows = db.execute(query).all()

    turn_indices: list[int] = []
    seen: set[int] = set()
    for turn_index, _distance in rows:
        if not isinstance(turn_index, int):
            continue
        if turn_index in excluded or turn_index in seen:
            continue
        seen.add(turn_index)
        turn_indices.append(turn_index)
        if len(turn_indices) >= limit:
            break
    return turn_indices


def _resolve_anchor_objects_for_context(
    db: Session,
    session_id: uuid.UUID,
    anchor_object_ids: list[str],
) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_anchor_id in anchor_object_ids:
        anchor_id = str(raw_anchor_id or "").strip()
        if not anchor_id or anchor_id in seen:
            continue
        seen.add(anchor_id)
        try:
            object_id = uuid.UUID(anchor_id)
        except (TypeError, ValueError, AttributeError):
            continue
        object_row = _get_object(db, session_id, object_id)
        if object_row is None:
            continue
        anchors.append({"object_id": str(object_id), "name": str(object_row.name or "")})
    return anchors


def _normalize_structural_signals_for_context(
    db: Session,
    session_id: uuid.UUID,
    raw_signals: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw_signals, list):
        return []

    formatted: list[dict[str, Any]] = []
    for raw_item in raw_signals:
        if not isinstance(raw_item, dict):
            continue
        trigger = str(raw_item.get("trigger") or "").strip()
        object_id = str(raw_item.get("object_id") or "").strip()
        turn_value = raw_item.get("turn")
        if not trigger or not object_id:
            continue
        entry: dict[str, Any] = {
            "trigger": trigger,
            "object_id": object_id,
            "turn": turn_value if isinstance(turn_value, int) else None,
        }
        try:
            parsed_object_id = uuid.UUID(object_id)
        except (TypeError, ValueError, AttributeError):
            formatted.append(entry)
            continue
        object_row = _get_object(db, session_id, parsed_object_id)
        if object_row is not None:
            entry["object_name"] = str(object_row.name or "")
        formatted.append(entry)
    return formatted


def _serialize_neighbor_turn_for_context(turn_row: models.TurnModel) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "turn_index": turn_row.turn_index,
        "user_input": _truncate_text(str(turn_row.user_input or ""), MAX_CONTEXT_TEXT_PER_ROW),
        "ai_text": _truncate_text(str(turn_row.ai_text or ""), MAX_CONTEXT_TEXT_PER_ROW),
    }
    turn_weight = _extract_turn_weight(getattr(turn_row, "ai_json", None))
    if turn_weight is not None:
        payload["turn_weight"] = turn_weight
    applied_ops = _extract_turn_applied_ops_for_context(getattr(turn_row, "ai_json", None))
    if applied_ops:
        payload["applied_ops"] = applied_ops
    return payload


def _build_turn_variants_for_context(
    db: Session,
    session_id: uuid.UUID,
    *,
    merged_turn_rows: list[models.TurnModel],
    turn_rows_by_index: dict[int, models.TurnModel],
    semantic_turn_indices: set[int],
    new_turn: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    compact_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []

    for row in merged_turn_rows:
        compact_payload: dict[str, Any] = {
            "turn_index": row.turn_index,
            "user_input": _truncate_text(str(row.user_input or ""), MAX_CONTEXT_TEXT_PER_ROW),
            "ai_text": _truncate_text(str(row.ai_text or ""), MAX_CONTEXT_TEXT_PER_ROW),
        }
        turn_weight = _extract_turn_weight(getattr(row, "ai_json", None))
        if turn_weight is not None:
            compact_payload["turn_weight"] = turn_weight
        compact_rows.append(compact_payload)

        full_payload = dict(compact_payload)
        applied_ops = _extract_turn_applied_ops_for_context(getattr(row, "ai_json", None))
        if applied_ops:
            full_payload["applied_ops"] = applied_ops

        semantic_context: dict[str, list[dict[str, Any]]] = {"prev": [], "next": []}
        for direction, turn_offset in (("prev", -1), ("next", 1)):
            neighbor_turn_index = row.turn_index + turn_offset
            if neighbor_turn_index < 1:
                continue
            neighbor_row = turn_rows_by_index.get(neighbor_turn_index)
            if neighbor_row is None:
                neighbor_row = db.get(models.TurnModel, (session_id, neighbor_turn_index))
                if neighbor_row is not None:
                    turn_rows_by_index[neighbor_turn_index] = neighbor_row
            if neighbor_row is None:
                continue
            if str(getattr(neighbor_row, "turn_kind", "player") or "player").strip() == "director":
                continue
            semantic_context[direction].append(_serialize_neighbor_turn_for_context(neighbor_row))
        if semantic_context["prev"] or semantic_context["next"]:
            full_payload["semantic_context"] = semantic_context

        if row.turn_index == new_turn - 1:
            category = "prev_turn"
        elif row.turn_index in semantic_turn_indices:
            category = "semantic_turn"
        else:
            category = "recent_turn"
        variant_rows.append(
            {
                "source_id": f"turn:{row.turn_index}",
                "category": category,
                "compact": compact_payload,
                "full": full_payload,
            }
        )

    return compact_rows, variant_rows


def _build_turn_context_pack(
    db: Session,
    session_id: uuid.UUID,
    *,
    new_turn: int,
    user_input: str,
) -> dict[str, Any]:
    from .application.turn_services import turn_context_service

    return turn_context_service.build_turn_context_pack(
        db,
        session_id,
        new_turn=new_turn,
        user_input=user_input,
    )



__all__ = [
    "_split_world_prompt_chunks",
    "_ensure_world_prompt_chunks_indexed",
    "_list_relevant_world_prompt_chunks",
    "_serialize_patch_ops",
    "_render_turn_ai_text",
    "_build_embedding_snippet",
    "_build_input_embedding_snippet",
    "_build_event_embedding_line",
    "_list_turn_event_embedding_lines",
    "_get_recent_ai_text_for_relevance",
    "_get_recent_scene_entities_for_relevance",
    "_build_relevance_query_text",
    "_build_turn_variants_for_context",
    "_get_latest_narrative_spine_row",
    "_list_recent_turn_payload_for_spine",
    "_update_narrative_spine",
    "_summarize_world_prompt_chunks",
    "_apply_elastic_field_budgets",
    "_apply_unified_context_scoring",
    "_collect_embedding_candidates",
    "_embed_query_for_relevance",
    "_list_relevant_objects_for_input",
    "_list_relevant_npcs_for_input",
    "_list_zone_npcs_with_relationships",
    "_get_relevant_player_for_input",
    "_get_player_inventory",
    "_list_orphaned_items_for_context",
    "_get_player_location_history",
    "_list_relevant_items_for_input",
    "_list_relevant_factions_for_input",
    "_list_relevant_links_for_input",
    "_list_relevant_quests_for_input",
    "_list_relevant_claims_for_input",
    "_list_zone_recent_claims",
    "_list_active_zone_actor_ids",
    "_list_active_npc_claim_links_for_knowledge",
    "_merge_npc_knowledge_subjects",
    "_build_zone_npc_knowledge",
    "_build_reaction_hints",
    "_enrich_reaction_hints",
    "_resolve_exact_name_object_ids_for_context",
    "_list_one_hop_link_candidates_for_context",
    "_list_relevant_memories_for_input",
    "_list_semantically_relevant_turn_indices",
    "_build_turn_context_pack",
    "CHRONICLE_OUTPUT_NAMESPACE",
    "CHRONICLE_INPUT_NAMESPACE",
    "SESSION_SUMMARY_OBJECT_TYPE",
    "SESSION_SUMMARY_LIVE_TURNS",
    "NARRATIVE_SPINE_OBJECT_TYPE",
    "_SPINE_UPDATER_MAX_TOKENS",
]
