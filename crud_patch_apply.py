from __future__ import annotations

from dataclasses import dataclass
import logging
import uuid
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from . import models, schemas
from .constants import (
    LOCATED_IN_LINK_TYPE,
    NPC_DEATH_PRESERVED_LINK_TYPES,
    NPC_OFFSTAGE_STATUS,
    QUEST_TERMINAL_STATUSES,
    REACTION_CONFLICT_LINK_TYPES,
    REACTION_SUPPORT_LINK_TYPES,
    SESSION_PLAYER_REF,
    TRACKING_QUEST_LINK_TYPE,
)
from .crud_embeddings_ops import (
    FACTION_PROFILE_EMBED_INSTRUCTION,
    ITEM_PROFILE_EMBED_INSTRUCTION,
    LINK_CONTENT_TYPES,
    _refresh_link_context_embedding,
    NPC_PROFILE_EMBED_INSTRUCTION,
    PLAYER_PROFILE_EMBED_INSTRUCTION,
    QUEST_PROFILE_EMBED_INSTRUCTION,
    ZONE_PROFILE_EMBED_INSTRUCTION,
    _extract_claim_text,
    _extract_link_context_text,
    _maybe_embed_texts,
    _upsert_claim_text_embedding,
    _upsert_faction_profile_embedding,
    _upsert_item_profile_embedding,
    _upsert_npc_profile_embedding,
    _upsert_player_profile_embedding,
    _upsert_quest_profile_embedding,
    _upsert_zone_profile_embedding,
)
from .crud_profiles import (
    _build_faction_profile_text,
    _build_item_profile_text,
    _build_npc_profile_text,
    _build_player_profile_text,
    _build_quest_profile_text,
    _build_zone_profile_text,
    _should_refresh_item_or_faction_profile_embedding,
    _should_refresh_npc_profile_embedding,
    _should_refresh_player_profile_embedding,
    _should_refresh_quest_profile_embedding,
    _should_refresh_zone_profile_embedding,
)
from .crud_shared import (
    PreparedObjectCreateOp,
    TurnApplyExternalPreparationRequired,
    TurnApplyExternalRequest,
    _close_player_active_located_in_links,
    _turn_apply_dedup_arbiter_key,
    current_turn_apply_external_artifacts,
    _get_active_located_in_links,
    _get_active_link,
    _get_session_player_object_id,
    _infer_actor_zone_id,
    _get_player_current_zone_id,
    _is_true,
    _normalize_json_preview,
    _require_object,
    _resolve_object_ref,
    _safe_int,
    _truncate_text,
)
from .db import (
    DEDUP_ARBITER_MIN_SIM,
    DEDUP_SIM_THRESHOLD,
    OPENROUTER_CHAT_MODEL,
    USE_CTX_WEIGHT_DECAY,
    USE_DEDUP_ARBITER,
    USE_EMBEDDINGS,
    ZONE_GLOBAL_DEDUP_THRESHOLD,
)
from .llm import openrouter_chat
from .llm_telemetry import telemetry_context

from .crud_entities import DEFAULT_EPHEMERAL_NPC_TTL

ITEM_DEDUP_THRESHOLD = DEDUP_SIM_THRESHOLD
FACTION_DEDUP_THRESHOLD = 0.90
QUEST_DEDUP_THRESHOLD = 0.85
_DEDUP_ARBITER_SYSTEM = (
    "You are a strict RPG entity dedup arbiter. "
    "Given two candidate profile texts and similarity metadata, decide if they are the same entity. "
    "Return JSON only: {\"same_entity\": true|false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}. "
    "Prefer false when uncertain."
)
_DEDUP_ARBITER_MAX_PROFILE_CHARS = 900

logger = logging.getLogger(__name__)
CTX_LAST_TOUCHED_TURN_KEY = "ctx_last_touched_turn"
NPC_PRESENCE_SINCE_TURN_KEY = "presence_since_turn"


class PatchApplyResult(dict[str, str]):
    def __init__(
        self,
        ref_map: dict[str, uuid.UUID | str] | None = None,
        *,
        applied_ops: list[dict[str, Any]] | None = None,
        applied_input_count: int = 0,
    ) -> None:
        super().__init__({str(key): str(value) for key, value in (ref_map or {}).items()})
        self.applied_ops = [dict(op) for op in (applied_ops or []) if isinstance(op, dict)]
        self.applied_input_count = max(int(applied_input_count), 0)


@dataclass(slots=True)
class _ObjectPatchApplyResult:
    player_object_id: uuid.UUID | None
    applied_ops: list[dict[str, Any]]


def _sanitize_applied_object_data(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if str(key) != CTX_LAST_TOUCHED_TURN_KEY
    }


def _build_applied_object_create_op(
    *,
    object_row: models.ObjectModel,
    object_data: dict[str, Any] | None,
    ref: str | None = None,
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "op": "object.create",
        "object_id": str(object_row.object_id),
        "type": str(object_row.type),
        "name": str(object_row.name),
    }
    raw_ref = str(ref or "").strip()
    if raw_ref:
        op["ref"] = raw_ref
    sanitized_data = _sanitize_applied_object_data(object_data)
    if sanitized_data:
        op["data"] = sanitized_data
    return op


def _build_applied_object_update_op(
    *,
    object_row: models.ObjectModel,
    old_name: str,
    old_data: dict[str, Any],
    new_name: str,
    new_data: dict[str, Any],
) -> dict[str, Any] | None:
    patch: dict[str, Any] = {}
    if old_name != new_name:
        patch["name"] = new_name

    old_clean = _sanitize_applied_object_data(old_data)
    new_clean = _sanitize_applied_object_data(new_data)
    for key in sorted(set(old_clean) | set(new_clean)):
        old_present = key in old_clean
        new_present = key in new_clean
        if old_present and new_present and old_clean[key] == new_clean[key]:
            continue
        patch[key] = new_clean[key] if new_present else None

    if not patch:
        return None
    return {
        "op": "object.update",
        "object": str(object_row.object_id),
        "patch": patch,
    }


def _build_applied_link_create_op(link_row: models.LinkModel) -> dict[str, Any]:
    op: dict[str, Any] = {
        "op": "link.create",
        "from": str(link_row.from_object_id),
        "to": str(link_row.to_object_id),
        "type": str(link_row.type),
    }
    link_data = getattr(link_row, "data", None)
    if isinstance(link_data, dict) and link_data:
        op["data"] = dict(link_data)
    return op


def _build_applied_link_close_op(
    link_row: models.LinkModel,
    *,
    from_object_id: uuid.UUID | None = None,
    to_object_id: uuid.UUID | None = None,
    link_type: str | None = None,
) -> dict[str, Any]:
    resolved_from_object_id = from_object_id or getattr(link_row, "from_object_id", None)
    resolved_to_object_id = to_object_id or getattr(link_row, "to_object_id", None)
    resolved_link_type = link_type or getattr(link_row, "type", None)
    return {
        "op": "link.close",
        "from": str(resolved_from_object_id),
        "to": str(resolved_to_object_id),
        "type": str(resolved_link_type),
    }


def _build_applied_event_create_op(
    *,
    event_type: str,
    scope_object_id: uuid.UUID | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "op": "event.create",
        "type": event_type,
        "payload": dict(payload),
    }
    if scope_object_id is not None:
        op["scope"] = str(scope_object_id)
    return op


def _raise_dedup_probe_failure(
    *,
    object_type: str,
    session_id: uuid.UUID,
    ref: str,
    exc: Exception,
) -> None:
    logger.exception(
        "Dedup probe failed for object_type=%s session_id=%s ref=%s",
        object_type,
        session_id,
        ref,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"dedup probe failed: {object_type}",
    ) from exc


def _build_profile_text_for_dedup(
    *,
    object_type: str,
    name: str,
    data: dict[str, Any],
) -> str:
    if object_type == "npc":
        return _build_npc_profile_text(name, data)
    if object_type == "zone":
        return _build_zone_profile_text(name, data)
    if object_type == "item":
        return _build_item_profile_text(name, data)
    if object_type == "faction":
        return _build_faction_profile_text(name, data)
    if object_type == "quest":
        return _build_quest_profile_text(name, data)
    if object_type == "player":
        return _build_player_profile_text(name, data)
    return _truncate_text(f"{name}. {_normalize_json_preview(data, 600)}", _DEDUP_ARBITER_MAX_PROFILE_CHARS)


def _coerce_same_entity(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "same", "match"}:
            return True
        if normalized in {"false", "no", "0", "different", "mismatch"}:
            return False
    return None


def _should_reuse_by_similarity_or_arbiter(
    *,
    object_type: str,
    similarity: float,
    threshold: float,
    incoming_profile_text: str | None,
    candidate_name: str,
    candidate_data: dict[str, Any],
) -> bool:
    if similarity >= threshold:
        return True
    if not USE_DEDUP_ARBITER:
        return False

    lower_bound = min(max(DEDUP_ARBITER_MIN_SIM, 0.0), threshold)
    # Arbiter is consulted only in the gray zone: [lower_bound, threshold).
    if similarity < lower_bound:
        return False

    incoming_text = _truncate_text(str(incoming_profile_text or "").strip(), _DEDUP_ARBITER_MAX_PROFILE_CHARS)
    candidate_text = _build_profile_text_for_dedup(
        object_type=object_type,
        name=str(candidate_name or "").strip(),
        data=dict(candidate_data or {}),
    )
    candidate_text = _truncate_text(candidate_text, _DEDUP_ARBITER_MAX_PROFILE_CHARS)
    if not incoming_text or not candidate_text:
        return False

    payload = {
        "object_type": object_type,
        "similarity": round(similarity, 6),
        "threshold": threshold,
        "incoming_profile": incoming_text,
        "candidate_profile": candidate_text,
    }
    artifacts = current_turn_apply_external_artifacts()
    if artifacts is not None:
        dedup_key = _turn_apply_dedup_arbiter_key(payload)
        if dedup_key not in artifacts.dedup_arbiter_decisions:
            raise TurnApplyExternalPreparationRequired(
                TurnApplyExternalRequest(
                    kind="dedup_arbiter",
                    dedup_payload=payload,
                )
            )
        return bool(artifacts.dedup_arbiter_decisions[dedup_key])

    return _call_dedup_arbiter(payload)


def _call_dedup_arbiter(payload: dict[str, Any]) -> bool:
    try:
        with telemetry_context(request_type="dedup_arbiter"):
            result = openrouter_chat.generate_json(
                model=OPENROUTER_CHAT_MODEL,
                system_prompt=_DEDUP_ARBITER_SYSTEM,
                user_prompt=_normalize_json_preview(payload, 7000),
                max_tokens=120,
            )
    except TurnApplyExternalPreparationRequired:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("Dedup arbiter failed, fallback to strict threshold", exc_info=True)
        return False

    if not isinstance(result, dict):
        logger.warning(
            "Dedup arbiter returned invalid payload type=%s, fallback to strict threshold (similarity=%s threshold=%s)",
            type(result).__name__,
            payload.get("similarity"),
            payload.get("threshold"),
        )
        return False

    decision = _coerce_same_entity(result.get("same_entity"))
    if decision is not None:
        return decision
    decision = _coerce_same_entity(result.get("same"))
    if decision is not None:
        return decision
    decision = _coerce_same_entity(result.get("decision"))
    if decision is None:
        logger.warning(
            "Dedup arbiter returned invalid decision payload, fallback to strict threshold (similarity=%s threshold=%s)",
            payload.get("similarity"),
            payload.get("threshold"),
        )
        return False
    return decision is True


def _apply_object_patch_to_row(
    *,
    db: Session,
    session_id: uuid.UUID,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
    object_row: models.ObjectModel,
    patch_data: dict[str, Any],
    fallback_scope_id: uuid.UUID | None,
    touched_object_ids: set[uuid.UUID],
    player_object_id: uuid.UUID | None,
    allow_name_update: bool,
) -> _ObjectPatchApplyResult:
    old_name = object_row.name
    effective_patch_data: dict[str, Any] = dict(patch_data or {})
    applied_ops: list[dict[str, Any]] = []
    if allow_name_update and object_row.type in {"zone", "item", "faction", "quest", "npc", "player"} and "name" in effective_patch_data:
        raw_name = effective_patch_data.pop("name")
        new_name = str(raw_name).strip() if raw_name is not None else ""
        if not new_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{object_row.type} name in object.update cannot be empty",
            )
        object_row.name = new_name
    if object_row.type == "npc":
        dropped_keys = [key for key in ("known_claims", "asserted_claims") if key in effective_patch_data]
        for key in dropped_keys:
            effective_patch_data.pop(key, None)
        if dropped_keys:
            logger.warning(
                "Dropping deprecated npc knowledge arrays on object.update in session_id=%s turn=%s npc=%s keys=%s",
                session_id,
                new_turn,
                object_row.object_id,
                dropped_keys,
            )
    old_data: dict[str, Any] = dict(object_row.data or {})
    claim_text_changed = False
    if object_row.type == "claim" and "text" in effective_patch_data:
        old_text = old_data.get("text")
        new_text = effective_patch_data.get("text")
        if (
            new_text != old_text
            and _claim_has_audience_links(
                db,
                session_id,
                claim_object_id=object_row.object_id,
            )
        ):
            logger.warning(
                "Skipping claim text change after audience exists in session_id=%s turn=%s claim=%s",
                session_id,
                new_turn,
                object_row.object_id,
            )
            effective_patch_data.pop("text", None)
        if "text" in effective_patch_data and effective_patch_data.get("text") != old_text:
            claim_text_changed = True
    updated_data: dict[str, Any] = dict(old_data)
    # RFC 7396: null values in the patch delete the key from data.
    for patch_key, patch_value in effective_patch_data.items():
        if patch_value is None:
            updated_data.pop(patch_key, None)
        elif object_row.type == "world_constitution" and patch_key == "structural_triggers" and isinstance(patch_value, list):
            existing_triggers = updated_data.get(patch_key, [])
            if isinstance(existing_triggers, list):
                merged_triggers = list(existing_triggers)
                for trigger in patch_value:
                    if trigger not in merged_triggers:
                        merged_triggers.append(trigger)
                updated_data[patch_key] = merged_triggers
            else:
                updated_data[patch_key] = patch_value
        else:
            updated_data[patch_key] = patch_value
    if object_row.type == "claim":
        if _resolve_claim_location_id(updated_data.get("location_id")) is None:
            asserted_speaker_ids = _get_claim_asserted_speaker_ids(
                db,
                session_id,
                claim_object_id=object_row.object_id,
            )
            if len(asserted_speaker_ids) > 1:
                logger.warning(
                    "Multiple asserted speakers for claim location backfill in session_id=%s turn=%s claim=%s speakers=%s",
                    session_id,
                    new_turn,
                    object_row.object_id,
                    asserted_speaker_ids,
                )
            if asserted_speaker_ids:
                inferred_zone_id = _infer_actor_zone_id(
                    db,
                    session_id,
                    asserted_speaker_ids[0],
                )
                if inferred_zone_id is not None:
                    updated_data["location_id"] = str(inferred_zone_id)
                else:
                    logger.warning(
                        "Unable to infer claim location on object.update in session_id=%s turn=%s claim=%s speaker=%s",
                        session_id,
                        new_turn,
                        object_row.object_id,
                        asserted_speaker_ids[0],
                    )
            else:
                logger.warning(
                    "Unable to backfill claim location without asserted speaker in session_id=%s turn=%s claim=%s",
                    session_id,
                    new_turn,
                    object_row.object_id,
                )
    updated_data = _touch_ctx_metadata(updated_data, new_turn=new_turn)
    object_row.data = updated_data
    touched_object_ids.add(object_row.object_id)
    if (
        object_row.type == "quest"
        and "status" in effective_patch_data
        and _is_terminal_quest_status(updated_data.get("status"))
        and not _is_terminal_quest_status(old_data.get("status"))
    ):
        if player_object_id is None:
            player_object_id = _get_session_player_object_id(db, session_id)
        closed_links = _close_active_tracking_quest_links(
            db,
            session_id,
            player_object_id=player_object_id,
            quest_object_id=object_row.object_id,
            closed_at_turn=new_turn,
        )
        closed_links_count = len(closed_links)
        if closed_links_count > 0:
            _add_patch_link_closed_events_for_rows(
                db,
                session_id=session_id,
                turn_index=new_turn,
                links=closed_links,
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
            )
            logger.info(
                "Closed %s active %s links for quest terminal transition in session_id=%s turn=%s quest=%s",
                closed_links_count,
                TRACKING_QUEST_LINK_TYPE,
                session_id,
                new_turn,
                object_row.object_id,
            )
            applied_ops.extend(
                _build_applied_link_close_op(
                    link_row,
                    from_object_id=player_object_id,
                    to_object_id=object_row.object_id,
                    link_type=TRACKING_QUEST_LINK_TYPE,
                )
                for link_row in closed_links
            )
    if object_row.type == "npc":
        old_status = _normalize_object_status(old_data.get("status"))
        new_status = _normalize_object_status(updated_data.get("status"))
        if old_status != "inactive" and new_status == "inactive":
            npc_deactivation_sources: set[uuid.UUID] = set()
            closed_links, total_active_before = _close_active_links_for_npc_deactivation(
                db,
                session_id,
                npc_object_id=object_row.object_id,
                closed_at_turn=new_turn,
                refresh_link_context_sources=npc_deactivation_sources,
            )
            closed_links_count = len(closed_links)
            old_weight = _extract_ctx_weight(old_data)
            if old_weight is None:
                old_weight = 1.0
            decay = closed_links_count / max(total_active_before, 1)
            decayed_weight = old_weight * (1.0 - decay * 0.5)
            updated_data["ctx_weight"] = round(min(max(decayed_weight, 0.0), 1.0), 6)
            if closed_links_count > 0:
                _add_patch_link_closed_events_for_rows(
                    db,
                    session_id=session_id,
                    turn_index=new_turn,
                    links=closed_links,
                    in_game_day=in_game_day,
                    in_game_minute=in_game_minute,
                )
                logger.info(
                    "Closed %s active interaction links for npc deactivation in session_id=%s turn=%s npc=%s",
                    closed_links_count,
                    session_id,
                    new_turn,
                    object_row.object_id,
                )
                applied_ops.extend(_build_applied_link_close_op(link_row) for link_row in closed_links)
            _refresh_link_context_embeddings_for_sources(
                db,
                session_id,
                source_object_ids=npc_deactivation_sources,
            )
    should_refresh_npc_embedding = _should_refresh_npc_profile_embedding(
        old_name=old_name,
        new_name=object_row.name,
        old_data=old_data,
        patch_data=effective_patch_data,
    )
    if object_row.type == "npc" and USE_EMBEDDINGS and should_refresh_npc_embedding:
        try:
            _upsert_npc_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                npc_name=object_row.name,
                npc_data=updated_data,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update npc_profile embedding for object_id=%s in session_id=%s",
                object_row.object_id,
                session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: npc_profile",
            ) from exc
    should_refresh_zone_embedding = _should_refresh_zone_profile_embedding(
        old_name=old_name,
        new_name=object_row.name,
        old_data=old_data,
        patch_data=effective_patch_data,
    )
    if object_row.type == "zone" and USE_EMBEDDINGS and should_refresh_zone_embedding:
        try:
            _upsert_zone_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                zone_name=object_row.name,
                zone_data=updated_data,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update zone_profile embedding for object_id=%s in session_id=%s",
                object_row.object_id,
                session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: zone_profile",
            ) from exc
    should_refresh_item_or_faction_embedding = _should_refresh_item_or_faction_profile_embedding(
        old_name=old_name,
        new_name=object_row.name,
        old_data=old_data,
        patch_data=effective_patch_data,
    )
    if object_row.type == "item" and USE_EMBEDDINGS and should_refresh_item_or_faction_embedding:
        try:
            _upsert_item_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                item_name=object_row.name,
                item_data=updated_data,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update item_profile embedding for object_id=%s in session_id=%s",
                object_row.object_id,
                session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: item_profile",
            ) from exc
    if object_row.type == "faction" and USE_EMBEDDINGS and should_refresh_item_or_faction_embedding:
        try:
            _upsert_faction_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                faction_name=object_row.name,
                faction_data=updated_data,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update faction_profile embedding for object_id=%s in session_id=%s",
                object_row.object_id,
                session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: faction_profile",
            ) from exc
    should_refresh_quest_embedding = _should_refresh_quest_profile_embedding(
        old_name=old_name,
        new_name=object_row.name,
        old_data=old_data,
        patch_data=effective_patch_data,
    )
    if object_row.type == "quest" and USE_EMBEDDINGS and should_refresh_quest_embedding:
        try:
            _upsert_quest_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                quest_name=object_row.name,
                quest_data=updated_data,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update quest_profile embedding for object_id=%s in session_id=%s",
                object_row.object_id,
                session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: quest_profile",
            ) from exc
    if object_row.type == "claim" and USE_EMBEDDINGS and claim_text_changed:
        try:
            _upsert_claim_text_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                claim_data=updated_data,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update claim_text embedding for object_id=%s in session_id=%s",
                object_row.object_id,
                session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: claim_text",
            ) from exc
    should_refresh_player_embedding = _should_refresh_player_profile_embedding(
        old_name=old_name,
        new_name=object_row.name,
        old_data=old_data,
        patch_data=effective_patch_data,
    )
    if object_row.type == "player" and USE_EMBEDDINGS and should_refresh_player_embedding:
        try:
            _upsert_player_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_row.object_id,
                player_name=object_row.name,
                player_data=updated_data,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update player_profile embedding for object_id=%s in session_id=%s",
                object_row.object_id,
                session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: player_profile",
            ) from exc
    applied_object_update = _build_applied_object_update_op(
        object_row=object_row,
        old_name=old_name,
        old_data=old_data,
        new_name=object_row.name,
        new_data=updated_data,
    )
    if applied_object_update is not None:
        applied_ops.insert(0, applied_object_update)
        _add_patch_object_updated_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
            object_row=object_row,
            patch_data=dict(applied_object_update.get("patch") or {}),
            fallback_scope_id=fallback_scope_id,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )
    return _ObjectPatchApplyResult(
        player_object_id=player_object_id,
        applied_ops=applied_ops,
    )


def _resolve_patch_link_event_scope(
    db: Session,
    *,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
) -> uuid.UUID | None:
    from_object = _require_object(db, session_id, from_object_id)
    if from_object.type == "zone":
        return from_object_id

    to_object = _require_object(db, session_id, to_object_id)
    if to_object.type == "zone":
        return to_object_id

    return _infer_actor_zone_id(db, session_id, from_object_id)


def _add_patch_link_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    event_type: str,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
    link_type: str,
    in_game_day: int,
    in_game_minute: int,
    link_data: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "from_object_id": str(from_object_id),
        "to_object_id": str(to_object_id),
        "type": link_type,
        "source": "turn_patch",
        "in_game_time": {"day": in_game_day, "minute": in_game_minute},
    }
    if link_data is not None:
        payload["data"] = dict(link_data)

    db.add(
        models.EventModel(
            session_id=session_id,
            turn_index=turn_index,
            type=event_type,
            scope_object_id=_resolve_patch_link_event_scope(
                db,
                session_id=session_id,
                from_object_id=from_object_id,
                to_object_id=to_object_id,
            ),
            payload=payload,
        )
    )


def _add_patch_move_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    player_object_id: uuid.UUID,
    from_zone_id: uuid.UUID | None,
    to_zone_id: uuid.UUID,
    to_zone_name: str,
    in_game_day: int,
    in_game_minute: int,
) -> None:
    payload: dict[str, Any] = {
        "player_object_id": str(player_object_id),
        "to_object_id": str(to_zone_id),
        "to_name": to_zone_name,
        "source": "turn_patch",
        "in_game_time": {"day": in_game_day, "minute": in_game_minute},
    }
    if from_zone_id is not None:
        payload["from_object_id"] = str(from_zone_id)
        try:
            payload["from_name"] = _require_object(db, session_id, from_zone_id).name
        except HTTPException:
            pass

    db.add(
        models.EventModel(
            session_id=session_id,
            turn_index=turn_index,
            type="MOVE",
            scope_object_id=to_zone_id,
            payload=payload,
        )
    )


def _coerce_patch_event_scope_uuid(value: Any) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except (TypeError, ValueError, AttributeError):
        return None


def _resolve_patch_object_event_scope(
    *,
    object_row: models.ObjectModel,
    object_data: dict[str, Any],
    fallback_scope_id: uuid.UUID | None,
) -> uuid.UUID | None:
    if object_row.type == "zone":
        return object_row.object_id
    for key in ("location_id", "zone_id"):
        scope_object_id = _coerce_patch_event_scope_uuid(object_data.get(key))
        if scope_object_id is not None:
            return scope_object_id
    return fallback_scope_id


def _add_patch_object_created_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    object_row: models.ObjectModel,
    object_data: dict[str, Any],
    fallback_scope_id: uuid.UUID | None,
    in_game_day: int,
    in_game_minute: int,
) -> None:
    db.add(
        models.EventModel(
            session_id=session_id,
            turn_index=turn_index,
            type="OBJECT_CREATED",
            scope_object_id=_resolve_patch_object_event_scope(
                object_row=object_row,
                object_data=object_data,
                fallback_scope_id=fallback_scope_id,
            ),
            payload={
                "object_id": str(object_row.object_id),
                "object_type": object_row.type,
                "name": object_row.name,
                "data": dict(object_data),
                "source": "turn_patch",
                "in_game_time": {"day": in_game_day, "minute": in_game_minute},
            },
        )
    )


def _add_patch_object_updated_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    object_row: models.ObjectModel,
    patch_data: dict[str, Any],
    fallback_scope_id: uuid.UUID | None,
    in_game_day: int,
    in_game_minute: int,
) -> None:
    db.add(
        models.EventModel(
            session_id=session_id,
            turn_index=turn_index,
            type="OBJECT_UPDATED",
            scope_object_id=_resolve_patch_object_event_scope(
                object_row=object_row,
                object_data=dict(getattr(object_row, "data", {}) or {}),
                fallback_scope_id=fallback_scope_id,
            ),
            payload={
                "object_id": str(object_row.object_id),
                "object_type": object_row.type,
                "name": object_row.name,
                "patch": dict(patch_data or {}),
                "source": "turn_patch",
                "in_game_time": {"day": in_game_day, "minute": in_game_minute},
            },
        )
    )


def _add_patch_link_closed_events_for_rows(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    links: list[models.LinkModel],
    in_game_day: int,
    in_game_minute: int,
) -> None:
    for link in links:
        from_object_id = getattr(link, "from_object_id", None)
        to_object_id = getattr(link, "to_object_id", None)
        link_type = str(getattr(link, "type", "")).strip()
        if not isinstance(from_object_id, uuid.UUID) or not isinstance(to_object_id, uuid.UUID) or not link_type:
            continue
        _add_patch_link_event(
            db,
            session_id=session_id,
            turn_index=turn_index,
            event_type="LINK_CLOSED",
            from_object_id=from_object_id,
            to_object_id=to_object_id,
            link_type=link_type,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )


def _find_ephemeral_npc_dedup_candidate(
    db: Session,
    session_id: uuid.UUID,
    profile_embedding: list[float],
    zone_id: uuid.UUID | None,
    incoming_profile_text: str | None = None,
) -> uuid.UUID | None:
    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(profile_embedding)

    query = (
        select(
            models.ObjectEmbeddingModel.object_id,
            distance_expr.label("distance"),
            models.ObjectModel.name,
            models.ObjectModel.data,
        )
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.ObjectEmbeddingModel.session_id,
                models.ObjectModel.object_id == models.ObjectEmbeddingModel.object_id,
            ),
        )
        .where(
            models.ObjectEmbeddingModel.session_id == session_id,
            models.ObjectEmbeddingModel.namespace == "npc_profile",
            models.ObjectModel.type == "npc",
            models.ObjectModel.data["ephemeral"].astext == "true",
            or_(
                models.ObjectModel.data["pinned"].astext.is_(None),
                models.ObjectModel.data["pinned"].astext != "true",
            ),
            or_(
                models.ObjectModel.data["status"].astext.is_(None),
                models.ObjectModel.data["status"].astext != "inactive",
            ),
        )
        .order_by(distance_expr.asc())
        .limit(1)
    )

    if zone_id is not None:
        active_zone_presence = (
            select(models.LinkModel.link_id)
            .where(
                models.LinkModel.session_id == models.ObjectModel.session_id,
                models.LinkModel.from_object_id == models.ObjectModel.object_id,
                models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                models.LinkModel.valid_to_turn.is_(None),
                models.LinkModel.to_object_id == zone_id,
            )
            .exists()
        )
        offstage_zone_match = and_(
            models.ObjectModel.data["status"].astext == NPC_OFFSTAGE_STATUS,
            models.ObjectModel.data["last_known_zone_id"].astext == str(zone_id),
        )
        query = query.where(or_(active_zone_presence, offstage_zone_match))

    row = db.execute(query).first()
    if row is None:
        return None

    object_id, distance, candidate_name, candidate_data = row
    if distance is None:
        return None

    similarity = 1.0 - float(distance)
    if _should_reuse_by_similarity_or_arbiter(
        object_type="npc",
        similarity=similarity,
        threshold=DEDUP_SIM_THRESHOLD,
        incoming_profile_text=incoming_profile_text,
        candidate_name=str(candidate_name or ""),
        candidate_data=dict(candidate_data or {}),
    ):
        return object_id
    return None


def _find_persistent_npc_dedup_candidate(
    db: Session,
    session_id: uuid.UUID,
    profile_embedding: list[float],
    incoming_profile_text: str | None = None,
) -> uuid.UUID | None:
    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(profile_embedding)
    row = db.execute(
        select(
            models.ObjectEmbeddingModel.object_id,
            distance_expr.label("distance"),
            models.ObjectModel.name,
            models.ObjectModel.data,
        )
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.ObjectEmbeddingModel.session_id,
                models.ObjectModel.object_id == models.ObjectEmbeddingModel.object_id,
            ),
        )
        .where(
            models.ObjectEmbeddingModel.session_id == session_id,
            models.ObjectEmbeddingModel.namespace == "npc_profile",
            models.ObjectModel.type == "npc",
            or_(
                models.ObjectModel.data["status"].astext.is_(None),
                models.ObjectModel.data["status"].astext != "inactive",
            ),
        )
        .order_by(distance_expr.asc())
        .limit(1)
    ).first()
    if row is None:
        return None

    object_id, distance, candidate_name, candidate_data = row
    if distance is None:
        return None

    similarity = 1.0 - float(distance)
    if _should_reuse_by_similarity_or_arbiter(
        object_type="npc",
        similarity=similarity,
        threshold=DEDUP_SIM_THRESHOLD,
        incoming_profile_text=incoming_profile_text,
        candidate_name=str(candidate_name or ""),
        candidate_data=dict(candidate_data or {}),
    ):
        return object_id
    return None


def _find_global_object_dedup_candidate(
    db: Session,
    session_id: uuid.UUID,
    profile_embedding: list[float],
    *,
    object_type: str,
    namespace: str,
    threshold: float,
    incoming_profile_text: str | None = None,
) -> uuid.UUID | None:
    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(profile_embedding)
    query = (
        select(
            models.ObjectEmbeddingModel.object_id,
            distance_expr.label("distance"),
            models.ObjectModel.name,
            models.ObjectModel.data,
        )
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.ObjectEmbeddingModel.session_id,
                models.ObjectModel.object_id == models.ObjectEmbeddingModel.object_id,
            ),
        )
        .where(
            models.ObjectEmbeddingModel.session_id == session_id,
            models.ObjectEmbeddingModel.namespace == namespace,
            models.ObjectModel.type == object_type,
        )
    )

    if object_type == "quest":
        status_expr = models.ObjectModel.data["status"].astext
        query = query.where(
            or_(
                status_expr.is_(None),
                func.lower(status_expr).notin_(QUEST_TERMINAL_STATUSES),
            )
        )
    else:
        query = query.where(
            or_(
                models.ObjectModel.data["status"].astext.is_(None),
                models.ObjectModel.data["status"].astext != "inactive",
            )
        )

    row = db.execute(query.order_by(distance_expr.asc()).limit(1)).first()
    if row is None:
        return None

    object_id, distance, candidate_name, candidate_data = row
    if distance is None:
        return None

    similarity = 1.0 - float(distance)
    if _should_reuse_by_similarity_or_arbiter(
        object_type=object_type,
        similarity=similarity,
        threshold=threshold,
        incoming_profile_text=incoming_profile_text,
        candidate_name=str(candidate_name or ""),
        candidate_data=dict(candidate_data or {}),
    ):
        return object_id
    return None


def _find_global_zone_dedup_candidate(
    db: Session,
    session_id: uuid.UUID,
    profile_embedding: list[float],
    incoming_profile_text: str | None = None,
) -> uuid.UUID | None:
    distance_expr = models.ObjectEmbeddingModel.embedding.cosine_distance(profile_embedding)
    query = (
        select(
            models.ObjectEmbeddingModel.object_id,
            distance_expr.label("distance"),
            models.ObjectModel.name,
            models.ObjectModel.data,
        )
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.ObjectEmbeddingModel.session_id,
                models.ObjectModel.object_id == models.ObjectEmbeddingModel.object_id,
            ),
        )
        .where(
            models.ObjectEmbeddingModel.session_id == session_id,
            models.ObjectEmbeddingModel.namespace == "zone_profile",
            models.ObjectModel.type == "zone",
            or_(
                models.ObjectModel.data["status"].astext.is_(None),
                models.ObjectModel.data["status"].astext != "inactive",
            ),
        )
        .order_by(distance_expr.asc())
        .limit(1)
    )
    row = db.execute(query).first()
    if row is None:
        return None

    object_id, distance, candidate_name, candidate_data = row
    if distance is None:
        return None

    similarity = 1.0 - float(distance)
    if _should_reuse_by_similarity_or_arbiter(
        object_type="zone",
        similarity=similarity,
        threshold=ZONE_GLOBAL_DEDUP_THRESHOLD,
        incoming_profile_text=incoming_profile_text,
        candidate_name=str(candidate_name or ""),
        candidate_data=dict(candidate_data or {}),
    ):
        return object_id
    return None


def _prepare_object_create_chunk(
    db: Session,
    session_id: uuid.UUID,
    *,
    new_turn: int,
    ops: list[schemas.PatchOp],
    start_index: int,
) -> tuple[dict[int, PreparedObjectCreateOp], int]:
    prepared_by_index: dict[int, PreparedObjectCreateOp] = {}
    if start_index >= len(ops) or not isinstance(ops[start_index], schemas.ObjectCreateOp):
        return prepared_by_index, start_index

    current_zone_id = _get_player_current_zone_id(db, session_id)
    embedding_requests: list[tuple[int, str, str, str | None]] = []
    index = start_index
    while index < len(ops):
        op = ops[index]
        if not isinstance(op, schemas.ObjectCreateOp):
            break

        object_data: dict[str, Any] = dict(op.data or {})
        is_npc = op.type == "npc"
        is_zone = op.type == "zone"
        is_item = op.type == "item"
        is_faction = op.type == "faction"
        is_quest = op.type == "quest"
        is_player = op.type == "player"
        is_ephemeral_npc = is_npc and _is_true(object_data.get("ephemeral"))

        if is_ephemeral_npc:
            if "pinned" not in object_data:
                object_data["pinned"] = False

            parsed_despawn_turn = _safe_int(object_data.get("despawn_turn"))
            if parsed_despawn_turn is None:
                object_data["despawn_turn"] = new_turn + DEFAULT_EPHEMERAL_NPC_TTL
            else:
                object_data["despawn_turn"] = parsed_despawn_turn

            if not isinstance(object_data.get("spawn"), dict):
                object_data["spawn"] = {
                    "reason": "unspecified",
                    "turn": new_turn,
                    "zone_id": str(current_zone_id) if current_zone_id else None,
                }
            object_data[NPC_PRESENCE_SINCE_TURN_KEY] = new_turn

        prepared = PreparedObjectCreateOp(
            object_data=object_data,
            current_zone_id=current_zone_id,
        )
        if USE_EMBEDDINGS:
            if is_player:
                prepared.player_profile_text = _build_player_profile_text(op.name, object_data)
                if prepared.player_profile_text:
                    embedding_requests.append(
                        (
                            index,
                            "player_profile_embedding",
                            prepared.player_profile_text,
                            PLAYER_PROFILE_EMBED_INSTRUCTION,
                        )
                    )
            if is_npc:
                prepared.npc_profile_text = _build_npc_profile_text(op.name, object_data)
                if prepared.npc_profile_text:
                    embedding_requests.append(
                        (
                            index,
                            "npc_profile_embedding",
                            prepared.npc_profile_text,
                            NPC_PROFILE_EMBED_INSTRUCTION,
                        )
                    )
            if is_zone:
                prepared.zone_profile_text = _build_zone_profile_text(op.name, object_data)
                if prepared.zone_profile_text:
                    embedding_requests.append(
                        (
                            index,
                            "zone_profile_embedding",
                            prepared.zone_profile_text,
                            ZONE_PROFILE_EMBED_INSTRUCTION,
                        )
                    )
            if is_item:
                prepared.item_profile_text = _build_item_profile_text(op.name, object_data)
                if prepared.item_profile_text:
                    embedding_requests.append(
                        (
                            index,
                            "item_profile_embedding",
                            prepared.item_profile_text,
                            ITEM_PROFILE_EMBED_INSTRUCTION,
                        )
                    )
            if is_faction:
                prepared.faction_profile_text = _build_faction_profile_text(op.name, object_data)
                if prepared.faction_profile_text:
                    embedding_requests.append(
                        (
                            index,
                            "faction_profile_embedding",
                            prepared.faction_profile_text,
                            FACTION_PROFILE_EMBED_INSTRUCTION,
                        )
                    )
            if is_quest:
                prepared.quest_profile_text = _build_quest_profile_text(op.name, object_data)
                if prepared.quest_profile_text:
                    embedding_requests.append(
                        (
                            index,
                            "quest_profile_embedding",
                            prepared.quest_profile_text,
                            QUEST_PROFILE_EMBED_INSTRUCTION,
                        )
                    )

        prepared_by_index[index] = prepared
        index += 1

    if USE_EMBEDDINGS and embedding_requests:
        try:
            grouped_requests: dict[str | None, list[tuple[int, str, str]]] = {}
            for op_index, attr_name, text, instruction in embedding_requests:
                grouped_requests.setdefault(instruction, []).append((op_index, attr_name, text))

            for instruction, requests in grouped_requests.items():
                unique_texts: list[str] = []
                text_to_index: dict[str, int] = {}
                for _op_index, _attr_name, text in requests:
                    if text in text_to_index:
                        continue
                    text_to_index[text] = len(unique_texts)
                    unique_texts.append(text)

                vectors = _maybe_embed_texts(unique_texts, instruction=instruction)
                if len(vectors) != len(unique_texts):
                    raise RuntimeError(
                        f"Batched embedding size mismatch: got {len(vectors)}, expected {len(unique_texts)}"
                    )

                for op_index, attr_name, text in requests:
                    embedding = vectors[text_to_index[text]]
                    setattr(prepared_by_index[op_index], attr_name, embedding)
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "Batched profile embedding precompute failed for session_id=%s create_ops=%s..%s",
                session_id,
                start_index,
                index - 1,
            )

    return prepared_by_index, index


def _get_npc_presence_floor_turn(
    db: Session,
    session_id: uuid.UUID,
    *,
    npc_object_id: uuid.UUID,
) -> int | None:
    spawn_turn: int | None = None
    npc_row = db.get(models.ObjectModel, (session_id, npc_object_id))
    if npc_row is not None and isinstance(npc_row.data, dict):
        presence_since_turn = _safe_int(npc_row.data.get(NPC_PRESENCE_SINCE_TURN_KEY))
        if presence_since_turn is not None:
            if presence_since_turn < 0:
                return 0
            return presence_since_turn
        spawn_payload = npc_row.data.get("spawn")
        if isinstance(spawn_payload, dict):
            spawn_turn = _safe_int(spawn_payload.get("turn"))
            if spawn_turn is not None and spawn_turn < 0:
                spawn_turn = 0

    earliest_presence_turn = db.execute(
        select(func.min(models.LinkModel.valid_from_turn))
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == npc_object_id,
            models.LinkModel.type == LOCATED_IN_LINK_TYPE,
        )
    ).scalar_one_or_none()
    if isinstance(earliest_presence_turn, bool):
        earliest_presence_turn = None
    if isinstance(earliest_presence_turn, int) and earliest_presence_turn < 0:
        earliest_presence_turn = 0

    if spawn_turn is None and not isinstance(earliest_presence_turn, int):
        return None
    if spawn_turn is None:
        if isinstance(earliest_presence_turn, int):
            return earliest_presence_turn
        return None
    if not isinstance(earliest_presence_turn, int):
        return spawn_turn
    return max(spawn_turn, earliest_presence_turn)


def _get_actor_active_zone_id(
    db: Session,
    session_id: uuid.UUID,
    *,
    actor_object_id: uuid.UUID,
) -> uuid.UUID | None:
    active_links = _get_active_located_in_links(db, session_id, actor_object_id)
    if len(active_links) > 1:
        logger.warning(
            "Multiple active located_in links for actor in session_id=%s actor=%s; using first link deterministically",
            session_id,
            actor_object_id,
        )
    if not active_links:
        return None
    return active_links[0].to_object_id


def _is_missing_claim_location(raw_location: Any) -> bool:
    if raw_location is None:
        return True
    text = str(raw_location).strip()
    if not text:
        return True
    return text.lower() == "null"


def _resolve_claim_location_id(raw_location: Any) -> uuid.UUID | None:
    if _is_missing_claim_location(raw_location):
        return None
    try:
        return uuid.UUID(str(raw_location).strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _get_claim_asserted_speaker_ids(
    db: Session,
    session_id: uuid.UUID,
    *,
    claim_object_id: uuid.UUID,
) -> list[uuid.UUID]:
    rows = db.execute(
        select(models.LinkModel.from_object_id)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.to_object_id == claim_object_id,
            models.LinkModel.type == "asserted",
            models.LinkModel.valid_to_turn.is_(None),
        )
        .order_by(models.LinkModel.valid_from_turn.asc(), models.LinkModel.created_at.asc())
    ).scalars().all()
    speaker_ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for speaker_id in rows:
        if not isinstance(speaker_id, uuid.UUID):
            continue
        if speaker_id in seen:
            continue
        seen.add(speaker_id)
        speaker_ids.append(speaker_id)
    return speaker_ids


def _backfill_claim_location_from_speaker(
    db: Session,
    session_id: uuid.UUID,
    *,
    claim_object_id: uuid.UUID,
    speaker_object_id: uuid.UUID,
    new_turn: int,
) -> uuid.UUID | None:
    claim_row = db.get(models.ObjectModel, (session_id, claim_object_id))
    if claim_row is None or claim_row.type != "claim":
        return None

    claim_data = dict(claim_row.data or {})
    if _resolve_claim_location_id(claim_data.get("location_id")) is not None:
        return None

    inferred_zone_id = _infer_actor_zone_id(
        db,
        session_id,
        speaker_object_id,
    )
    if inferred_zone_id is None:
        logger.warning(
            "Unable to infer claim location_id in session_id=%s turn=%s claim=%s speaker=%s",
            session_id,
            new_turn,
            claim_object_id,
            speaker_object_id,
        )
        return None

    claim_data["location_id"] = str(inferred_zone_id)
    claim_row.data = claim_data
    return inferred_zone_id


def _get_claim_first_turn(
    db: Session,
    session_id: uuid.UUID,
    *,
    claim_object_id: uuid.UUID,
) -> int | None:
    first_turn = db.execute(
        select(func.min(models.LinkModel.valid_from_turn))
        .join(
            models.ObjectModel,
            and_(
                models.ObjectModel.session_id == models.LinkModel.session_id,
                models.ObjectModel.object_id == models.LinkModel.to_object_id,
            ),
        )
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.to_object_id == claim_object_id,
            models.LinkModel.type.in_(("heard", "asserted")),
            models.ObjectModel.type == "claim",
        )
    ).scalar_one_or_none()
    if isinstance(first_turn, bool):
        return None
    if not isinstance(first_turn, int):
        return None
    return max(first_turn, 0)


def _resolve_claim_zone_id_for_link_validation(
    db: Session,
    session_id: uuid.UUID,
    *,
    claim_object_id: uuid.UUID,
    claim_data: dict[str, Any],
    new_turn: int,
) -> uuid.UUID | None:
    claim_zone_id = _resolve_claim_location_id(claim_data.get("location_id"))
    if claim_zone_id is not None:
        return claim_zone_id

    asserted_speaker_ids = _get_claim_asserted_speaker_ids(
        db,
        session_id,
        claim_object_id=claim_object_id,
    )
    if len(asserted_speaker_ids) > 1:
        logger.warning(
            "Multiple asserted speakers while resolving claim zone for heard link in session_id=%s turn=%s claim=%s speakers=%s",
            session_id,
            new_turn,
            claim_object_id,
            asserted_speaker_ids,
        )
    if not asserted_speaker_ids:
        return None

    inferred_zone_id = _infer_actor_zone_id(
        db,
        session_id,
        asserted_speaker_ids[0],
    )
    if inferred_zone_id is None:
        logger.warning(
            "Unable to infer claim zone for heard link in session_id=%s turn=%s claim=%s speaker=%s",
            session_id,
            new_turn,
            claim_object_id,
            asserted_speaker_ids[0],
        )
    return inferred_zone_id


def _claim_has_audience_links(
    db: Session,
    session_id: uuid.UUID,
    *,
    claim_object_id: uuid.UUID,
) -> bool:
    link_id = db.execute(
        select(models.LinkModel.link_id)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.to_object_id == claim_object_id,
            models.LinkModel.type.in_(("heard", "asserted")),
        )
        .limit(1)
    ).scalar_one_or_none()
    return link_id is not None


def _validate_claim_audience_link_create(
    db: Session,
    session_id: uuid.UUID,
    *,
    new_turn: int,
    link_type: str,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
) -> bool:
    to_object = _require_object(db, session_id, to_object_id)
    if to_object.type != "claim":
        logger.warning(
            "Skipping %s link.create with non-claim target in session_id=%s turn=%s from=%s to=%s target_type=%s",
            link_type,
            session_id,
            new_turn,
            from_object_id,
            to_object_id,
            to_object.type,
        )
        return False
    if link_type != "heard":
        return True

    claim_data = dict(to_object.data or {})
    claim_first_turn = _get_claim_first_turn(
        db,
        session_id,
        claim_object_id=to_object_id,
    )
    if claim_first_turn is None:
        claim_first_turn = new_turn
    if new_turn != claim_first_turn:
        logger.warning(
            "Skipping heard link.create outside claim first turn in session_id=%s turn=%s from=%s claim=%s claim_first_turn=%s",
            session_id,
            new_turn,
            from_object_id,
            to_object_id,
            claim_first_turn,
        )
        return False

    claim_zone_id = _resolve_claim_zone_id_for_link_validation(
        db,
        session_id,
        claim_object_id=to_object_id,
        claim_data=claim_data,
        new_turn=new_turn,
    )
    if claim_zone_id is None:
        logger.warning(
            "Skipping heard link.create with unresolved claim zone in session_id=%s turn=%s from=%s claim=%s",
            session_id,
            new_turn,
            from_object_id,
            to_object_id,
        )
        return False

    actor_zone_id = _get_actor_active_zone_id(
        db,
        session_id,
        actor_object_id=from_object_id,
    )
    if actor_zone_id is None:
        logger.warning(
            "Skipping heard link.create for actor without active zone in session_id=%s turn=%s actor=%s claim=%s claim_zone=%s",
            session_id,
            new_turn,
            from_object_id,
            to_object_id,
            claim_zone_id,
        )
        return False
    if actor_zone_id != claim_zone_id:
        logger.warning(
            "Skipping heard link.create with zone mismatch in session_id=%s turn=%s actor=%s claim=%s actor_zone=%s claim_zone=%s",
            session_id,
            new_turn,
            from_object_id,
            to_object_id,
            actor_zone_id,
            claim_zone_id,
        )
        return False

    actor_presence_floor_turn = _get_npc_presence_floor_turn(
        db,
        session_id,
        npc_object_id=from_object_id,
    )
    if (
        isinstance(actor_presence_floor_turn, int)
        and claim_first_turn < actor_presence_floor_turn
    ):
        logger.warning(
            "Skipping heard link.create with temporal mismatch in session_id=%s turn=%s actor=%s claim=%s claim_first_turn=%s actor_presence_floor=%s",
            session_id,
            new_turn,
            from_object_id,
            to_object_id,
            claim_first_turn,
            actor_presence_floor_turn,
        )
        return False
    return True


def _normalize_quest_status(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip().casefold()
        return text or None
    text = str(raw_value).strip().casefold()
    return text or None


def _is_terminal_quest_status(raw_value: Any) -> bool:
    normalized = _normalize_quest_status(raw_value)
    if normalized is None:
        return False
    return normalized in QUEST_TERMINAL_STATUSES


def _normalize_object_status(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip().casefold()
        return text or None
    text = str(raw_value).strip().casefold()
    return text or None


def _prepare_reused_ephemeral_npc_patch_data(
    *,
    existing_data: dict[str, Any] | None,
    patch_data: dict[str, Any],
    new_turn: int,
    current_zone_id: uuid.UUID | None,
) -> dict[str, Any]:
    prepared_patch = dict(patch_data or {})
    current_status = _normalize_object_status((existing_data or {}).get("status"))
    if current_status != NPC_OFFSTAGE_STATUS:
        return prepared_patch
    if "status" not in prepared_patch:
        prepared_patch["status"] = "active"
    if "despawn_turn" not in prepared_patch:
        prepared_patch["despawn_turn"] = new_turn + DEFAULT_EPHEMERAL_NPC_TTL
    if "despawned_turn" not in prepared_patch:
        prepared_patch["despawned_turn"] = None
    if "despawn_reason" not in prepared_patch:
        prepared_patch["despawn_reason"] = None
    if current_zone_id is not None and "last_known_zone_id" not in prepared_patch:
        prepared_patch["last_known_zone_id"] = str(current_zone_id)
    prepared_patch[NPC_PRESENCE_SINCE_TURN_KEY] = new_turn
    return prepared_patch


def _extract_ctx_weight(raw_data: dict[str, Any] | None) -> float | None:
    if not isinstance(raw_data, dict):
        return None
    raw_value = raw_data.get("ctx_weight")
    if isinstance(raw_value, bool):
        return None
    parsed: float | None = None
    if isinstance(raw_value, (int, float)):
        parsed = float(raw_value)
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if text:
            try:
                parsed = float(text)
            except ValueError:
                parsed = None
    if parsed is None or parsed != parsed:
        return None
    return round(min(max(parsed, 0.0), 1.0), 6)


def _touch_ctx_metadata(
    raw_data: dict[str, Any] | None,
    *,
    new_turn: int,
) -> dict[str, Any]:
    data = dict(raw_data or {})
    if not USE_CTX_WEIGHT_DECAY:
        return data
    data[CTX_LAST_TOUCHED_TURN_KEY] = max(int(new_turn), 0)
    return data


def _apply_ctx_touches_for_object_ids(
    db: Session,
    session_id: uuid.UUID,
    *,
    object_ids: set[uuid.UUID],
    new_turn: int,
) -> None:
    if not USE_CTX_WEIGHT_DECAY or not object_ids:
        return
    rows = db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.object_id.in_(tuple(object_ids)),
        )
    ).scalars().all()
    touched_turn = max(int(new_turn), 0)
    for object_row in rows:
        old_data = dict(object_row.data or {})
        if _safe_int(old_data.get(CTX_LAST_TOUCHED_TURN_KEY)) == touched_turn:
            continue
        object_row.data = _touch_ctx_metadata(old_data, new_turn=touched_turn)


def _collect_proposed_tracking_targets(
    ops: list[schemas.PatchOp],
    *,
    player_object_id: uuid.UUID,
) -> set[uuid.UUID]:
    quest_targets: set[uuid.UUID] = set()
    for op in ops:
        if not isinstance(op, schemas.LinkCreateOp):
            continue
        if op.type != TRACKING_QUEST_LINK_TYPE:
            continue
        from_is_player = op.from_ref == SESSION_PLAYER_REF or (
            isinstance(op.from_ref, uuid.UUID) and op.from_ref == player_object_id
        )
        if not from_is_player:
            continue
        if isinstance(op.to, uuid.UUID):
            quest_targets.add(op.to)
    return quest_targets


def _assert_quest_reopen_tracking_contract(
    db: Session,
    session_id: uuid.UUID,
    *,
    ops: list[schemas.PatchOp],
) -> None:
    player_object_id: uuid.UUID | None = None
    proposed_targets: set[uuid.UUID] | None = None
    for op in ops:
        if not isinstance(op, schemas.ObjectUpdateOp):
            continue
        if not isinstance(op.object, uuid.UUID):
            continue
        patch_data = dict(op.patch or {})
        if "status" not in patch_data:
            continue
        new_status = _normalize_quest_status(patch_data.get("status"))
        if new_status is None or _is_terminal_quest_status(new_status):
            continue

        object_row = db.get(models.ObjectModel, (session_id, op.object))
        if object_row is None or object_row.type != "quest":
            continue
        old_status = _normalize_quest_status(dict(object_row.data or {}).get("status"))
        if old_status not in QUEST_TERMINAL_STATUSES:
            continue

        if player_object_id is None:
            player_object_id = _get_session_player_object_id(db, session_id)
        if proposed_targets is None:
            proposed_targets = _collect_proposed_tracking_targets(
                ops,
                player_object_id=player_object_id,
            )
        has_active_tracking = (
            _get_active_link(
                db,
                session_id,
                player_object_id,
                op.object,
                TRACKING_QUEST_LINK_TYPE,
            )
            is not None
        )
        if has_active_tracking:
            continue
        if op.object in proposed_targets:
            continue
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Quest reopen requires link.create type='tracking_quest' from session player "
                f"to quest {op.object}"
            ),
        )


def _close_active_tracking_quest_links(
    db: Session,
    session_id: uuid.UUID,
    *,
    player_object_id: uuid.UUID,
    quest_object_id: uuid.UUID,
    closed_at_turn: int,
) -> list[models.LinkModel]:
    active_links = db.execute(
        select(models.LinkModel)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.from_object_id == player_object_id,
            models.LinkModel.to_object_id == quest_object_id,
            models.LinkModel.type == TRACKING_QUEST_LINK_TYPE,
            models.LinkModel.valid_to_turn.is_(None),
        )
        .order_by(models.LinkModel.created_at.asc())
    ).scalars().all()
    for link in active_links:
        close_turn = closed_at_turn
        link_valid_from = getattr(link, "valid_from_turn", close_turn)
        if isinstance(link_valid_from, int) and close_turn < link_valid_from:
            close_turn = link_valid_from
        link.valid_to_turn = close_turn
    return active_links


def _close_active_links_for_npc_deactivation(
    db: Session,
    session_id: uuid.UUID,
    *,
    npc_object_id: uuid.UUID,
    closed_at_turn: int,
    refresh_link_context_sources: set[uuid.UUID] | None = None,
) -> tuple[list[models.LinkModel], int]:
    rows = db.execute(
        select(models.LinkModel)
        .where(
            models.LinkModel.session_id == session_id,
            models.LinkModel.valid_to_turn.is_(None),
            or_(
                models.LinkModel.from_object_id == npc_object_id,
                models.LinkModel.to_object_id == npc_object_id,
            ),
        )
        .order_by(models.LinkModel.created_at.asc())
    ).scalars().all()
    total_active_before = len(rows)
    if not rows:
        return [], 0
    closable_rows = [
        link
        for link in rows
        if str(getattr(link, "type", "")).strip() not in NPC_DEATH_PRESERVED_LINK_TYPES
    ]
    if not closable_rows:
        return [], total_active_before
    _close_link_rows(
        closable_rows,
        closed_at_turn=closed_at_turn,
        refresh_link_context_sources=refresh_link_context_sources,
    )
    return closable_rows, total_active_before


def _close_link_rows(
    links: list[models.LinkModel],
    *,
    closed_at_turn: int,
    refresh_link_context_sources: set[uuid.UUID] | None = None,
) -> int:
    for link in links:
        close_turn = closed_at_turn
        link_valid_from = getattr(link, "valid_from_turn", close_turn)
        if isinstance(link_valid_from, int) and close_turn < link_valid_from:
            close_turn = link_valid_from
        link.valid_to_turn = close_turn
        if (
            refresh_link_context_sources is not None
            and str(getattr(link, "type", "")).strip() in LINK_CONTENT_TYPES
            and isinstance(getattr(link, "from_object_id", None), uuid.UUID)
        ):
            # link_context embeddings are indexed per source object and built from
            # outgoing edges only, so refresh uses the closed link's from_object_id.
            refresh_link_context_sources.add(link.from_object_id)
    return len(links)


def _close_active_links_for_edge(
    db: Session,
    session_id: uuid.UUID,
    *,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
    link_type: str,
    closed_at_turn: int,
    refresh_link_context_sources: set[uuid.UUID] | None = None,
) -> list[models.LinkModel]:
    rows = list(
        db.execute(
            select(models.LinkModel)
            .where(
                models.LinkModel.session_id == session_id,
                models.LinkModel.from_object_id == from_object_id,
                models.LinkModel.to_object_id == to_object_id,
                models.LinkModel.type == link_type,
                models.LinkModel.valid_to_turn.is_(None),
            )
            .order_by(models.LinkModel.created_at.asc())
        ).scalars().all()
    )
    if not rows:
        return []
    _close_link_rows(
        rows,
        closed_at_turn=closed_at_turn,
        refresh_link_context_sources=refresh_link_context_sources,
    )
    return rows


def _close_other_active_carried_by_links(
    db: Session,
    session_id: uuid.UUID,
    *,
    item_object_id: uuid.UUID,
    new_owner_object_id: uuid.UUID,
    closed_at_turn: int,
    refresh_link_context_sources: set[uuid.UUID] | None = None,
) -> list[models.LinkModel]:
    rows = list(
        db.execute(
            select(models.LinkModel)
            .where(
                models.LinkModel.session_id == session_id,
                models.LinkModel.from_object_id == item_object_id,
                models.LinkModel.type == "carried_by",
                models.LinkModel.valid_to_turn.is_(None),
                models.LinkModel.to_object_id != new_owner_object_id,
            )
            .order_by(models.LinkModel.created_at.asc())
        ).scalars().all()
    )
    if not rows:
        return []
    _close_link_rows(
        rows,
        closed_at_turn=closed_at_turn,
        refresh_link_context_sources=refresh_link_context_sources,
    )
    return rows


def _close_opposite_social_links(
    db: Session,
    session_id: uuid.UUID,
    *,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
    link_type: str,
    bidirectional: bool,
    closed_at_turn: int,
    refresh_link_context_sources: set[uuid.UUID] | None = None,
) -> list[models.LinkModel]:
    if link_type in REACTION_SUPPORT_LINK_TYPES:
        opposite_types = REACTION_CONFLICT_LINK_TYPES
    elif link_type in REACTION_CONFLICT_LINK_TYPES:
        opposite_types = REACTION_SUPPORT_LINK_TYPES
    else:
        return []

    closed_rows: list[models.LinkModel] = []
    for opposite_type in sorted(opposite_types):
        closed_rows.extend(
            _close_active_links_for_edge(
                db,
                session_id,
                from_object_id=from_object_id,
                to_object_id=to_object_id,
                link_type=opposite_type,
                closed_at_turn=closed_at_turn,
                refresh_link_context_sources=refresh_link_context_sources,
            )
        )
        if bidirectional and from_object_id != to_object_id:
            closed_rows.extend(
                _close_active_links_for_edge(
                    db,
                    session_id,
                    from_object_id=to_object_id,
                    to_object_id=from_object_id,
                    link_type=opposite_type,
                    closed_at_turn=closed_at_turn,
                    refresh_link_context_sources=refresh_link_context_sources,
                )
            )
    return closed_rows


def _refresh_link_context_embeddings_for_sources(
    db: Session,
    session_id: uuid.UUID,
    *,
    source_object_ids: set[uuid.UUID],
) -> None:
    if not USE_EMBEDDINGS or not source_object_ids:
        return
    for from_object_id in sorted(source_object_ids, key=str):
        try:
            _refresh_link_context_embedding(
                db=db,
                session_id=session_id,
                from_object_id=from_object_id,
            )
        except TurnApplyExternalPreparationRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to refresh link_context embedding for session_id=%s from=%s",
                session_id,
                from_object_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="embedding write failed: link_context",
            ) from exc


def apply_patch_ops(
    db: Session,
    session_id: uuid.UUID,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
    ops: list[schemas.PatchOp],
) -> PatchApplyResult:
    ref_map: dict[str, uuid.UUID] = {}
    applied_ops: list[dict[str, Any]] = []
    applied_input_count = 0
    touched_object_ids: set[uuid.UUID] = set()
    prepared_create_chunk_by_index: dict[int, PreparedObjectCreateOp] = {}
    prepared_create_chunk_end = 0
    player_object_id: uuid.UUID | None = None
    _assert_quest_reopen_tracking_contract(
        db,
        session_id,
        ops=ops,
    )
    existing_world_constitution_id = db.execute(
        select(models.ObjectModel.object_id)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "world_constitution",
        )
        .order_by(models.ObjectModel.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    for op_index, op in enumerate(ops):
        op_applied_ops_start = len(applied_ops)
        if isinstance(op, schemas.ObjectCreateOp):
            if op.type == "world_constitution" and existing_world_constitution_id is not None:
                logger.warning(
                    "Merging duplicate world_constitution create in session_id=%s turn=%s ref=%s",
                    session_id,
                    new_turn,
                    op.ref,
                )
                ref_map[op.ref] = existing_world_constitution_id
                existing_row = _require_object(db, session_id, existing_world_constitution_id)
                old_name = str(getattr(existing_row, "name", "") or "")
                old_data = dict(existing_row.data or {})
                merged_data = dict(existing_row.data or {})
                for k, v in (op.data or {}).items():
                    if k == "structural_triggers" and isinstance(v, list):
                        existing_triggers = merged_data.get(k, [])
                        if isinstance(existing_triggers, list):
                            merged_triggers = list(existing_triggers)
                            for trigger in v:
                                if trigger not in merged_triggers:
                                    merged_triggers.append(trigger)
                            merged_data[k] = merged_triggers
                            continue
                    merged_data[k] = v
                merged_data = _touch_ctx_metadata(merged_data, new_turn=new_turn)
                existing_row.data = merged_data
                touched_object_ids.add(existing_world_constitution_id)
                applied_object_update = _build_applied_object_update_op(
                    object_row=existing_row,
                    old_name=old_name,
                    old_data=old_data,
                    new_name=str(getattr(existing_row, "name", "") or ""),
                    new_data=merged_data,
                )
                if applied_object_update is not None:
                    applied_ops.append(applied_object_update)
                    _add_patch_object_updated_event(
                        db,
                        session_id=session_id,
                        turn_index=new_turn,
                        object_row=existing_row,
                        patch_data=dict(applied_object_update.get("patch") or {}),
                        fallback_scope_id=None,
                        in_game_day=in_game_day,
                        in_game_minute=in_game_minute,
                    )
                if len(applied_ops) > op_applied_ops_start:
                    applied_input_count += 1
                continue

            if op_index >= prepared_create_chunk_end:
                prepared_create_chunk_by_index, prepared_create_chunk_end = _prepare_object_create_chunk(
                    db=db,
                    session_id=session_id,
                    new_turn=new_turn,
                    ops=ops,
                    start_index=op_index,
                )

            prepared_create = prepared_create_chunk_by_index.get(op_index)
            if op.ref in ref_map:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Duplicate temporary ref: {op.ref}",
                )

            object_data: dict[str, Any] = (
                dict(prepared_create.object_data)
                if prepared_create is not None
                else dict(op.data or {})
            )
            object_data = _touch_ctx_metadata(object_data, new_turn=new_turn)
            current_zone_id = (
                prepared_create.current_zone_id
                if prepared_create is not None
                else _get_player_current_zone_id(db, session_id)
            )
            npc_profile_text: str | None = prepared_create.npc_profile_text if prepared_create is not None else None
            npc_profile_embedding: list[float] | None = (
                prepared_create.npc_profile_embedding if prepared_create is not None else None
            )
            npc_dedup_zone_id: uuid.UUID | None = None
            zone_profile_text: str | None = prepared_create.zone_profile_text if prepared_create is not None else None
            zone_profile_embedding: list[float] | None = (
                prepared_create.zone_profile_embedding if prepared_create is not None else None
            )
            zone_dedup_current_zone_id: uuid.UUID | None = None
            item_profile_text: str | None = prepared_create.item_profile_text if prepared_create is not None else None
            item_profile_embedding: list[float] | None = (
                prepared_create.item_profile_embedding if prepared_create is not None else None
            )
            faction_profile_text: str | None = (
                prepared_create.faction_profile_text if prepared_create is not None else None
            )
            faction_profile_embedding: list[float] | None = (
                prepared_create.faction_profile_embedding if prepared_create is not None else None
            )
            quest_profile_text: str | None = prepared_create.quest_profile_text if prepared_create is not None else None
            quest_profile_embedding: list[float] | None = (
                prepared_create.quest_profile_embedding if prepared_create is not None else None
            )
            player_profile_text: str | None = (
                prepared_create.player_profile_text if prepared_create is not None else None
            )
            player_profile_embedding: list[float] | None = (
                prepared_create.player_profile_embedding if prepared_create is not None else None
            )
            is_npc = op.type == "npc"
            is_zone = op.type == "zone"
            is_item = op.type == "item"
            is_faction = op.type == "faction"
            is_quest = op.type == "quest"
            is_player = op.type == "player"
            is_ephemeral_npc = is_npc and _is_true(object_data.get("ephemeral"))
            skip_item_dedup = is_item and (
                _is_true(object_data.get("stackable")) or _is_true(object_data.get("unique_instance"))
            )
            if is_npc:
                dropped_keys: list[str] = []
                for key in ("known_claims", "asserted_claims"):
                    if key in object_data:
                        object_data.pop(key, None)
                        dropped_keys.append(key)
                if dropped_keys:
                    logger.warning(
                        "Dropping deprecated npc knowledge arrays on object.create in session_id=%s turn=%s ref=%s keys=%s",
                        session_id,
                        new_turn,
                        op.ref,
                        dropped_keys,
                    )

            if is_npc and USE_EMBEDDINGS and npc_profile_embedding is not None:
                try:
                    npc_dedup_zone_id = current_zone_id
                    if is_ephemeral_npc:
                        existing_npc_id = _find_ephemeral_npc_dedup_candidate(
                            db=db,
                            session_id=session_id,
                            profile_embedding=npc_profile_embedding,
                            zone_id=npc_dedup_zone_id,
                            incoming_profile_text=npc_profile_text,
                        )
                    else:
                        existing_npc_id = _find_persistent_npc_dedup_candidate(
                            db=db,
                            session_id=session_id,
                            profile_embedding=npc_profile_embedding,
                            incoming_profile_text=npc_profile_text,
                        )
                    if existing_npc_id is not None:
                        ref_map[op.ref] = existing_npc_id
                        touched_object_ids.add(existing_npc_id)
                        db.add(
                            models.EventModel(
                                session_id=session_id,
                                turn_index=new_turn,
                                type="NPC_REUSED",
                                scope_object_id=npc_dedup_zone_id,
                                payload={
                                    "npc_id": str(existing_npc_id),
                                    "ref": op.ref,
                                    "in_game_time": {
                                        "day": in_game_day,
                                        "minute": in_game_minute,
                                    },
                                },
                            )
                        )
                        reused_object_row = _require_object(db, session_id, existing_npc_id)
                        if is_ephemeral_npc:
                            object_data = _prepare_reused_ephemeral_npc_patch_data(
                                existing_data=dict(getattr(reused_object_row, "data", None) or {}),
                                patch_data=object_data,
                                new_turn=new_turn,
                                current_zone_id=current_zone_id,
                            )
                        patch_result = _apply_object_patch_to_row(
                            db=db,
                            session_id=session_id,
                            new_turn=new_turn,
                            in_game_day=in_game_day,
                            in_game_minute=in_game_minute,
                            object_row=reused_object_row,
                            patch_data=object_data,
                            fallback_scope_id=current_zone_id,
                            touched_object_ids=touched_object_ids,
                            player_object_id=player_object_id,
                            allow_name_update=False,
                        )
                        player_object_id = patch_result.player_object_id
                        applied_ops.extend(patch_result.applied_ops)
                        if len(applied_ops) > op_applied_ops_start:
                            applied_input_count += 1
                        continue
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    _raise_dedup_probe_failure(
                        object_type="npc",
                        session_id=session_id,
                        ref=op.ref,
                        exc=exc,
                    )

            if is_zone and USE_EMBEDDINGS and zone_profile_embedding is not None:
                try:
                    zone_dedup_current_zone_id = current_zone_id
                    existing_zone_id = _find_global_zone_dedup_candidate(
                        db=db,
                        session_id=session_id,
                        profile_embedding=zone_profile_embedding,
                        incoming_profile_text=zone_profile_text,
                    )
                    if existing_zone_id is not None:
                        ref_map[op.ref] = existing_zone_id
                        touched_object_ids.add(existing_zone_id)
                        db.add(
                            models.EventModel(
                                session_id=session_id,
                                turn_index=new_turn,
                                type="ZONE_REUSED",
                                scope_object_id=zone_dedup_current_zone_id,
                                payload={
                                    "zone_id": str(existing_zone_id),
                                    "ref": op.ref,
                                    "in_game_time": {
                                        "day": in_game_day,
                                        "minute": in_game_minute,
                                    },
                                },
                            )
                        )
                        reused_object_row = _require_object(db, session_id, existing_zone_id)
                        patch_result = _apply_object_patch_to_row(
                            db=db,
                            session_id=session_id,
                            new_turn=new_turn,
                            in_game_day=in_game_day,
                            in_game_minute=in_game_minute,
                            object_row=reused_object_row,
                            patch_data=object_data,
                            fallback_scope_id=current_zone_id,
                            touched_object_ids=touched_object_ids,
                            player_object_id=player_object_id,
                            allow_name_update=False,
                        )
                        player_object_id = patch_result.player_object_id
                        applied_ops.extend(patch_result.applied_ops)
                        if len(applied_ops) > op_applied_ops_start:
                            applied_input_count += 1
                        continue
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    _raise_dedup_probe_failure(
                        object_type="zone",
                        session_id=session_id,
                        ref=op.ref,
                        exc=exc,
                    )

            if is_item and USE_EMBEDDINGS and item_profile_embedding is not None and not skip_item_dedup:
                try:
                    existing_item_id = _find_global_object_dedup_candidate(
                        db=db,
                        session_id=session_id,
                        profile_embedding=item_profile_embedding,
                        object_type="item",
                        namespace="item_profile",
                        threshold=ITEM_DEDUP_THRESHOLD,
                        incoming_profile_text=item_profile_text,
                    )
                    if existing_item_id is not None:
                        ref_map[op.ref] = existing_item_id
                        touched_object_ids.add(existing_item_id)
                        db.add(
                            models.EventModel(
                                session_id=session_id,
                                turn_index=new_turn,
                                type="ITEM_REUSED",
                                scope_object_id=current_zone_id,
                                payload={
                                    "item_id": str(existing_item_id),
                                    "ref": op.ref,
                                    "in_game_time": {"day": in_game_day, "minute": in_game_minute},
                                },
                            )
                        )
                        reused_object_row = _require_object(db, session_id, existing_item_id)
                        patch_result = _apply_object_patch_to_row(
                            db=db,
                            session_id=session_id,
                            new_turn=new_turn,
                            in_game_day=in_game_day,
                            in_game_minute=in_game_minute,
                            object_row=reused_object_row,
                            patch_data=object_data,
                            fallback_scope_id=current_zone_id,
                            touched_object_ids=touched_object_ids,
                            player_object_id=player_object_id,
                            allow_name_update=False,
                        )
                        player_object_id = patch_result.player_object_id
                        applied_ops.extend(patch_result.applied_ops)
                        if len(applied_ops) > op_applied_ops_start:
                            applied_input_count += 1
                        continue
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    _raise_dedup_probe_failure(
                        object_type="item",
                        session_id=session_id,
                        ref=op.ref,
                        exc=exc,
                    )

            if is_faction and USE_EMBEDDINGS and faction_profile_embedding is not None:
                try:
                    existing_faction_id = _find_global_object_dedup_candidate(
                        db=db,
                        session_id=session_id,
                        profile_embedding=faction_profile_embedding,
                        object_type="faction",
                        namespace="faction_profile",
                        threshold=FACTION_DEDUP_THRESHOLD,
                        incoming_profile_text=faction_profile_text,
                    )
                    if existing_faction_id is not None:
                        ref_map[op.ref] = existing_faction_id
                        touched_object_ids.add(existing_faction_id)
                        db.add(
                            models.EventModel(
                                session_id=session_id,
                                turn_index=new_turn,
                                type="FACTION_REUSED",
                                scope_object_id=current_zone_id,
                                payload={
                                    "faction_id": str(existing_faction_id),
                                    "ref": op.ref,
                                    "in_game_time": {"day": in_game_day, "minute": in_game_minute},
                                },
                            )
                        )
                        reused_object_row = _require_object(db, session_id, existing_faction_id)
                        patch_result = _apply_object_patch_to_row(
                            db=db,
                            session_id=session_id,
                            new_turn=new_turn,
                            in_game_day=in_game_day,
                            in_game_minute=in_game_minute,
                            object_row=reused_object_row,
                            patch_data=object_data,
                            fallback_scope_id=current_zone_id,
                            touched_object_ids=touched_object_ids,
                            player_object_id=player_object_id,
                            allow_name_update=False,
                        )
                        player_object_id = patch_result.player_object_id
                        applied_ops.extend(patch_result.applied_ops)
                        if len(applied_ops) > op_applied_ops_start:
                            applied_input_count += 1
                        continue
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    _raise_dedup_probe_failure(
                        object_type="faction",
                        session_id=session_id,
                        ref=op.ref,
                        exc=exc,
                    )

            if is_quest and USE_EMBEDDINGS and quest_profile_embedding is not None:
                try:
                    existing_quest_id = _find_global_object_dedup_candidate(
                        db=db,
                        session_id=session_id,
                        profile_embedding=quest_profile_embedding,
                        object_type="quest",
                        namespace="quest_profile",
                        threshold=QUEST_DEDUP_THRESHOLD,
                        incoming_profile_text=quest_profile_text,
                    )
                    if existing_quest_id is not None:
                        ref_map[op.ref] = existing_quest_id
                        touched_object_ids.add(existing_quest_id)
                        db.add(
                            models.EventModel(
                                session_id=session_id,
                                turn_index=new_turn,
                                type="QUEST_REUSED",
                                scope_object_id=current_zone_id,
                                payload={
                                    "quest_id": str(existing_quest_id),
                                    "ref": op.ref,
                                    "in_game_time": {"day": in_game_day, "minute": in_game_minute},
                                },
                            )
                        )
                        reused_object_row = _require_object(db, session_id, existing_quest_id)
                        patch_result = _apply_object_patch_to_row(
                            db=db,
                            session_id=session_id,
                            new_turn=new_turn,
                            in_game_day=in_game_day,
                            in_game_minute=in_game_minute,
                            object_row=reused_object_row,
                            patch_data=object_data,
                            fallback_scope_id=current_zone_id,
                            touched_object_ids=touched_object_ids,
                            player_object_id=player_object_id,
                            allow_name_update=False,
                        )
                        player_object_id = patch_result.player_object_id
                        applied_ops.extend(patch_result.applied_ops)
                        if len(applied_ops) > op_applied_ops_start:
                            applied_input_count += 1
                        continue
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    _raise_dedup_probe_failure(
                        object_type="quest",
                        session_id=session_id,
                        ref=op.ref,
                        exc=exc,
                    )

            object_row = models.ObjectModel(
                session_id=session_id,
                type=op.type,
                name=op.name,
                data=object_data,
            )
            db.add(object_row)
            db.flush()
            ref_map[op.ref] = object_row.object_id
            touched_object_ids.add(object_row.object_id)
            if op.type == "world_constitution":
                existing_world_constitution_id = object_row.object_id
            _add_patch_object_created_event(
                db,
                session_id=session_id,
                turn_index=new_turn,
                object_row=object_row,
                object_data=object_data,
                fallback_scope_id=current_zone_id,
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
            )
            applied_ops.append(
                _build_applied_object_create_op(
                    object_row=object_row,
                    object_data=object_data,
                    ref=op.ref,
                )
            )

            if is_npc and USE_EMBEDDINGS:
                try:
                    _upsert_npc_profile_embedding(
                        db=db,
                        session_id=session_id,
                        object_id=object_row.object_id,
                        npc_name=op.name,
                        npc_data=object_data,
                        precomputed_text=npc_profile_text,
                        precomputed_embedding=npc_profile_embedding,
                    )
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to write npc_profile embedding for object_id=%s in session_id=%s",
                        object_row.object_id,
                        session_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="embedding write failed: npc_profile",
                    ) from exc
            if is_zone and USE_EMBEDDINGS:
                try:
                    _upsert_zone_profile_embedding(
                        db=db,
                        session_id=session_id,
                        object_id=object_row.object_id,
                        zone_name=op.name,
                        zone_data=object_data,
                        precomputed_text=zone_profile_text,
                        precomputed_embedding=zone_profile_embedding,
                    )
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to write zone_profile embedding for object_id=%s in session_id=%s",
                        object_row.object_id,
                        session_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="embedding write failed: zone_profile",
                    ) from exc
            if is_item and USE_EMBEDDINGS:
                try:
                    _upsert_item_profile_embedding(
                        db=db,
                        session_id=session_id,
                        object_id=object_row.object_id,
                        item_name=op.name,
                        item_data=object_data,
                        precomputed_text=item_profile_text,
                        precomputed_embedding=item_profile_embedding,
                    )
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to write item_profile embedding for object_id=%s in session_id=%s",
                        object_row.object_id,
                        session_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="embedding write failed: item_profile",
                    ) from exc
            if is_faction and USE_EMBEDDINGS:
                try:
                    _upsert_faction_profile_embedding(
                        db=db,
                        session_id=session_id,
                        object_id=object_row.object_id,
                        faction_name=op.name,
                        faction_data=object_data,
                        precomputed_text=faction_profile_text,
                        precomputed_embedding=faction_profile_embedding,
                    )
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to write faction_profile embedding for object_id=%s in session_id=%s",
                        object_row.object_id,
                        session_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="embedding write failed: faction_profile",
                    ) from exc
            if is_quest and USE_EMBEDDINGS:
                try:
                    _upsert_quest_profile_embedding(
                        db=db,
                        session_id=session_id,
                        object_id=object_row.object_id,
                        quest_name=op.name,
                        quest_data=object_data,
                        precomputed_text=quest_profile_text,
                        precomputed_embedding=quest_profile_embedding,
                    )
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to write quest_profile embedding for object_id=%s in session_id=%s",
                        object_row.object_id,
                        session_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="embedding write failed: quest_profile",
                    ) from exc
            if is_player and USE_EMBEDDINGS:
                try:
                    _upsert_player_profile_embedding(
                        db=db,
                        session_id=session_id,
                        object_id=object_row.object_id,
                        player_name=op.name,
                        player_data=object_data,
                        precomputed_text=player_profile_text,
                        precomputed_embedding=player_profile_embedding,
                    )
                except TurnApplyExternalPreparationRequired:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to write player_profile embedding for object_id=%s in session_id=%s",
                        object_row.object_id,
                        session_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="embedding write failed: player_profile",
                    ) from exc
            if len(applied_ops) > op_applied_ops_start:
                applied_input_count += 1
            continue

        if isinstance(op, schemas.ObjectUpdateOp):
            object_id = _resolve_object_ref(db, session_id, op.object, ref_map)
            object_row = _require_object(db, session_id, object_id)
            patch_data: dict[str, Any] = dict(op.patch or {})
            patch_result = _apply_object_patch_to_row(
                db=db,
                session_id=session_id,
                new_turn=new_turn,
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
                object_row=object_row,
                patch_data=patch_data,
                fallback_scope_id=None,
                touched_object_ids=touched_object_ids,
                player_object_id=player_object_id,
                allow_name_update=True,
            )
            player_object_id = patch_result.player_object_id
            applied_ops.extend(patch_result.applied_ops)
            if len(applied_ops) > op_applied_ops_start:
                applied_input_count += 1
            continue

        if isinstance(op, schemas.LinkCloseOp):
            if op.type == LOCATED_IN_LINK_TYPE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Use player.move op for located_in links",
                )
            if op.type == TRACKING_QUEST_LINK_TYPE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Use quest status transitions to close {TRACKING_QUEST_LINK_TYPE} links",
                )

            from_object_id = _resolve_object_ref(db, session_id, op.from_ref, ref_map)
            to_object_id = _resolve_object_ref(db, session_id, op.to, ref_map)
            touched_object_ids.add(from_object_id)
            touched_object_ids.add(to_object_id)
            if from_object_id == to_object_id:
                logger.warning(
                    "Skipping self-referencing link.close in session_id=%s turn=%s object=%s type=%s",
                    session_id,
                    new_turn,
                    from_object_id,
                    op.type,
                )
                continue

            link_close_sources: set[uuid.UUID] = set()
            forward_closed_rows = _close_active_links_for_edge(
                db,
                session_id,
                from_object_id=from_object_id,
                to_object_id=to_object_id,
                link_type=op.type,
                closed_at_turn=new_turn,
                refresh_link_context_sources=link_close_sources,
            )
            reverse_closed_rows: list[models.LinkModel] = []
            if op.bidirectional:
                reverse_closed_rows = _close_active_links_for_edge(
                    db,
                    session_id,
                    from_object_id=to_object_id,
                    to_object_id=from_object_id,
                    link_type=op.type,
                    closed_at_turn=new_turn,
                    refresh_link_context_sources=link_close_sources,
                )
            closed_count = len(forward_closed_rows) + len(reverse_closed_rows)

            if closed_count == 0:
                logger.info(
                    "Skipping link.close with no active links in session_id=%s turn=%s from=%s to=%s type=%s",
                    session_id,
                    new_turn,
                    from_object_id,
                    to_object_id,
                    op.type,
                )
            else:
                if forward_closed_rows:
                    _add_patch_link_closed_events_for_rows(
                        db,
                        session_id=session_id,
                        turn_index=new_turn,
                        links=forward_closed_rows,
                        in_game_day=in_game_day,
                        in_game_minute=in_game_minute,
                    )
                    applied_ops.extend(_build_applied_link_close_op(link_row) for link_row in forward_closed_rows)
                if reverse_closed_rows:
                    _add_patch_link_closed_events_for_rows(
                        db,
                        session_id=session_id,
                        turn_index=new_turn,
                        links=reverse_closed_rows,
                        in_game_day=in_game_day,
                        in_game_minute=in_game_minute,
                    )
                    applied_ops.extend(_build_applied_link_close_op(link_row) for link_row in reverse_closed_rows)
            _refresh_link_context_embeddings_for_sources(
                db,
                session_id,
                source_object_ids=link_close_sources,
            )
            if len(applied_ops) > op_applied_ops_start:
                applied_input_count += 1
            continue

        if isinstance(op, schemas.LinkCreateOp):
            if op.type == LOCATED_IN_LINK_TYPE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Use player.move op for located_in links",
                )

            from_object_id = _resolve_object_ref(db, session_id, op.from_ref, ref_map)
            to_object_id = _resolve_object_ref(db, session_id, op.to, ref_map)
            touched_object_ids.add(from_object_id)
            touched_object_ids.add(to_object_id)
            if op.type == TRACKING_QUEST_LINK_TYPE:
                if player_object_id is None:
                    player_object_id = _get_session_player_object_id(db, session_id)
                if op.bidirectional:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="tracking_quest links cannot be bidirectional",
                    )
                if from_object_id != player_object_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="tracking_quest link must originate from session player",
                    )
                to_object = _require_object(db, session_id, to_object_id)
                if to_object.type != "quest":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="tracking_quest link target must be a quest object",
                    )

            if from_object_id == to_object_id:
                logger.warning(
                    "Skipping self-referencing link.create in session_id=%s turn=%s object=%s type=%s",
                    session_id,
                    new_turn,
                    from_object_id,
                    op.type,
                )
                continue
            if op.type in {"heard", "asserted"} and not _validate_claim_audience_link_create(
                db,
                session_id,
                new_turn=new_turn,
                link_type=op.type,
                from_object_id=from_object_id,
                to_object_id=to_object_id,
            ):
                continue
            if op.type == "asserted":
                _backfill_claim_location_from_speaker(
                    db,
                    session_id,
                    claim_object_id=to_object_id,
                    speaker_object_id=from_object_id,
                    new_turn=new_turn,
                )
            link_data = dict(op.data or {})
            forward_created = False
            reverse_created = False
            forward_link_row: models.LinkModel | None = None
            reverse_link_row: models.LinkModel | None = None
            link_create_sources: set[uuid.UUID] = set()
            carried_by_closed_rows: list[models.LinkModel] = []
            opposite_closed_rows: list[models.LinkModel] = []

            if op.type == "carried_by":
                carried_by_closed_rows = _close_other_active_carried_by_links(
                    db,
                    session_id,
                    item_object_id=from_object_id,
                    new_owner_object_id=to_object_id,
                    closed_at_turn=new_turn,
                    refresh_link_context_sources=link_create_sources,
                )

            opposite_closed_rows = _close_opposite_social_links(
                db,
                session_id,
                from_object_id=from_object_id,
                to_object_id=to_object_id,
                link_type=op.type,
                bidirectional=op.bidirectional,
                closed_at_turn=new_turn,
                refresh_link_context_sources=link_create_sources,
            )

            active_duplicate = _get_active_link(
                db,
                session_id,
                from_object_id,
                to_object_id,
                op.type,
            )
            if active_duplicate is not None:
                logger.info(
                    "Skipping duplicate active link.create in session_id=%s turn=%s from=%s to=%s type=%s",
                    session_id,
                    new_turn,
                    from_object_id,
                    to_object_id,
                    op.type,
                )
            else:
                forward_link_row = models.LinkModel(
                    session_id=session_id,
                    from_object_id=from_object_id,
                    to_object_id=to_object_id,
                    type=op.type,
                    data=link_data,
                    valid_from_turn=new_turn,
                    valid_to_turn=None,
                )
                db.add(forward_link_row)
                forward_created = True

            if op.bidirectional and from_object_id != to_object_id:
                reverse_duplicate = _get_active_link(
                    db,
                    session_id,
                    to_object_id,
                    from_object_id,
                    op.type,
                )
                if reverse_duplicate is not None:
                    logger.info(
                        "Skipping duplicate reverse link.create in session_id=%s turn=%s from=%s to=%s type=%s",
                        session_id,
                        new_turn,
                        to_object_id,
                        from_object_id,
                        op.type,
                    )
                else:
                    reverse_link_row = models.LinkModel(
                        session_id=session_id,
                        from_object_id=to_object_id,
                        to_object_id=from_object_id,
                        type=op.type,
                        data=link_data,
                        valid_from_turn=new_turn,
                        valid_to_turn=None,
                    )
                    db.add(reverse_link_row)
                    reverse_created = True
            if carried_by_closed_rows:
                _add_patch_link_closed_events_for_rows(
                    db,
                    session_id=session_id,
                    turn_index=new_turn,
                    links=carried_by_closed_rows,
                    in_game_day=in_game_day,
                    in_game_minute=in_game_minute,
                )
                applied_ops.extend(_build_applied_link_close_op(link_row) for link_row in carried_by_closed_rows)
            if opposite_closed_rows:
                _add_patch_link_closed_events_for_rows(
                    db,
                    session_id=session_id,
                    turn_index=new_turn,
                    links=opposite_closed_rows,
                    in_game_day=in_game_day,
                    in_game_minute=in_game_minute,
                )
                applied_ops.extend(_build_applied_link_close_op(link_row) for link_row in opposite_closed_rows)
            if forward_created:
                _add_patch_link_event(
                    db,
                    session_id=session_id,
                    turn_index=new_turn,
                    event_type="LINK_CREATED",
                    from_object_id=from_object_id,
                    to_object_id=to_object_id,
                    link_type=op.type,
                    in_game_day=in_game_day,
                    in_game_minute=in_game_minute,
                    link_data=link_data,
                )
                if forward_link_row is not None:
                    applied_ops.append(_build_applied_link_create_op(forward_link_row))
            if reverse_created:
                _add_patch_link_event(
                    db,
                    session_id=session_id,
                    turn_index=new_turn,
                    event_type="LINK_CREATED",
                    from_object_id=to_object_id,
                    to_object_id=from_object_id,
                    link_type=op.type,
                    in_game_day=in_game_day,
                    in_game_minute=in_game_minute,
                    link_data=link_data,
                )
                if reverse_link_row is not None:
                    applied_ops.append(_build_applied_link_create_op(reverse_link_row))
            if op.type in LINK_CONTENT_TYPES and _extract_link_context_text(link_data):
                if forward_created:
                    link_create_sources.add(from_object_id)
                if reverse_created:
                    link_create_sources.add(to_object_id)
            _refresh_link_context_embeddings_for_sources(
                db,
                session_id,
                source_object_ids=link_create_sources,
            )
            if len(applied_ops) > op_applied_ops_start:
                applied_input_count += 1
            continue

        if isinstance(op, schemas.PlayerMoveOp):
            player_object_id = _resolve_object_ref(db, session_id, op.player, ref_map)
            player_object = _require_object(db, session_id, player_object_id)
            if player_object.type != "player":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="player.move target must reference a player object",
                )

            to_object_id = _resolve_object_ref(db, session_id, op.to, ref_map)
            to_object = _require_object(db, session_id, to_object_id)
            if to_object.type != "zone":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"player.move destination must be a zone, got {to_object.type}",
                )

            active_locations = _close_player_active_located_in_links(
                db,
                session_id,
                player_object_id,
                closed_at_turn=new_turn,
            )
            touched_object_ids.add(player_object_id)
            touched_object_ids.add(to_object_id)
            for active_location in active_locations:
                old_zone_id = getattr(active_location, "to_object_id", None)
                if isinstance(old_zone_id, uuid.UUID):
                    touched_object_ids.add(old_zone_id)
            if not active_locations:
                logger.warning(
                    "Recovering player.move with no active located_in link for session_id=%s player=%s turn=%s",
                    session_id,
                    player_object_id,
                    new_turn,
                )
            previous_zone_id: uuid.UUID | None = None
            for active_location in active_locations:
                candidate_zone_id = getattr(active_location, "to_object_id", None)
                if isinstance(candidate_zone_id, uuid.UUID):
                    previous_zone_id = candidate_zone_id
                    break

            new_location_link = models.LinkModel(
                session_id=session_id,
                from_object_id=player_object_id,
                to_object_id=to_object_id,
                type=LOCATED_IN_LINK_TYPE,
                data={},
                valid_from_turn=new_turn,
                valid_to_turn=None,
            )
            db.add(new_location_link)
            for active_location in active_locations:
                old_zone_id = getattr(active_location, "to_object_id", None)
                if not isinstance(old_zone_id, uuid.UUID):
                    continue
                applied_ops.append(
                    _build_applied_link_close_op(
                        active_location,
                        from_object_id=player_object_id,
                        to_object_id=old_zone_id,
                        link_type=LOCATED_IN_LINK_TYPE,
                    )
                )
                _add_patch_link_event(
                    db,
                    session_id=session_id,
                    turn_index=new_turn,
                    event_type="LINK_CLOSED",
                    from_object_id=player_object_id,
                    to_object_id=old_zone_id,
                    link_type=LOCATED_IN_LINK_TYPE,
                    in_game_day=in_game_day,
                    in_game_minute=in_game_minute,
                )
            _add_patch_link_event(
                db,
                session_id=session_id,
                turn_index=new_turn,
                event_type="LINK_CREATED",
                from_object_id=player_object_id,
                to_object_id=to_object_id,
                link_type=LOCATED_IN_LINK_TYPE,
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
                link_data={},
            )
            applied_ops.append(_build_applied_link_create_op(new_location_link))
            _add_patch_move_event(
                db,
                session_id=session_id,
                turn_index=new_turn,
                player_object_id=player_object_id,
                from_zone_id=previous_zone_id,
                to_zone_id=to_object_id,
                to_zone_name=to_object.name,
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
            )
            if len(applied_ops) > op_applied_ops_start:
                applied_input_count += 1
            continue

        if isinstance(op, schemas.EventCreateOp):
            scope_object_id = (
                _resolve_object_ref(db, session_id, op.scope, ref_map)
                if op.scope is not None
                else None
            )
            if isinstance(scope_object_id, uuid.UUID):
                touched_object_ids.add(scope_object_id)
            payload_with_time = dict(op.payload or {})
            payload_with_time.setdefault(
                "in_game_time",
                {"day": in_game_day, "minute": in_game_minute},
            )
            event_row = models.EventModel(
                session_id=session_id,
                turn_index=new_turn,
                type=op.type,
                scope_object_id=scope_object_id,
                payload=payload_with_time,
            )
            db.add(event_row)
            applied_ops.append(
                _build_applied_event_create_op(
                    event_type=op.type,
                    scope_object_id=scope_object_id,
                    payload=payload_with_time,
                )
            )
            continue

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported patch operation",
        )

    _apply_ctx_touches_for_object_ids(
        db,
        session_id,
        object_ids=touched_object_ids,
        new_turn=new_turn,
    )
    db.flush()
    return PatchApplyResult(
        ref_map,
        applied_ops=applied_ops,
        applied_input_count=applied_input_count,
    )



__all__ = [
    "PatchApplyResult",
    "_find_ephemeral_npc_dedup_candidate",
    "_find_persistent_npc_dedup_candidate",
    "_find_global_object_dedup_candidate",
    "_find_global_zone_dedup_candidate",
    "_prepare_object_create_chunk",
    "apply_patch_ops",
]
