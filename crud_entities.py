from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import logging
import uuid
from typing import Any, Iterable

from fastapi import HTTPException, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, SessionTransactionOrigin

from . import models, schemas
from .constants import (
    ADJACENT_LINK_TYPE,
    LOCATED_IN_LINK_TYPE,
    NPC_OFFSTAGE_STATUS,
    TRACKING_QUEST_LINK_TYPE,
)
from .crud_context import (
    CHRONICLE_INPUT_NAMESPACE,
    CHRONICLE_OUTPUT_NAMESPACE,
    RELEVANCE_QUERY_EMBED_INSTRUCTION,
    _ensure_world_prompt_chunks_indexed,
)
from .crud_embeddings_ops import (
    FACTION_PROFILE_EMBED_INSTRUCTION,
    ITEM_PROFILE_EMBED_INSTRUCTION,
    LINK_CONTENT_TYPES,
    PLAYER_PROFILE_EMBED_INSTRUCTION,
    NPC_PROFILE_EMBED_INSTRUCTION,
    QUEST_PROFILE_EMBED_INSTRUCTION,
    ZONE_PROFILE_EMBED_INSTRUCTION,
    CLAIM_TEXT_EMBED_INSTRUCTION,
    LINK_CONTEXT_EMBED_INSTRUCTION,
    LINK_CONTEXT_NAMESPACE,
    _extract_claim_text,
    _extract_link_context_text,
    _list_active_link_context_snippets,
    _maybe_embed_texts,
    _refresh_link_context_embedding,
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
)
from .crud_shared import (
    DEFAULT_TIME_SCALE_MINUTES,
    _acquire_session_turn_lock,
    _build_session_snapshot_dump,
    _coerce_state_payload,
    _create_internal_turn_row,
    _extract_in_game_time,
    _get_active_located_in_links,
    _get_object,
    _get_active_link,
    _infer_actor_zone_id,
    _get_session_player_object_id,
    _rollback_read_only_autobegin_transaction,
    _recover_abandoned_pending_turn_locked,
    _require_object,
    _require_session,
    _safe_int,
    _truncate_text,
)
from .db import DEFAULT_SPAWN_ZONE_NAME, EMBED_SNIPPET_MAX_CHARS, SessionLocal, USE_EMBEDDINGS

DEFAULT_EPHEMERAL_NPC_TTL = 3
CHRONICLE_EVENT_EMBED_INSTRUCTION = "Represent this game event for retrieval"

logger = logging.getLogger(__name__)
_OBJECT_PROFILE_REFRESH_TYPES = frozenset({"player", "npc", "zone", "item", "faction", "quest"})


@dataclass(frozen=True)
class _ChronicleEmbeddingCandidate:
    zone_id: uuid.UUID | None
    in_game_day: int | None
    in_game_minute: int | None
    text_snippet: str
    text_hash: str


@dataclass(frozen=True)
class TtlCleanupResult:
    cleaned_count: int
    applied_ops: list[dict[str, Any]]


def _coerce_ttl_cleanup_result(value: TtlCleanupResult | int | None) -> TtlCleanupResult:
    if isinstance(value, TtlCleanupResult):
        return value
    coerced_count = _safe_int(value)
    return TtlCleanupResult(
        cleaned_count=max(coerced_count or 0, 0),
        applied_ops=[],
    )


def _enqueue_session_bootstrap_indexing_event(
    db: Session,
    *,
    session_id: uuid.UUID,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_SESSION_BOOTSTRAP_INDEX,
        payload={},
        session_id=session_id,
        turn_index=0,
        trace_id=get_trace_id(),
        dedupe_key=f"session_bootstrap_index:{session_id}",
    )


def _enqueue_player_profile_refresh_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_PLAYER_PROFILE_REFRESH,
        payload={"object_id": str(object_id)},
        session_id=session_id,
        turn_index=None,
        trace_id=get_trace_id(),
        dedupe_key=None,
    )


def _enqueue_object_profile_refresh_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_OBJECT_PROFILE_REFRESH,
        payload={"object_id": str(object_id)},
        session_id=session_id,
        turn_index=None,
        trace_id=get_trace_id(),
        dedupe_key=None,
    )


def _enqueue_link_context_refresh_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_LINK_CONTEXT_REFRESH,
        payload={"from_object_id": str(from_object_id)},
        session_id=session_id,
        turn_index=None,
        trace_id=get_trace_id(),
        dedupe_key=None,
    )


def _enqueue_turn_chronicle_sync_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    dedupe: bool = True,
) -> None:
    from . import crud_continuity as _continuity
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    if USE_EMBEDDINGS:
        _outbox_runtime.enqueue_outbox_event(
            db,
            event_type=_outbox_runtime.EVENT_TURN_CHRONICLE_SYNC,
            payload={},
            session_id=session_id,
            turn_index=turn_index,
            trace_id=get_trace_id(),
            dedupe_key=f"turn_chronicle_sync:{session_id}:{turn_index}" if dedupe else None,
        )
    _continuity._enqueue_turn_memory_sync_event(
        db,
        session_id=session_id,
        turn_index=turn_index,
        dedupe=dedupe,
    )


def _coerce_event_scope_uuid(value: Any) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except (TypeError, ValueError, AttributeError):
        return None


def _resolve_internal_object_event_scope(
    *,
    object_row: models.ObjectModel,
) -> uuid.UUID | None:
    if object_row.type == "zone":
        return object_row.object_id
    object_data = dict(object_row.data or {})
    for key in ("location_id", "zone_id"):
        scope_object_id = _coerce_event_scope_uuid(object_data.get(key))
        if scope_object_id is not None:
            return scope_object_id
    return None


def _resolve_internal_link_event_scope(
    db: Session,
    *,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
    from_object: models.ObjectModel | Any | None,
    to_object: models.ObjectModel | Any | None,
) -> uuid.UUID | None:
    if from_object is not None and getattr(from_object, "type", None) == "zone":
        return from_object_id

    if to_object is not None and getattr(to_object, "type", None) == "zone":
        return to_object_id

    return _infer_actor_zone_id(db, session_id, from_object_id)


def _add_internal_object_created_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    object_row: models.ObjectModel,
    object_data: dict[str, Any] | None = None,
    scope_object_id: uuid.UUID | None = None,
) -> None:
    payload_data = dict(object_data if isinstance(object_data, dict) else (getattr(object_row, "data", {}) or {}))
    db.add(
        models.EventModel(
            session_id=session_id,
            turn_index=turn_index,
            type="OBJECT_CREATED",
            scope_object_id=scope_object_id
            if isinstance(scope_object_id, uuid.UUID)
            else _resolve_internal_object_event_scope(object_row=object_row),
            payload={
                "object_id": str(object_row.object_id),
                "object_type": str(getattr(object_row, "type", "") or ""),
                "name": str(getattr(object_row, "name", "") or ""),
                "data": payload_data,
            },
        )
    )


def _add_internal_object_updated_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    object_row: models.ObjectModel,
    patch_data: dict[str, Any],
    scope_object_id: uuid.UUID | None = None,
) -> None:
    payload_patch = dict(patch_data or {})
    db.add(
        models.EventModel(
            session_id=session_id,
            turn_index=turn_index,
            type="OBJECT_UPDATED",
            scope_object_id=scope_object_id
            if isinstance(scope_object_id, uuid.UUID)
            else _resolve_internal_object_event_scope(object_row=object_row),
            payload={
                "object_id": str(object_row.object_id),
                "object_type": str(getattr(object_row, "type", "") or ""),
                "name": str(getattr(object_row, "name", "") or ""),
                "patch": payload_patch,
            },
        )
    )


def _add_internal_link_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    event_type: str,
    from_object_id: uuid.UUID,
    to_object_id: uuid.UUID,
    link_type: str,
    from_object: models.ObjectModel | Any | None,
    to_object: models.ObjectModel | Any | None,
    link_data: dict[str, Any] | None = None,
    valid_from_turn: int | None = None,
    valid_to_turn: int | None = None,
    source: str | None = None,
    in_game_day: int | None = None,
    in_game_minute: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "from_object_id": str(from_object_id),
        "to_object_id": str(to_object_id),
        "type": str(link_type or ""),
    }
    if link_data is not None:
        payload["data"] = dict(link_data)
    if valid_from_turn is not None:
        payload["valid_from_turn"] = max(int(valid_from_turn), 0)
    if valid_to_turn is not None or event_type == "LINK_CREATED":
        payload["valid_to_turn"] = max(int(valid_to_turn), 0) if valid_to_turn is not None else None
    if source is not None:
        payload["source"] = str(source)
    if in_game_day is not None and in_game_minute is not None:
        payload["in_game_time"] = {"day": int(in_game_day), "minute": int(in_game_minute)}

    db.add(
        models.EventModel(
            session_id=session_id,
            turn_index=turn_index,
            type=event_type,
            scope_object_id=_resolve_internal_link_event_scope(
                db,
                session_id=session_id,
                from_object_id=from_object_id,
                to_object_id=to_object_id,
                from_object=from_object,
                to_object=to_object,
            ),
            payload=payload,
        )
    )


def _enqueue_zone_profile_refresh_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_ZONE_PROFILE_REFRESH,
        payload={"object_id": str(object_id)},
        session_id=session_id,
        turn_index=None,
        trace_id=get_trace_id(),
        dedupe_key=None,
    )


def _enqueue_claim_text_refresh_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_CLAIM_TEXT_REFRESH,
        payload={"object_id": str(object_id)},
        session_id=session_id,
        turn_index=None,
        trace_id=get_trace_id(),
        dedupe_key=None,
    )


def _enqueue_chronicle_append_event(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    namespace: str,
    snippet_text: str,
    zone_id: uuid.UUID | None,
    in_game_day: int | None,
    in_game_minute: int | None,
    append_to_existing: bool = False,
) -> None:
    from . import outbox_runtime as _outbox_runtime
    from .observability import get_trace_id

    namespace_value = str(namespace or "").strip()
    if namespace_value not in {CHRONICLE_OUTPUT_NAMESPACE, CHRONICLE_INPUT_NAMESPACE}:
        raise ValueError(f"Unsupported chronicle namespace: {namespace_value}")

    snippet_value = str(snippet_text or "").strip()
    if not snippet_value:
        return

    _outbox_runtime.enqueue_outbox_event(
        db,
        event_type=_outbox_runtime.EVENT_CHRONICLE_APPEND,
        payload={
            "namespace": namespace_value,
            "snippet_text": snippet_value,
            "zone_id": str(zone_id) if zone_id is not None else None,
            "in_game_day": in_game_day,
            "in_game_minute": in_game_minute,
            "append_to_existing": bool(append_to_existing),
        },
        session_id=session_id,
        turn_index=turn_index,
        trace_id=get_trace_id(),
        dedupe_key=None,
    )


def _run_session_bootstrap_indexing_outbox_event(
    *,
    session_id: uuid.UUID,
) -> None:
    if not USE_EMBEDDINGS:
        return

    try:
        player_object_id: uuid.UUID | None = None
        player_name = ""
        player_data: dict[str, Any] = {}

        read_db = SessionLocal()
        try:
            try:
                _require_session(read_db, session_id)
            except HTTPException as exc:
                if exc.status_code == status.HTTP_404_NOT_FOUND:
                    return
                raise

            try:
                player_object_id = _get_session_player_object_id(read_db, session_id)
            except HTTPException:
                player_object_id = None
            if player_object_id is not None:
                player_row = _get_object(read_db, session_id, player_object_id)
                if player_row is not None and getattr(player_row, "type", None) == "player":
                    player_name = str(getattr(player_row, "name", "") or "")
                    player_data = dict(getattr(player_row, "data", {}) or {})
        finally:
            if read_db.in_transaction():
                read_db.rollback()
            read_db.close()

        if player_object_id is not None:
            player_profile_text = _build_player_profile_text(player_name, player_data)
            player_vectors = _maybe_embed_texts(
                [player_profile_text],
                instruction=PLAYER_PROFILE_EMBED_INSTRUCTION,
            )
            if len(player_vectors) != 1:
                raise RuntimeError(
                    f"player bootstrap embedding size mismatch: got {len(player_vectors)}, expected 1"
                )

            write_db = SessionLocal()
            try:
                with write_db.begin():
                    current_session = _require_session(write_db, session_id)
                    current_player_id = _get_session_player_object_id(write_db, session_id)
                    if current_player_id != player_object_id:
                        logger.info(
                            "Session bootstrap player embedding skipped because player_object_id changed for session_id=%s",
                            session_id,
                        )
                    else:
                        current_player = _get_object(write_db, session_id, player_object_id)
                        current_player_name = str(getattr(current_player, "name", "") or "")
                        current_player_data = dict(getattr(current_player, "data", {}) or {})
                        if (
                            current_player_name != player_name
                            or current_player_data != player_data
                            or getattr(current_player, "type", None) != "player"
                        ):
                            logger.info(
                                "Session bootstrap player embedding skipped because player state changed for session_id=%s",
                                getattr(current_session, "id", session_id),
                            )
                        else:
                            _upsert_player_profile_embedding(
                                db=write_db,
                                session_id=session_id,
                                object_id=player_object_id,
                                player_name=player_name,
                                player_data=player_data,
                                precomputed_text=player_profile_text,
                                precomputed_embedding=player_vectors[0],
                            )
            finally:
                write_db.close()

        world_db = SessionLocal()
        try:
            try:
                session_row = _require_session(world_db, session_id)
            except HTTPException as exc:
                if exc.status_code == status.HTTP_404_NOT_FOUND:
                    return
                raise
            world_prompt = str(getattr(session_row, "world_prompt", "") or "").strip()
            if not world_prompt:
                return

            if world_db.in_transaction():
                world_db.rollback()

            _ensure_world_prompt_chunks_indexed(
                db=world_db,
                session_id=session_id,
                world_prompt=world_prompt,
            )
        finally:
            if world_db.in_transaction():
                world_db.rollback()
            world_db.close()
    except Exception:  # noqa: BLE001
        logger.warning(
            "Session bootstrap indexing failed for session_id=%s",
            session_id,
            exc_info=True,
        )
        raise


def _run_player_profile_refresh_outbox_event(
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    if not USE_EMBEDDINGS:
        return

    _run_object_profile_refresh_outbox_event(
        session_id=session_id,
        object_id=object_id,
        expected_type="player",
        instruction=PLAYER_PROFILE_EMBED_INSTRUCTION,
        build_text=lambda name, data: _build_player_profile_text(name, data),
        apply_embedding=lambda db, name, data, text, embedding: _upsert_player_profile_embedding(
            db=db,
            session_id=session_id,
            object_id=object_id,
            player_name=name,
            player_data=data,
            precomputed_text=text,
            precomputed_embedding=embedding,
        ),
        failure_log="Player profile refresh failed for session_id=%s object_id=%s",
    )


def _run_link_context_refresh_outbox_event(
    *,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
) -> None:
    if not USE_EMBEDDINGS:
        return

    try:
        while True:
            prepared_text = _read_link_context_refresh_text(
                session_id=session_id,
                from_object_id=from_object_id,
            )
            embedding: list[float] | None = None
            if prepared_text:
                embedding = _maybe_embed_texts(
                    [prepared_text],
                    instruction=LINK_CONTEXT_EMBED_INSTRUCTION,
                )[0]

            write_db = SessionLocal()
            try:
                with write_db.begin():
                    try:
                        _require_session(write_db, session_id)
                    except HTTPException as exc:
                        if exc.status_code == status.HTTP_404_NOT_FOUND:
                            return
                        raise

                    current_text = _build_link_context_refresh_text(
                        db=write_db,
                        session_id=session_id,
                        from_object_id=from_object_id,
                    )
                    if current_text != prepared_text:
                        continue

                    _refresh_link_context_embedding(
                        write_db,
                        session_id,
                        from_object_id=from_object_id,
                        precomputed_text=prepared_text,
                        precomputed_embedding=embedding,
                    )
                    return
            finally:
                write_db.close()
    except Exception:  # noqa: BLE001
        logger.warning(
            "Link context refresh failed for session_id=%s from_object_id=%s",
            session_id,
            from_object_id,
            exc_info=True,
        )
        raise


def _run_zone_profile_refresh_outbox_event(
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    if not USE_EMBEDDINGS:
        return

    _run_object_profile_refresh_outbox_event(
        session_id=session_id,
        object_id=object_id,
        expected_type="zone",
        instruction=ZONE_PROFILE_EMBED_INSTRUCTION,
        build_text=lambda name, data: _build_zone_profile_text(name, data),
        apply_embedding=lambda db, name, data, text, embedding: _upsert_zone_profile_embedding(
            db=db,
            session_id=session_id,
            object_id=object_id,
            zone_name=name,
            zone_data=data,
            precomputed_text=text,
            precomputed_embedding=embedding,
        ),
        failure_log="Zone profile refresh failed for session_id=%s object_id=%s",
    )


def _run_claim_text_refresh_outbox_event(
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    if not USE_EMBEDDINGS:
        return

    _run_object_profile_refresh_outbox_event(
        session_id=session_id,
        object_id=object_id,
        expected_type="claim",
        instruction=CLAIM_TEXT_EMBED_INSTRUCTION,
        build_text=lambda _name, data: _extract_claim_text(data),
        apply_embedding=lambda db, _name, data, text, embedding: _upsert_claim_text_embedding(
            db=db,
            session_id=session_id,
            object_id=object_id,
            claim_data=data,
            precomputed_text=text,
            precomputed_embedding=embedding,
        ),
        failure_log="Claim text refresh failed for session_id=%s object_id=%s",
    )


def _run_object_profile_refresh_outbox_event_by_object_id(
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
) -> None:
    read_db = SessionLocal()
    try:
        try:
            _require_session(read_db, session_id)
        except HTTPException as exc:
            if exc.status_code == status.HTTP_404_NOT_FOUND:
                return
            raise

        object_row = _get_object(read_db, session_id, object_id)
        object_type = str(getattr(object_row, "type", "") or "").strip()
    finally:
        if read_db.in_transaction():
            read_db.rollback()
        read_db.close()

    if object_type == "player":
        _run_player_profile_refresh_outbox_event(session_id=session_id, object_id=object_id)
        return
    if object_type == "zone":
        _run_zone_profile_refresh_outbox_event(session_id=session_id, object_id=object_id)
        return
    if object_type == "npc":
        _run_object_profile_refresh_outbox_event(
            session_id=session_id,
            object_id=object_id,
            expected_type="npc",
            instruction=NPC_PROFILE_EMBED_INSTRUCTION,
            build_text=lambda name, data: _build_npc_profile_text(name, data),
            apply_embedding=lambda db, name, data, text, embedding: _upsert_npc_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_id,
                npc_name=name,
                npc_data=data,
                precomputed_text=text,
                precomputed_embedding=embedding,
            ),
            failure_log="NPC profile refresh failed for session_id=%s object_id=%s",
        )
        return
    if object_type == "item":
        _run_object_profile_refresh_outbox_event(
            session_id=session_id,
            object_id=object_id,
            expected_type="item",
            instruction=ITEM_PROFILE_EMBED_INSTRUCTION,
            build_text=lambda name, data: _build_item_profile_text(name, data),
            apply_embedding=lambda db, name, data, text, embedding: _upsert_item_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_id,
                item_name=name,
                item_data=data,
                precomputed_text=text,
                precomputed_embedding=embedding,
            ),
            failure_log="Item profile refresh failed for session_id=%s object_id=%s",
        )
        return
    if object_type == "faction":
        _run_object_profile_refresh_outbox_event(
            session_id=session_id,
            object_id=object_id,
            expected_type="faction",
            instruction=FACTION_PROFILE_EMBED_INSTRUCTION,
            build_text=lambda name, data: _build_faction_profile_text(name, data),
            apply_embedding=lambda db, name, data, text, embedding: _upsert_faction_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_id,
                faction_name=name,
                faction_data=data,
                precomputed_text=text,
                precomputed_embedding=embedding,
            ),
            failure_log="Faction profile refresh failed for session_id=%s object_id=%s",
        )
        return
    if object_type == "quest":
        _run_object_profile_refresh_outbox_event(
            session_id=session_id,
            object_id=object_id,
            expected_type="quest",
            instruction=QUEST_PROFILE_EMBED_INSTRUCTION,
            build_text=lambda name, data: _build_quest_profile_text(name, data),
            apply_embedding=lambda db, name, data, text, embedding: _upsert_quest_profile_embedding(
                db=db,
                session_id=session_id,
                object_id=object_id,
                quest_name=name,
                quest_data=data,
                precomputed_text=text,
                precomputed_embedding=embedding,
            ),
            failure_log="Quest profile refresh failed for session_id=%s object_id=%s",
        )
        return


def _run_chronicle_append_outbox_event(
    *,
    session_id: uuid.UUID,
    turn_index: int,
    namespace: str,
    snippet_text: str,
    zone_id: uuid.UUID | None,
    in_game_day: int | None,
    in_game_minute: int | None,
    append_to_existing: bool,
) -> None:
    if not USE_EMBEDDINGS:
        return

    write_db = SessionLocal()
    try:
        try:
            _require_session(write_db, session_id)
        except HTTPException as exc:
            if exc.status_code == status.HTTP_404_NOT_FOUND:
                return
            raise

        if write_db.get(models.TurnModel, (session_id, turn_index)) is None:
            return

        if write_db.in_transaction():
            write_db.rollback()

        index_turn_embedding(
            db=write_db,
            session_id=session_id,
            turn_index=turn_index,
            zone_id=zone_id,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
            snippet_text=snippet_text,
            append_to_existing=append_to_existing,
            namespace=namespace,
        )
    finally:
        if write_db.in_transaction():
            write_db.rollback()
        write_db.close()


def _read_object_refresh_snapshot(
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    expected_type: str,
) -> tuple[str, dict[str, Any]] | None:
    read_db = SessionLocal()
    try:
        try:
            _require_session(read_db, session_id)
        except HTTPException as exc:
            if exc.status_code == status.HTTP_404_NOT_FOUND:
                return None
            raise

        object_row = _get_object(read_db, session_id, object_id)
        if object_row is None or getattr(object_row, "type", None) != expected_type:
            return None
        return (
            str(getattr(object_row, "name", "") or ""),
            dict(getattr(object_row, "data", {}) or {}),
        )
    finally:
        if read_db.in_transaction():
            read_db.rollback()
        read_db.close()


def _run_object_profile_refresh_outbox_event(
    *,
    session_id: uuid.UUID,
    object_id: uuid.UUID,
    expected_type: str,
    instruction: str,
    build_text: Any,
    apply_embedding: Any,
    failure_log: str,
) -> None:
    try:
        while True:
            snapshot = _read_object_refresh_snapshot(
                session_id=session_id,
                object_id=object_id,
                expected_type=expected_type,
            )
            if snapshot is None:
                return
            object_name, object_data = snapshot
            prepared_text = str(build_text(object_name, object_data) or "").strip()
            if not prepared_text:
                return
            prepared_embedding = _maybe_embed_texts(
                [prepared_text],
                instruction=instruction,
            )[0]

            write_db = SessionLocal()
            try:
                with write_db.begin():
                    try:
                        _require_session(write_db, session_id)
                    except HTTPException as exc:
                        if exc.status_code == status.HTTP_404_NOT_FOUND:
                            return
                        raise

                    current_row = _get_object(write_db, session_id, object_id)
                    if current_row is None or getattr(current_row, "type", None) != expected_type:
                        return
                    current_name = str(getattr(current_row, "name", "") or "")
                    current_data = dict(getattr(current_row, "data", {}) or {})
                    current_text = str(build_text(current_name, current_data) or "").strip()
                    if current_text != prepared_text:
                        continue

                    apply_embedding(
                        write_db,
                        current_name,
                        current_data,
                        prepared_text,
                        prepared_embedding,
                    )
                    return
            finally:
                write_db.close()
    except Exception:  # noqa: BLE001
        logger.warning(
            failure_log,
            session_id,
            object_id,
            exc_info=True,
        )
        raise


def _build_link_context_refresh_text(
    *,
    db: Session,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
) -> str | None:
    from_object = db.get(models.ObjectModel, (session_id, from_object_id))
    if from_object is None:
        return None
    snippets = _list_active_link_context_snippets(
        db=db,
        session_id=session_id,
        from_object_id=from_object_id,
        from_name=from_object.name,
    )
    if not snippets:
        return ""
    return _truncate_text("\n".join(snippets), EMBED_SNIPPET_MAX_CHARS)


def _read_link_context_refresh_text(
    *,
    session_id: uuid.UUID,
    from_object_id: uuid.UUID,
) -> str | None:
    read_db = SessionLocal()
    try:
        try:
            _require_session(read_db, session_id)
        except HTTPException as exc:
            if exc.status_code == status.HTTP_404_NOT_FOUND:
                return None
            raise
        return _build_link_context_refresh_text(
            db=read_db,
            session_id=session_id,
            from_object_id=from_object_id,
        )
    finally:
        if read_db.in_transaction():
            read_db.rollback()
        read_db.close()


def _cleanup_ephemeral_npcs(
    db: Session,
    session_id: uuid.UUID,
    current_turn: int,
    *,
    in_game_day: int | None = None,
    in_game_minute: int | None = None,
) -> TtlCleanupResult:
    status_expr = models.ObjectModel.data["status"].astext
    candidates = db.execute(
        select(models.ObjectModel).where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "npc",
            models.ObjectModel.data["ephemeral"].astext == "true",
            or_(
                models.ObjectModel.data["pinned"].astext.is_(None),
                models.ObjectModel.data["pinned"].astext != "true",
            ),
            models.ObjectModel.data["despawn_turn"].astext.is_not(None),
            func.jsonb_typeof(models.ObjectModel.data["despawn_turn"]) != "null",
            or_(
                status_expr.is_(None),
                and_(
                    status_expr != "inactive",
                    status_expr != NPC_OFFSTAGE_STATUS,
                ),
            ),
        )
    ).scalars().all()

    cleaned_count = 0
    applied_ops: list[dict[str, Any]] = []
    for npc in candidates:
        despawn_turn = _safe_int((npc.data or {}).get("despawn_turn"))
        if despawn_turn is None or despawn_turn > current_turn:
            continue

        active_links = _get_active_located_in_links(db, session_id, npc.object_id)
        scope_object_id = active_links[0].to_object_id if active_links else None
        for link in active_links:
            link.valid_to_turn = current_turn
            from_object = _get_object(db, session_id, link.from_object_id)
            to_object = _get_object(db, session_id, link.to_object_id)
            _add_internal_link_event(
                db,
                session_id=session_id,
                turn_index=current_turn,
                event_type="LINK_CLOSED",
                from_object_id=link.from_object_id,
                to_object_id=link.to_object_id,
                link_type=link.type,
                from_object=from_object,
                to_object=to_object,
                valid_to_turn=current_turn,
                source="ttl_cleanup",
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
            )
            applied_ops.append(
                {
                    "op": "link.close",
                    "from": str(link.from_object_id),
                    "to": str(link.to_object_id),
                    "type": link.type,
                }
            )

        npc_data = dict(npc.data or {})
        npc_data["status"] = NPC_OFFSTAGE_STATUS
        npc_data["despawned_turn"] = current_turn
        npc_data["despawn_reason"] = "ttl"
        if isinstance(scope_object_id, uuid.UUID):
            npc_data["last_known_zone_id"] = str(scope_object_id)
        npc.data = npc_data
        patch_data: dict[str, Any] = {
            "status": NPC_OFFSTAGE_STATUS,
            "despawned_turn": current_turn,
            "despawn_reason": "ttl",
        }
        if isinstance(scope_object_id, uuid.UUID):
            patch_data["last_known_zone_id"] = str(scope_object_id)
        applied_ops.append(
            {
                "op": "object.update",
                "object": str(npc.object_id),
                "patch": patch_data,
            }
        )
        _add_internal_object_updated_event(
            db,
            session_id=session_id,
            turn_index=current_turn,
            object_row=npc,
            patch_data=patch_data,
            scope_object_id=scope_object_id,
        )
        if USE_EMBEDDINGS:
            _enqueue_object_profile_refresh_event(
                db,
                session_id=session_id,
                object_id=npc.object_id,
            )

        despawn_payload: dict[str, Any] = {
            "npc_id": str(npc.object_id),
            "reason": "ttl",
            "source": "ttl_cleanup",
        }
        if in_game_day is not None and in_game_minute is not None:
            despawn_payload["in_game_time"] = {"day": int(in_game_day), "minute": int(in_game_minute)}

        db.add(
            models.EventModel(
                session_id=session_id,
                turn_index=current_turn,
                type="NPC_DESPAWN",
                scope_object_id=scope_object_id,
                payload=despawn_payload,
            )
        )
        cleaned_count += 1

    if cleaned_count:
        db.flush()
    return TtlCleanupResult(cleaned_count=cleaned_count, applied_ops=applied_ops)

def create_session_with_defaults(db: Session, payload: schemas.SessionCreateIn) -> schemas.SessionCreateOut:
    result: schemas.SessionCreateOut | None = None
    with db.begin():
        state = models.default_session_state()
        if payload.world_name:
            state["world_name"] = payload.world_name
        if payload.difficulty:
            state["difficulty"] = payload.difficulty
        state.setdefault("time", {"day": 0, "minute": 0})
        state.setdefault("time_scale", DEFAULT_TIME_SCALE_MINUTES)

        session_row = models.SessionModel(
            world_prompt=payload.world_prompt,
            state_json=state,
            turn_index=0,
        )
        db.add(session_row)
        db.flush()

        player_object = models.ObjectModel(
            session_id=session_row.id,
            type="player",
            name="Player",
            data={},
        )
        tavern_object = models.ObjectModel(
            session_id=session_row.id,
            type="zone",
            name=DEFAULT_SPAWN_ZONE_NAME,
            data={},
        )
        db.add_all([player_object, tavern_object])
        db.flush()

        located_in_link = models.LinkModel(
            session_id=session_row.id,
            from_object_id=player_object.object_id,
            to_object_id=tavern_object.object_id,
            type=LOCATED_IN_LINK_TYPE,
            data={},
            valid_from_turn=0,
            valid_to_turn=None,
        )
        db.add(located_in_link)

        state_payload = dict(session_row.state_json)
        state_payload["player_object_id"] = str(player_object.object_id)
        session_row.state_json = state_payload

        # Create a system turn row for turn_index=0 so the SESSION_CREATED
        # event has a valid FK target in the turns table.
        turn_zero = models.TurnModel(
            session_id=session_row.id,
            turn_index=0,
            user_input="(session created)",
            ai_text=f"(system) Player starts in {tavern_object.name}",
            ai_json={
                "status": "system",
                "note": "session_init",
                "applied_ops": [
                    {
                        "op": "object.create",
                        "ref": str(player_object.object_id),
                        "type": "player",
                        "name": player_object.name,
                    },
                    {
                        "op": "object.create",
                        "ref": str(tavern_object.object_id),
                        "type": "zone",
                        "name": tavern_object.name,
                    },
                    {
                        "op": "link.create",
                        "from": str(player_object.object_id),
                        "to": str(tavern_object.object_id),
                        "type": LOCATED_IN_LINK_TYPE,
                    },
                ],
            },
        )
        db.add(turn_zero)
        db.flush()
        _add_internal_object_created_event(
            db,
            session_id=session_row.id,
            turn_index=0,
            object_row=player_object,
            object_data={},
            scope_object_id=tavern_object.object_id,
        )
        _add_internal_object_created_event(
            db,
            session_id=session_row.id,
            turn_index=0,
            object_row=tavern_object,
            object_data={},
        )
        _add_internal_link_event(
            db,
            session_id=session_row.id,
            turn_index=0,
            event_type="LINK_CREATED",
            from_object_id=player_object.object_id,
            to_object_id=tavern_object.object_id,
            link_type=LOCATED_IN_LINK_TYPE,
            from_object=player_object,
            to_object=tavern_object,
            link_data={},
            valid_from_turn=0,
            valid_to_turn=None,
            source="session_bootstrap",
            in_game_day=0,
            in_game_minute=0,
        )

        created_event = models.EventModel(
            session_id=session_row.id,
            turn_index=0,
            type="SESSION_CREATED",
            scope_object_id=tavern_object.object_id,
            payload={
                "player_object_id": str(player_object.object_id),
                "spawn_location_id": str(tavern_object.object_id),
            },
        )
        db.add(created_event)

        if USE_EMBEDDINGS:
            _enqueue_session_bootstrap_indexing_event(
                db,
                session_id=session_row.id,
            )
        _enqueue_turn_chronicle_sync_event(
            db,
            session_id=session_row.id,
            turn_index=0,
        )

        snapshot_row = db.get(models.SessionSnapshotModel, (session_row.id, 0))
        snapshot_dump = _build_session_snapshot_dump(db, session_row)
        if snapshot_row is None:
            db.add(
                models.SessionSnapshotModel(
                    session_id=session_row.id,
                    turn_index=0,
                    dump_json=snapshot_dump,
                )
            )
        else:
            snapshot_row.dump_json = snapshot_dump

        db.flush()

        result = schemas.SessionCreateOut(
            session_id=session_row.id,
            player_object_id=player_object.object_id,
            tavern_object_id=tavern_object.object_id,
        )

    if result is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create session")

    return result


def get_session(db: Session, session_id: uuid.UUID) -> models.SessionModel:
    return _require_session(db, session_id)


def reindex_world_prompt(
    db: Session,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    """Force re-index world prompt chunks for the given session.

    Useful when the initial indexing failed during session creation
    (see fix #1) or when the world prompt changes externally.
    """
    had_active_tx = db.in_transaction()
    session_row = _require_session(db, session_id)
    world_prompt = (session_row.world_prompt or "").strip()
    if not world_prompt:
        return {"status": "skip", "reason": "no_world_prompt"}
    if not USE_EMBEDDINGS:
        return {"status": "skip", "reason": "embeddings_disabled"}
    # Route-level reads trigger SQLAlchemy autobegin. If this function started
    # that implicit transaction itself, clear it so the indexer can own/commit
    # its dedicated write transactions.
    if not had_active_tx and db.in_transaction():
        db.rollback()
    try:
        source_hash = _ensure_world_prompt_chunks_indexed(
            db=db,
            session_id=session_id,
            world_prompt=world_prompt,
        )
        return {"status": "ok", "source_hash": source_hash}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "reindex_world_prompt failed for session_id=%s",
            session_id,
            exc_info=True,
        )
        return {"status": "error", "reason": str(exc)}


def delete_session(db: Session, session_id: uuid.UUID) -> None:
    with db.begin():
        session_row = _require_session(db, session_id)
        db.delete(session_row)


def get_session_token_stats(
    db: Session,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    """Aggregate LLM token usage across all turns for a session."""
    _require_session(db, session_id)

    turn_rows = db.execute(
        select(models.TurnModel.ai_json)
        .where(
            models.TurnModel.session_id == session_id,
            models.TurnModel.ai_json.is_not(None),
        )
        .order_by(models.TurnModel.turn_index.asc())
    ).scalars().all()

    total_turns = len(turn_rows)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    librarian_count = 0

    def _coerce_usage_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, float) and value.is_integer():
            return max(int(value), 0)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return max(int(text), 0)
            except ValueError:
                return None
        return None

    def _extract_usage_totals(raw_usage: Any) -> tuple[int, int, int] | None:
        if not isinstance(raw_usage, dict):
            return None
        prompt = _coerce_usage_int(raw_usage.get("prompt_tokens"))
        completion = _coerce_usage_int(raw_usage.get("completion_tokens"))
        total = _coerce_usage_int(raw_usage.get("total_tokens"))
        if total is None and (prompt is not None or completion is not None):
            total = (prompt or 0) + (completion or 0)
        if prompt is None and completion is None and total is None:
            return None
        return (prompt or 0, completion or 0, total or 0)

    def _accumulate_usage(raw_usage: Any) -> bool:
        nonlocal total_prompt_tokens, total_completion_tokens, total_tokens
        parsed = _extract_usage_totals(raw_usage)
        if parsed is None:
            return False
        prompt, completion, total = parsed
        total_prompt_tokens += prompt
        total_completion_tokens += completion
        total_tokens += total
        return True

    for ai_json in turn_rows:
        if not isinstance(ai_json, dict):
            continue
        counted_usage = False

        llm_usage = ai_json.get("llm_usage")
        if isinstance(llm_usage, dict):
            # xAI buckets.
            counted_usage = _accumulate_usage(llm_usage.get("narrator")) or counted_usage
            counted_usage = _accumulate_usage(llm_usage.get("librarian")) or counted_usage

            # DeepSeek bucket can appear either as aggregated deepseek_total
            # (preferred) or as deepseek_patch in split mode.
            if _accumulate_usage(llm_usage.get("deepseek_total")):
                counted_usage = True
            elif _accumulate_usage(llm_usage.get("deepseek_patch")):
                counted_usage = True

        if not counted_usage:
            # Backward compatibility for old payloads.
            counted_usage = _accumulate_usage(ai_json.get("narrator_usage")) or counted_usage
            counted_usage = _accumulate_usage(ai_json.get("_xai_usage")) or counted_usage

        # Librarian usage
        if ai_json.get("librarian_used"):
            librarian_count += 1

    return {
        "total_turns": total_turns,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_turn": round(total_tokens / max(total_turns, 1), 1),
        "librarian_invocations": librarian_count,
        "librarian_rate": round(librarian_count / max(total_turns, 1), 3),
    }


def create_session_snapshot(db: Session, session_id: uuid.UUID) -> models.SessionSnapshotModel:
    with db.begin():
        _acquire_session_turn_lock(db, session_id)
        session_row = _require_session(db, session_id, for_update=True)
        _recover_abandoned_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=dict(getattr(session_row, "state_json", {}) or {}),
        )
        snapshot_turn = session_row.turn_index
        dump_payload = _build_session_snapshot_dump(db, session_row)

        snapshot_row = db.get(models.SessionSnapshotModel, (session_id, snapshot_turn))
        if snapshot_row is None:
            snapshot_row = models.SessionSnapshotModel(
                session_id=session_id,
                turn_index=snapshot_turn,
                dump_json=dump_payload,
            )
            db.add(snapshot_row)
        else:
            snapshot_row.dump_json = dump_payload
        db.flush()
        return snapshot_row


def get_session_snapshot(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
) -> models.SessionSnapshotModel:
    _require_session(db, session_id)
    snapshot_row = db.get(models.SessionSnapshotModel, (session_id, turn_index))
    if snapshot_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session snapshot not found")
    return snapshot_row


def list_session_snapshots(
    db: Session,
    session_id: uuid.UUID,
    *,
    limit: int = 50,
) -> Iterable[models.SessionSnapshotModel]:
    _require_session(db, session_id)
    bounded_limit = min(max(limit, 1), 500)
    query = (
        select(models.SessionSnapshotModel)
        .where(models.SessionSnapshotModel.session_id == session_id)
        .order_by(models.SessionSnapshotModel.turn_index.desc())
        .limit(bounded_limit)
    )
    return db.execute(query).scalars().all()


def create_object(db: Session, session_id: uuid.UUID, payload: schemas.ObjectCreateIn) -> models.ObjectModel:
    with db.begin():
        _acquire_session_turn_lock(db, session_id)
        session_row = _require_session(db, session_id, for_update=True)
        state_payload = _recover_abandoned_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=_coerce_state_payload(getattr(session_row, "state_json", {})),
        )
        in_game_day, in_game_minute = _extract_in_game_time(state_payload)
        new_turn = max(_safe_int(getattr(session_row, "turn_index", 0)) or 0, 0) + 1
        object_row = models.ObjectModel(
            session_id=session_id,
            type=payload.type,
            name=payload.name,
            data=payload.data,
        )
        db.add(object_row)
        db.flush()
        _create_internal_turn_row(
            db,
            session_id,
            session_row,
            turn_index=new_turn,
            user_input=f"[internal object.create {payload.type} {payload.name}]",
            ai_text=f"(internal) created {payload.type} {payload.name}",
            note="internal_object_create",
            applied_ops=[
                {
                    "op": "object.create",
                    "ref": str(object_row.object_id),
                    "type": payload.type,
                    "name": payload.name,
                    "data": dict(payload.data or {}),
                }
            ],
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )
        db.flush()
        _add_internal_object_created_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
            object_row=object_row,
            object_data=dict(payload.data or {}),
        )
        _enqueue_turn_chronicle_sync_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
        )
        if USE_EMBEDDINGS:
            if payload.type in _OBJECT_PROFILE_REFRESH_TYPES:
                _enqueue_object_profile_refresh_event(
                    db,
                    session_id=session_id,
                    object_id=object_row.object_id,
                )
            elif payload.type == "claim":
                _enqueue_claim_text_refresh_event(
                    db,
                    session_id=session_id,
                    object_id=object_row.object_id,
                )
        return object_row


def get_object(db: Session, session_id: uuid.UUID, object_id: uuid.UUID) -> models.ObjectModel:
    object_row = _require_object(db, session_id, object_id)
    if str(getattr(object_row, "type", "") or "") not in schemas.API_OBJECT_TYPE_VALUES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Object not found")
    return object_row


def list_objects(
    db: Session,
    session_id: uuid.UUID,
    *,
    name: str | None = None,
    type_: str | None = None,
) -> Iterable[models.ObjectModel]:
    _require_session(db, session_id)

    normalized_type = str(type_ or "").strip() if type_ is not None else None
    if normalized_type is not None and normalized_type not in schemas.API_OBJECT_TYPE_VALUES:
        return []

    query = select(models.ObjectModel).where(
        models.ObjectModel.session_id == session_id,
        models.ObjectModel.type.in_(schemas.API_OBJECT_TYPE_VALUES),
    )
    if name:
        query = query.where(models.ObjectModel.name == name)
    if normalized_type:
        query = query.where(models.ObjectModel.type == normalized_type)

    query = query.order_by(models.ObjectModel.created_at.asc())
    return db.execute(query).scalars().all()


def create_link(db: Session, session_id: uuid.UUID, payload: schemas.LinkCreateIn) -> models.LinkModel:
    with db.begin():
        _acquire_session_turn_lock(db, session_id)
        session_row = _require_session(db, session_id, for_update=True)
        state_payload = _recover_abandoned_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=_coerce_state_payload(getattr(session_row, "state_json", {})),
        )
        in_game_day, in_game_minute = _extract_in_game_time(state_payload)
        new_turn = max(_safe_int(getattr(session_row, "turn_index", 0)) or 0, 0) + 1

        if payload.valid_from_turn < 0 or (payload.valid_to_turn is not None and payload.valid_to_turn < 0):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="valid_from_turn and valid_to_turn must be non-negative",
            )
        if payload.valid_to_turn is not None and payload.valid_to_turn < payload.valid_from_turn:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="valid_to_turn must be greater than or equal to valid_from_turn",
            )
        if payload.from_object_id == payload.to_object_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Self-referencing links are not allowed (from_object_id == to_object_id)",
            )
        from_object = _require_object(db, session_id, payload.from_object_id)
        to_object = _require_object(db, session_id, payload.to_object_id)
        if payload.type == TRACKING_QUEST_LINK_TYPE:
            session_player_id = _get_session_player_object_id(db, session_id)
            if payload.from_object_id != session_player_id:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="tracking_quest link must originate from session player",
                )
            if to_object.type != "quest":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="tracking_quest link target must be a quest object",
                )

        if (
            payload.type == LOCATED_IN_LINK_TYPE
            and from_object.type == "player"
            and payload.valid_to_turn is None
        ):
            active_location = db.execute(
                select(models.LinkModel).where(
                    models.LinkModel.session_id == session_id,
                    models.LinkModel.from_object_id == payload.from_object_id,
                    models.LinkModel.type == LOCATED_IN_LINK_TYPE,
                    models.LinkModel.valid_to_turn.is_(None),
                )
            ).scalar_one_or_none()
            if active_location is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Player already has an active located_in link",
                )

        if payload.valid_to_turn is None:
            active_duplicate = _get_active_link(
                db,
                session_id,
                payload.from_object_id,
                payload.to_object_id,
                payload.type,
            )
            if active_duplicate is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Active link already exists",
                )

        link_row = models.LinkModel(
            session_id=session_id,
            from_object_id=payload.from_object_id,
            to_object_id=payload.to_object_id,
            type=payload.type,
            data=payload.data,
            valid_from_turn=payload.valid_from_turn,
            valid_to_turn=payload.valid_to_turn,
        )
        db.add(link_row)
        reverse_link: models.LinkModel | None = None

        if payload.type == ADJACENT_LINK_TYPE and payload.from_object_id != payload.to_object_id:
            reverse_duplicate = (
                _get_active_link(
                    db,
                    session_id,
                    payload.to_object_id,
                    payload.from_object_id,
                    ADJACENT_LINK_TYPE,
                )
                if payload.valid_to_turn is None
                else None
            )
            if reverse_duplicate is None:
                reverse_link = models.LinkModel(
                    session_id=session_id,
                    from_object_id=payload.to_object_id,
                    to_object_id=payload.from_object_id,
                    type=ADJACENT_LINK_TYPE,
                    data=payload.data,
                    valid_from_turn=payload.valid_from_turn,
                    valid_to_turn=payload.valid_to_turn,
                )
                db.add(reverse_link)

        if (
            USE_EMBEDDINGS
            and payload.valid_to_turn is None
            and payload.type in LINK_CONTENT_TYPES
            and _extract_link_context_text(dict(payload.data or {}))
        ):
            _enqueue_link_context_refresh_event(
                db,
                session_id=session_id,
                from_object_id=payload.from_object_id,
            )

        try:
            db.flush()
        except IntegrityError as exc:
            error_text = str(getattr(exc, "orig", exc)).lower()
            if "uq_links_active_located_in_per_from" in error_text:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Player already has an active located_in link",
                ) from exc
            if "uq_links_active_per_edge_type" in error_text:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Active link already exists",
                ) from exc
            raise
        applied_ops: list[dict[str, Any]] = [
            {
                "op": "link.create",
                "from": str(payload.from_object_id),
                "to": str(payload.to_object_id),
                "type": payload.type,
                "data": dict(payload.data or {}),
                "valid_from_turn": payload.valid_from_turn,
                "valid_to_turn": payload.valid_to_turn,
            }
        ]
        if reverse_link is not None:
            applied_ops.append(
                {
                    "op": "link.create",
                    "from": str(payload.to_object_id),
                    "to": str(payload.from_object_id),
                    "type": ADJACENT_LINK_TYPE,
                    "data": dict(payload.data or {}),
                    "valid_from_turn": payload.valid_from_turn,
                    "valid_to_turn": payload.valid_to_turn,
                }
            )
        _create_internal_turn_row(
            db,
            session_id,
            session_row,
            turn_index=new_turn,
            user_input=f"[internal link.create {payload.type}] {from_object.name} -> {to_object.name}",
            ai_text=f"(internal) {from_object.name} {payload.type} {to_object.name}",
            note="internal_link_create",
            applied_ops=applied_ops,
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )
        db.flush()
        _add_internal_link_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
            event_type="LINK_CREATED",
            from_object_id=payload.from_object_id,
            to_object_id=payload.to_object_id,
            link_type=payload.type,
            from_object=from_object,
            to_object=to_object,
            link_data=dict(payload.data or {}),
            valid_from_turn=payload.valid_from_turn,
            valid_to_turn=payload.valid_to_turn,
        )
        if reverse_link is not None:
            _add_internal_link_event(
                db,
                session_id=session_id,
                turn_index=new_turn,
                event_type="LINK_CREATED",
                from_object_id=payload.to_object_id,
                to_object_id=payload.from_object_id,
                link_type=ADJACENT_LINK_TYPE,
                from_object=to_object,
                to_object=from_object,
                link_data=dict(payload.data or {}),
                valid_from_turn=payload.valid_from_turn,
                valid_to_turn=payload.valid_to_turn,
            )
        _enqueue_turn_chronicle_sync_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
        )
        return link_row


def list_links(
    db: Session,
    session_id: uuid.UUID,
    *,
    type_: str | None = None,
    from_object_id: uuid.UUID | None = None,
    to_object_id: uuid.UUID | None = None,
    active_only: bool = False,
) -> Iterable[models.LinkModel]:
    _require_session(db, session_id)

    query = select(models.LinkModel).where(models.LinkModel.session_id == session_id)
    if type_:
        query = query.where(models.LinkModel.type == type_)
    if from_object_id:
        query = query.where(models.LinkModel.from_object_id == from_object_id)
    if to_object_id:
        query = query.where(models.LinkModel.to_object_id == to_object_id)
    if active_only:
        query = query.where(models.LinkModel.valid_to_turn.is_(None))

    query = query.order_by(models.LinkModel.created_at.asc())
    return db.execute(query).scalars().all()


def list_events(db: Session, session_id: uuid.UUID, *, limit: int = 50) -> Iterable[models.EventModel]:
    _require_session(db, session_id)
    bounded_limit = min(max(limit, 1), 500)

    query = (
        select(models.EventModel)
        .where(models.EventModel.session_id == session_id)
        .order_by(models.EventModel.turn_index.desc(), models.EventModel.created_at.desc())
        .limit(bounded_limit)
    )
    return db.execute(query).scalars().all()


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


def _should_use_dedicated_chronicle_writer(db: Session) -> bool:
    if not db.in_transaction():
        return False
    if _session_transaction_origin(db) != SessionTransactionOrigin.AUTOBEGIN:
        return False
    return not _session_has_pending_state(db)


def _independent_session_bind(db: Session) -> Any:
    bind = db.get_bind()
    engine = getattr(bind, "engine", None)
    return engine or bind


def _build_turn_embedding_candidate(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
    zone_id: uuid.UUID | None,
    in_game_day: int | None,
    in_game_minute: int | None,
    *,
    snippet: str,
    append_to_existing: bool,
    namespace: str,
) -> _ChronicleEmbeddingCandidate | None:
    existing = db.execute(
        select(models.ChronicleChunkModel).where(
            models.ChronicleChunkModel.session_id == session_id,
            models.ChronicleChunkModel.turn_index == turn_index,
            models.ChronicleChunkModel.namespace == namespace,
        )
    ).scalar_one_or_none()

    final_zone_id = zone_id
    final_in_game_day = in_game_day
    final_in_game_minute = in_game_minute
    final_snippet = snippet
    if existing is not None:
        if final_zone_id is None:
            final_zone_id = existing.zone_id
        if final_in_game_day is None:
            final_in_game_day = existing.in_game_day
        if final_in_game_minute is None:
            final_in_game_minute = existing.in_game_minute

        if append_to_existing:
            base = (existing.text_snippet or "").strip()
            if base:
                base_lines = [line.strip() for line in base.splitlines() if line.strip()]
                new_lines = [line.strip() for line in final_snippet.splitlines() if line.strip()]
                if not new_lines:
                    final_snippet = base
                else:
                    existing_lines = set(base_lines)
                    appended_lines: list[str] = []
                    appended_seen: set[str] = set()
                    for line in new_lines:
                        if line in existing_lines or line in appended_seen:
                            continue
                        appended_seen.add(line)
                        appended_lines.append(line)
                    if appended_lines:
                        final_snippet = "\n".join([base, *appended_lines])
                    else:
                        final_snippet = base

    final_snippet = _truncate_text(final_snippet, EMBED_SNIPPET_MAX_CHARS)
    text_hash = hashlib.sha256(final_snippet.encode("utf-8")).hexdigest()

    metadata_unchanged = (
        existing is not None
        and existing.zone_id == final_zone_id
        and existing.in_game_day == final_in_game_day
        and existing.in_game_minute == final_in_game_minute
    )
    if existing is not None and existing.text_hash == text_hash and metadata_unchanged:
        return None

    return _ChronicleEmbeddingCandidate(
        zone_id=final_zone_id,
        in_game_day=final_in_game_day,
        in_game_minute=final_in_game_minute,
        text_snippet=final_snippet,
        text_hash=text_hash,
    )


def _upsert_turn_embedding_candidate(
    db: Session,
    *,
    session_id: uuid.UUID,
    turn_index: int,
    namespace: str,
    candidate: _ChronicleEmbeddingCandidate,
    embedding: list[float],
) -> None:
    bind = db.get_bind()
    dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "")).lower()
    if dialect_name == "postgresql":
        insert_stmt = pg_insert(models.ChronicleChunkModel).values(
            session_id=session_id,
            turn_index=turn_index,
            namespace=namespace,
            zone_id=candidate.zone_id,
            in_game_day=candidate.in_game_day,
            in_game_minute=candidate.in_game_minute,
            text_snippet=candidate.text_snippet,
            text_hash=candidate.text_hash,
            embedding=embedding,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["session_id", "turn_index", "namespace"],
            set_={
                "zone_id": candidate.zone_id,
                "in_game_day": candidate.in_game_day,
                "in_game_minute": candidate.in_game_minute,
                "text_snippet": candidate.text_snippet,
                "text_hash": candidate.text_hash,
                "embedding": embedding,
            },
        )
        db.execute(upsert_stmt)
        return

    existing = db.execute(
        select(models.ChronicleChunkModel).where(
            models.ChronicleChunkModel.session_id == session_id,
            models.ChronicleChunkModel.turn_index == turn_index,
            models.ChronicleChunkModel.namespace == namespace,
        )
    ).scalar_one_or_none()
    if existing is None:
        db.add(
            models.ChronicleChunkModel(
                session_id=session_id,
                turn_index=turn_index,
                namespace=namespace,
                zone_id=candidate.zone_id,
                in_game_day=candidate.in_game_day,
                in_game_minute=candidate.in_game_minute,
                text_snippet=candidate.text_snippet,
                text_hash=candidate.text_hash,
                embedding=embedding,
            )
        )
        return

    existing.zone_id = candidate.zone_id
    existing.in_game_day = candidate.in_game_day
    existing.in_game_minute = candidate.in_game_minute
    existing.text_snippet = candidate.text_snippet
    existing.text_hash = candidate.text_hash
    existing.embedding = embedding


def _index_turn_embedding_in_session(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
    zone_id: uuid.UUID | None,
    in_game_day: int | None,
    in_game_minute: int | None,
    *,
    snippet: str,
    append_to_existing: bool,
    namespace: str,
) -> None:
    candidate = _build_turn_embedding_candidate(
        db,
        session_id,
        turn_index,
        zone_id,
        in_game_day,
        in_game_minute,
        snippet=snippet,
        append_to_existing=append_to_existing,
        namespace=namespace,
    )
    if candidate is None:
        return

    embedding = _maybe_embed_texts(
        [candidate.text_snippet],
        instruction=CHRONICLE_EVENT_EMBED_INSTRUCTION,
    )[0]
    _upsert_turn_embedding_candidate(
        db,
        session_id=session_id,
        turn_index=turn_index,
        namespace=namespace,
        candidate=candidate,
        embedding=embedding,
    )


def _index_turn_embedding_without_transaction(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
    zone_id: uuid.UUID | None,
    in_game_day: int | None,
    in_game_minute: int | None,
    *,
    snippet: str,
    append_to_existing: bool,
    namespace: str,
) -> None:
    while True:
        candidate = _build_turn_embedding_candidate(
            db,
            session_id,
            turn_index,
            zone_id,
            in_game_day,
            in_game_minute,
            snippet=snippet,
            append_to_existing=append_to_existing,
            namespace=namespace,
        )
        if candidate is None:
            return
        _rollback_read_only_autobegin_transaction(db)
        embedding = _maybe_embed_texts(
            [candidate.text_snippet],
            instruction=CHRONICLE_EVENT_EMBED_INSTRUCTION,
        )[0]
        with db.begin():
            current_candidate = _build_turn_embedding_candidate(
                db,
                session_id,
                turn_index,
                zone_id,
                in_game_day,
                in_game_minute,
                snippet=snippet,
                append_to_existing=append_to_existing,
                namespace=namespace,
            )
            if current_candidate is None:
                return
            if current_candidate != candidate:
                continue
            _upsert_turn_embedding_candidate(
                db,
                session_id=session_id,
                turn_index=turn_index,
                namespace=namespace,
                candidate=candidate,
                embedding=embedding,
            )
            return


def index_turn_embedding(
    db: Session,
    session_id: uuid.UUID,
    turn_index: int,
    zone_id: uuid.UUID | None,
    in_game_day: int | None,
    in_game_minute: int | None,
    ai_text: str | None = None,
    snippet_text: str | None = None,
    append_to_existing: bool = False,
    namespace: str = CHRONICLE_OUTPUT_NAMESPACE,
) -> None:
    raw_snippet = snippet_text if snippet_text is not None else ai_text
    snippet = (raw_snippet or "").strip()
    if not snippet or not USE_EMBEDDINGS:
        return

    if _should_use_dedicated_chronicle_writer(db):
        writer_db = Session(bind=_independent_session_bind(db), expire_on_commit=False)
        try:
            _index_turn_embedding_without_transaction(
                writer_db,
                session_id,
                turn_index,
                zone_id,
                in_game_day,
                in_game_minute,
                snippet=snippet,
                append_to_existing=append_to_existing,
                namespace=namespace,
            )
        finally:
            writer_db.close()
        return

    if not db.in_transaction():
        _index_turn_embedding_without_transaction(
            db,
            session_id,
            turn_index,
            zone_id,
            in_game_day,
            in_game_minute,
            snippet=snippet,
            append_to_existing=append_to_existing,
            namespace=namespace,
        )
        return

    with nullcontext():
        _index_turn_embedding_in_session(
            db,
            session_id,
            turn_index,
            zone_id,
            in_game_day,
            in_game_minute,
            snippet=snippet,
            append_to_existing=append_to_existing,
            namespace=namespace,
        )


def semantic_retrieve(
    db: Session,
    session_id: uuid.UUID,
    query_text: str,
    *,
    k: int,
    max_chars: int,
    namespaces: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not USE_EMBEDDINGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Embeddings are disabled",
        )
    _require_session(db, session_id)
    _rollback_read_only_autobegin_transaction(db)

    try:
        query_embedding = _maybe_embed_texts(
            [f"Query: {query_text}"],
            instruction=RELEVANCE_QUERY_EMBED_INSTRUCTION,
        )[0]
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings provider is unavailable",
        ) from exc
    top_k = min(max(k, 1), 20)
    snippet_limit = max(max_chars, 1)

    selected_namespaces: list[str]
    if namespaces is None:
        selected_namespaces = [CHRONICLE_OUTPUT_NAMESPACE, CHRONICLE_INPUT_NAMESPACE]
    else:
        selected_namespaces = []
        for raw_namespace in namespaces:
            namespace = str(raw_namespace or "").strip()
            if namespace and namespace not in selected_namespaces:
                selected_namespaces.append(namespace)
    if not selected_namespaces:
        return []

    namespace_count = len(selected_namespaces)
    base_quota = top_k // namespace_count
    remainder = top_k % namespace_count
    namespace_quotas: dict[str, int] = {}
    for idx, namespace in enumerate(selected_namespaces):
        namespace_quotas[namespace] = base_quota + (1 if idx < remainder else 0)

    distance_expr = models.ChronicleChunkModel.embedding.cosine_distance(query_embedding)
    candidates_by_namespace: dict[str, list[tuple[models.ChronicleChunkModel, float]]] = {}
    for namespace in selected_namespaces:
        rows = db.execute(
            select(models.ChronicleChunkModel, distance_expr.label("distance"))
            .where(
                models.ChronicleChunkModel.session_id == session_id,
                models.ChronicleChunkModel.namespace == namespace,
            )
            .order_by(distance_expr.asc())
            .limit(top_k)
        ).all()
        parsed_rows: list[tuple[models.ChronicleChunkModel, float]] = []
        for row, distance in rows:
            parsed_rows.append((row, float(distance) if distance is not None else 1e9))
        candidates_by_namespace[namespace] = parsed_rows

    selected_rows: list[tuple[models.ChronicleChunkModel, float]] = []
    seen_keys: set[tuple[int, str]] = set()
    for namespace in selected_namespaces:
        quota = namespace_quotas.get(namespace, 0)
        if quota <= 0:
            continue
        for row, distance in candidates_by_namespace.get(namespace, [])[:quota]:
            row_key = (row.turn_index, row.namespace)
            if row_key in seen_keys:
                continue
            seen_keys.add(row_key)
            selected_rows.append((row, distance))

    if len(selected_rows) < top_k:
        leftovers: list[tuple[models.ChronicleChunkModel, float]] = []
        for namespace in selected_namespaces:
            quota = namespace_quotas.get(namespace, 0)
            candidate_rows = candidates_by_namespace.get(namespace, [])
            for row, distance in candidate_rows[quota:]:
                row_key = (row.turn_index, row.namespace)
                if row_key in seen_keys:
                    continue
                leftovers.append((row, distance))
        leftovers.sort(key=lambda item: (item[1], -item[0].turn_index, item[0].namespace))
        for row, distance in leftovers[: max(top_k - len(selected_rows), 0)]:
            row_key = (row.turn_index, row.namespace)
            if row_key in seen_keys:
                continue
            seen_keys.add(row_key)
            selected_rows.append((row, distance))

    selected_rows.sort(key=lambda item: (item[1], -item[0].turn_index, item[0].namespace))
    selected_chunks = [row for row, _distance in selected_rows[:top_k]]

    result: list[dict[str, Any]] = []
    for row in selected_chunks:
        result.append(
            {
                "turn_index": row.turn_index,
                "namespace": row.namespace,
                "zone_id": row.zone_id,
                "in_game_time": {
                    "day": row.in_game_day,
                    "minute": row.in_game_minute,
                },
                "snippet": row.text_snippet[:snippet_limit],
            }
        )
    return result



__all__ = [
    "create_session_with_defaults",
    "get_session",
    "reindex_world_prompt",
    "get_session_token_stats",
    "delete_session",
    "create_session_snapshot",
    "get_session_snapshot",
    "list_session_snapshots",
    "create_object",
    "get_object",
    "list_objects",
    "create_link",
    "list_links",
    "list_events",
    "index_turn_embedding",
    "semantic_retrieve",
]
