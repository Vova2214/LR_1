from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from .. import schemas
from .. import crud_claims as claims_ops
from .. import crud_entities as entity_ops
from .. import crud_graph_ops as graph_ops
from .. import crud_lore_adaptation as lore_ops
from .. import crud_movement as movement_ops
from .. import crud_telemetry as telemetry_ops
from .. import crud_core as core_ops
from ..persistence.graph_repository import graph_repository
from ..persistence.session_read_repository import session_read_repository


class SessionLifecycleService:
    def create_session(self, db: Session, payload: schemas.SessionCreateIn) -> Any:
        return entity_ops.create_session_with_defaults(db, payload)

    def get_session(self, db: Session, session_id: uuid.UUID) -> Any:
        return entity_ops.get_session(db, session_id)

    def upload_lore(self, db: Session, session_id: uuid.UUID, payload: schemas.LoreUploadIn) -> dict[str, Any]:
        return lore_ops.upload_lore(db, session_id, payload)

    def get_lore_adaptation(self, db: Session, session_id: uuid.UUID) -> dict[str, Any]:
        return lore_ops.get_lore_adaptation(db, session_id)

    def answer_lore_gap(
        self,
        db: Session,
        session_id: uuid.UUID,
        gap_id: str,
        payload: schemas.LoreGapAnswerIn,
    ) -> dict[str, Any]:
        return lore_ops.answer_lore_gap(db, session_id, gap_id, payload)

    def auto_resolve_lore_gaps(self, db: Session, session_id: uuid.UUID) -> dict[str, Any]:
        return lore_ops.auto_resolve_lore_gaps(db, session_id)

    def delete_session(self, db: Session, session_id: uuid.UUID) -> None:
        entity_ops.delete_session(db, session_id)

    def create_session_snapshot(self, db: Session, session_id: uuid.UUID) -> Any:
        return entity_ops.create_session_snapshot(db, session_id)

    def get_session_snapshot(self, db: Session, session_id: uuid.UUID, turn_index: int) -> Any:
        return entity_ops.get_session_snapshot(db, session_id, turn_index)

    def list_session_snapshots(self, db: Session, session_id: uuid.UUID, *, limit: int) -> list[Any]:
        return entity_ops.list_session_snapshots(db, session_id, limit=limit)

    def list_pending_graph_ops(self, db: Session, session_id: uuid.UUID) -> list[dict]:
        return graph_repository.get_pending_graph_ops(
            db,
            session_id=session_id,
        )

    def apply_pending_graph_ops(self, db: Session, session_id: uuid.UUID) -> schemas.PendingGraphOpsApplyOut:
        return graph_ops.apply_pending_graph_ops(db, session_id)

    def reindex_world_prompt(self, db: Session, session_id: uuid.UUID) -> dict:
        return entity_ops.reindex_world_prompt(db, session_id)

    def get_session_token_stats(self, db: Session, session_id: uuid.UUID) -> dict:
        return entity_ops.get_session_token_stats(db, session_id)

    def get_session_timeline(
        self,
        db: Session,
        session_id: uuid.UUID,
        *,
        from_turn: int,
        to_turn: int | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        session_row = core_ops._require_session(db, session_id)
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
        self,
        db: Session,
        session_id: uuid.UUID,
        *,
        from_turn: int,
        to_turn: int,
    ) -> dict[str, Any]:
        session_row = core_ops._require_session(db, session_id)
        return session_read_repository.get_session_diff(
            db,
            session_id=session_id,
            session_row=session_row,
            from_turn=from_turn,
            to_turn=to_turn,
        )

    def get_relationship_graph(
        self,
        db: Session,
        session_id: uuid.UUID,
        *,
        zone_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        core_ops._require_session(db, session_id)
        return session_read_repository.get_relationship_graph(
            db,
            session_id=session_id,
            zone_id=zone_id,
        )


class TelemetryService:
    def get_session_llm_telemetry(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        limit: int,
        offset: int,
        turn_index: int | None,
        request_type: str | None,
        provider: str | None,
        from_ts: datetime | None,
        to_ts: datetime | None,
        cost_mode: schemas.CostMode,
    ) -> dict[str, Any]:
        return telemetry_ops.list_session_llm_telemetry(
            db,
            session_id=session_id,
            limit=limit,
            offset=offset,
            turn_index=turn_index,
            request_type=request_type,
            provider=provider,
            from_ts=from_ts,
            to_ts=to_ts,
            cost_mode=cost_mode,
        )

    def get_llm_telemetry_summary(
        self,
        db: Session,
        *,
        days: int,
        cost_mode: schemas.CostMode,
    ) -> dict[str, Any]:
        return telemetry_ops.get_llm_telemetry_summary(
            db,
            days=days,
            cost_mode=cost_mode,
        )

    def list_outbox_events(
        self,
        db: Session,
        *,
        status_filter: str | None,
        event_type: str | None,
        session_id: uuid.UUID | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        return telemetry_ops.list_outbox_events(
            db,
            status_filter=status_filter,
            event_type=event_type,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

    def list_active_system_prompts(self, db: Session) -> list[Any]:
        return telemetry_ops.list_active_system_prompts(db)

    def activate_system_prompt(self, db: Session, *, module: str, version: int) -> Any:
        return telemetry_ops.activate_system_prompt(db, module=module, version=version)


class EntityService:
    def create_object(self, db: Session, session_id: uuid.UUID, payload: schemas.ObjectCreateIn) -> Any:
        return entity_ops.create_object(db, session_id, payload)

    def get_object(self, db: Session, session_id: uuid.UUID, object_id: uuid.UUID) -> Any:
        return entity_ops.get_object(db, session_id, object_id)

    def list_objects(
        self,
        db: Session,
        session_id: uuid.UUID,
        *,
        name: str | None,
        type_: str | None,
    ) -> list[Any]:
        return entity_ops.list_objects(db, session_id, name=name, type_=type_)

    def create_link(self, db: Session, session_id: uuid.UUID, payload: schemas.LinkCreateIn) -> Any:
        return entity_ops.create_link(db, session_id, payload)

    def list_links(
        self,
        db: Session,
        session_id: uuid.UUID,
        *,
        type_: str | None,
        from_object_id: uuid.UUID | None,
        to_object_id: uuid.UUID | None,
        active_only: bool,
    ) -> list[Any]:
        return entity_ops.list_links(
            db,
            session_id,
            type_=type_,
            from_object_id=from_object_id,
            to_object_id=to_object_id,
            active_only=active_only,
        )

    def list_events(self, db: Session, session_id: uuid.UUID, *, limit: int) -> list[Any]:
        return entity_ops.list_events(db, session_id, limit=limit)


class SemanticToolService:
    def retrieve(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        query_text: str,
        k: int,
        max_chars: int,
        namespaces: list[str] | None,
    ) -> list[dict[str, Any]]:
        return entity_ops.semantic_retrieve(
            db=db,
            session_id=session_id,
            query_text=query_text,
            k=k,
            max_chars=max_chars,
            namespaces=namespaces,
        )


class DebugCommandService:
    def move_player(self, db: Session, session_id: uuid.UUID, payload: schemas.MoveIn) -> schemas.MoveOut:
        return movement_ops.move_player(db, session_id, payload)

    def create_claim(
        self,
        db: Session,
        session_id: uuid.UUID,
        payload: schemas.ClaimCreateIn,
    ) -> schemas.ClaimCreateOut:
        return claims_ops.create_claim(db, session_id, payload)


session_lifecycle_service = SessionLifecycleService()
telemetry_service = TelemetryService()
entity_service = EntityService()
semantic_tool_service = SemanticToolService()
debug_command_service = DebugCommandService()
