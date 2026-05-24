from __future__ import annotations

import copy
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base, EMBEDDING_DIM
from .vector_compat import Vector


DEFAULT_SESSION_STATE: dict[str, Any] = {
    "world_name": "Unnamed",
    "difficulty": "normal",
    "rules": {},
    "player_object_id": None,
}

DEFAULT_SESSION_STATE_SQL = (
    "'{\"world_name\":\"Unnamed\",\"difficulty\":\"normal\",\"rules\":{},"
    "\"player_object_id\": null}'::jsonb"
)
# NOTE: This tuple includes internal object types (__world_prompt_chunk,
# __memory_event, etc.) that are created directly by server-side code.
# The API-facing Pydantic schema (schemas.ALLOWED_OBJECT_TYPES) intentionally
# excludes most internal types so the AI cannot create them via patch ops.
ALLOWED_OBJECT_TYPES: tuple[str, ...] = (
    "player",
    "npc",
    "zone",
    "item",
    "faction",
    "quest",
    "claim",
    "world_constitution",
    "__world_prompt_chunk",
    "__memory_event",
    "__memory_fact",
    "__memory_bundle",
    "__story_obligation",
    "__memory_conflict_edge",
    "__memory_review_report",
    "__memory_evaluation_report",
    "__session_summary",
    "__narrative_spine",
    "__entity_memory",
    "__callback_memory",
    "__consequence_window",
    "__director_agenda",
    "__director_action",
)
ALLOWED_OBJECT_TYPES_SQL = "type IN (" + ", ".join(f"'{item}'" for item in ALLOWED_OBJECT_TYPES) + ")"


def default_session_state() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_SESSION_STATE)


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    world_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    state_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=default_session_state,
        server_default=text(DEFAULT_SESSION_STATE_SQL),
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))

    __table_args__: tuple[Any, ...] = (
        CheckConstraint(
            "NOT (state_json ? 'current_turn')",
            name="ck_sessions_state_json_no_current_turn",
        ),
    )


class ObjectModel(Base):
    __tablename__ = "objects"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    object_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__: tuple[Any, ...] = (
        PrimaryKeyConstraint("session_id", "object_id", name="pk_objects"),
        CheckConstraint(ALLOWED_OBJECT_TYPES_SQL, name="ck_objects_type_allowed"),
        CheckConstraint("jsonb_typeof(data) = 'object'", name="ck_objects_data_is_object"),
        Index("ix_objects_session_type", "session_id", "type"),
        Index(
            "uq_objects_world_constitution_singleton",
            "session_id",
            unique=True,
            postgresql_where=text("type = 'world_constitution'"),
        ),
        Index("ix_objects_session_name", "session_id", "name"),
        Index("ix_objects_session_location", "session_id", text("(data->>'location_id')")),
        Index("ix_objects_session_type_status", "session_id", "type", text("(data->>'status')")),
        Index("ix_objects_session_anchor_object_id", "session_id", text("(data->>'anchor_object_id')")),
        Index("ix_objects_session_fact_key", "session_id", text("(data->>'fact_key')")),
        Index(
            "uq_objects_consequence_window_key",
            "session_id",
            text("(data->>'window_key')"),
            unique=True,
            postgresql_where=text("type = '__consequence_window'"),
        ),
        Index(
            "uq_objects_director_agenda_key",
            "session_id",
            text("(data->>'agenda_key')"),
            unique=True,
            postgresql_where=text("type = '__director_agenda'"),
        ),
        Index(
            "uq_objects_director_action_key",
            "session_id",
            text("(data->>'action_key')"),
            unique=True,
            postgresql_where=text("type = '__director_action'"),
        ),
        Index(
            "ix_objects_consequence_window_runtime",
            "session_id",
            text("(data->>'status')"),
            text("(data->>'earliest_turn')"),
            text("(data->>'latest_turn')"),
            postgresql_where=text("type = '__consequence_window'"),
        ),
        Index(
            "ix_objects_director_agenda_runtime",
            "session_id",
            text("(data->>'status')"),
            text("(data->>'suppressed_until_turn')"),
            text("(data->>'priority_score')"),
            postgresql_where=text("type = '__director_agenda'"),
        ),
        Index(
            "ix_objects_data_gin_path",
            "data",
            postgresql_using="gin",
            postgresql_ops={"data": "jsonb_path_ops"},
        ),
        # NOTE: no unique constraint on (session_id, type, name) by design.
        # Name-based dedup is handled by semantic embeddings (NPC, zone, etc.)
        # when USE_EMBEDDINGS is enabled.  With embeddings off, duplicates
        # are possible but acceptable for the graph-based data model.
    )


class LinkModel(Base):
    __tablename__ = "links"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    link_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    from_object_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    to_object_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    valid_from_turn: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    valid_to_turn: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__: tuple[Any, ...] = (
        PrimaryKeyConstraint("session_id", "link_id", name="pk_links"),
        ForeignKeyConstraint(
            ["session_id", "from_object_id"],
            ["objects.session_id", "objects.object_id"],
            ondelete="CASCADE",
            name="fk_links_from_object",
        ),
        ForeignKeyConstraint(
            ["session_id", "to_object_id"],
            ["objects.session_id", "objects.object_id"],
            ondelete="CASCADE",
            name="fk_links_to_object",
        ),
        CheckConstraint("jsonb_typeof(data) = 'object'", name="ck_links_data_is_object"),
        CheckConstraint("valid_from_turn >= 0", name="ck_links_valid_from_turn_non_negative"),
        CheckConstraint(
            "(valid_to_turn IS NULL) OR (valid_from_turn <= valid_to_turn)",
            name="ck_links_valid_turn_window",
        ),
        Index("ix_links_session_type", "session_id", "type"),
        Index("ix_links_session_from", "session_id", "from_object_id"),
        Index("ix_links_session_to", "session_id", "to_object_id"),
        Index("ix_links_session_type_valid_to", "session_id", "type", "valid_to_turn"),
        Index(
            "uq_links_active_per_edge_type",
            "session_id",
            "from_object_id",
            "to_object_id",
            "type",
            unique=True,
            postgresql_where=text("valid_to_turn IS NULL"),
        ),
        Index(
            "uq_links_active_located_in_per_from",
            "session_id",
            "from_object_id",
            unique=True,
            postgresql_where=text("type = 'located_in' AND valid_to_turn IS NULL"),
        ),
    )


class EventModel(Base):
    __tablename__ = "events"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    scope_object_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__: tuple[Any, ...] = (
        PrimaryKeyConstraint("session_id", "event_id", name="pk_events"),
        CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_events_payload_is_object"),
        ForeignKeyConstraint(
            ["session_id", "turn_index"],
            ["turns.session_id", "turns.turn_index"],
            ondelete="CASCADE",
            name="fk_events_turn",
        ),
        ForeignKeyConstraint(
            ["session_id", "scope_object_id"],
            ["objects.session_id", "objects.object_id"],
            # SET NULL is intentional: the architecture guarantees "no data is
            # ever deleted, only closed", so this should never fire in normal
            # operation.  If an object is somehow removed (e.g. CASCADE from
            # session deletion), we keep the event row and accept losing the
            # spatial scope rather than cascade-deleting historical events.
            ondelete="SET NULL",
            name="fk_events_scope_object",
        ),
        Index("ix_events_session_turn", "session_id", "turn_index"),
    )


class TurnModel(Base):
    __tablename__ = "turns"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    turn_kind: Mapped[str] = mapped_column(Text, nullable=False, default="player", server_default=text("'player'"))
    user_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_object_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    triggered_by_turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    root_turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("session_id", "turn_index", name="pk_turns"),
        CheckConstraint(
            "turn_kind IN ('player', 'director')",
            name="ck_turns_turn_kind_allowed",
        ),
        CheckConstraint(
            "("
            "(turn_kind = 'director' AND (user_input IS NULL OR btrim(user_input) = '')) "
            "OR "
            "(turn_kind = 'player' AND user_input IS NOT NULL AND btrim(user_input) <> '')"
            ")",
            name="ck_turns_user_input_by_kind",
        ),
        Index("ix_turns_session_turn_kind", "session_id", "turn_kind"),
        Index("ix_turns_session_root_turn", "session_id", "root_turn_index"),
    )


class LlmTelemetryModel(Base):
    __tablename__ = "llm_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    origin_trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    request_type: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_cents: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    pricing_revision: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="ok", server_default=text("'ok'"))
    error_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    __table_args__ = (
        CheckConstraint(
            "(prompt_tokens IS NULL) OR (prompt_tokens >= 0)",
            name="ck_llm_telemetry_prompt_tokens_non_negative",
        ),
        CheckConstraint(
            "(completion_tokens IS NULL) OR (completion_tokens >= 0)",
            name="ck_llm_telemetry_completion_tokens_non_negative",
        ),
        CheckConstraint(
            "(total_tokens IS NULL) OR (total_tokens >= 0)",
            name="ck_llm_telemetry_total_tokens_non_negative",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="ck_llm_telemetry_latency_ms_non_negative",
        ),
        CheckConstraint(
            "status IN ('ok', 'error')",
            name="ck_llm_telemetry_status_allowed",
        ),
        CheckConstraint(
            "jsonb_typeof(meta_json) = 'object'",
            name="ck_llm_telemetry_meta_json_is_object",
        ),
        Index("ix_llm_telemetry_created_at", text("created_at DESC")),
        Index(
            "ix_llm_telemetry_session_turn_created",
            "session_id",
            "turn_index",
            text("created_at DESC"),
        ),
        Index("ix_llm_telemetry_request_type_created", "request_type", text("created_at DESC")),
        Index("ix_llm_telemetry_model_created", "model_name", text("created_at DESC")),
        Index("ix_llm_telemetry_trace_id", "trace_id"),
    )


class LlmTelemetryDailyRollupModel(Base):
    __tablename__ = "llm_telemetry_daily_rollups"

    day_utc: Mapped[date] = mapped_column(Date, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    request_type: Mapped[str] = mapped_column(Text, nullable=False)
    calls_total: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default=text("0"))
    errors_total: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default=text("0"))
    prompt_tokens_total: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default=text("0"))
    completion_tokens_total: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default=text("0"))
    total_tokens_total: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default=text("0"))
    cost_cents_total_snapshot: Mapped[float] = mapped_column(
        Numeric(14, 4), nullable=False, default=0.0, server_default=text("0")
    )
    latency_ms_avg: Mapped[float] = mapped_column(
        Numeric(12, 2), nullable=False, default=0.0, server_default=text("0")
    )
    latency_ms_p95: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint(
            "day_utc",
            "provider",
            "model_name",
            "request_type",
            name="pk_llm_telemetry_daily_rollups",
        ),
    )


class OutboxEventModel(Base):
    __tablename__ = "outbox_events"

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending", server_default=text("'pending'"))
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True
    )
    turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=8, server_default=text("8"))
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    dedupe_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'processed', 'failed')",
            name="ck_outbox_events_status_allowed",
        ),
        CheckConstraint("attempts >= 0", name="ck_outbox_events_attempts_non_negative"),
        CheckConstraint("max_attempts >= 1", name="ck_outbox_events_max_attempts_positive"),
        CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_outbox_events_payload_is_object"),
        Index("ix_outbox_events_status_available", "status", "available_at"),
        Index("ix_outbox_events_event_type_status", "event_type", "status"),
        Index("ix_outbox_events_session_turn", "session_id", "turn_index"),
        Index(
            "uq_outbox_events_dedupe_key",
            "dedupe_key",
            unique=True,
            postgresql_where=text("dedupe_key IS NOT NULL"),
        ),
    )


class SystemPromptRegistryModel(Base):
    __tablename__ = "system_prompts_registry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    module: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("uq_system_prompts_registry_module_version", "module", "version", unique=True),
        Index(
            "uq_system_prompts_registry_active_module",
            "module",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
        Index("ix_system_prompts_registry_module_active", "module", "is_active"),
    )


class SessionSnapshotModel(Base):
    __tablename__ = "session_snapshots"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    dump_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("session_id", "turn_index", name="pk_session_snapshots"),
    )


class ChronicleChunkModel(Base):
    __tablename__ = "chronicle_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    namespace: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="chronicle_output",
        server_default=text("'chronicle_output'"),
    )
    zone_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    in_game_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    in_game_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_snippet: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__: tuple[Any, ...] = (
        ForeignKeyConstraint(
            ["session_id", "zone_id"],
            ["objects.session_id", "objects.object_id"],
            # SET NULL: same reasoning as EventModel.scope_object_id — keep
            # the chronicle chunk even if the zone object is ever removed.
            ondelete="SET NULL",
            name="fk_chronicle_chunks_zone_object",
        ),
        Index(
            "uq_chronicle_chunks_session_turn_namespace",
            "session_id",
            "turn_index",
            "namespace",
            unique=True,
        ),
        Index("ix_chronicle_chunks_session_namespace_turn", "session_id", "namespace", "turn_index"),
    )
    if EMBEDDING_DIM <= 2000:
        __table_args__ += (
            Index(
                "ix_chronicle_chunks_embedding_hnsw",
                "embedding",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )


class ObjectEmbeddingModel(Base):
    __tablename__ = "object_embeddings"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    object_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__: tuple[Any, ...] = (
        PrimaryKeyConstraint("session_id", "object_id", "namespace", name="pk_object_embeddings"),
        ForeignKeyConstraint(
            ["session_id", "object_id"],
            ["objects.session_id", "objects.object_id"],
            ondelete="CASCADE",
            name="fk_object_embeddings_object",
        ),
        Index("ix_object_embeddings_session_namespace", "session_id", "namespace"),
    )
    if EMBEDDING_DIM <= 2000:
        __table_args__ += (
            Index(
                "ix_object_embeddings_embedding_hnsw",
                "embedding",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )
