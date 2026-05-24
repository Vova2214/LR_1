"""Add llm telemetry, outbox events, and system prompt registry.

Revision ID: 0023_llm_obs_outbox_prompts
Revises: 0022_obj_data_gin
Create Date: 2026-03-05 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0023_llm_obs_outbox_prompts"
down_revision: Union[str, None] = "0022_obj_data_gin"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_telemetry",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column("origin_trace_id", sa.Text(), nullable=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("turn_index", sa.Integer(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("request_type", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_cents", sa.Numeric(12, 4), nullable=True),
        sa.Column("pricing_revision", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'ok'")),
        sa.Column("error_type", sa.Text(), nullable=True),
        sa.Column(
            "meta_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="SET NULL"),
        sa.CheckConstraint(
            "(prompt_tokens IS NULL) OR (prompt_tokens >= 0)",
            name="ck_llm_telemetry_prompt_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "(completion_tokens IS NULL) OR (completion_tokens >= 0)",
            name="ck_llm_telemetry_completion_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "(total_tokens IS NULL) OR (total_tokens >= 0)",
            name="ck_llm_telemetry_total_tokens_non_negative",
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_llm_telemetry_latency_ms_non_negative"),
        sa.CheckConstraint("status IN ('ok', 'error')", name="ck_llm_telemetry_status_allowed"),
        sa.CheckConstraint("jsonb_typeof(meta_json) = 'object'", name="ck_llm_telemetry_meta_json_is_object"),
    )
    op.execute("CREATE INDEX ix_llm_telemetry_created_at ON llm_telemetry (created_at DESC)")
    op.execute(
        "CREATE INDEX ix_llm_telemetry_session_turn_created "
        "ON llm_telemetry (session_id, turn_index, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_llm_telemetry_request_type_created "
        "ON llm_telemetry (request_type, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_llm_telemetry_model_created "
        "ON llm_telemetry (model_name, created_at DESC)"
    )
    op.create_index("ix_llm_telemetry_trace_id", "llm_telemetry", ["trace_id"], unique=False)

    op.create_table(
        "llm_telemetry_daily_rollups",
        sa.Column("day_utc", sa.Date(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("request_type", sa.Text(), nullable=False),
        sa.Column("calls_total", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("errors_total", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("prompt_tokens_total", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("completion_tokens_total", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_tokens_total", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("cost_cents_total_snapshot", sa.Numeric(14, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("latency_ms_avg", sa.Numeric(12, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("latency_ms_p95", sa.Numeric(12, 2), nullable=True),
        sa.PrimaryKeyConstraint(
            "day_utc",
            "provider",
            "model_name",
            "request_type",
            name="pk_llm_telemetry_daily_rollups",
        ),
    )

    op.create_table(
        "outbox_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("turn_index", sa.Integer(), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default=sa.text("8")),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column("dedupe_key", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'processed', 'failed')",
            name="ck_outbox_events_status_allowed",
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_outbox_events_attempts_non_negative"),
        sa.CheckConstraint("max_attempts >= 1", name="ck_outbox_events_max_attempts_positive"),
        sa.CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_outbox_events_payload_is_object"),
    )
    op.create_index("ix_outbox_events_status_available", "outbox_events", ["status", "available_at"], unique=False)
    op.create_index("ix_outbox_events_event_type_status", "outbox_events", ["event_type", "status"], unique=False)
    op.create_index("ix_outbox_events_session_turn", "outbox_events", ["session_id", "turn_index"], unique=False)
    op.create_index(
        "uq_outbox_events_dedupe_key",
        "outbox_events",
        ["dedupe_key"],
        unique=True,
        postgresql_where=sa.text("dedupe_key IS NOT NULL"),
    )

    op.create_table(
        "system_prompts_registry",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("module", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index(
        "uq_system_prompts_registry_module_version",
        "system_prompts_registry",
        ["module", "version"],
        unique=True,
    )
    op.create_index(
        "uq_system_prompts_registry_active_module",
        "system_prompts_registry",
        ["module"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "ix_system_prompts_registry_module_active",
        "system_prompts_registry",
        ["module", "is_active"],
        unique=False,
    )

    prompt_rows = [
        (
            "narrator_base",
            "You are a strict game narrator. Return ONLY a valid JSON object with narration, choices, proposed_updates, memory_seeds, consequence_seeds, resolved_consequence_ids, zone_scope, in_game_time, and turn_weight.",
        ),
        (
            "narrator_text_only",
            "You are a game narrator for a text RPG. Return only narration text. Keep it concrete and consistent with world rules.",
        ),
        (
            "librarian_validator",
            "You are a strict validator-corrector. Return only valid narrator JSON and fix schema/logic errors conservatively.",
        ),
        (
            "deepseek_patch_generator",
            "You are a structured data generator for a text RPG game engine. Return ONLY valid narrator JSON and coherent proposed_updates.",
        ),
        (
            "deepseek_consequence_extension",
            "Consequence extension: emit bounded consequence_seeds and resolved_consequence_ids aligned with current turn context.",
        ),
    ]

    bind = op.get_bind()
    for module_name, prompt_text in prompt_rows:
        bind.execute(
            sa.text(
                """
                INSERT INTO system_prompts_registry (
                    module,
                    version,
                    prompt_text,
                    is_active,
                    created_at,
                    activated_at,
                    created_by,
                    notes
                )
                VALUES (
                    :module,
                    1,
                    :prompt_text,
                    false,
                    now(),
                    NULL,
                    'migration:0023',
                    'default seed'
                )
                ON CONFLICT (module, version) DO NOTHING
                """
            ),
            {"module": module_name, "prompt_text": prompt_text},
        )


def downgrade() -> None:
    op.drop_index("ix_system_prompts_registry_module_active", table_name="system_prompts_registry")
    op.drop_index("uq_system_prompts_registry_active_module", table_name="system_prompts_registry")
    op.drop_index("uq_system_prompts_registry_module_version", table_name="system_prompts_registry")
    op.drop_table("system_prompts_registry")

    op.drop_index("uq_outbox_events_dedupe_key", table_name="outbox_events")
    op.drop_index("ix_outbox_events_session_turn", table_name="outbox_events")
    op.drop_index("ix_outbox_events_event_type_status", table_name="outbox_events")
    op.drop_index("ix_outbox_events_status_available", table_name="outbox_events")
    op.drop_table("outbox_events")

    op.drop_table("llm_telemetry_daily_rollups")

    op.drop_index("ix_llm_telemetry_trace_id", table_name="llm_telemetry")
    op.execute("DROP INDEX IF EXISTS ix_llm_telemetry_model_created")
    op.execute("DROP INDEX IF EXISTS ix_llm_telemetry_request_type_created")
    op.execute("DROP INDEX IF EXISTS ix_llm_telemetry_session_turn_created")
    op.execute("DROP INDEX IF EXISTS ix_llm_telemetry_created_at")
    op.drop_table("llm_telemetry")
