"""Add state-first director foundation schema and indexes.

Revision ID: 0029_state_first_director_foundation
Revises: 0028_memory_eval_report
Create Date: 2026-04-03 15:30:00.000000

"""

from __future__ import annotations

import hashlib
import uuid
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0029_state_first_dir_found"
down_revision: Union[str, None] = "0028_memory_eval_report"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PREV_ALLOWED = (
    "player,npc,zone,item,faction,quest,claim,world_constitution,"
    "__world_prompt_chunk,__memory_event,__memory_fact,__memory_bundle,"
    "__story_obligation,__memory_conflict_edge,__memory_review_report,"
    "__memory_evaluation_report,__session_summary,__narrative_spine,"
    "__entity_memory,__callback_memory"
)
NEW_ALLOWED = (
    "player,npc,zone,item,faction,quest,claim,world_constitution,"
    "__world_prompt_chunk,__memory_event,__memory_fact,__memory_bundle,"
    "__story_obligation,__memory_conflict_edge,__memory_review_report,"
    "__memory_evaluation_report,__session_summary,__narrative_spine,"
    "__entity_memory,__callback_memory,__consequence_window,"
    "__director_agenda,__director_action"
)


def _legacy_window_key(session_id: uuid.UUID, seed_id: str) -> str:
    digest = hashlib.sha256(f"{session_id}:{seed_id}".encode("utf-8")).hexdigest()
    return f"legacy:{digest}"


def _collect_legacy_types(bind: sa.engine.Connection, base_allowed: list[str]) -> list[str]:
    known = set(base_allowed)
    rows = bind.execute(sa.text("SELECT DISTINCT type FROM objects WHERE type IS NOT NULL")).all()

    legacy: list[str] = []
    for (raw_value,) in rows:
        if not isinstance(raw_value, str):
            continue
        candidate = raw_value.strip()
        if not candidate or candidate in known or candidate in legacy:
            continue
        legacy.append(candidate)
    legacy.sort()
    return legacy


def _constraint_sql_from_values(values: list[str]) -> str:
    escaped = ",".join("'" + item.replace("'", "''") + "'" for item in values)
    return f"CHECK (type IN ({escaped}))"


def _apply_objects_constraint(values_csv: str) -> None:
    bind = op.get_bind()
    base_allowed = [item for item in values_csv.split(",") if item]
    legacy_allowed = _collect_legacy_types(bind, base_allowed)
    effective_allowed = [*base_allowed, *legacy_allowed]

    op.execute("ALTER TABLE objects DROP CONSTRAINT IF EXISTS ck_objects_type_allowed")
    op.execute(
        f"""
        ALTER TABLE objects ADD CONSTRAINT ck_objects_type_allowed
        {_constraint_sql_from_values(effective_allowed)}
        """
    )


def _coerce_legacy_window_payload(
    session_id: uuid.UUID,
    raw_seed: object,
) -> dict[str, object] | None:
    if not isinstance(raw_seed, dict):
        return None

    seed_id = str(raw_seed.get("id") or "").strip()
    if not seed_id:
        return None

    earliest_turn = raw_seed.get("earliest_turn")
    latest_turn = raw_seed.get("latest_turn")
    if not isinstance(earliest_turn, int) or isinstance(earliest_turn, bool):
        return None
    if not isinstance(latest_turn, int) or isinstance(latest_turn, bool):
        return None
    if latest_turn < earliest_turn:
        return None

    priority = str(raw_seed.get("priority") or "med").strip().lower()
    if priority not in {"low", "med", "high"}:
        priority = "med"

    shows = raw_seed.get("shows", 0)
    if not isinstance(shows, int) or isinstance(shows, bool):
        shows = 0
    max_shows = raw_seed.get("max_shows", 2)
    if not isinstance(max_shows, int) or isinstance(max_shows, bool):
        max_shows = 2
    depth = raw_seed.get("depth", 0)
    if not isinstance(depth, int) or isinstance(depth, bool):
        depth = 0

    return {
        "window_key": _legacy_window_key(session_id, seed_id),
        "status": "open",
        "priority": priority,
        "earliest_turn": earliest_turn,
        "latest_turn": latest_turn,
        "shows": max(shows, 0),
        "max_shows": max(max_shows, 0),
        "depth": max(depth, 0),
        "target_object_ids": [],
        "anchor_object_ids": [
            str(item).strip()
            for item in list(raw_seed.get("anchor_object_ids") or [])
            if str(item).strip()
        ],
        "source_turn": None,
        "source_event_ids": [],
        "source_fact_keys": [],
        "opened_obligation_keys": [],
        "created_by_turn_kind": "legacy_backfill",
        "provenance_status": "legacy_backfill",
        "reason_kind": "legacy_backfill",
        "text": str(raw_seed.get("text") or "").strip(),
    }


def _backfill_legacy_consequence_windows() -> None:
    bind = op.get_bind()
    objects_table = sa.table(
        "objects",
        sa.column("session_id", sa.UUID()),
        sa.column("object_id", sa.UUID()),
        sa.column("type", sa.Text()),
        sa.column("name", sa.Text()),
        sa.column("data", sa.JSON()),
    )

    existing_rows = bind.execute(
        sa.text(
            """
            SELECT session_id, data->>'window_key'
            FROM objects
            WHERE type = '__consequence_window'
            """
        )
    ).all()
    existing_keys = {
        (str(session_id), str(window_key or "").strip())
        for session_id, window_key in existing_rows
        if str(window_key or "").strip()
    }

    session_rows = bind.execute(sa.text("SELECT id, state_json FROM sessions")).all()
    insert_rows: list[dict[str, object]] = []
    for raw_session_id, raw_state_json in session_rows:
        try:
            session_id = raw_session_id if isinstance(raw_session_id, uuid.UUID) else uuid.UUID(str(raw_session_id))
        except (TypeError, ValueError, AttributeError):
            continue
        state_json = raw_state_json if isinstance(raw_state_json, dict) else {}
        raw_pending = state_json.get("pending_consequences")
        if not isinstance(raw_pending, list):
            continue
        for raw_seed in raw_pending:
            payload = _coerce_legacy_window_payload(session_id, raw_seed)
            if payload is None:
                continue
            window_key = str(payload.get("window_key") or "").strip()
            dedupe_key = (str(session_id), window_key)
            if not window_key or dedupe_key in existing_keys:
                continue
            existing_keys.add(dedupe_key)
            insert_rows.append(
                {
                    "session_id": session_id,
                    "object_id": uuid.uuid4(),
                    "type": "__consequence_window",
                    "name": f"consequence_window:{window_key}",
                    "data": payload,
                }
            )

    if insert_rows:
        bind.execute(sa.insert(objects_table), insert_rows)


def upgrade() -> None:
    _apply_objects_constraint(NEW_ALLOWED)

    op.add_column(
        "turns",
        sa.Column("turn_kind", sa.Text(), nullable=False, server_default=sa.text("'player'")),
    )
    op.add_column("turns", sa.Column("actor_object_id", sa.UUID(), nullable=True))
    op.add_column("turns", sa.Column("triggered_by_turn_index", sa.Integer(), nullable=True))
    op.add_column("turns", sa.Column("root_turn_index", sa.Integer(), nullable=True))
    op.alter_column("turns", "user_input", existing_type=sa.Text(), nullable=True)

    op.execute("UPDATE turns SET turn_kind = 'player' WHERE turn_kind IS NULL")
    op.execute("UPDATE turns SET root_turn_index = turn_index WHERE root_turn_index IS NULL")

    op.create_check_constraint(
        "ck_turns_turn_kind_allowed",
        "turns",
        "turn_kind IN ('player', 'director')",
    )
    op.create_check_constraint(
        "ck_turns_user_input_by_kind",
        "turns",
        "("
        "(turn_kind = 'director' AND (user_input IS NULL OR btrim(user_input) = '')) "
        "OR "
        "(turn_kind = 'player' AND user_input IS NOT NULL AND btrim(user_input) <> '')"
        ")",
    )
    op.create_index("ix_turns_session_turn_kind", "turns", ["session_id", "turn_kind"])
    op.create_index("ix_turns_session_root_turn", "turns", ["session_id", "root_turn_index"])

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_objects_consequence_window_key
        ON objects (session_id, (data->>'window_key'))
        WHERE type = '__consequence_window'
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_objects_director_agenda_key
        ON objects (session_id, (data->>'agenda_key'))
        WHERE type = '__director_agenda'
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_objects_director_action_key
        ON objects (session_id, (data->>'action_key'))
        WHERE type = '__director_action'
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_objects_consequence_window_runtime
        ON objects (session_id, (data->>'status'), (data->>'earliest_turn'), (data->>'latest_turn'))
        WHERE type = '__consequence_window'
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_objects_director_agenda_runtime
        ON objects (session_id, (data->>'status'), (data->>'suppressed_until_turn'), (data->>'priority_score'))
        WHERE type = '__director_agenda'
        """
    )
    _backfill_legacy_consequence_windows()

    op.alter_column("turns", "turn_kind", server_default=None)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_objects_director_agenda_runtime")
    op.execute("DROP INDEX IF EXISTS ix_objects_consequence_window_runtime")
    op.execute("DROP INDEX IF EXISTS uq_objects_director_action_key")
    op.execute("DROP INDEX IF EXISTS uq_objects_director_agenda_key")
    op.execute("DROP INDEX IF EXISTS uq_objects_consequence_window_key")

    op.drop_index("ix_turns_session_root_turn", table_name="turns")
    op.drop_index("ix_turns_session_turn_kind", table_name="turns")
    op.drop_constraint("ck_turns_user_input_by_kind", "turns", type_="check")
    op.drop_constraint("ck_turns_turn_kind_allowed", "turns", type_="check")

    op.execute("UPDATE turns SET user_input = '[legacy_turn]' WHERE user_input IS NULL")
    op.alter_column("turns", "user_input", existing_type=sa.Text(), nullable=False)
    op.drop_column("turns", "root_turn_index")
    op.drop_column("turns", "triggered_by_turn_index")
    op.drop_column("turns", "actor_object_id")
    op.drop_column("turns", "turn_kind")

    _apply_objects_constraint(PREV_ALLOWED)
