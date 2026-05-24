"""Allow continuity memory object types and indexes.

Revision ID: 0026_continuity_memory_objects
Revises: 0025_world_constitution_singleton
Create Date: 2026-03-07 20:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0026_continuity_memory_objects"
down_revision: Union[str, None] = "0025_world_const_singleton"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PREV_ALLOWED = (
    "player,npc,zone,item,faction,quest,claim,world_constitution,"
    "__world_prompt_chunk,__memory_seed,__session_summary,__narrative_spine,__memory_consolidated"
)
NEW_ALLOWED = PREV_ALLOWED + ",__entity_memory,__callback_memory"


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


def _apply_constraint(values_csv: str) -> None:
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


def upgrade() -> None:
    _apply_constraint(NEW_ALLOWED)
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_objects_session_anchor_object_id
        ON objects (session_id, (data->>'anchor_object_id'))
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_objects_session_fact_key
        ON objects (session_id, (data->>'fact_key'))
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_objects_session_fact_key")
    op.execute("DROP INDEX IF EXISTS ix_objects_session_anchor_object_id")
    _apply_constraint(PREV_ALLOWED)
