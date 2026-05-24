"""Extend objects.type allowlist with summary/consolidation internal objects.

Revision ID: 0016_internal_summary
Revises: 0015_objects_type_check
Create Date: 2026-03-03 15:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0016_internal_summary"
down_revision: Union[str, None] = "0015_objects_type_check"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PREV_ALLOWED = "player,npc,zone,item,quest,claim,faction,__memory_seed,__world_prompt_chunk"
NEW_ALLOWED = (
    "player,npc,zone,item,quest,claim,faction,"
    "__memory_seed,__world_prompt_chunk,__session_summary,__memory_consolidated"
)

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


def upgrade() -> None:
    bind = op.get_bind()
    base_allowed = [item for item in NEW_ALLOWED.split(",") if item]
    legacy_allowed = _collect_legacy_types(bind, base_allowed)
    effective_allowed = [*base_allowed, *legacy_allowed]

    op.execute("ALTER TABLE objects DROP CONSTRAINT IF EXISTS ck_objects_type_allowed")
    op.execute(
        f"""
        ALTER TABLE objects ADD CONSTRAINT ck_objects_type_allowed
        {_constraint_sql_from_values(effective_allowed)}
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    base_allowed = [item for item in PREV_ALLOWED.split(",") if item]
    legacy_allowed = _collect_legacy_types(bind, base_allowed)
    effective_allowed = [*base_allowed, *legacy_allowed]

    op.execute("ALTER TABLE objects DROP CONSTRAINT IF EXISTS ck_objects_type_allowed")
    op.execute(
        f"""
        ALTER TABLE objects ADD CONSTRAINT ck_objects_type_allowed
        {_constraint_sql_from_values(effective_allowed)}
        """
    )
