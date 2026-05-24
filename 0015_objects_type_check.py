"""Enforce allowed values for objects.type

Revision ID: 0015_objects_type_check
Revises: 0014_jsonb_data_object_checks
Create Date: 2026-03-03 10:35:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0015_objects_type_check"
down_revision: Union[str, None] = "0014_jsonb_data_object_checks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Keep internal system object types that are created outside patch ops.
ALLOWED = "player,npc,zone,item,quest,claim,faction,__memory_seed,__world_prompt_chunk"


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def _build_constraint_sql(allowed_values: list[str]) -> str:
    values_sql = ",".join(_sql_quote(item) for item in allowed_values)
    return f"CHECK (type IN ({values_sql}))"


def upgrade() -> None:
    bind = op.get_bind()
    base_allowed = [item for item in ALLOWED.split(",") if item]
    legacy_allowed = _collect_legacy_types(bind, base_allowed)
    effective_allowed = [*base_allowed, *legacy_allowed]

    op.execute("ALTER TABLE objects DROP CONSTRAINT IF EXISTS ck_objects_type_allowed")
    op.execute(
        f"""
        ALTER TABLE objects ADD CONSTRAINT ck_objects_type_allowed
        {_build_constraint_sql(effective_allowed)}
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE objects DROP CONSTRAINT ck_objects_type_allowed")
