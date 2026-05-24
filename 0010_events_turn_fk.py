"""Add events turn_index foreign key to turns

Revision ID: 0010_events_turn_fk
Revises: 0009_state_json_refs
Create Date: 2026-03-02 07:00:00.000000

"""

from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0010_events_turn_fk"
down_revision: Union[str, None] = "0009_state_json_refs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


EVENTS_TURN_FK = "fk_events_turn"


def _get_fk(inspector: Any, table_name: str, fk_name: str) -> dict | None:
    for fk in inspector.get_foreign_keys(table_name):
        if fk.get("name") == fk_name:
            return fk
    return None


def _is_events_turn_fk(fk: dict | None) -> bool:
    if not fk:
        return False
    if fk.get("referred_table") != "turns":
        return False
    constrained = fk.get("constrained_columns") or []
    referred = fk.get("referred_columns") or []
    return constrained == ["session_id", "turn_index"] and referred == ["session_id", "turn_index"]


def _fk_has_cascade(fk: dict | None) -> bool:
    if not fk:
        return False
    options = fk.get("options") or {}
    ondelete = options.get("ondelete")
    if not isinstance(ondelete, str):
        return False
    return ondelete.strip().upper() == "CASCADE"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "events" not in table_names or "turns" not in table_names:
        return

    # Prevent upgrade failures when legacy rows reference missing turns.
    op.execute(
        """
        DELETE FROM events e
        WHERE NOT EXISTS (
            SELECT 1
            FROM turns t
            WHERE t.session_id = e.session_id
              AND t.turn_index = e.turn_index
        )
        """
    )

    existing_named = _get_fk(inspector, "events", EVENTS_TURN_FK)
    existing_equivalent = None
    for fk in inspector.get_foreign_keys("events"):
        if _is_events_turn_fk(fk):
            existing_equivalent = fk
            break

    existing = existing_named or existing_equivalent
    if existing is not None and _fk_has_cascade(existing):
        return

    if existing is not None and existing.get("name"):
        op.drop_constraint(existing["name"], "events", type_="foreignkey")

    op.create_foreign_key(
        EVENTS_TURN_FK,
        "events",
        "turns",
        ["session_id", "turn_index"],
        ["session_id", "turn_index"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "events" not in table_names:
        return

    existing = _get_fk(inspector, "events", EVENTS_TURN_FK)
    if existing is not None:
        op.drop_constraint(EVENTS_TURN_FK, "events", type_="foreignkey")
        return

    for fk in inspector.get_foreign_keys("events"):
        if _is_events_turn_fk(fk) and fk.get("name"):
            op.drop_constraint(fk["name"], "events", type_="foreignkey")
            return
