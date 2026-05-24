"""Set nullable object scope FKs to ON DELETE SET NULL

Revision ID: 0008_set_null_object_scope_fks
Revises: 0007_chronicle_input_namespace
Create Date: 2026-03-02 05:00:00.000000

"""

from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0008_set_null_object_scope_fks"
down_revision: Union[str, None] = "0007_chronicle_input_namespace"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


EVENTS_SCOPE_FK = "fk_events_scope_object"
CHRONICLE_ZONE_FK = "fk_chronicle_chunks_zone_object"


def _get_fk(inspector: Any, table_name: str, fk_name: str) -> dict | None:
    for fk in inspector.get_foreign_keys(table_name):
        if fk.get("name") == fk_name:
            return fk
    return None


def _fk_has_set_null(fk: dict | None) -> bool:
    if not fk:
        return False
    options = fk.get("options") or {}
    ondelete = options.get("ondelete")
    if not isinstance(ondelete, str):
        return False
    return ondelete.strip().upper() == "SET NULL"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_names = set(inspector.get_table_names())
    if "events" in table_names:
        fk = _get_fk(inspector, "events", EVENTS_SCOPE_FK)
        if fk is None or not _fk_has_set_null(fk):
            if fk is not None:
                op.drop_constraint(EVENTS_SCOPE_FK, "events", type_="foreignkey")
            op.create_foreign_key(
                EVENTS_SCOPE_FK,
                "events",
                "objects",
                ["session_id", "scope_object_id"],
                ["session_id", "object_id"],
                ondelete="SET NULL",
            )

    if "chronicle_chunks" in table_names:
        fk = _get_fk(inspector, "chronicle_chunks", CHRONICLE_ZONE_FK)
        if fk is None or not _fk_has_set_null(fk):
            if fk is not None:
                op.drop_constraint(CHRONICLE_ZONE_FK, "chronicle_chunks", type_="foreignkey")
            op.create_foreign_key(
                CHRONICLE_ZONE_FK,
                "chronicle_chunks",
                "objects",
                ["session_id", "zone_id"],
                ["session_id", "object_id"],
                ondelete="SET NULL",
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_names = set(inspector.get_table_names())
    if "events" in table_names:
        fk = _get_fk(inspector, "events", EVENTS_SCOPE_FK)
        if fk is not None:
            op.drop_constraint(EVENTS_SCOPE_FK, "events", type_="foreignkey")
        op.create_foreign_key(
            EVENTS_SCOPE_FK,
            "events",
            "objects",
            ["session_id", "scope_object_id"],
            ["session_id", "object_id"],
        )

    if "chronicle_chunks" in table_names:
        fk = _get_fk(inspector, "chronicle_chunks", CHRONICLE_ZONE_FK)
        if fk is not None:
            op.drop_constraint(CHRONICLE_ZONE_FK, "chronicle_chunks", type_="foreignkey")
        op.create_foreign_key(
            CHRONICLE_ZONE_FK,
            "chronicle_chunks",
            "objects",
            ["session_id", "zone_id"],
            ["session_id", "object_id"],
        )
