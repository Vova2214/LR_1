"""Fix events scope FK delete behavior

Revision ID: 0005_fix_events_scope_fk
Revises: 0004_add_hnsw_large_vec
Create Date: 2026-03-01 19:40:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_fix_events_scope_fk"
down_revision: Union[str, None] = "0004_add_hnsw_large_vec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("fk_events_scope_object", "events", type_="foreignkey")
    op.create_foreign_key(
        "fk_events_scope_object",
        "events",
        "objects",
        ["session_id", "scope_object_id"],
        ["session_id", "object_id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_events_scope_object", "events", type_="foreignkey")
    op.create_foreign_key(
        "fk_events_scope_object",
        "events",
        "objects",
        ["session_id", "scope_object_id"],
        ["session_id", "object_id"],
        ondelete="SET NULL",
    )
