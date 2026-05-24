"""Ensure session_snapshots table exists

Revision ID: 0006_ensure_session_snapshots
Revises: 0005_fix_events_scope_fk
Create Date: 2026-03-01 21:20:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006_ensure_session_snapshots"
down_revision: Union[str, None] = "0005_fix_events_scope_fk"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("session_snapshots"):
        return

    op.create_table(
        "session_snapshots",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("dump_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id", "turn_index", name="pk_session_snapshots"),
    )


def downgrade() -> None:
    # No-op: table may be managed by the initial migration in existing installations.
    return
