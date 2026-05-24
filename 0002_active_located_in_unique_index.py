"""Add unique active located_in index

Revision ID: 0002_active_located_uq
Revises: 0001_initial_graph_schema
Create Date: 2026-03-01 03:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0002_active_located_uq"
down_revision: Union[str, None] = "0001_initial_graph_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Backfill: keep one active located_in per (session_id, from_object_id)
    # so unique partial index can be created on existing datasets.
    op.execute(
        """
        WITH ranked AS (
            SELECT
                session_id,
                link_id,
                valid_from_turn,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id, from_object_id
                    ORDER BY valid_from_turn DESC, created_at DESC, link_id DESC
                ) AS rn
            FROM links
            WHERE type = 'located_in' AND valid_to_turn IS NULL
        )
        UPDATE links AS l
        SET valid_to_turn = l.valid_from_turn
        FROM ranked AS r
        WHERE r.rn > 1
          AND l.session_id = r.session_id
          AND l.link_id = r.link_id
          AND l.valid_to_turn IS NULL
        """
    )

    op.create_index(
        "uq_links_active_located_in_per_from",
        "links",
        ["session_id", "from_object_id"],
        unique=True,
        postgresql_where=sa.text("type = 'located_in' AND valid_to_turn IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_links_active_located_in_per_from", table_name="links")
