"""Add unique active link index per edge tuple

Revision ID: 0011_active_link_edge_uq
Revises: 0010_events_turn_fk
Create Date: 2026-03-02 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0011_active_link_edge_uq"
down_revision: Union[str, None] = "0010_events_turn_fk"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Backfill: keep one active link per (session_id, from, to, type)
    # so unique partial index can be created on existing datasets.
    op.execute(
        """
        WITH ranked AS (
            SELECT
                session_id,
                link_id,
                valid_from_turn,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id, from_object_id, to_object_id, type
                    ORDER BY valid_from_turn DESC, created_at DESC, link_id DESC
                ) AS rn
            FROM links
            WHERE valid_to_turn IS NULL
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
        "uq_links_active_per_edge_type",
        "links",
        ["session_id", "from_object_id", "to_object_id", "type"],
        unique=True,
        postgresql_where=sa.text("valid_to_turn IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_links_active_per_edge_type", table_name="links")

