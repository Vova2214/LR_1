"""Enforce a singleton world_constitution object per session.

Revision ID: 0025_world_const_singleton
Revises: 0024_tracking_quest_backfill
Create Date: 2026-03-05 22:40:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0025_world_const_singleton"
down_revision: Union[str, None] = "0024_tracking_quest_backfill"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        WITH ranked AS (
            SELECT
                session_id,
                object_id,
                row_number() OVER (
                    PARTITION BY session_id
                    ORDER BY created_at DESC, object_id DESC
                ) AS rn
            FROM objects
            WHERE type = 'world_constitution'
        )
        DELETE FROM objects AS o
        USING ranked AS r
        WHERE o.session_id = r.session_id
          AND o.object_id = r.object_id
          AND r.rn > 1
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_objects_world_constitution_singleton
        ON objects (session_id)
        WHERE type = 'world_constitution'
        """
    )


def downgrade() -> None:
    op.drop_index("uq_objects_world_constitution_singleton", table_name="objects")
