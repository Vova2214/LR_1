"""Add expression indexes for objects location/status lookup.

Revision ID: 0021_obj_expr_idx
Revises: 0020_embed_hnsw_params
Create Date: 2026-03-04 20:55:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0021_obj_expr_idx"
down_revision: Union[str, None] = "0020_embed_hnsw_params"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_objects_session_location
        ON objects (session_id, (data->>'location_id'))
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_objects_session_type_status
        ON objects (session_id, type, (data->>'status'))
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_objects_session_type_status")
    op.execute("DROP INDEX IF EXISTS ix_objects_session_location")
