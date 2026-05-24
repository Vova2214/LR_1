"""Add GIN index for objects.data JSONB lookups.

Revision ID: 0022_obj_data_gin
Revises: 0021_obj_expr_idx
Create Date: 2026-03-04 21:20:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0022_obj_data_gin"
down_revision: Union[str, None] = "0021_obj_expr_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_objects_data_gin_path
        ON objects USING gin (data jsonb_path_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_objects_data_gin_path")
