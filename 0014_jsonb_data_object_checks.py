"""Enforce objects.data and links.data as JSON objects

Revision ID: 0014_jsonb_data_object_checks
Revises: 0013_links_valid_turn_window
Create Date: 2026-03-02 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014_jsonb_data_object_checks"
down_revision: Union[str, None] = "0013_links_valid_turn_window"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE objects
        SET data = '{}'::jsonb
        WHERE data IS NULL OR jsonb_typeof(data) <> 'object'
        """
    )
    op.execute(
        """
        UPDATE links
        SET data = '{}'::jsonb
        WHERE data IS NULL OR jsonb_typeof(data) <> 'object'
        """
    )

    op.create_check_constraint(
        "ck_objects_data_is_object",
        "objects",
        "jsonb_typeof(data) = 'object'",
    )
    op.create_check_constraint(
        "ck_links_data_is_object",
        "links",
        "jsonb_typeof(data) = 'object'",
    )


def downgrade() -> None:
    op.drop_constraint("ck_links_data_is_object", "links", type_="check")
    op.drop_constraint("ck_objects_data_is_object", "objects", type_="check")

