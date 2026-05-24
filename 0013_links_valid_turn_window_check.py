"""Enforce valid_from_turn <= valid_to_turn for links

Revision ID: 0013_links_valid_turn_window
Revises: 0012_session_turn_consistency
Create Date: 2026-03-02 15:15:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013_links_valid_turn_window"
down_revision: Union[str, None] = "0012_session_turn_consistency"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Repair invalid historical rows before adding strict constraint.
    # For impossible windows (to < from), close at from-turn.
    op.execute(
        """
        UPDATE links
        SET valid_to_turn = valid_from_turn
        WHERE valid_to_turn IS NOT NULL
          AND valid_to_turn < valid_from_turn
        """
    )

    op.create_check_constraint(
        "ck_links_valid_turn_window",
        "links",
        "(valid_to_turn IS NULL) OR (valid_from_turn <= valid_to_turn)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_links_valid_turn_window", "links", type_="check")

