"""Backfill tracking_quest links for active quests.

Revision ID: 0024_tracking_quest_backfill
Revises: 0023_llm_obs_outbox_prompts
Create Date: 2026-03-05 03:00:00.000000

"""

from __future__ import annotations

import uuid
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0024_tracking_quest_backfill"
down_revision: Union[str, None] = "0023_llm_obs_outbox_prompts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TERMINAL_STATUSES = (
    "inactive",
    "completed",
    "done",
    "closed",
    "failed",
    "archived",
)


def upgrade() -> None:
    bind = op.get_bind()

    session_rows = bind.execute(
        sa.text(
            """
            SELECT id, state_json ->> 'player_object_id' AS player_object_id
            FROM sessions
            """
        )
    ).all()

    insert_rows: list[dict[str, object]] = []
    for session_id, player_object_id_raw in session_rows:
        try:
            player_object_id = uuid.UUID(str(player_object_id_raw or "").strip())
        except (TypeError, ValueError, AttributeError):
            continue

        quest_rows = bind.execute(
            sa.text(
                """
                SELECT o.object_id
                FROM objects AS o
                WHERE o.session_id = :session_id
                  AND o.type = 'quest'
                  AND (
                        o.data ->> 'status' IS NULL
                        OR lower(o.data ->> 'status') NOT IN :terminal_statuses
                      )
                  AND NOT EXISTS (
                        SELECT 1
                        FROM links AS l
                        WHERE l.session_id = o.session_id
                          AND l.from_object_id = :player_object_id
                          AND l.to_object_id = o.object_id
                          AND l.type = 'tracking_quest'
                          AND l.valid_to_turn IS NULL
                  )
                """
            ).bindparams(sa.bindparam("terminal_statuses", expanding=True)),
            {
                "session_id": session_id,
                "player_object_id": player_object_id,
                "terminal_statuses": list(_TERMINAL_STATUSES),
            },
        ).all()

        for (quest_object_id,) in quest_rows:
            insert_rows.append(
                {
                    "session_id": session_id,
                    "link_id": uuid.uuid4(),
                    "from_object_id": player_object_id,
                    "to_object_id": quest_object_id,
                    "type": "tracking_quest",
                }
            )

    if insert_rows:
        bind.execute(
            sa.text(
                """
                INSERT INTO links (
                    session_id,
                    link_id,
                    from_object_id,
                    to_object_id,
                    type,
                    data,
                    valid_from_turn,
                    valid_to_turn,
                    created_at
                )
                VALUES (
                    :session_id,
                    :link_id,
                    :from_object_id,
                    :to_object_id,
                    :type,
                    '{}'::jsonb,
                    0,
                    NULL,
                    now()
                )
                """
            ),
            insert_rows,
        )

    op.create_index(
        "ix_links_session_from_type_valid_to",
        "links",
        ["session_id", "from_object_id", "type", "valid_to_turn"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_links_session_from_type_valid_to", table_name="links")
