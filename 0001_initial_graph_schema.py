"""Initial graph schema

Revision ID: 0001_initial_graph_schema
Revises: 
Create Date: 2026-03-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_graph_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("world_prompt", sa.Text(), nullable=True),
        sa.Column(
            "state_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text(
                "'{\"world_name\":\"Unnamed\",\"difficulty\":\"normal\",\"rules\":{},\"player_object_id\": null,\"current_turn\": 0}'::jsonb"
            ),
        ),
        sa.Column("turn_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "objects",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("object_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id", "object_id", name="pk_objects"),
    )
    op.create_index("ix_objects_session_name", "objects", ["session_id", "name"], unique=False)
    op.create_index("ix_objects_session_type", "objects", ["session_id", "type"], unique=False)

    op.create_table(
        "links",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("link_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("from_object_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("to_object_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column(
            "data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("valid_from_turn", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("valid_to_turn", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id", "from_object_id"],
            ["objects.session_id", "objects.object_id"],
            ondelete="CASCADE",
            name="fk_links_from_object",
        ),
        sa.ForeignKeyConstraint(
            ["session_id", "to_object_id"],
            ["objects.session_id", "objects.object_id"],
            ondelete="CASCADE",
            name="fk_links_to_object",
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id", "link_id", name="pk_links"),
    )
    op.create_index("ix_links_session_from", "links", ["session_id", "from_object_id"], unique=False)
    op.create_index("ix_links_session_to", "links", ["session_id", "to_object_id"], unique=False)
    op.create_index("ix_links_session_type", "links", ["session_id", "type"], unique=False)
    op.create_index(
        "ix_links_session_type_valid_to",
        "links",
        ["session_id", "type", "valid_to_turn"],
        unique=False,
    )

    op.create_table(
        "events",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("scope_object_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id", "scope_object_id"],
            ["objects.session_id", "objects.object_id"],
            name="fk_events_scope_object",
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id", "event_id", name="pk_events"),
    )
    op.create_index("ix_events_session_turn", "events", ["session_id", "turn_index"], unique=False)

    op.create_table(
        "turns",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("ai_text", sa.Text(), nullable=True),
        sa.Column("ai_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id", "turn_index", name="pk_turns"),
    )

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
    op.drop_table("session_snapshots")
    op.drop_table("turns")
    op.drop_index("ix_events_session_turn", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_links_session_type_valid_to", table_name="links")
    op.drop_index("ix_links_session_type", table_name="links")
    op.drop_index("ix_links_session_to", table_name="links")
    op.drop_index("ix_links_session_from", table_name="links")
    op.drop_table("links")
    op.drop_index("ix_objects_session_type", table_name="objects")
    op.drop_index("ix_objects_session_name", table_name="objects")
    op.drop_table("objects")
    op.drop_table("sessions")
