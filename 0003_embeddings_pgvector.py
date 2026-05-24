"""Add pgvector embeddings tables

Revision ID: 0003_embeddings_pgvector
Revises: 0002_active_located_uq
Create Date: 2026-03-01 04:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

try:
    from src.vector_compat import Vector
except Exception:  # pragma: no cover
    class Vector(sa.types.UserDefinedType):
        cache_ok = True

        def __init__(self, dim: int):
            self.dim = dim

        def get_col_spec(self, **kw):
            return f"VECTOR({self.dim})"

# revision identifiers, used by Alembic.
revision: str = "0003_embeddings_pgvector"
down_revision: Union[str, None] = "0002_active_located_uq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
EMBEDDING_DIM = 4096
IVFFLAT_MAX_DIMS = 2000


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "chronicle_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("in_game_day", sa.Integer(), nullable=True),
        sa.Column("in_game_minute", sa.Integer(), nullable=True),
        sa.Column("text_snippet", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["session_id", "zone_id"],
            ["objects.session_id", "objects.object_id"],
            name="fk_chronicle_chunks_zone_object",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "turn_index", name="uq_chronicle_chunks_session_turn"),
    )
    op.create_index("ix_chronicle_chunks_session_turn_desc", "chronicle_chunks", ["session_id", "turn_index"], unique=False)
    if EMBEDDING_DIM <= IVFFLAT_MAX_DIMS:
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chronicle_chunks_embedding_ivfflat "
            "ON chronicle_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )

    op.create_table(
        "object_embeddings",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("object_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["session_id", "object_id"],
            ["objects.session_id", "objects.object_id"],
            ondelete="CASCADE",
            name="fk_object_embeddings_object",
        ),
        sa.PrimaryKeyConstraint("session_id", "object_id", "namespace", name="pk_object_embeddings"),
    )
    op.create_index(
        "ix_object_embeddings_session_namespace",
        "object_embeddings",
        ["session_id", "namespace"],
        unique=False,
    )
    if EMBEDDING_DIM <= IVFFLAT_MAX_DIMS:
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_object_embeddings_embedding_ivfflat "
            "ON object_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_object_embeddings_embedding_ivfflat")
    op.drop_index("ix_object_embeddings_session_namespace", table_name="object_embeddings")
    op.drop_table("object_embeddings")

    op.execute("DROP INDEX IF EXISTS ix_chronicle_chunks_embedding_ivfflat")
    op.drop_index("ix_chronicle_chunks_session_turn_desc", table_name="chronicle_chunks")
    op.drop_table("chronicle_chunks")
