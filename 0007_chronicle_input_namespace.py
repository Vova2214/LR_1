"""Add namespace to chronicle_chunks for input/output embeddings

Revision ID: 0007_chronicle_input_namespace
Revises: 0006_ensure_session_snapshots
Create Date: 2026-03-01 23:55:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0007_chronicle_input_namespace"
down_revision: Union[str, None] = "0006_ensure_session_snapshots"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


OLD_UNIQUE_NAME = "uq_chronicle_chunks_session_turn"
NEW_UNIQUE_NAME = "uq_chronicle_chunks_session_turn_namespace"
NEW_INDEX_NAME = "ix_chronicle_chunks_session_namespace_turn"
DEFAULT_NAMESPACE = "chronicle_output"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("chronicle_chunks"):
        return

    column_names = {column["name"] for column in inspector.get_columns("chronicle_chunks")}
    if "namespace" not in column_names:
        op.add_column(
            "chronicle_chunks",
            sa.Column(
                "namespace",
                sa.Text(),
                nullable=False,
                server_default=sa.text(f"'{DEFAULT_NAMESPACE}'"),
            ),
        )
        bind.execute(
            sa.text("UPDATE chronicle_chunks SET namespace = :value WHERE namespace IS NULL"),
            {"value": DEFAULT_NAMESPACE},
        )

    unique_constraints = {constraint["name"] for constraint in inspector.get_unique_constraints("chronicle_chunks")}
    if OLD_UNIQUE_NAME in unique_constraints:
        op.drop_constraint(OLD_UNIQUE_NAME, "chronicle_chunks", type_="unique")
    if NEW_UNIQUE_NAME not in unique_constraints:
        op.create_unique_constraint(
            NEW_UNIQUE_NAME,
            "chronicle_chunks",
            ["session_id", "turn_index", "namespace"],
        )

    existing_indexes = {index["name"] for index in inspector.get_indexes("chronicle_chunks")}
    if NEW_INDEX_NAME not in existing_indexes:
        op.create_index(
            NEW_INDEX_NAME,
            "chronicle_chunks",
            ["session_id", "namespace", "turn_index"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("chronicle_chunks"):
        return

    existing_indexes = {index["name"] for index in inspector.get_indexes("chronicle_chunks")}
    if NEW_INDEX_NAME in existing_indexes:
        op.drop_index(NEW_INDEX_NAME, table_name="chronicle_chunks")

    unique_constraints = {constraint["name"] for constraint in inspector.get_unique_constraints("chronicle_chunks")}
    if NEW_UNIQUE_NAME in unique_constraints:
        op.drop_constraint(NEW_UNIQUE_NAME, "chronicle_chunks", type_="unique")
    if OLD_UNIQUE_NAME not in unique_constraints:
        op.create_unique_constraint(
            OLD_UNIQUE_NAME,
            "chronicle_chunks",
            ["session_id", "turn_index"],
        )

    column_names = {column["name"] for column in inspector.get_columns("chronicle_chunks")}
    if "namespace" in column_names:
        op.drop_column("chronicle_chunks", "namespace")
