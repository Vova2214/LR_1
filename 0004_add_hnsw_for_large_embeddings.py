"""No-op guard migration for large embedding dimensions

Revision ID: 0004_add_hnsw_large_vec
Revises: 0003_embeddings_pgvector
Create Date: 2026-03-01 06:10:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_add_hnsw_large_vec"
down_revision: Union[str, None] = "0003_embeddings_pgvector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Best-effort ANN acceleration: create HNSW indexes when the current
    # PostgreSQL/pgvector stack supports the configured vector type/dimension.
    # If not supported, keep migration chain moving without hard failure.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
                BEGIN
                    EXECUTE
                        'CREATE INDEX IF NOT EXISTS ix_chronicle_chunks_embedding_hnsw '
                        || 'ON chronicle_chunks USING hnsw (embedding vector_cosine_ops)';
                EXCEPTION
                    WHEN feature_not_supported OR undefined_function OR invalid_parameter_value OR program_limit_exceeded THEN
                        RAISE NOTICE 'Skipping ix_chronicle_chunks_embedding_hnsw: %', SQLERRM;
                END;

                BEGIN
                    EXECUTE
                        'CREATE INDEX IF NOT EXISTS ix_object_embeddings_embedding_hnsw '
                        || 'ON object_embeddings USING hnsw (embedding vector_cosine_ops)';
                EXCEPTION
                    WHEN feature_not_supported OR undefined_function OR invalid_parameter_value OR program_limit_exceeded THEN
                        RAISE NOTICE 'Skipping ix_object_embeddings_embedding_hnsw: %', SQLERRM;
                END;
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_object_embeddings_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chronicle_chunks_embedding_hnsw")
