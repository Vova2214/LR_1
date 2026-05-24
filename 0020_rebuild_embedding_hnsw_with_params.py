"""Rebuild embedding HNSW indexes with explicit construction params.

Revision ID: 0020_embed_hnsw_params
Revises: 0019_narrative_spine_type
Create Date: 2026-03-04 20:30:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0020_embed_hnsw_params"
down_revision: Union[str, None] = "0019_narrative_spine_type"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_object_embeddings_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chronicle_chunks_embedding_hnsw")
    op.execute(
        """
        DO $$
        BEGIN
            BEGIN
                CREATE INDEX IF NOT EXISTS ix_chronicle_chunks_embedding_hnsw
                ON chronicle_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            EXCEPTION
                WHEN program_limit_exceeded OR feature_not_supported OR undefined_object OR undefined_function OR invalid_parameter_value THEN
                    RAISE NOTICE 'Skipping ix_chronicle_chunks_embedding_hnsw: %', SQLERRM;
            END;

            BEGIN
                CREATE INDEX IF NOT EXISTS ix_object_embeddings_embedding_hnsw
                ON object_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            EXCEPTION
                WHEN program_limit_exceeded OR feature_not_supported OR undefined_object OR undefined_function OR invalid_parameter_value THEN
                    RAISE NOTICE 'Skipping ix_object_embeddings_embedding_hnsw: %', SQLERRM;
            END;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_object_embeddings_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chronicle_chunks_embedding_hnsw")
    op.execute(
        """
        DO $$
        BEGIN
            BEGIN
                CREATE INDEX IF NOT EXISTS ix_chronicle_chunks_embedding_hnsw
                ON chronicle_chunks
                USING hnsw (embedding vector_cosine_ops);
            EXCEPTION
                WHEN program_limit_exceeded OR feature_not_supported OR undefined_object OR undefined_function OR invalid_parameter_value THEN
                    RAISE NOTICE 'Skipping ix_chronicle_chunks_embedding_hnsw downgrade recreate: %', SQLERRM;
            END;

            BEGIN
                CREATE INDEX IF NOT EXISTS ix_object_embeddings_embedding_hnsw
                ON object_embeddings
                USING hnsw (embedding vector_cosine_ops);
            EXCEPTION
                WHEN program_limit_exceeded OR feature_not_supported OR undefined_object OR undefined_function OR invalid_parameter_value THEN
                    RAISE NOTICE 'Skipping ix_object_embeddings_embedding_hnsw downgrade recreate: %', SQLERRM;
            END;
        END
        $$;
        """
    )
