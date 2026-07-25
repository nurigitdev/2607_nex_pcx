"""Add reranked vector search profile.

Revision ID: 20260725_0032
Revises: 20260720_0031
Create Date: 2026-07-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260725_0032"
down_revision: str | None = "20260720_0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE search_profiles
        DROP CONSTRAINT IF EXISTS search_profiles_profile_kind_check
        """)
    op.execute("""
        ALTER TABLE search_profiles
        ADD CONSTRAINT search_profiles_profile_kind_check
        CHECK (profile_kind IN ('embedding', 'keyword', 'hybrid', 'rerank'))
        """)
    op.execute("""
        INSERT INTO search_profiles (
            search_profile_name,
            profile_kind,
            embedding_profile_name,
            strategy_name,
            display_name,
            is_active,
            runtime_parameters
        )
        VALUES (
            'reranked_vector_cosine',
            'rerank',
            NULL,
            'reranked_vector_cosine',
            'Reranked Vector Cosine',
            true,
            '{
                "source_strategy": "vector_cosine",
                "retrieval_strategy": "reranked",
                "reranker_profile_name": "qwen3_reranker_4b",
                "reranker_model_id": "Qwen/Qwen3-Reranker-4B",
                "candidate_multiplier": 4
            }'::jsonb
        )
        ON CONFLICT (search_profile_name) DO UPDATE
        SET
            profile_kind = EXCLUDED.profile_kind,
            strategy_name = EXCLUDED.strategy_name,
            display_name = EXCLUDED.display_name,
            is_active = EXCLUDED.is_active,
            runtime_parameters = EXCLUDED.runtime_parameters,
            updated_at = now()
        """)


def downgrade() -> None:
    op.execute("DELETE FROM search_profiles WHERE search_profile_name = 'reranked_vector_cosine'")
    op.execute("""
        ALTER TABLE search_profiles
        DROP CONSTRAINT IF EXISTS search_profiles_profile_kind_check
        """)
    op.execute("""
        ALTER TABLE search_profiles
        ADD CONSTRAINT search_profiles_profile_kind_check
        CHECK (profile_kind IN ('embedding', 'keyword', 'hybrid'))
        """)
