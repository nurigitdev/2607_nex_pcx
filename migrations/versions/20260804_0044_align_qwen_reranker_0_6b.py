"""Align reranked search profile with Qwen3 Reranker 0.6B.

Revision ID: 20260804_0044
Revises: 20260730_0043
Create Date: 2026-08-04
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260804_0044"
down_revision: str | None = "20260730_0043"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        UPDATE search_profiles
        SET
            runtime_parameters = jsonb_set(
                jsonb_set(
                    COALESCE(runtime_parameters, '{}'::jsonb),
                    '{reranker_profile_name}',
                    to_jsonb('qwen3_reranker_0_6b'::text),
                    true
                ),
                '{reranker_model_id}',
                to_jsonb('Qwen/Qwen3-Reranker-0.6B'::text),
                true
            ),
            updated_at = now()
        WHERE search_profile_name = 'reranked_vector_cosine'
        """)


def downgrade() -> None:
    op.execute("""
        UPDATE search_profiles
        SET
            runtime_parameters = jsonb_set(
                jsonb_set(
                    COALESCE(runtime_parameters, '{}'::jsonb),
                    '{reranker_profile_name}',
                    to_jsonb('qwen3_reranker_4b'::text),
                    true
                ),
                '{reranker_model_id}',
                to_jsonb('Qwen/Qwen3-Reranker-4B'::text),
                true
            ),
            updated_at = now()
        WHERE search_profile_name = 'reranked_vector_cosine'
        """)
