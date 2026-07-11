"""Create embedding provider contract sample set schema.

Revision ID: 20260711_0017
Revises: 20260711_0016
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0017"
down_revision: str | None = "20260711_0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_provider_contract_sample_sets (
            sample_set_name TEXT PRIMARY KEY,
            description TEXT,
            input_type TEXT NOT NULL
                CHECK (input_type IN ('query', 'document')),
            sample_texts JSONB NOT NULL
                CHECK (
                    jsonb_typeof(sample_texts) = 'array'
                    AND jsonb_array_length(sample_texts) > 0
                ),
            is_active BOOLEAN NOT NULL DEFAULT true,
            is_default BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE UNIQUE INDEX idx_embedding_provider_contract_sample_sets_default
        ON embedding_provider_contract_sample_sets (is_default)
        WHERE is_default
        """)
    op.execute("""
        INSERT INTO embedding_provider_contract_sample_sets (
            sample_set_name,
            description,
            input_type,
            sample_texts,
            is_active,
            is_default
        )
        VALUES (
            'default_route_contract',
            'Default provider route embedding contract check sample set',
            'document',
            '["NeX-PCX embedding provider contract check sample."]'::jsonb,
            true,
            true
        )
        ON CONFLICT (sample_set_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_provider_contract_sample_sets")
