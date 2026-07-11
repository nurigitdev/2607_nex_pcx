"""Create embedding provider route contract snapshot schema.

Revision ID: 20260711_0015
Revises: 20260710_0014
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0015"
down_revision: str | None = "20260710_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_provider_route_contract_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            route_id BIGINT NOT NULL
                REFERENCES embedding_provider_routes(route_id) ON DELETE CASCADE,
            profile_name TEXT NOT NULL
                REFERENCES embedding_profiles(profile_name) ON DELETE CASCADE,
            provider_name TEXT NOT NULL,
            provider_mode TEXT NOT NULL
                CHECK (provider_mode IN ('mock', 'remote')),
            passed BOOLEAN NOT NULL,
            status TEXT NOT NULL
                CHECK (
                    status IN (
                        'passed',
                        'mismatch',
                        'invalid_route',
                        'embedding_failed',
                        'health_ready',
                        'health_not_ready',
                        'health_mismatch',
                        'health_unreachable',
                        'health_skipped',
                        'health_unsupported'
                    )
                ),
            elapsed_ms INT NOT NULL CHECK (elapsed_ms >= 0),
            input_type TEXT NOT NULL,
            sample_text_count INT NOT NULL CHECK (sample_text_count > 0),
            expected_dimension INT CHECK (expected_dimension IS NULL OR expected_dimension > 0),
            provider_type TEXT,
            provider_model_id TEXT,
            model_key TEXT,
            dimension INT CHECK (dimension IS NULL OR dimension > 0),
            input_count INT CHECK (input_count IS NULL OR input_count >= 0),
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            validation_errors JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(validation_errors) = 'array'),
            error_message TEXT,
            checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_route_contract_snapshots_route_time
        ON embedding_provider_route_contract_snapshots (route_id, checked_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_route_contract_snapshots_profile_time
        ON embedding_provider_route_contract_snapshots (profile_name, checked_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_route_contract_snapshots_status_time
        ON embedding_provider_route_contract_snapshots (status, checked_at DESC)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_provider_route_contract_snapshots")
