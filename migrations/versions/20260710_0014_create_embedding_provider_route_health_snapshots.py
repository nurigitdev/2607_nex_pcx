"""Create embedding provider route health snapshot schema.

Revision ID: 20260710_0014
Revises: 20260710_0013
Create Date: 2026-07-10
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260710_0014"
down_revision: str | None = "20260710_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_provider_route_health_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            route_id BIGINT NOT NULL
                REFERENCES embedding_provider_routes(route_id) ON DELETE CASCADE,
            profile_name TEXT NOT NULL
                REFERENCES embedding_profiles(profile_name) ON DELETE CASCADE,
            provider_name TEXT NOT NULL,
            provider_mode TEXT NOT NULL
                CHECK (provider_mode IN ('mock', 'remote')),
            checked BOOLEAN NOT NULL,
            ready BOOLEAN NOT NULL,
            status TEXT NOT NULL
                CHECK (
                    status IN (
                        'ready',
                        'not_ready',
                        'mismatch',
                        'unreachable',
                        'skipped',
                        'unsupported'
                    )
                ),
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            provider_type TEXT,
            provider_model_id TEXT,
            model_key TEXT,
            profile_names JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(profile_names) = 'array'),
            dimension INT CHECK (dimension IS NULL OR dimension > 0),
            device TEXT,
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            validation_errors JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(validation_errors) = 'array'),
            error_message TEXT,
            checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_route_health_snapshots_route_time
        ON embedding_provider_route_health_snapshots (route_id, checked_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_route_health_snapshots_profile_time
        ON embedding_provider_route_health_snapshots (profile_name, checked_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_route_health_snapshots_status_time
        ON embedding_provider_route_health_snapshots (status, checked_at DESC)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_provider_route_health_snapshots")
