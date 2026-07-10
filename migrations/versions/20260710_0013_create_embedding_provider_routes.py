"""Create embedding provider routing schema.

Revision ID: 20260710_0013
Revises: 20260709_0012
Create Date: 2026-07-10
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260710_0013"
down_revision: str | None = "20260709_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_provider_routes (
            route_id BIGSERIAL PRIMARY KEY,
            profile_name TEXT NOT NULL REFERENCES embedding_profiles(profile_name)
                ON DELETE CASCADE,
            provider_name TEXT NOT NULL,
            provider_mode TEXT NOT NULL DEFAULT 'remote'
                CHECK (provider_mode IN ('mock', 'remote')),
            provider_base_url TEXT,
            timeout_seconds NUMERIC(8, 3) NOT NULL DEFAULT 30.0
                CHECK (timeout_seconds > 0),
            priority INT NOT NULL DEFAULT 100 CHECK (priority >= 0),
            is_active BOOLEAN NOT NULL DEFAULT true,
            health_check_enabled BOOLEAN NOT NULL DEFAULT true,
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (profile_name, provider_name),
            CHECK (
                provider_mode <> 'remote'
                OR (
                    provider_base_url IS NOT NULL
                    AND length(trim(provider_base_url)) > 0
                )
            )
        )
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_routes_profile_active
        ON embedding_provider_routes (profile_name, is_active, priority, route_id)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_routes_mode
        ON embedding_provider_routes (provider_mode, is_active)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_provider_routes")
