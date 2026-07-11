"""Create provider route preflight schedule schema.

Revision ID: 20260711_0018
Revises: 20260711_0017
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0018"
down_revision: str | None = "20260711_0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_provider_preflight_schedules (
            schedule_name TEXT PRIMARY KEY,
            description TEXT,
            profile_name TEXT,
            active_only BOOLEAN NOT NULL DEFAULT true,
            interval_minutes INTEGER NOT NULL
                CHECK (interval_minutes BETWEEN 1 AND 10080),
            is_enabled BOOLEAN NOT NULL DEFAULT false,
            next_run_at TIMESTAMPTZ,
            last_run_at TIMESTAMPTZ,
            last_status TEXT NOT NULL DEFAULT 'never_run'
                CHECK (last_status IN ('never_run', 'succeeded', 'failed', 'error')),
            last_result JSONB NOT NULL DEFAULT '{}'::jsonb,
            run_count INTEGER NOT NULL DEFAULT 0 CHECK (run_count >= 0),
            failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_preflight_schedules_due
        ON embedding_provider_preflight_schedules (is_enabled, next_run_at)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_preflight_schedules_profile
        ON embedding_provider_preflight_schedules (profile_name)
        """)
    op.execute("""
        INSERT INTO embedding_provider_preflight_schedules (
            schedule_name,
            description,
            profile_name,
            active_only,
            interval_minutes,
            is_enabled,
            next_run_at
        )
        VALUES (
            'default_provider_route_preflight',
            'Default disabled schedule for provider route preflight checks',
            NULL,
            true,
            60,
            false,
            now()
        )
        ON CONFLICT (schedule_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_provider_preflight_schedules")
