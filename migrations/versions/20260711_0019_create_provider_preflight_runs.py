"""Create provider route preflight run history schema.

Revision ID: 20260711_0019
Revises: 20260711_0018
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0019"
down_revision: str | None = "20260711_0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_provider_preflight_runs (
            run_id BIGSERIAL PRIMARY KEY,
            schedule_name TEXT
                REFERENCES embedding_provider_preflight_schedules(schedule_name)
                ON DELETE SET NULL,
            trigger_source TEXT NOT NULL
                CHECK (trigger_source IN ('manual_api', 'scheduled_cli')),
            profile_name TEXT,
            active_only BOOLEAN NOT NULL DEFAULT true,
            status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed', 'error')),
            route_count INTEGER NOT NULL DEFAULT 0 CHECK (route_count >= 0),
            passed_count INTEGER NOT NULL DEFAULT 0 CHECK (passed_count >= 0),
            failed_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
            sample_set_name TEXT,
            input_type TEXT,
            sample_text_count INTEGER NOT NULL DEFAULT 0 CHECK (sample_text_count >= 0),
            elapsed_ms INTEGER CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            result JSONB NOT NULL DEFAULT '{}'::jsonb,
            error_message TEXT,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_preflight_runs_completed
        ON embedding_provider_preflight_runs (completed_at DESC, run_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_preflight_runs_schedule_completed
        ON embedding_provider_preflight_runs (schedule_name, completed_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_provider_preflight_runs_status_completed
        ON embedding_provider_preflight_runs (status, completed_at DESC)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_provider_preflight_runs")
