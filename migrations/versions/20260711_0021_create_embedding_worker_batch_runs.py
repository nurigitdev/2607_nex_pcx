"""Create embedding worker batch run history schema.

Revision ID: 20260711_0021
Revises: 20260711_0020
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0021"
down_revision: str | None = "20260711_0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_worker_batch_runs (
            batch_run_id BIGSERIAL PRIMARY KEY,
            worker_name TEXT NOT NULL,
            profile_name TEXT,
            provider_source TEXT NOT NULL
                CHECK (provider_source IN ('route', 'runtime')),
            provider_mode TEXT NOT NULL,
            remote_provider_url TEXT,
            require_route_readiness BOOLEAN NOT NULL DEFAULT false,
            readiness_gate_failure_mode TEXT NOT NULL DEFAULT 'fail'
                CHECK (readiness_gate_failure_mode IN ('fail', 'defer')),
            readiness_gate_defer_seconds INTEGER NOT NULL DEFAULT 300
                CHECK (readiness_gate_defer_seconds > 0),
            limit_requested INTEGER NOT NULL CHECK (limit_requested BETWEEN 1 AND 100),
            result_count INTEGER NOT NULL DEFAULT 0 CHECK (result_count >= 0),
            processed_count INTEGER NOT NULL DEFAULT 0 CHECK (processed_count >= 0),
            succeeded_count INTEGER NOT NULL DEFAULT 0 CHECK (succeeded_count >= 0),
            failed_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
            deferred_count INTEGER NOT NULL DEFAULT 0 CHECK (deferred_count >= 0),
            idle_count INTEGER NOT NULL DEFAULT 0 CHECK (idle_count >= 0),
            stopped_reason TEXT NOT NULL
                CHECK (stopped_reason IN ('limit_reached', 'queue_empty')),
            job_ids BIGINT[] NOT NULL DEFAULT '{}'::bigint[],
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            elapsed_ms INTEGER NOT NULL DEFAULT 0 CHECK (elapsed_ms >= 0),
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_embedding_worker_batch_runs_completed
        ON embedding_worker_batch_runs (completed_at DESC, batch_run_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_worker_batch_runs_worker_completed
        ON embedding_worker_batch_runs (worker_name, completed_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_embedding_worker_batch_runs_profile_completed
        ON embedding_worker_batch_runs (profile_name, completed_at DESC)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_worker_batch_runs")
