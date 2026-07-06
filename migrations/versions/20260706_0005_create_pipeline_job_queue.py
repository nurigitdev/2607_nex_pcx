"""Create pipeline job queue schema.

Revision ID: 20260706_0005
Revises: 20260705_0004
Create Date: 2026-07-06
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260706_0005"
down_revision: str | None = "20260705_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE pipeline_jobs (
            job_id BIGSERIAL PRIMARY KEY,
            job_type TEXT NOT NULL
                CHECK (
                    job_type IN (
                        'document_ingestion',
                        'text_extraction',
                        'parsing',
                        'chunking',
                        'embedding',
                        'vector_indexing'
                    )
                ),
            file_id BIGINT REFERENCES files(file_id) ON DELETE CASCADE,
            document_id BIGINT REFERENCES documents(document_id) ON DELETE CASCADE,
            parent_job_id BIGINT REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE,
            requested_by_user_id BIGINT REFERENCES app_users(user_id),
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK (
                    status IN (
                        'queued',
                        'running',
                        'succeeded',
                        'failed',
                        'canceled',
                        'skipped'
                    )
                ),
            stage TEXT NOT NULL DEFAULT 'upload_saved'
                CHECK (
                    stage IN (
                        'upload_saved',
                        'text_extraction',
                        'parsing',
                        'chunking',
                        'embedding',
                        'vector_indexing',
                        'completed'
                    )
                ),
            priority INT NOT NULL DEFAULT 100 CHECK (priority >= 0),
            total_units INT NOT NULL DEFAULT 0 CHECK (total_units >= 0),
            processed_units INT NOT NULL DEFAULT 0 CHECK (processed_units >= 0),
            progress_percent NUMERIC(5, 2) NOT NULL DEFAULT 0
                CHECK (progress_percent >= 0 AND progress_percent <= 100),
            current_message TEXT,
            attempts INT NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_attempts INT NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
            lease_owner TEXT,
            lease_expires_at TIMESTAMPTZ,
            heartbeat_at TIMESTAMPTZ,
            error_code TEXT,
            error_message TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (processed_units <= total_units),
            CHECK (attempts <= max_attempts),
            CHECK (
                (
                    status = 'running'
                    AND lease_owner IS NOT NULL
                    AND lease_expires_at IS NOT NULL
                    AND heartbeat_at IS NOT NULL
                )
                OR (status <> 'running')
            )
        )
        """)
    op.execute("""
        CREATE INDEX idx_pipeline_jobs_claim
        ON pipeline_jobs (status, priority, queued_at)
        WHERE status = 'queued'
        """)
    op.execute("CREATE INDEX idx_pipeline_jobs_file ON pipeline_jobs (file_id)")
    op.execute("CREATE INDEX idx_pipeline_jobs_document ON pipeline_jobs (document_id)")
    op.execute("""
        CREATE INDEX idx_pipeline_jobs_lease
        ON pipeline_jobs (lease_expires_at)
        WHERE status = 'running'
        """)

    op.execute("""
        CREATE TABLE pipeline_job_events (
            event_id BIGSERIAL PRIMARY KEY,
            job_id BIGINT NOT NULL REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE,
            event_type TEXT NOT NULL
                CHECK (
                    event_type IN (
                        'created',
                        'claimed',
                        'heartbeat',
                        'stage_started',
                        'progress',
                        'stage_succeeded',
                        'failed',
                        'retried',
                        'canceled'
                    )
                ),
            stage TEXT
                CHECK (
                    stage IS NULL
                    OR stage IN (
                        'upload_saved',
                        'text_extraction',
                        'parsing',
                        'chunking',
                        'embedding',
                        'vector_indexing',
                        'completed'
                    )
                ),
            status TEXT
                CHECK (
                    status IS NULL
                    OR status IN (
                        'queued',
                        'running',
                        'succeeded',
                        'failed',
                        'canceled',
                        'skipped'
                    )
                ),
            message TEXT,
            event_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_pipeline_job_events_job
        ON pipeline_job_events (job_id, created_at)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pipeline_job_events")
    op.execute("DROP TABLE IF EXISTS pipeline_jobs")
