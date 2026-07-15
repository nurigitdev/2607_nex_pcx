"""Create extraction quality snapshot schema.

Revision ID: 20260715_0029
Revises: 20260715_0028
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260715_0029"
down_revision: str | None = "20260715_0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE extraction_quality_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
            artifact_id BIGINT NOT NULL REFERENCES extraction_artifacts(artifact_id)
                ON DELETE CASCADE,
            extraction_run_id BIGINT REFERENCES extraction_runs(extraction_run_id)
                ON DELETE SET NULL,
            artifact_type TEXT NOT NULL CHECK (length(btrim(artifact_type)) > 0),
            extraction_profile_name TEXT,
            extractor_name TEXT,
            extractor_version TEXT,
            status TEXT NOT NULL CHECK (status IN ('passed', 'warning', 'failed')),
            content_length INT CHECK (content_length IS NULL OR content_length >= 0),
            content_lines INT CHECK (content_lines IS NULL OR content_lines >= 0),
            block_count INT NOT NULL CHECK (block_count >= 0),
            source_anchor_count INT NOT NULL CHECK (source_anchor_count >= 0),
            source_anchor_coverage_percent NUMERIC(5, 2)
                CHECK (
                    source_anchor_coverage_percent IS NULL OR
                    (
                        source_anchor_coverage_percent >= 0 AND
                        source_anchor_coverage_percent <= 100
                    )
                ),
            issue_count INT NOT NULL CHECK (issue_count >= 0),
            warning_count INT NOT NULL CHECK (warning_count >= 0),
            failed_count INT NOT NULL CHECK (failed_count >= 0),
            block_summary JSONB NOT NULL CHECK (jsonb_typeof(block_summary) = 'object'),
            quality_payload JSONB NOT NULL CHECK (jsonb_typeof(quality_payload) = 'object'),
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_extraction_quality_snapshots_document_time
        ON extraction_quality_snapshots (document_id, created_at DESC, snapshot_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_extraction_quality_snapshots_artifact_time
        ON extraction_quality_snapshots (artifact_id, created_at DESC, snapshot_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_extraction_quality_snapshots_status_time
        ON extraction_quality_snapshots (status, created_at DESC, snapshot_id DESC)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS extraction_quality_snapshots")
