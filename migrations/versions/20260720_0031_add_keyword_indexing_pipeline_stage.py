"""Add keyword indexing pipeline stage.

Revision ID: 20260720_0031
Revises: 20260720_0030
Create Date: 2026-07-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260720_0031"
down_revision: str | None = "20260720_0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


PIPELINE_JOB_STAGES_WITH_KEYWORD_INDEXING = (
    "'upload_saved'",
    "'text_extraction'",
    "'parsing'",
    "'chunking'",
    "'keyword_indexing'",
    "'embedding'",
    "'vector_indexing'",
    "'completed'",
)
PIPELINE_JOB_STAGES_WITHOUT_KEYWORD_INDEXING = (
    "'upload_saved'",
    "'text_extraction'",
    "'parsing'",
    "'chunking'",
    "'embedding'",
    "'vector_indexing'",
    "'completed'",
)


def upgrade() -> None:
    _replace_stage_constraints(PIPELINE_JOB_STAGES_WITH_KEYWORD_INDEXING)


def downgrade() -> None:
    op.execute("""
        UPDATE pipeline_job_events
        SET stage = 'chunking'
        WHERE stage = 'keyword_indexing'
        """)
    op.execute("""
        UPDATE pipeline_jobs
        SET stage = 'chunking'
        WHERE stage = 'keyword_indexing'
        """)
    _replace_stage_constraints(PIPELINE_JOB_STAGES_WITHOUT_KEYWORD_INDEXING)


def _replace_stage_constraints(stages: tuple[str, ...]) -> None:
    stage_list = ", ".join(stages)
    op.execute("ALTER TABLE pipeline_job_events DROP CONSTRAINT pipeline_job_events_stage_check")
    op.execute("ALTER TABLE pipeline_jobs DROP CONSTRAINT pipeline_jobs_stage_check")
    op.execute(f"""
        ALTER TABLE pipeline_jobs
        ADD CONSTRAINT pipeline_jobs_stage_check
        CHECK (stage IN ({stage_list}))
        """)
    op.execute(f"""
        ALTER TABLE pipeline_job_events
        ADD CONSTRAINT pipeline_job_events_stage_check
        CHECK (stage IS NULL OR stage IN ({stage_list}))
        """)
