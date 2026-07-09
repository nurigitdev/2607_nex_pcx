"""Add search log review tag and memo metadata.

Revision ID: 20260709_0011
Revises: 20260707_0010
Create Date: 2026-07-09
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260709_0011"
down_revision: str | None = "20260707_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE search_logs
            ADD COLUMN review_tags JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(review_tags) = 'array'),
            ADD COLUMN review_memo TEXT,
            ADD COLUMN reviewed_by_user_id BIGINT REFERENCES app_users(user_id),
            ADD COLUMN reviewed_at TIMESTAMPTZ
        """)
    op.execute("""
        CREATE INDEX idx_search_logs_review_tags
        ON search_logs USING GIN (review_tags)
        """)
    op.execute("""
        CREATE INDEX idx_search_logs_reviewed_at
        ON search_logs (reviewed_at DESC)
        WHERE reviewed_at IS NOT NULL
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_search_logs_reviewed_at")
    op.execute("DROP INDEX IF EXISTS idx_search_logs_review_tags")
    op.execute("""
        ALTER TABLE search_logs
            DROP COLUMN IF EXISTS reviewed_at,
            DROP COLUMN IF EXISTS reviewed_by_user_id,
            DROP COLUMN IF EXISTS review_memo,
            DROP COLUMN IF EXISTS review_tags
        """)
