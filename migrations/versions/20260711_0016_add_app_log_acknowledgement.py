"""Add app log acknowledgement fields.

Revision ID: 20260711_0016
Revises: 20260711_0015
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0016"
down_revision: str | None = "20260711_0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE app_logs
        ADD COLUMN acknowledged_at TIMESTAMPTZ,
        ADD COLUMN acknowledged_by TEXT,
        ADD COLUMN acknowledgement_note TEXT
        """)
    op.execute("""
        CREATE INDEX idx_app_logs_acknowledged_at
        ON app_logs (acknowledged_at, occurred_at DESC)
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_app_logs_acknowledged_at")
    op.execute("""
        ALTER TABLE app_logs
        DROP COLUMN IF EXISTS acknowledgement_note,
        DROP COLUMN IF EXISTS acknowledged_by,
        DROP COLUMN IF EXISTS acknowledged_at
        """)
