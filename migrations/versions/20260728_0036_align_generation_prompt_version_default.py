"""Align generation run prompt version default with template-aware contract.

Revision ID: 20260728_0036
Revises: 20260728_0035
Create Date: 2026-07-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260728_0036"
down_revision: str | None = "20260728_0035"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE generation_runs
        ALTER COLUMN prompt_version SET DEFAULT 'grounded_answer_v1_prompt_v1'
        """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE generation_runs
        ALTER COLUMN prompt_version SET DEFAULT 'grounded_answer_v1'
        """)
