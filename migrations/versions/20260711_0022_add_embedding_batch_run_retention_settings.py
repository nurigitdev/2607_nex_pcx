"""Add embedding batch run retention settings.

Revision ID: 20260711_0022
Revises: 20260711_0021
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0022"
down_revision: str | None = "20260711_0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO app_log_settings
            (setting_name, setting_value, value_type, description)
        VALUES
            (
                'embedding_batch_run_retention_enabled',
                'true',
                'bool',
                'Enable embedding batch run retention cleanup actions'
            ),
            (
                'embedding_batch_run_retention_days',
                '30',
                'int',
                'Number of days to retain embedding worker batch run history'
            ),
            (
                'embedding_batch_run_cleanup_batch_size',
                '1000',
                'int',
                'Maximum embedding batch run rows cleaned up in one action'
            )
        ON CONFLICT (setting_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM app_log_settings
        WHERE setting_name IN (
            'embedding_batch_run_retention_enabled',
            'embedding_batch_run_retention_days',
            'embedding_batch_run_cleanup_batch_size'
        )
        """)
