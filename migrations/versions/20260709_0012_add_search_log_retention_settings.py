"""Add search log retention settings.

Revision ID: 20260709_0012
Revises: 20260709_0011
Create Date: 2026-07-09
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260709_0012"
down_revision: str | None = "20260709_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO app_log_settings
            (setting_name, setting_value, value_type, description)
        VALUES
            (
                'search_log_retention_enabled',
                'true',
                'bool',
                'Enable search log retention cleanup actions'
            ),
            (
                'search_log_retention_days',
                '30',
                'int',
                'Number of days to retain search_logs and dependent rows'
            ),
            (
                'search_log_cleanup_batch_size',
                '1000',
                'int',
                'Maximum search_logs rows cleaned up in one admin action'
            )
        ON CONFLICT (setting_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM app_log_settings
        WHERE setting_name IN (
            'search_log_retention_enabled',
            'search_log_retention_days',
            'search_log_cleanup_batch_size'
        )
        """)
