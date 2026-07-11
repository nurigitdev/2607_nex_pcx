"""Add provider route retention settings.

Revision ID: 20260711_0020
Revises: 20260711_0019
Create Date: 2026-07-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260711_0020"
down_revision: str | None = "20260711_0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO app_log_settings
            (setting_name, setting_value, value_type, description)
        VALUES
            (
                'provider_route_retention_enabled',
                'true',
                'bool',
                'Enable provider route operational retention cleanup actions'
            ),
            (
                'provider_route_retention_days',
                '30',
                'int',
                'Number of days to retain provider route snapshots and preflight runs'
            ),
            (
                'provider_route_cleanup_batch_size',
                '1000',
                'int',
                'Maximum provider route operational rows cleaned up per table in one action'
            )
        ON CONFLICT (setting_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM app_log_settings
        WHERE setting_name IN (
            'provider_route_retention_enabled',
            'provider_route_retention_days',
            'provider_route_cleanup_batch_size'
        )
        """)
