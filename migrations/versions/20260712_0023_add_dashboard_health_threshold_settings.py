"""Add dashboard health threshold settings.

Revision ID: 20260712_0023
Revises: 20260711_0022
Create Date: 2026-07-12
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260712_0023"
down_revision: str | None = "20260711_0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO app_log_settings
            (setting_name, setting_value, value_type, description)
        VALUES
            (
                'dashboard_pipeline_stale_critical_threshold',
                '1',
                'int',
                'Pipeline stale lease count needed for a critical dashboard signal'
            ),
            (
                'dashboard_pipeline_exhausted_critical_threshold',
                '1',
                'int',
                'Pipeline exhausted failure count needed for a critical dashboard signal'
            ),
            (
                'dashboard_pipeline_retryable_warning_threshold',
                '1',
                'int',
                'Pipeline retryable failure count needed for a warning dashboard signal'
            ),
            (
                'dashboard_embedding_stale_critical_threshold',
                '1',
                'int',
                'Embedding stale lease count needed for a critical dashboard signal'
            ),
            (
                'dashboard_embedding_exhausted_critical_threshold',
                '1',
                'int',
                'Embedding exhausted failure count needed for a critical dashboard signal'
            ),
            (
                'dashboard_embedding_retryable_warning_threshold',
                '1',
                'int',
                'Embedding retryable failure count needed for a warning dashboard signal'
            ),
            (
                'dashboard_provider_alert_warning_threshold',
                '1',
                'int',
                'Provider alert count needed for a warning dashboard signal'
            ),
            (
                'dashboard_app_error_warning_threshold',
                '1',
                'int',
                'App error log count needed for a warning dashboard signal'
            ),
            (
                'dashboard_parsing_failure_warning_threshold',
                '1',
                'int',
                'Parsing failure count needed for a warning dashboard signal'
            )
        ON CONFLICT (setting_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM app_log_settings
        WHERE setting_name IN (
            'dashboard_pipeline_stale_critical_threshold',
            'dashboard_pipeline_exhausted_critical_threshold',
            'dashboard_pipeline_retryable_warning_threshold',
            'dashboard_embedding_stale_critical_threshold',
            'dashboard_embedding_exhausted_critical_threshold',
            'dashboard_embedding_retryable_warning_threshold',
            'dashboard_provider_alert_warning_threshold',
            'dashboard_app_error_warning_threshold',
            'dashboard_parsing_failure_warning_threshold'
        )
        """)
