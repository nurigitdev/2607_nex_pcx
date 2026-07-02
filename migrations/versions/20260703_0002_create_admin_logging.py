"""Create admin logging tables.

Revision ID: 20260703_0002
Revises: 20260702_0001
Create Date: 2026-07-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260703_0002"
down_revision: str | None = "20260702_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE app_log_settings (
            setting_name TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL,
            value_type TEXT NOT NULL
                CHECK (value_type IN ('bool', 'int', 'text')),
            description TEXT,
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE app_logs (
            log_id BIGSERIAL PRIMARY KEY,
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            level TEXT NOT NULL
                CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
            event_type TEXT NOT NULL,
            source TEXT,
            message TEXT NOT NULL,
            detail JSONB NOT NULL DEFAULT '{}'::jsonb,
            traceback TEXT,
            request_path TEXT,
            correlation_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_app_logs_occurred_at ON app_logs (occurred_at DESC)")
    op.execute("CREATE INDEX idx_app_logs_level ON app_logs (level)")
    op.execute(
        """
        INSERT INTO app_log_settings
            (setting_name, setting_value, value_type, description)
        VALUES
            ('logging_enabled', 'true', 'bool', 'Enable database-backed application logging'),
            ('min_log_level', 'INFO', 'text', 'Minimum level stored in app_logs'),
            ('log_retention_days', '7', 'int', 'Number of days to retain app_logs rows'),
            ('admin_log_page_size', '100', 'int', 'Default number of log rows shown in admin UI')
        ON CONFLICT (setting_name) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS app_logs")
    op.execute("DROP TABLE IF EXISTS app_log_settings")
