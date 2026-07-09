import pytest

from app.core.admin_logging import list_logs, log_event
from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def test_admin_logging_tables_and_default_settings(migrated_database_url: str) -> None:
    settings_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM app_log_settings
        WHERE setting_name IN (
            'logging_enabled',
            'min_log_level',
            'log_retention_days',
            'admin_log_page_size'
        )
        """,
    )
    retention = fetch_one(
        migrated_database_url,
        """
        SELECT setting_value
        FROM app_log_settings
        WHERE setting_name = 'log_retention_days'
        """,
    )

    assert settings_count["count"] == 4
    assert retention["setting_value"] == "7"


def test_log_event_persists_and_purges_expired_rows(migrated_database_url: str) -> None:
    correlation_id = "integration-admin-logging"
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM app_logs WHERE correlation_id = %s", (correlation_id,))
            cursor.execute(
                """
                INSERT INTO app_logs (
                    occurred_at,
                    level,
                    event_type,
                    source,
                    message,
                    correlation_id
                )
                VALUES (
                    now() - interval '8 days',
                    'ERROR',
                    'old_event',
                    'test',
                    'expired',
                    %s
                )
                """,
                (correlation_id,),
            )

    log_id = log_event(
        migrated_database_url,
        level="ERROR",
        event_type="integration_event",
        source="test",
        message="stored",
        detail={"ok": True},
        correlation_id=correlation_id,
    )

    stored = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM app_logs
        WHERE correlation_id = %s
        """,
        (correlation_id,),
    )
    latest = list_logs(migrated_database_url, level="ERROR", limit=1)[0]

    assert log_id is not None
    assert stored["count"] == 1
    assert latest["event_type"] == "integration_event"
