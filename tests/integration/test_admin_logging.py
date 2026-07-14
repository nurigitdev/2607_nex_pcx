import pytest

from app.core.admin_logging import (
    InvalidAdminLogError,
    acknowledge_log,
    count_provider_route_alert_logs,
    get_provider_route_change_log,
    list_logs,
    list_provider_route_change_logs,
    log_event,
)
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
    acknowledgement_columns = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'app_logs'
          AND column_name IN (
              'acknowledged_at',
              'acknowledged_by',
              'acknowledgement_note'
          )
        """,
    )

    assert settings_count["count"] == 4
    assert retention["setting_value"] == "7"
    assert acknowledgement_columns["count"] == 3


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


def test_provider_route_alert_count_tracks_acknowledgement(
    migrated_database_url: str,
) -> None:
    correlation_id = "integration-provider-route-alert-count"
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM app_logs WHERE correlation_id = %s", (correlation_id,))

    unacknowledged_before = count_provider_route_alert_logs(
        migrated_database_url,
        level="ERROR",
        acknowledged=False,
    )
    acknowledged_before = count_provider_route_alert_logs(
        migrated_database_url,
        level="ERROR",
        acknowledged=True,
    )
    log_id = log_event(
        migrated_database_url,
        level="ERROR",
        event_type="embedding_provider_route_health_alert",
        source="integration-test",
        message="provider route alert count",
        correlation_id=correlation_id,
    )

    assert log_id is not None
    assert (
        count_provider_route_alert_logs(
            migrated_database_url,
            level="ERROR",
            acknowledged=False,
        )
        == unacknowledged_before + 1
    )

    acknowledged = acknowledge_log(
        migrated_database_url,
        log_id,
        acknowledged_by="integration-test",
    )

    assert acknowledged is not None
    assert (
        count_provider_route_alert_logs(
            migrated_database_url,
            level="ERROR",
            acknowledged=True,
        )
        == acknowledged_before + 1
    )


def test_provider_route_change_logs_support_filters(
    migrated_database_url: str,
) -> None:
    correlation_ids = [
        "integration-provider-route-change-created",
        "integration-provider-route-change-updated",
    ]
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM app_logs WHERE correlation_id = ANY(%s)",
                (correlation_ids,),
            )

    try:
        created_log_id = log_event(
            migrated_database_url,
            level="INFO",
            event_type="embedding_provider_route_created",
            source="integration-test",
            message="provider route created",
            detail={
                "action": "created",
                "route_id": 900001,
                "profile_name": "integration_profile",
                "provider_name": "integration_provider",
                "changed_fields": ["provider_base_url"],
            },
            correlation_id=correlation_ids[0],
        )
        updated_log_id = log_event(
            migrated_database_url,
            level="INFO",
            event_type="embedding_provider_route_updated",
            source="integration-test",
            message="provider route renamed",
            detail={
                "action": "updated",
                "route_id": 900001,
                "profile_name": "integration_profile",
                "provider_name": "integration_provider_v2",
                "previous_profile_name": "integration_profile",
                "previous_provider_name": "integration_provider",
                "changed_fields": ["provider_name"],
            },
            correlation_id=correlation_ids[1],
        )

        assert created_log_id is not None
        assert updated_log_id is not None
        provider_logs = list_provider_route_change_logs(
            migrated_database_url,
            provider_name="integration_provider",
            limit=10,
        )
        route_logs = list_provider_route_change_logs(
            migrated_database_url,
            route_id=900001,
            limit=10,
        )
        created_log = get_provider_route_change_log(
            migrated_database_url,
            created_log_id,
        )

        assert {log["log_id"] for log in provider_logs} == {
            created_log_id,
            updated_log_id,
        }
        assert {log["log_id"] for log in route_logs} == {
            created_log_id,
            updated_log_id,
        }
        assert created_log is not None
        assert created_log["detail"]["changed_fields"] == ["provider_base_url"]
        assert get_provider_route_change_log(migrated_database_url, 999999999) is None
        with pytest.raises(InvalidAdminLogError):
            list_provider_route_change_logs(migrated_database_url, limit=0)
        with pytest.raises(InvalidAdminLogError):
            get_provider_route_change_log(migrated_database_url, 0)
    finally:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM app_logs WHERE correlation_id = ANY(%s)",
                    (correlation_ids,),
                )
