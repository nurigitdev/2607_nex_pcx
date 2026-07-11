"""Retention cleanup helpers for provider route operational records."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect


@dataclass(frozen=True)
class ProviderRouteRetentionSettings:
    enabled: bool = True
    retention_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass(frozen=True)
class ProviderRouteRetentionSettingsInput:
    enabled: bool = True
    retention_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass(frozen=True)
class ProviderRouteCleanupResult:
    enabled: bool
    dry_run: bool
    retention_days: int
    cleanup_batch_size: int
    expired_health_snapshot_count: int
    expired_contract_snapshot_count: int
    expired_preflight_run_count: int
    deleted_health_snapshot_count: int
    deleted_contract_snapshot_count: int
    deleted_preflight_run_count: int
    cutoff_at: datetime

    @property
    def expired_count(self) -> int:
        return (
            self.expired_health_snapshot_count
            + self.expired_contract_snapshot_count
            + self.expired_preflight_run_count
        )

    @property
    def deleted_count(self) -> int:
        return (
            self.deleted_health_snapshot_count
            + self.deleted_contract_snapshot_count
            + self.deleted_preflight_run_count
        )


class InvalidProviderRouteRetentionError(ValueError):
    """Raised when provider route retention settings are invalid."""


def provider_route_retention_settings_from_rows(
    rows: list[dict[str, Any]],
) -> ProviderRouteRetentionSettings:
    values = {row["setting_name"]: row["setting_value"] for row in rows}
    defaults = ProviderRouteRetentionSettings()
    return ProviderRouteRetentionSettings(
        enabled=_parse_bool(
            values.get("provider_route_retention_enabled", str(defaults.enabled)),
            defaults.enabled,
        ),
        retention_days=_parse_positive_int(
            values.get("provider_route_retention_days", str(defaults.retention_days)),
            defaults.retention_days,
        ),
        cleanup_batch_size=_parse_positive_int(
            values.get(
                "provider_route_cleanup_batch_size",
                str(defaults.cleanup_batch_size),
            ),
            defaults.cleanup_batch_size,
        ),
    )


def validate_provider_route_retention_settings_input(
    settings_input: ProviderRouteRetentionSettingsInput,
) -> ProviderRouteRetentionSettingsInput:
    if settings_input.retention_days <= 0 or settings_input.retention_days > 3650:
        raise InvalidProviderRouteRetentionError("retention_days must be between 1 and 3650")
    if settings_input.cleanup_batch_size <= 0 or settings_input.cleanup_batch_size > 100000:
        raise InvalidProviderRouteRetentionError("cleanup_batch_size must be between 1 and 100000")
    return ProviderRouteRetentionSettingsInput(
        enabled=bool(settings_input.enabled),
        retention_days=settings_input.retention_days,
        cleanup_batch_size=settings_input.cleanup_batch_size,
    )


def load_provider_route_retention_settings(
    database_url: str,
) -> ProviderRouteRetentionSettings:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT setting_name, setting_value
                FROM app_log_settings
                WHERE setting_name IN (
                    'provider_route_retention_enabled',
                    'provider_route_retention_days',
                    'provider_route_cleanup_batch_size'
                )
                """)
            rows = cursor.fetchall()
    return provider_route_retention_settings_from_rows([dict(row) for row in rows])


def update_provider_route_retention_settings(
    database_url: str,
    settings_input: ProviderRouteRetentionSettingsInput,
) -> ProviderRouteRetentionSettings:
    validated = validate_provider_route_retention_settings_input(settings_input)
    rows = (
        (
            "provider_route_retention_enabled",
            "true" if validated.enabled else "false",
            "bool",
            "Enable provider route operational retention cleanup actions",
        ),
        (
            "provider_route_retention_days",
            str(validated.retention_days),
            "int",
            "Number of days to retain provider route snapshots and preflight runs",
        ),
        (
            "provider_route_cleanup_batch_size",
            str(validated.cleanup_batch_size),
            "int",
            "Maximum provider route operational rows cleaned up per table in one action",
        ),
    )
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO app_log_settings (
                    setting_name,
                    setting_value,
                    value_type,
                    description,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (setting_name) DO UPDATE
                SET setting_value = EXCLUDED.setting_value,
                    value_type = EXCLUDED.value_type,
                    description = EXCLUDED.description,
                    updated_at = now()
                """,
                rows,
            )
    return ProviderRouteRetentionSettings(
        enabled=validated.enabled,
        retention_days=validated.retention_days,
        cleanup_batch_size=validated.cleanup_batch_size,
    )


def cleanup_expired_provider_route_records(
    database_url: str,
    *,
    dry_run: bool = True,
) -> ProviderRouteCleanupResult:
    retention_settings = load_provider_route_retention_settings(database_url)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT now() - (%s::int * interval '1 day') AS cutoff_at
                """,
                (retention_settings.retention_days,),
            )
            cutoff_at = cursor.fetchone()["cutoff_at"]

            expired_health_snapshot_count = _count_expired(
                cursor,
                table_name="embedding_provider_route_health_snapshots",
                timestamp_column="checked_at",
                cutoff_at=cutoff_at,
            )
            expired_contract_snapshot_count = _count_expired(
                cursor,
                table_name="embedding_provider_route_contract_snapshots",
                timestamp_column="checked_at",
                cutoff_at=cutoff_at,
            )
            expired_preflight_run_count = _count_expired(
                cursor,
                table_name="embedding_provider_preflight_runs",
                timestamp_column="completed_at",
                cutoff_at=cutoff_at,
            )

            deleted_health_snapshot_count = 0
            deleted_contract_snapshot_count = 0
            deleted_preflight_run_count = 0
            if retention_settings.enabled and not dry_run:
                deleted_health_snapshot_count = _delete_expired(
                    cursor,
                    table_name="embedding_provider_route_health_snapshots",
                    id_column="snapshot_id",
                    timestamp_column="checked_at",
                    cutoff_at=cutoff_at,
                    limit=retention_settings.cleanup_batch_size,
                )
                deleted_contract_snapshot_count = _delete_expired(
                    cursor,
                    table_name="embedding_provider_route_contract_snapshots",
                    id_column="snapshot_id",
                    timestamp_column="checked_at",
                    cutoff_at=cutoff_at,
                    limit=retention_settings.cleanup_batch_size,
                )
                deleted_preflight_run_count = _delete_expired(
                    cursor,
                    table_name="embedding_provider_preflight_runs",
                    id_column="run_id",
                    timestamp_column="completed_at",
                    cutoff_at=cutoff_at,
                    limit=retention_settings.cleanup_batch_size,
                )

    return ProviderRouteCleanupResult(
        enabled=retention_settings.enabled,
        dry_run=dry_run,
        retention_days=retention_settings.retention_days,
        cleanup_batch_size=retention_settings.cleanup_batch_size,
        expired_health_snapshot_count=expired_health_snapshot_count,
        expired_contract_snapshot_count=expired_contract_snapshot_count,
        expired_preflight_run_count=expired_preflight_run_count,
        deleted_health_snapshot_count=deleted_health_snapshot_count,
        deleted_contract_snapshot_count=deleted_contract_snapshot_count,
        deleted_preflight_run_count=deleted_preflight_run_count,
        cutoff_at=cutoff_at,
    )


def _count_expired(cursor, *, table_name: str, timestamp_column: str, cutoff_at: datetime) -> int:
    cursor.execute(
        f"""
        SELECT count(*) AS expired_count
        FROM {table_name}
        WHERE {timestamp_column} < %s
        """,
        (cutoff_at,),
    )
    return int(cursor.fetchone()["expired_count"] or 0)


def _delete_expired(
    cursor,
    *,
    table_name: str,
    id_column: str,
    timestamp_column: str,
    cutoff_at: datetime,
    limit: int,
) -> int:
    cursor.execute(
        f"""
        WITH doomed AS (
            SELECT {id_column}
            FROM {table_name}
            WHERE {timestamp_column} < %s
            ORDER BY {timestamp_column} ASC, {id_column} ASC
            LIMIT %s
        )
        DELETE FROM {table_name}
        WHERE {id_column} IN (SELECT {id_column} FROM doomed)
        """,
        (cutoff_at, limit),
    )
    return cursor.rowcount or 0


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
