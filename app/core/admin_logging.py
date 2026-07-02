"""Database-backed admin logging."""

from dataclasses import dataclass
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from app.core.database import connect

LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


@dataclass(frozen=True)
class LogSettings:
    enabled: bool = True
    min_level: str = "INFO"
    retention_days: int = 7
    page_size: int = 100


def normalize_level(level: str) -> str:
    normalized = level.upper()
    if normalized not in LOG_LEVELS:
        msg = f"Unsupported log level: {level}"
        raise ValueError(msg)
    return normalized


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def settings_from_rows(rows: list[dict[str, Any]]) -> LogSettings:
    values = {row["setting_name"]: row["setting_value"] for row in rows}
    min_level = values.get("min_log_level", "INFO").upper()
    if min_level not in LOG_LEVELS:
        min_level = "INFO"
    return LogSettings(
        enabled=parse_bool(values.get("logging_enabled", "true")),
        min_level=min_level,
        retention_days=parse_positive_int(values.get("log_retention_days", "7"), 7),
        page_size=parse_positive_int(values.get("admin_log_page_size", "100"), 100),
    )


def load_log_settings(connection: Connection) -> LogSettings:
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT setting_name, setting_value
            FROM app_log_settings
            WHERE setting_name IN (
                'logging_enabled',
                'min_log_level',
                'log_retention_days',
                'admin_log_page_size'
            )
            """)
        return settings_from_rows([dict(row) for row in cursor.fetchall()])


def should_store_log(level: str, settings: LogSettings) -> bool:
    normalized = normalize_level(level)
    return settings.enabled and LOG_LEVELS[normalized] >= LOG_LEVELS[settings.min_level]


def purge_expired_logs(connection: Connection, retention_days: int) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM app_logs
            WHERE occurred_at < now() - (%s::int * interval '1 day')
            """,
            (retention_days,),
        )
        return cursor.rowcount or 0


def log_event(
    database_url: str,
    *,
    level: str,
    event_type: str,
    message: str,
    source: str | None = None,
    detail: dict[str, Any] | None = None,
    traceback: str | None = None,
    request_path: str | None = None,
    correlation_id: str | None = None,
) -> int | None:
    normalized_level = normalize_level(level)
    with connect(database_url) as connection:
        settings = load_log_settings(connection)
        if not should_store_log(normalized_level, settings):
            return None
        purge_expired_logs(connection, settings.retention_days)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO app_logs (
                    level,
                    event_type,
                    source,
                    message,
                    detail,
                    traceback,
                    request_path,
                    correlation_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING log_id
                """,
                (
                    normalized_level,
                    event_type,
                    source,
                    message,
                    Jsonb(detail or {}),
                    traceback,
                    request_path,
                    correlation_id,
                ),
            )
            row = cursor.fetchone()
            return int(row["log_id"]) if row else None


def list_logs(
    database_url: str,
    *,
    level: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with connect(database_url) as connection:
        settings = load_log_settings(connection)
        row_limit = limit or settings.page_size
        with connection.cursor() as cursor:
            if level:
                cursor.execute(
                    """
                    SELECT *
                    FROM app_logs
                    WHERE level = %s
                    ORDER BY occurred_at DESC
                    LIMIT %s
                    """,
                    (normalize_level(level), row_limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM app_logs
                    ORDER BY occurred_at DESC
                    LIMIT %s
                    """,
                    (row_limit,),
                )
            return [dict(row) for row in cursor.fetchall()]
