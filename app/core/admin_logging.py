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
PROVIDER_ROUTE_ALERT_EVENT_TYPES = (
    "embedding_provider_route_health_alert",
    "embedding_provider_route_contract_alert",
)
PROVIDER_ROUTE_CHANGE_EVENT_TYPES = (
    "embedding_provider_route_created",
    "embedding_provider_route_updated",
    "embedding_provider_route_activation_changed",
)


@dataclass(frozen=True)
class LogSettings:
    enabled: bool = True
    min_level: str = "INFO"
    retention_days: int = 7
    page_size: int = 100


class InvalidAdminLogError(ValueError):
    """Raised when admin log filters or actions are invalid."""


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


def list_provider_route_alert_logs(
    database_url: str,
    *,
    level: str | None = None,
    acknowledged: bool | None = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with connect(database_url) as connection:
        settings = load_log_settings(connection)
        row_limit = _normalize_limit(settings.page_size if limit is None else limit)
        where_clauses = ["event_type = ANY(%s)"]
        params: list[object] = [list(PROVIDER_ROUTE_ALERT_EVENT_TYPES)]
        if level:
            where_clauses.append("level = %s")
            params.append(normalize_level(level))
        if acknowledged is True:
            where_clauses.append("acknowledged_at IS NOT NULL")
        elif acknowledged is False:
            where_clauses.append("acknowledged_at IS NULL")

        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM app_logs
                WHERE {' AND '.join(where_clauses)}
                ORDER BY occurred_at DESC, log_id DESC
                LIMIT %s
                """,
                (*params, row_limit),
            )
            return [dict(row) for row in cursor.fetchall()]


def list_provider_route_change_logs(
    database_url: str,
    *,
    profile_name: str | None = None,
    provider_name: str | None = None,
    route_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with connect(database_url) as connection:
        settings = load_log_settings(connection)
        row_limit = _normalize_limit(settings.page_size if limit is None else limit)
        where_clauses = ["event_type = ANY(%s)"]
        params: list[object] = [list(PROVIDER_ROUTE_CHANGE_EVENT_TYPES)]
        if profile_name is not None:
            profile = _validate_nonblank(profile_name, "profile_name")
            where_clauses.append(
                "(detail->>'profile_name' = %s OR detail->>'previous_profile_name' = %s)"
            )
            params.extend([profile, profile])
        if provider_name is not None:
            provider = _validate_nonblank(provider_name, "provider_name")
            where_clauses.append(
                "(detail->>'provider_name' = %s OR detail->>'previous_provider_name' = %s)"
            )
            params.extend([provider, provider])
        if route_id is not None:
            if route_id <= 0:
                raise InvalidAdminLogError("route_id must be greater than 0")
            where_clauses.append("detail->>'route_id' = %s")
            params.append(str(route_id))

        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM app_logs
                WHERE {' AND '.join(where_clauses)}
                ORDER BY occurred_at DESC, log_id DESC
                LIMIT %s
                """,
                (*params, row_limit),
            )
            return [dict(row) for row in cursor.fetchall()]


def count_provider_route_alert_logs(
    database_url: str,
    *,
    level: str | None = None,
    acknowledged: bool | None = False,
) -> int:
    with connect(database_url) as connection:
        where_clauses = ["event_type = ANY(%s)"]
        params: list[object] = [list(PROVIDER_ROUTE_ALERT_EVENT_TYPES)]
        if level:
            where_clauses.append("level = %s")
            params.append(normalize_level(level))
        if acknowledged is True:
            where_clauses.append("acknowledged_at IS NOT NULL")
        elif acknowledged is False:
            where_clauses.append("acknowledged_at IS NULL")

        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT count(*) AS alert_count
                FROM app_logs
                WHERE {' AND '.join(where_clauses)}
                """,
                tuple(params),
            )
            row = cursor.fetchone()
            return int(row["alert_count"]) if row else 0


def acknowledge_log(
    database_url: str,
    log_id: int,
    *,
    acknowledged_by: str = "operator",
    acknowledgement_note: str | None = None,
) -> dict[str, Any] | None:
    if log_id <= 0:
        raise InvalidAdminLogError("log_id must be greater than 0")
    acknowledged_by = _validate_nonblank(acknowledged_by, "acknowledged_by")
    note = acknowledgement_note.strip() if acknowledgement_note else None
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE app_logs
                SET acknowledged_at = COALESCE(acknowledged_at, now()),
                    acknowledged_by = %s,
                    acknowledgement_note = %s
                WHERE log_id = %s
                  AND event_type = ANY(%s)
                RETURNING *
                """,
                (
                    acknowledged_by,
                    note,
                    log_id,
                    list(PROVIDER_ROUTE_ALERT_EVENT_TYPES),
                ),
            )
            row = cursor.fetchone()
            return dict(row) if row else None


def _normalize_limit(limit: int) -> int:
    if limit <= 0:
        raise InvalidAdminLogError("limit must be greater than 0")
    if limit > 500:
        raise InvalidAdminLogError("limit must be less than or equal to 500")
    return limit


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidAdminLogError(f"{field_name} is required")
    return normalized
