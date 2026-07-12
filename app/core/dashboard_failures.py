"""Recent operational failure aggregation for the dashboard."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.admin_logging import PROVIDER_ROUTE_ALERT_EVENT_TYPES
from app.core.database import connect

MAX_RECENT_FAILURE_LIMIT = 50


@dataclass(frozen=True)
class DashboardFailureRecord:
    source: str
    severity: str
    title: str
    message: str | None
    occurred_at: datetime
    status: str | None
    action_url: str | None
    reference_id: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DashboardFailureSummary:
    total_count: int
    pipeline_failure_count: int
    embedding_failure_count: int
    parsing_failure_count: int
    app_error_count: int
    provider_alert_count: int
    failures: tuple[DashboardFailureRecord, ...]


class InvalidDashboardFailureError(ValueError):
    """Raised when dashboard failure aggregation inputs are invalid."""


def _validate_limit(limit: int) -> int:
    if limit < 1:
        raise InvalidDashboardFailureError("limit must be at least 1")
    if limit > MAX_RECENT_FAILURE_LIMIT:
        raise InvalidDashboardFailureError(
            f"limit must be less than or equal to {MAX_RECENT_FAILURE_LIMIT}"
        )
    return limit


def _count(cursor: Any, sql: str, params: tuple[object, ...] = ()) -> int:
    cursor.execute(sql, params)
    row = cursor.fetchone() or {}
    return int(row["count"])


def _record_from_row(row: dict[str, Any]) -> DashboardFailureRecord:
    return DashboardFailureRecord(
        source=str(row["source"]),
        severity=str(row["severity"]),
        title=str(row["title"]),
        message=row.get("message"),
        occurred_at=row["occurred_at"],
        status=row.get("status"),
        action_url=row.get("action_url"),
        reference_id=str(row["reference_id"]) if row.get("reference_id") is not None else None,
        metadata=dict(row.get("metadata") or {}),
    )


def get_dashboard_recent_failures(
    database_url: str,
    *,
    limit: int = 10,
) -> DashboardFailureSummary:
    validated_limit = _validate_limit(limit)
    records: list[DashboardFailureRecord] = []

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            pipeline_failure_count = _count(
                cursor,
                """
                SELECT COUNT(*)::int AS count
                FROM pipeline_jobs
                WHERE status IN ('failed', 'canceled')
                """,
            )
            embedding_failure_count = _count(
                cursor,
                """
                SELECT COUNT(*)::int AS count
                FROM embedding_jobs
                WHERE status = 'failed'
                """,
            )
            parsing_failure_count = _count(
                cursor,
                """
                SELECT COUNT(*)::int AS count
                FROM files
                WHERE parse_status = 'failed'
                """,
            )
            app_error_count = _count(
                cursor,
                """
                SELECT COUNT(*)::int AS count
                FROM app_logs
                WHERE level IN ('ERROR', 'CRITICAL')
                    AND event_type NOT IN (%s, %s)
                """,
                PROVIDER_ROUTE_ALERT_EVENT_TYPES,
            )
            provider_alert_count = _count(
                cursor,
                """
                SELECT COUNT(*)::int AS count
                FROM app_logs
                WHERE event_type IN (%s, %s)
                    AND acknowledged_at IS NULL
                """,
                PROVIDER_ROUTE_ALERT_EVENT_TYPES,
            )

            cursor.execute(
                """
                SELECT
                    'pipeline' AS source,
                    CASE WHEN status = 'canceled' THEN 'warning' ELSE 'error' END AS severity,
                    concat('Pipeline job #', job_id) AS title,
                    error_message AS message,
                    COALESCE(finished_at, updated_at, queued_at) AS occurred_at,
                    status,
                    concat('/admin/jobs?job_id=', job_id) AS action_url,
                    job_id::text AS reference_id,
                    jsonb_build_object(
                        'job_type', job_type,
                        'stage', stage,
                        'error_code', error_code,
                        'attempts', attempts,
                        'max_attempts', max_attempts
                    ) AS metadata
                FROM pipeline_jobs
                WHERE status IN ('failed', 'canceled')
                ORDER BY COALESCE(finished_at, updated_at, queued_at) DESC
                LIMIT %s
                """,
                (validated_limit,),
            )
            records.extend(_record_from_row(dict(row)) for row in cursor.fetchall())

            cursor.execute(
                """
                SELECT
                    'embedding' AS source,
                    'error' AS severity,
                    concat('Embedding job #', job_id) AS title,
                    error_message AS message,
                    COALESCE(last_error_at, finished_at, updated_at, created_at) AS occurred_at,
                    status,
                    concat(
                        '/admin/embedding-jobs?status=failed&profile_name=',
                        profile_name,
                        '&job_id=',
                        job_id
                    ) AS action_url,
                    job_id::text AS reference_id,
                    jsonb_build_object(
                        'chunk_id', chunk_id,
                        'profile_name', profile_name,
                        'error_code', error_code,
                        'attempts', attempts,
                        'max_attempts', max_attempts
                    ) AS metadata
                FROM embedding_jobs
                WHERE status = 'failed'
                ORDER BY COALESCE(last_error_at, finished_at, updated_at, created_at) DESC
                LIMIT %s
                """,
                (validated_limit,),
            )
            records.extend(_record_from_row(dict(row)) for row in cursor.fetchall())

            cursor.execute(
                """
                SELECT
                    'parsing' AS source,
                    'warning' AS severity,
                    original_file_name AS title,
                    parse_error_message AS message,
                    updated_at AS occurred_at,
                    parse_status AS status,
                    '/documents?parse_status=failed' AS action_url,
                    file_id::text AS reference_id,
                    jsonb_build_object(
                        'file_id', file_id,
                        'document_group', document_group,
                        'file_ext', file_ext,
                        'storage_path', storage_path
                    ) AS metadata
                FROM files
                WHERE parse_status = 'failed'
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (validated_limit,),
            )
            records.extend(_record_from_row(dict(row)) for row in cursor.fetchall())

            cursor.execute(
                """
                SELECT
                    'app_log' AS source,
                    lower(level) AS severity,
                    event_type AS title,
                    message,
                    occurred_at,
                    level AS status,
                    concat('/admin/logs?level=', level) AS action_url,
                    log_id::text AS reference_id,
                    COALESCE(detail, '{}'::jsonb) || jsonb_build_object(
                        'log_id', log_id,
                        'source', source,
                        'request_path', request_path,
                        'correlation_id', correlation_id
                    ) AS metadata
                FROM app_logs
                WHERE level IN ('ERROR', 'CRITICAL')
                    AND event_type NOT IN (%s, %s)
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (*PROVIDER_ROUTE_ALERT_EVENT_TYPES, validated_limit),
            )
            records.extend(_record_from_row(dict(row)) for row in cursor.fetchall())

            cursor.execute(
                """
                SELECT
                    'provider_alert' AS source,
                    lower(level) AS severity,
                    event_type AS title,
                    message,
                    occurred_at,
                    CASE
                        WHEN acknowledged_at IS NULL THEN 'unacknowledged'
                        ELSE 'acknowledged'
                    END AS status,
                    '/admin/embedding-provider-routes' AS action_url,
                    log_id::text AS reference_id,
                    COALESCE(detail, '{}'::jsonb) || jsonb_build_object(
                        'log_id', log_id,
                        'source', source,
                        'request_path', request_path,
                        'correlation_id', correlation_id
                    ) AS metadata
                FROM app_logs
                WHERE event_type IN (%s, %s)
                    AND acknowledged_at IS NULL
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (*PROVIDER_ROUTE_ALERT_EVENT_TYPES, validated_limit),
            )
            records.extend(_record_from_row(dict(row)) for row in cursor.fetchall())

    failures = tuple(
        sorted(records, key=lambda failure: failure.occurred_at, reverse=True)[:validated_limit]
    )
    return DashboardFailureSummary(
        total_count=(
            pipeline_failure_count
            + embedding_failure_count
            + parsing_failure_count
            + app_error_count
            + provider_alert_count
        ),
        pipeline_failure_count=pipeline_failure_count,
        embedding_failure_count=embedding_failure_count,
        parsing_failure_count=parsing_failure_count,
        app_error_count=app_error_count,
        provider_alert_count=provider_alert_count,
        failures=failures,
    )
