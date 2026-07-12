"""Recent operational failure aggregation for the dashboard."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.admin_logging import PROVIDER_ROUTE_ALERT_EVENT_TYPES
from app.core.database import connect

MAX_RECENT_FAILURE_LIMIT = 50
FAILURE_SOURCES = {"pipeline", "embedding", "parsing", "app_log", "provider_alert"}


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


@dataclass(frozen=True)
class DashboardFailureDetail:
    source: str
    reference_id: str
    title: str
    severity: str
    status: str | None
    message: str | None
    occurred_at: datetime
    action_url: str | None
    summary: dict[str, Any]
    context: dict[str, Any]
    raw: dict[str, Any]


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


def _validate_source(source: str) -> str:
    normalized = source.strip().lower()
    if normalized not in FAILURE_SOURCES:
        raise InvalidDashboardFailureError(f"Unsupported failure source: {source}")
    return normalized


def _validate_reference_id(reference_id: int | str) -> int:
    try:
        parsed = int(str(reference_id).strip())
    except ValueError as exc:
        raise InvalidDashboardFailureError("reference_id must be a positive integer") from exc
    if parsed <= 0:
        raise InvalidDashboardFailureError("reference_id must be a positive integer")
    return parsed


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


def get_dashboard_failure_detail(
    database_url: str,
    *,
    source: str,
    reference_id: int | str,
) -> DashboardFailureDetail | None:
    normalized_source = _validate_source(source)
    numeric_reference_id = _validate_reference_id(reference_id)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            if normalized_source == "pipeline":
                return _get_pipeline_failure_detail(cursor, numeric_reference_id)
            if normalized_source == "embedding":
                return _get_embedding_failure_detail(cursor, numeric_reference_id)
            if normalized_source == "parsing":
                return _get_parsing_failure_detail(cursor, numeric_reference_id)
            if normalized_source == "app_log":
                return _get_app_log_failure_detail(cursor, numeric_reference_id)
            return _get_provider_alert_failure_detail(cursor, numeric_reference_id)


def _get_pipeline_failure_detail(
    cursor: Any,
    job_id: int,
) -> DashboardFailureDetail | None:
    cursor.execute(
        """
        SELECT
            pj.*,
            f.original_file_name,
            d.document_title
        FROM pipeline_jobs pj
        LEFT JOIN files f ON f.file_id = pj.file_id
        LEFT JOIN documents d ON d.document_id = pj.document_id
        WHERE pj.job_id = %s
            AND pj.status IN ('failed', 'canceled')
        """,
        (job_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    job = dict(row)
    cursor.execute(
        """
        SELECT *
        FROM pipeline_job_events
        WHERE job_id = %s
        ORDER BY created_at DESC, event_id DESC
        LIMIT 10
        """,
        (job_id,),
    )
    events = [dict(event_row) for event_row in cursor.fetchall()]
    occurred_at = job.get("finished_at") or job.get("updated_at") or job["queued_at"]
    return DashboardFailureDetail(
        source="pipeline",
        reference_id=str(job["job_id"]),
        title=f"Pipeline job #{job['job_id']}",
        severity="warning" if job["status"] == "canceled" else "error",
        status=job["status"],
        message=job.get("error_message"),
        occurred_at=occurred_at,
        action_url=f"/admin/jobs?job_id={job['job_id']}",
        summary={
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "error_code": job.get("error_code"),
            "error_message": job.get("error_message"),
            "progress_percent": job.get("progress_percent"),
            "updated_at": job.get("updated_at"),
        },
        context={
            "job_type": job["job_type"],
            "attempts": job["attempts"],
            "max_attempts": job["max_attempts"],
            "file_id": job.get("file_id"),
            "original_file_name": job.get("original_file_name"),
            "document_id": job.get("document_id"),
            "document_title": job.get("document_title"),
            "event_count": len(events),
        },
        raw={"job": job, "events": events},
    )


def _get_embedding_failure_detail(
    cursor: Any,
    job_id: int,
) -> DashboardFailureDetail | None:
    cursor.execute(
        """
        SELECT
            ej.*,
            c.document_id,
            c.chunk_seq,
            c.heading_path,
            c.parser_name,
            c.parser_version,
            d.document_title,
            f.file_id,
            f.original_file_name
        FROM embedding_jobs ej
        LEFT JOIN chunks c ON c.chunk_id = ej.chunk_id
        LEFT JOIN documents d ON d.document_id = c.document_id
        LEFT JOIN files f ON f.file_id = d.file_id
        WHERE ej.job_id = %s
            AND ej.status = 'failed'
        """,
        (job_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    job = dict(row)
    occurred_at = (
        job.get("last_error_at")
        or job.get("finished_at")
        or job.get("updated_at")
        or job["created_at"]
    )
    return DashboardFailureDetail(
        source="embedding",
        reference_id=str(job["job_id"]),
        title=f"Embedding job #{job['job_id']}",
        severity="error",
        status=job["status"],
        message=job.get("error_message"),
        occurred_at=occurred_at,
        action_url=(
            f"/admin/embedding-jobs?status=failed&profile_name={job['profile_name']}"
            f"&job_id={job['job_id']}"
        ),
        summary={
            "job_id": job["job_id"],
            "status": job["status"],
            "profile_name": job["profile_name"],
            "error_code": job.get("error_code"),
            "error_message": job.get("error_message"),
            "last_error_at": job.get("last_error_at"),
        },
        context={
            "chunk_id": job["chunk_id"],
            "document_id": job.get("document_id"),
            "chunk_seq": job.get("chunk_seq"),
            "heading_path": job.get("heading_path"),
            "document_title": job.get("document_title"),
            "original_file_name": job.get("original_file_name"),
            "attempts": job["attempts"],
            "max_attempts": job["max_attempts"],
        },
        raw={"embedding_job": job},
    )


def _get_parsing_failure_detail(
    cursor: Any,
    file_id: int,
) -> DashboardFailureDetail | None:
    cursor.execute(
        """
        SELECT
            f.*,
            d.document_id,
            d.document_title,
            d.owner_user_id,
            d.owner_org_unit_id,
            d.access_scope
        FROM files f
        LEFT JOIN documents d ON d.file_id = f.file_id
        WHERE f.file_id = %s
            AND f.parse_status = 'failed'
        ORDER BY d.document_id
        LIMIT 1
        """,
        (file_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    file_row = dict(row)
    return DashboardFailureDetail(
        source="parsing",
        reference_id=str(file_row["file_id"]),
        title=file_row["original_file_name"],
        severity="warning",
        status=file_row["parse_status"],
        message=file_row.get("parse_error_message"),
        occurred_at=file_row["updated_at"],
        action_url="/documents?parse_status=failed",
        summary={
            "file_id": file_row["file_id"],
            "parse_status": file_row["parse_status"],
            "parse_error_message": file_row.get("parse_error_message"),
            "updated_at": file_row["updated_at"],
        },
        context={
            "original_file_name": file_row["original_file_name"],
            "file_ext": file_row.get("file_ext"),
            "mime_type": file_row.get("mime_type"),
            "file_size_bytes": file_row.get("file_size_bytes"),
            "document_group": file_row.get("document_group"),
            "security_level": file_row.get("security_level"),
            "document_id": file_row.get("document_id"),
            "document_title": file_row.get("document_title"),
            "access_scope": file_row.get("access_scope"),
            "storage_path": file_row.get("storage_path"),
        },
        raw={"file": file_row},
    )


def _get_app_log_failure_detail(
    cursor: Any,
    log_id: int,
) -> DashboardFailureDetail | None:
    cursor.execute(
        """
        SELECT *
        FROM app_logs
        WHERE log_id = %s
            AND level IN ('ERROR', 'CRITICAL')
            AND event_type NOT IN (%s, %s)
        """,
        (log_id, *PROVIDER_ROUTE_ALERT_EVENT_TYPES),
    )
    row = cursor.fetchone()
    if not row:
        return None
    log = dict(row)
    return _log_failure_detail(
        source="app_log",
        log=log,
        action_url=f"/admin/logs?level={log['level']}",
    )


def _get_provider_alert_failure_detail(
    cursor: Any,
    log_id: int,
) -> DashboardFailureDetail | None:
    cursor.execute(
        """
        SELECT *
        FROM app_logs
        WHERE log_id = %s
            AND event_type IN (%s, %s)
        """,
        (log_id, *PROVIDER_ROUTE_ALERT_EVENT_TYPES),
    )
    row = cursor.fetchone()
    if not row:
        return None
    log = dict(row)
    return _log_failure_detail(
        source="provider_alert",
        log=log,
        action_url="/admin/embedding-provider-routes",
    )


def _log_failure_detail(
    *,
    source: str,
    log: dict[str, Any],
    action_url: str,
) -> DashboardFailureDetail:
    return DashboardFailureDetail(
        source=source,
        reference_id=str(log["log_id"]),
        title=log["event_type"],
        severity=str(log["level"]).lower(),
        status=(
            "acknowledged"
            if log.get("acknowledged_at") is not None
            else ("unacknowledged" if source == "provider_alert" else log["level"])
        ),
        message=log.get("message"),
        occurred_at=log["occurred_at"],
        action_url=action_url,
        summary={
            "log_id": log["log_id"],
            "level": log["level"],
            "event_type": log["event_type"],
            "message": log.get("message"),
            "occurred_at": log["occurred_at"],
        },
        context={
            "source": log.get("source"),
            "request_path": log.get("request_path"),
            "correlation_id": log.get("correlation_id"),
            "acknowledged_at": log.get("acknowledged_at"),
            "acknowledged_by": log.get("acknowledged_by"),
            "acknowledgement_note": log.get("acknowledgement_note"),
            "traceback_present": bool(log.get("traceback")),
        },
        raw={"log": log},
    )
