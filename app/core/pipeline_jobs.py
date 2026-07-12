"""Pipeline job repository helpers backed by PostgreSQL."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect

PIPELINE_JOB_TYPES = {
    "document_ingestion",
    "text_extraction",
    "parsing",
    "chunking",
    "embedding",
    "vector_indexing",
}
PIPELINE_JOB_STATUSES = {
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
    "skipped",
}
PIPELINE_JOB_STAGES = {
    "upload_saved",
    "text_extraction",
    "parsing",
    "chunking",
    "embedding",
    "vector_indexing",
    "completed",
}
PIPELINE_EVENT_TYPES = {
    "created",
    "claimed",
    "heartbeat",
    "stage_started",
    "progress",
    "stage_succeeded",
    "failed",
    "retried",
    "canceled",
}
DEFAULT_LEASE_SECONDS = 300


@dataclass(frozen=True)
class PipelineJobInput:
    job_type: str
    file_id: int | None = None
    document_id: int | None = None
    parent_job_id: int | None = None
    requested_by_user_id: int | None = None
    priority: int = 100
    total_units: int = 0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PipelineJobRecord:
    job_id: int
    job_type: str
    file_id: int | None
    document_id: int | None
    parent_job_id: int | None
    requested_by_user_id: int | None
    status: str
    stage: str
    priority: int
    total_units: int
    processed_units: int
    progress_percent: Decimal
    current_message: str | None
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    heartbeat_at: datetime | None
    error_code: str | None
    error_message: str | None
    metadata: dict[str, Any]
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True)
class PipelineJobEventRecord:
    event_id: int
    job_id: int
    event_type: str
    stage: str | None
    status: str | None
    message: str | None
    event_metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class PipelineJobListItem:
    job: PipelineJobRecord
    original_file_name: str | None
    document_title: str | None
    requested_by_login_id: str | None
    requested_by_display_name: str | None


@dataclass(frozen=True)
class PipelineQueueStageSummary:
    stage: str
    total_count: int
    queued_count: int
    running_count: int
    failed_count: int
    average_progress_percent: Decimal | None
    oldest_queued_at: datetime | None


@dataclass(frozen=True)
class PipelineQueueTypeSummary:
    job_type: str
    total_count: int
    queued_count: int
    running_count: int
    failed_count: int


@dataclass(frozen=True)
class PipelineQueueSummary:
    stage_summaries: tuple[PipelineQueueStageSummary, ...]
    type_summaries: tuple[PipelineQueueTypeSummary, ...]
    total_count: int
    queued_count: int
    running_count: int
    stale_running_count: int
    reclaimable_stale_running_count: int
    failed_count: int
    retryable_failed_count: int
    exhausted_failed_count: int
    canceled_count: int
    retryable_canceled_count: int
    exhausted_canceled_count: int
    succeeded_count: int
    skipped_count: int
    oldest_queued_at: datetime | None
    oldest_stale_lease_expires_at: datetime | None

    @property
    def claimable_count(self) -> int:
        return (
            self.queued_count
            + self.reclaimable_stale_running_count
            + self.retryable_failed_count
            + self.retryable_canceled_count
        )

    @property
    def attention_count(self) -> int:
        return (
            self.stale_running_count
            + self.failed_count
            + self.canceled_count
        )


class InvalidPipelineJobError(ValueError):
    """Raised when a pipeline job operation is invalid before reaching the DB."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidPipelineJobError(f"{field_name} must be greater than 0")


def _validate_job_type(job_type: str) -> None:
    if job_type not in PIPELINE_JOB_TYPES:
        raise InvalidPipelineJobError(f"Unsupported pipeline job type: {job_type}")


def _validate_stage(stage: str | None) -> None:
    if stage is not None and stage not in PIPELINE_JOB_STAGES:
        raise InvalidPipelineJobError(f"Unsupported pipeline stage: {stage}")


def _validate_event_type(event_type: str) -> None:
    if event_type not in PIPELINE_EVENT_TYPES:
        raise InvalidPipelineJobError(f"Unsupported pipeline event type: {event_type}")


def _validate_status(status: str | None) -> None:
    if status is not None and status not in PIPELINE_JOB_STATUSES:
        raise InvalidPipelineJobError(f"Unsupported pipeline job status: {status}")


def _validate_worker(worker_name: str) -> str:
    worker = worker_name.strip()
    if not worker:
        raise InvalidPipelineJobError("worker_name is required")
    return worker


def _validate_lease_seconds(lease_seconds: int) -> None:
    if lease_seconds <= 0:
        raise InvalidPipelineJobError("lease_seconds must be greater than 0")


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise InvalidPipelineJobError("limit must be greater than 0")
    if limit > 500:
        raise InvalidPipelineJobError("limit must be less than or equal to 500")


def validate_pipeline_job_input(job_input: PipelineJobInput) -> None:
    _validate_job_type(job_input.job_type)
    _require_positive_id(job_input.file_id, "file_id")
    _require_positive_id(job_input.document_id, "document_id")
    _require_positive_id(job_input.parent_job_id, "parent_job_id")
    _require_positive_id(job_input.requested_by_user_id, "requested_by_user_id")
    if job_input.priority < 0:
        raise InvalidPipelineJobError("priority must be greater than or equal to 0")
    if job_input.total_units < 0:
        raise InvalidPipelineJobError("total_units must be greater than or equal to 0")


def _row_to_pipeline_job_record(row: dict[str, Any]) -> PipelineJobRecord:
    return PipelineJobRecord(
        job_id=int(row["job_id"]),
        job_type=str(row["job_type"]),
        file_id=int(row["file_id"]) if row.get("file_id") is not None else None,
        document_id=int(row["document_id"]) if row.get("document_id") is not None else None,
        parent_job_id=int(row["parent_job_id"]) if row.get("parent_job_id") is not None else None,
        requested_by_user_id=(
            int(row["requested_by_user_id"])
            if row.get("requested_by_user_id") is not None
            else None
        ),
        status=str(row["status"]),
        stage=str(row["stage"]),
        priority=int(row["priority"]),
        total_units=int(row["total_units"]),
        processed_units=int(row["processed_units"]),
        progress_percent=Decimal(row["progress_percent"]),
        current_message=row["current_message"],
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        heartbeat_at=row["heartbeat_at"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        metadata=dict(row["metadata"] or {}),
        queued_at=row["queued_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=row["updated_at"],
    )


def _row_to_pipeline_job_event_record(row: dict[str, Any]) -> PipelineJobEventRecord:
    return PipelineJobEventRecord(
        event_id=int(row["event_id"]),
        job_id=int(row["job_id"]),
        event_type=str(row["event_type"]),
        stage=row["stage"],
        status=row["status"],
        message=row["message"],
        event_metadata=dict(row["event_metadata"] or {}),
        created_at=row["created_at"],
    )


def _row_to_pipeline_job_list_item(row: dict[str, Any]) -> PipelineJobListItem:
    return PipelineJobListItem(
        job=_row_to_pipeline_job_record(row),
        original_file_name=row["original_file_name"],
        document_title=row["document_title"],
        requested_by_login_id=row["requested_by_login_id"],
        requested_by_display_name=row["requested_by_display_name"],
    )


def _row_to_pipeline_queue_stage_summary(row: dict[str, Any]) -> PipelineQueueStageSummary:
    return PipelineQueueStageSummary(
        stage=str(row["stage"]),
        total_count=int(row["total_count"]),
        queued_count=int(row["queued_count"]),
        running_count=int(row["running_count"]),
        failed_count=int(row["failed_count"]),
        average_progress_percent=row["average_progress_percent"],
        oldest_queued_at=row["oldest_queued_at"],
    )


def _row_to_pipeline_queue_type_summary(row: dict[str, Any]) -> PipelineQueueTypeSummary:
    return PipelineQueueTypeSummary(
        job_type=str(row["job_type"]),
        total_count=int(row["total_count"]),
        queued_count=int(row["queued_count"]),
        running_count=int(row["running_count"]),
        failed_count=int(row["failed_count"]),
    )


def _select_pipeline_job_columns(alias: str = "pipeline_jobs") -> str:
    return f"""
        {alias}.job_id,
        {alias}.job_type,
        {alias}.file_id,
        {alias}.document_id,
        {alias}.parent_job_id,
        {alias}.requested_by_user_id,
        {alias}.status,
        {alias}.stage,
        {alias}.priority,
        {alias}.total_units,
        {alias}.processed_units,
        {alias}.progress_percent,
        {alias}.current_message,
        {alias}.attempts,
        {alias}.max_attempts,
        {alias}.lease_owner,
        {alias}.lease_expires_at,
        {alias}.heartbeat_at,
        {alias}.error_code,
        {alias}.error_message,
        {alias}.metadata,
        {alias}.queued_at,
        {alias}.started_at,
        {alias}.finished_at,
        {alias}.updated_at
    """


def get_pipeline_job_in_connection(
    connection: Connection,
    job_id: int,
) -> PipelineJobRecord | None:
    _require_positive_id(job_id, "job_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_select_pipeline_job_columns()}
            FROM pipeline_jobs
            WHERE job_id = %s
            """,
            (job_id,),
        )
        row = cursor.fetchone()
    return _row_to_pipeline_job_record(dict(row)) if row else None


def get_pipeline_job(database_url: str, job_id: int) -> PipelineJobRecord | None:
    with connect(database_url) as connection:
        return get_pipeline_job_in_connection(connection, job_id)


def list_pipeline_jobs(
    database_url: str,
    *,
    status: str | None = None,
    limit: int = 100,
) -> list[PipelineJobListItem]:
    _validate_status(status)
    _validate_limit(limit)
    where_clause = ""
    params: tuple[object, ...] = (limit,)
    if status is not None:
        where_clause = "WHERE pj.status = %s"
        params = (status, limit)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    {_select_pipeline_job_columns("pj")},
                    f.original_file_name,
                    d.document_title,
                    u.login_id AS requested_by_login_id,
                    u.display_name AS requested_by_display_name
                FROM pipeline_jobs pj
                LEFT JOIN files f ON f.file_id = pj.file_id
                LEFT JOIN documents d ON d.document_id = pj.document_id
                LEFT JOIN app_users u ON u.user_id = pj.requested_by_user_id
                {where_clause}
                ORDER BY pj.queued_at DESC, pj.job_id DESC
                LIMIT %s
                """,
                params,
            )
            rows = cursor.fetchall()
    return [_row_to_pipeline_job_list_item(dict(row)) for row in rows]


def list_pipeline_job_events(
    database_url: str,
    job_id: int,
    *,
    limit: int = 100,
) -> list[PipelineJobEventRecord]:
    _require_positive_id(job_id, "job_id")
    _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    event_id,
                    job_id,
                    event_type,
                    stage,
                    status,
                    message,
                    event_metadata,
                    created_at
                FROM pipeline_job_events
                WHERE job_id = %s
                ORDER BY created_at ASC, event_id ASC
                LIMIT %s
                """,
                (job_id, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_pipeline_job_event_record(dict(row)) for row in rows]


def get_pipeline_queue_summary(database_url: str) -> PipelineQueueSummary:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*)::int AS total_count,
                    COUNT(*) FILTER (WHERE status = 'queued')::int AS queued_count,
                    COUNT(*) FILTER (WHERE status = 'running')::int AS running_count,
                    COUNT(*) FILTER (
                        WHERE status = 'running'
                          AND lease_expires_at < now()
                    )::int AS stale_running_count,
                    COUNT(*) FILTER (
                        WHERE status = 'running'
                          AND lease_expires_at < now()
                          AND attempts < max_attempts
                    )::int AS reclaimable_stale_running_count,
                    COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count,
                    COUNT(*) FILTER (
                        WHERE status = 'failed'
                          AND attempts < max_attempts
                    )::int AS retryable_failed_count,
                    COUNT(*) FILTER (
                        WHERE status = 'failed'
                          AND attempts >= max_attempts
                    )::int AS exhausted_failed_count,
                    COUNT(*) FILTER (WHERE status = 'canceled')::int AS canceled_count,
                    COUNT(*) FILTER (
                        WHERE status = 'canceled'
                          AND attempts < max_attempts
                    )::int AS retryable_canceled_count,
                    COUNT(*) FILTER (
                        WHERE status = 'canceled'
                          AND attempts >= max_attempts
                    )::int AS exhausted_canceled_count,
                    COUNT(*) FILTER (WHERE status = 'succeeded')::int AS succeeded_count,
                    COUNT(*) FILTER (WHERE status = 'skipped')::int AS skipped_count,
                    MIN(queued_at) FILTER (WHERE status = 'queued') AS oldest_queued_at,
                    MIN(lease_expires_at) FILTER (
                        WHERE status = 'running'
                          AND lease_expires_at < now()
                    ) AS oldest_stale_lease_expires_at
                FROM pipeline_jobs
                """
            )
            summary_row = dict(cursor.fetchone() or {})

            cursor.execute(
                """
                SELECT
                    stage,
                    COUNT(*)::int AS total_count,
                    COUNT(*) FILTER (WHERE status = 'queued')::int AS queued_count,
                    COUNT(*) FILTER (WHERE status = 'running')::int AS running_count,
                    COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count,
                    AVG(progress_percent) AS average_progress_percent,
                    MIN(queued_at) FILTER (WHERE status = 'queued') AS oldest_queued_at
                FROM pipeline_jobs
                GROUP BY stage
                ORDER BY total_count DESC, stage ASC
                LIMIT 8
                """
            )
            stage_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT
                    job_type,
                    COUNT(*)::int AS total_count,
                    COUNT(*) FILTER (WHERE status = 'queued')::int AS queued_count,
                    COUNT(*) FILTER (WHERE status = 'running')::int AS running_count,
                    COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count
                FROM pipeline_jobs
                GROUP BY job_type
                ORDER BY total_count DESC, job_type ASC
                LIMIT 8
                """
            )
            type_rows = cursor.fetchall()

    return PipelineQueueSummary(
        stage_summaries=tuple(
            _row_to_pipeline_queue_stage_summary(dict(row)) for row in stage_rows
        ),
        type_summaries=tuple(
            _row_to_pipeline_queue_type_summary(dict(row)) for row in type_rows
        ),
        total_count=int(summary_row["total_count"]),
        queued_count=int(summary_row["queued_count"]),
        running_count=int(summary_row["running_count"]),
        stale_running_count=int(summary_row["stale_running_count"]),
        reclaimable_stale_running_count=int(
            summary_row["reclaimable_stale_running_count"]
        ),
        failed_count=int(summary_row["failed_count"]),
        retryable_failed_count=int(summary_row["retryable_failed_count"]),
        exhausted_failed_count=int(summary_row["exhausted_failed_count"]),
        canceled_count=int(summary_row["canceled_count"]),
        retryable_canceled_count=int(summary_row["retryable_canceled_count"]),
        exhausted_canceled_count=int(summary_row["exhausted_canceled_count"]),
        succeeded_count=int(summary_row["succeeded_count"]),
        skipped_count=int(summary_row["skipped_count"]),
        oldest_queued_at=summary_row["oldest_queued_at"],
        oldest_stale_lease_expires_at=summary_row["oldest_stale_lease_expires_at"],
    )


def create_pipeline_job_in_connection(
    connection: Connection,
    job_input: PipelineJobInput,
) -> PipelineJobRecord:
    validate_pipeline_job_input(job_input)
    metadata = job_input.metadata or {}
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO pipeline_jobs (
                job_type,
                file_id,
                document_id,
                parent_job_id,
                requested_by_user_id,
                priority,
                total_units,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_select_pipeline_job_columns()}
            """,
            (
                job_input.job_type,
                job_input.file_id,
                job_input.document_id,
                job_input.parent_job_id,
                job_input.requested_by_user_id,
                job_input.priority,
                job_input.total_units,
                Json(metadata),
            ),
        )
        job = _row_to_pipeline_job_record(dict(cursor.fetchone()))

    record_pipeline_event_in_connection(
        connection,
        job.job_id,
        "created",
        stage=job.stage,
        status=job.status,
        message="Pipeline job created",
        event_metadata={"job_type": job.job_type},
    )
    return job


def create_pipeline_job(
    database_url: str,
    job_input: PipelineJobInput,
) -> PipelineJobRecord:
    with connect(database_url) as connection:
        return create_pipeline_job_in_connection(connection, job_input)


def record_pipeline_event_in_connection(
    connection: Connection,
    job_id: int,
    event_type: str,
    *,
    stage: str | None = None,
    status: str | None = None,
    message: str | None = None,
    event_metadata: dict[str, Any] | None = None,
) -> PipelineJobEventRecord:
    _require_positive_id(job_id, "job_id")
    _validate_event_type(event_type)
    _validate_stage(stage)
    _validate_status(status)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO pipeline_job_events (
                job_id,
                event_type,
                stage,
                status,
                message,
                event_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING
                event_id,
                job_id,
                event_type,
                stage,
                status,
                message,
                event_metadata,
                created_at
            """,
            (
                job_id,
                event_type,
                stage,
                status,
                message,
                Json(event_metadata or {}),
            ),
        )
        return _row_to_pipeline_job_event_record(dict(cursor.fetchone()))


def record_pipeline_event(
    database_url: str,
    job_id: int,
    event_type: str,
    *,
    stage: str | None = None,
    status: str | None = None,
    message: str | None = None,
    event_metadata: dict[str, Any] | None = None,
) -> PipelineJobEventRecord:
    with connect(database_url) as connection:
        return record_pipeline_event_in_connection(
            connection,
            job_id,
            event_type,
            stage=stage,
            status=status,
            message=message,
            event_metadata=event_metadata,
        )


def claim_next_pipeline_job_in_connection(
    connection: Connection,
    worker_name: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> PipelineJobRecord | None:
    worker = _validate_worker(worker_name)
    _validate_lease_seconds(lease_seconds)
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT job_id
            FROM pipeline_jobs
            WHERE status = 'queued'
               OR (
                   status = 'running'
                   AND lease_expires_at < now()
                   AND attempts < max_attempts
               )
            ORDER BY
                CASE WHEN status = 'queued' THEN 0 ELSE 1 END,
                priority ASC,
                queued_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """)
        row = cursor.fetchone()
        if row is None:
            return None

        cursor.execute(
            f"""
            UPDATE pipeline_jobs
            SET status = 'running',
                lease_owner = %s,
                lease_expires_at = now() + (%s * interval '1 second'),
                heartbeat_at = now(),
                attempts = attempts + 1,
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_pipeline_job_columns()}
            """,
            (worker, lease_seconds, row["job_id"]),
        )
        job = _row_to_pipeline_job_record(dict(cursor.fetchone()))

    record_pipeline_event_in_connection(
        connection,
        job.job_id,
        "claimed",
        stage=job.stage,
        status=job.status,
        message=f"Claimed by {worker}",
        event_metadata={"worker_name": worker, "lease_seconds": lease_seconds},
    )
    return job


def claim_next_pipeline_job(
    database_url: str,
    worker_name: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> PipelineJobRecord | None:
    with connect(database_url) as connection:
        return claim_next_pipeline_job_in_connection(
            connection,
            worker_name,
            lease_seconds=lease_seconds,
        )


def heartbeat_pipeline_job_in_connection(
    connection: Connection,
    job_id: int,
    worker_name: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> PipelineJobRecord | None:
    _require_positive_id(job_id, "job_id")
    worker = _validate_worker(worker_name)
    _validate_lease_seconds(lease_seconds)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE pipeline_jobs
            SET lease_expires_at = now() + (%s * interval '1 second'),
                heartbeat_at = now(),
                updated_at = now()
            WHERE job_id = %s
              AND status = 'running'
              AND lease_owner = %s
            RETURNING {_select_pipeline_job_columns()}
            """,
            (lease_seconds, job_id, worker),
        )
        row = cursor.fetchone()
    if row is None:
        return None

    job = _row_to_pipeline_job_record(dict(row))
    record_pipeline_event_in_connection(
        connection,
        job.job_id,
        "heartbeat",
        stage=job.stage,
        status=job.status,
        message=f"Heartbeat from {worker}",
        event_metadata={"worker_name": worker, "lease_seconds": lease_seconds},
    )
    return job


def heartbeat_pipeline_job(
    database_url: str,
    job_id: int,
    worker_name: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> PipelineJobRecord | None:
    with connect(database_url) as connection:
        return heartbeat_pipeline_job_in_connection(
            connection,
            job_id,
            worker_name,
            lease_seconds=lease_seconds,
        )


def update_pipeline_progress_in_connection(
    connection: Connection,
    job_id: int,
    *,
    processed_units: int,
    total_units: int | None = None,
    stage: str | None = None,
    current_message: str | None = None,
) -> PipelineJobRecord | None:
    _require_positive_id(job_id, "job_id")
    _validate_stage(stage)
    if processed_units < 0:
        raise InvalidPipelineJobError("processed_units must be greater than or equal to 0")

    current_job = get_pipeline_job_in_connection(connection, job_id)
    if current_job is None:
        return None

    effective_total_units = current_job.total_units if total_units is None else total_units
    if effective_total_units < 0:
        raise InvalidPipelineJobError("total_units must be greater than or equal to 0")
    if processed_units > effective_total_units:
        raise InvalidPipelineJobError("processed_units must be less than or equal to total_units")

    progress_percent = Decimal("0.00")
    if effective_total_units > 0:
        progress_percent = Decimal(processed_units * 100) / Decimal(effective_total_units)
        progress_percent = progress_percent.quantize(Decimal("0.01"))

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE pipeline_jobs
            SET stage = COALESCE(%s, stage),
                total_units = %s,
                processed_units = %s,
                progress_percent = %s,
                current_message = %s,
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_pipeline_job_columns()}
            """,
            (
                stage,
                effective_total_units,
                processed_units,
                progress_percent,
                current_message,
                job_id,
            ),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    job = _row_to_pipeline_job_record(dict(row))
    record_pipeline_event_in_connection(
        connection,
        job.job_id,
        "progress",
        stage=job.stage,
        status=job.status,
        message=current_message,
        event_metadata={
            "processed_units": job.processed_units,
            "total_units": job.total_units,
            "progress_percent": str(job.progress_percent),
        },
    )
    return job


def update_pipeline_progress(
    database_url: str,
    job_id: int,
    *,
    processed_units: int,
    total_units: int | None = None,
    stage: str | None = None,
    current_message: str | None = None,
) -> PipelineJobRecord | None:
    with connect(database_url) as connection:
        return update_pipeline_progress_in_connection(
            connection,
            job_id,
            processed_units=processed_units,
            total_units=total_units,
            stage=stage,
            current_message=current_message,
        )


def mark_pipeline_succeeded_in_connection(
    connection: Connection,
    job_id: int,
    *,
    message: str | None = None,
) -> PipelineJobRecord | None:
    _require_positive_id(job_id, "job_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE pipeline_jobs
            SET status = 'succeeded',
                stage = 'completed',
                processed_units = CASE
                    WHEN total_units > 0 THEN total_units
                    ELSE processed_units
                END,
                progress_percent = 100,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                error_code = NULL,
                error_message = NULL,
                finished_at = COALESCE(finished_at, now()),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_pipeline_job_columns()}
            """,
            (job_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    job = _row_to_pipeline_job_record(dict(row))
    record_pipeline_event_in_connection(
        connection,
        job.job_id,
        "stage_succeeded",
        stage=job.stage,
        status=job.status,
        message=message or "Pipeline job succeeded",
    )
    return job


def mark_pipeline_succeeded(
    database_url: str,
    job_id: int,
    *,
    message: str | None = None,
) -> PipelineJobRecord | None:
    with connect(database_url) as connection:
        return mark_pipeline_succeeded_in_connection(
            connection,
            job_id,
            message=message,
        )


def mark_pipeline_failed_in_connection(
    connection: Connection,
    job_id: int,
    *,
    error_code: str,
    error_message: str,
) -> PipelineJobRecord | None:
    _require_positive_id(job_id, "job_id")
    if not error_code.strip():
        raise InvalidPipelineJobError("error_code is required")
    if not error_message.strip():
        raise InvalidPipelineJobError("error_message is required")

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE pipeline_jobs
            SET status = 'failed',
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                error_code = %s,
                error_message = %s,
                finished_at = now(),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_pipeline_job_columns()}
            """,
            (error_code.strip(), error_message.strip(), job_id),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    job = _row_to_pipeline_job_record(dict(row))
    record_pipeline_event_in_connection(
        connection,
        job.job_id,
        "failed",
        stage=job.stage,
        status=job.status,
        message=job.error_message,
        event_metadata={"error_code": job.error_code},
    )
    return job


def mark_pipeline_failed(
    database_url: str,
    job_id: int,
    *,
    error_code: str,
    error_message: str,
) -> PipelineJobRecord | None:
    with connect(database_url) as connection:
        return mark_pipeline_failed_in_connection(
            connection,
            job_id,
            error_code=error_code,
            error_message=error_message,
        )


def retry_pipeline_job_in_connection(
    connection: Connection,
    job_id: int,
    *,
    message: str | None = None,
) -> PipelineJobRecord | None:
    _require_positive_id(job_id, "job_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE pipeline_jobs
            SET status = 'queued',
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                error_code = NULL,
                error_message = NULL,
                finished_at = NULL,
                queued_at = now(),
                updated_at = now()
            WHERE job_id = %s
              AND status IN ('failed', 'canceled')
              AND attempts < max_attempts
            RETURNING {_select_pipeline_job_columns()}
            """,
            (job_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    job = _row_to_pipeline_job_record(dict(row))
    record_pipeline_event_in_connection(
        connection,
        job.job_id,
        "retried",
        stage=job.stage,
        status=job.status,
        message=message or "Pipeline job queued for retry",
    )
    return job


def retry_pipeline_job(
    database_url: str,
    job_id: int,
    *,
    message: str | None = None,
) -> PipelineJobRecord | None:
    with connect(database_url) as connection:
        return retry_pipeline_job_in_connection(
            connection,
            job_id,
            message=message,
        )
