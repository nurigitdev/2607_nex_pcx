"""Persistence helpers for embedding worker batch run summaries."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect
from app.core.embedding_worker import (
    EMBEDDING_WORKER_BATCH_STOP_LIMIT_REACHED,
    EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY,
    MAX_EMBEDDING_WORKER_BATCH_LIMIT,
    READINESS_GATE_FAILURE_MODES,
)

EMBEDDING_WORKER_BATCH_PROVIDER_SOURCES = ("route", "runtime")
EMBEDDING_WORKER_BATCH_STOP_REASONS = (
    EMBEDDING_WORKER_BATCH_STOP_LIMIT_REACHED,
    EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY,
)
MAX_EMBEDDING_WORKER_BATCH_RUN_LIMIT = 200


@dataclass(frozen=True)
class EmbeddingWorkerBatchRunInput:
    worker_name: str
    provider_source: str
    provider_mode: str
    limit_requested: int
    result_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    deferred_count: int
    idle_count: int
    stopped_reason: str
    profile_name: str | None = None
    remote_provider_url: str | None = None
    require_route_readiness: bool = False
    readiness_gate_failure_mode: str = "fail"
    readiness_gate_defer_seconds: int = 300
    job_ids: tuple[int, ...] = ()
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class EmbeddingWorkerBatchRunRecord:
    batch_run_id: int
    worker_name: str
    profile_name: str | None
    provider_source: str
    provider_mode: str
    remote_provider_url: str | None
    require_route_readiness: bool
    readiness_gate_failure_mode: str
    readiness_gate_defer_seconds: int
    limit_requested: int
    result_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    deferred_count: int
    idle_count: int
    stopped_reason: str
    job_ids: tuple[int, ...]
    runtime_metadata: dict[str, Any]
    elapsed_ms: int
    started_at: datetime
    completed_at: datetime
    created_at: datetime


class InvalidEmbeddingWorkerBatchRunError(ValueError):
    """Raised when embedding worker batch run summary data is invalid."""


def record_embedding_worker_batch_run(
    database_url: str,
    run_input: EmbeddingWorkerBatchRunInput,
) -> EmbeddingWorkerBatchRunRecord:
    validated = validate_embedding_worker_batch_run_input(run_input)
    completed_at = validated.completed_at or datetime.now(UTC)
    started_at = validated.started_at or completed_at
    elapsed_ms = validated.elapsed_ms
    if elapsed_ms is None:
        elapsed_ms = max(0, int((completed_at - started_at).total_seconds() * 1000))

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_worker_batch_runs (
                    worker_name,
                    profile_name,
                    provider_source,
                    provider_mode,
                    remote_provider_url,
                    require_route_readiness,
                    readiness_gate_failure_mode,
                    readiness_gate_defer_seconds,
                    limit_requested,
                    result_count,
                    processed_count,
                    succeeded_count,
                    failed_count,
                    deferred_count,
                    idle_count,
                    stopped_reason,
                    job_ids,
                    runtime_metadata,
                    elapsed_ms,
                    started_at,
                    completed_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING *
                """,
                (
                    validated.worker_name,
                    validated.profile_name,
                    validated.provider_source,
                    validated.provider_mode,
                    validated.remote_provider_url,
                    validated.require_route_readiness,
                    validated.readiness_gate_failure_mode,
                    validated.readiness_gate_defer_seconds,
                    validated.limit_requested,
                    validated.result_count,
                    validated.processed_count,
                    validated.succeeded_count,
                    validated.failed_count,
                    validated.deferred_count,
                    validated.idle_count,
                    validated.stopped_reason,
                    list(validated.job_ids),
                    Json(validated.runtime_metadata),
                    elapsed_ms,
                    started_at,
                    completed_at,
                ),
            )
            return _row_to_batch_run_record(dict(cursor.fetchone()))


def list_embedding_worker_batch_runs(
    database_url: str,
    *,
    worker_name: str | None = None,
    profile_name: str | None = None,
    stopped_reason: str | None = None,
    limit: int = 50,
) -> list[EmbeddingWorkerBatchRunRecord]:
    _validate_limit(limit)
    where_clauses = []
    params: list[object] = []
    if worker_name is not None:
        where_clauses.append("worker_name = %s")
        params.append(_validate_nonblank(worker_name, "worker_name"))
    if profile_name is not None:
        where_clauses.append("profile_name = %s")
        params.append(_validate_nonblank(profile_name, "profile_name"))
    if stopped_reason is not None:
        where_clauses.append("stopped_reason = %s")
        params.append(_validate_stopped_reason(stopped_reason))

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM embedding_worker_batch_runs
                {where_sql}
                ORDER BY completed_at DESC, batch_run_id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_batch_run_record(dict(row)) for row in rows]


def get_embedding_worker_batch_run(
    database_url: str,
    batch_run_id: int,
) -> EmbeddingWorkerBatchRunRecord | None:
    if batch_run_id <= 0:
        raise InvalidEmbeddingWorkerBatchRunError("batch_run_id must be greater than 0")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM embedding_worker_batch_runs
                WHERE batch_run_id = %s
                """,
                (batch_run_id,),
            )
            row = cursor.fetchone()
    return _row_to_batch_run_record(dict(row)) if row is not None else None


def validate_embedding_worker_batch_run_input(
    run_input: EmbeddingWorkerBatchRunInput,
) -> EmbeddingWorkerBatchRunInput:
    worker_name = _validate_nonblank(run_input.worker_name, "worker_name")
    profile_name = (
        _validate_nonblank(run_input.profile_name, "profile_name")
        if run_input.profile_name is not None
        else None
    )
    provider_source = _validate_provider_source(run_input.provider_source)
    provider_mode = _validate_nonblank(run_input.provider_mode, "provider_mode").lower()
    remote_provider_url = (
        _validate_nonblank(run_input.remote_provider_url, "remote_provider_url")
        if run_input.remote_provider_url is not None
        else None
    )
    readiness_gate_failure_mode = _validate_readiness_gate_failure_mode(
        run_input.readiness_gate_failure_mode
    )
    if run_input.readiness_gate_defer_seconds <= 0:
        raise InvalidEmbeddingWorkerBatchRunError(
            "readiness_gate_defer_seconds must be greater than 0"
        )
    _validate_batch_limit(run_input.limit_requested)
    _validate_batch_counts(run_input)
    stopped_reason = _validate_stopped_reason(run_input.stopped_reason)
    job_ids = _validate_job_ids(run_input.job_ids)
    if run_input.elapsed_ms is not None and run_input.elapsed_ms < 0:
        raise InvalidEmbeddingWorkerBatchRunError("elapsed_ms must be greater than or equal to 0")
    if (
        run_input.started_at is not None
        and run_input.completed_at is not None
        and run_input.completed_at < run_input.started_at
    ):
        raise InvalidEmbeddingWorkerBatchRunError(
            "completed_at must be greater than or equal to started_at"
        )

    return EmbeddingWorkerBatchRunInput(
        worker_name=worker_name,
        profile_name=profile_name,
        provider_source=provider_source,
        provider_mode=provider_mode,
        remote_provider_url=remote_provider_url,
        require_route_readiness=bool(run_input.require_route_readiness),
        readiness_gate_failure_mode=readiness_gate_failure_mode,
        readiness_gate_defer_seconds=run_input.readiness_gate_defer_seconds,
        limit_requested=run_input.limit_requested,
        result_count=run_input.result_count,
        processed_count=run_input.processed_count,
        succeeded_count=run_input.succeeded_count,
        failed_count=run_input.failed_count,
        deferred_count=run_input.deferred_count,
        idle_count=run_input.idle_count,
        stopped_reason=stopped_reason,
        job_ids=job_ids,
        runtime_metadata=dict(run_input.runtime_metadata),
        elapsed_ms=run_input.elapsed_ms,
        started_at=run_input.started_at,
        completed_at=run_input.completed_at,
    )


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise InvalidEmbeddingWorkerBatchRunError("limit must be greater than 0")
    if limit > MAX_EMBEDDING_WORKER_BATCH_RUN_LIMIT:
        raise InvalidEmbeddingWorkerBatchRunError(
            f"limit must be less than or equal to {MAX_EMBEDDING_WORKER_BATCH_RUN_LIMIT}"
        )


def _validate_batch_limit(limit_requested: int) -> None:
    if limit_requested <= 0:
        raise InvalidEmbeddingWorkerBatchRunError("limit_requested must be greater than 0")
    if limit_requested > MAX_EMBEDDING_WORKER_BATCH_LIMIT:
        raise InvalidEmbeddingWorkerBatchRunError(
            "limit_requested must be less than or equal to " f"{MAX_EMBEDDING_WORKER_BATCH_LIMIT}"
        )


def _validate_batch_counts(run_input: EmbeddingWorkerBatchRunInput) -> None:
    count_fields = (
        "result_count",
        "processed_count",
        "succeeded_count",
        "failed_count",
        "deferred_count",
        "idle_count",
    )
    for field_name in count_fields:
        if getattr(run_input, field_name) < 0:
            raise InvalidEmbeddingWorkerBatchRunError(
                f"{field_name} must be greater than or equal to 0"
            )
    if run_input.result_count > run_input.limit_requested:
        raise InvalidEmbeddingWorkerBatchRunError(
            "result_count must be less than or equal to limit_requested"
        )
    if run_input.processed_count + run_input.idle_count != run_input.result_count:
        raise InvalidEmbeddingWorkerBatchRunError(
            "processed_count plus idle_count must equal result_count"
        )
    if (
        run_input.succeeded_count + run_input.failed_count + run_input.deferred_count
        > run_input.processed_count
    ):
        raise InvalidEmbeddingWorkerBatchRunError(
            "terminal and deferred counts must not exceed processed_count"
        )
    if len(run_input.job_ids) > run_input.processed_count:
        raise InvalidEmbeddingWorkerBatchRunError("job_ids count must not exceed processed_count")


def _validate_provider_source(provider_source: str) -> str:
    normalized = _validate_nonblank(provider_source, "provider_source").lower()
    if normalized not in EMBEDDING_WORKER_BATCH_PROVIDER_SOURCES:
        raise InvalidEmbeddingWorkerBatchRunError(f"Unsupported provider_source: {normalized}")
    return normalized


def _validate_readiness_gate_failure_mode(mode: str) -> str:
    normalized = _validate_nonblank(mode, "readiness_gate_failure_mode").lower()
    if normalized not in READINESS_GATE_FAILURE_MODES:
        raise InvalidEmbeddingWorkerBatchRunError(
            f"Unsupported readiness_gate_failure_mode: {normalized}"
        )
    return normalized


def _validate_stopped_reason(stopped_reason: str) -> str:
    normalized = _validate_nonblank(stopped_reason, "stopped_reason").lower()
    if normalized not in EMBEDDING_WORKER_BATCH_STOP_REASONS:
        raise InvalidEmbeddingWorkerBatchRunError(f"Unsupported stopped_reason: {normalized}")
    return normalized


def _validate_job_ids(job_ids: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(int(job_id) for job_id in job_ids)
    if any(job_id <= 0 for job_id in normalized):
        raise InvalidEmbeddingWorkerBatchRunError("job_ids must be greater than 0")
    return normalized


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingWorkerBatchRunError(f"{field_name} is required")
    return normalized


def _row_to_batch_run_record(row: dict[str, Any]) -> EmbeddingWorkerBatchRunRecord:
    return EmbeddingWorkerBatchRunRecord(
        batch_run_id=int(row["batch_run_id"]),
        worker_name=str(row["worker_name"]),
        profile_name=row["profile_name"],
        provider_source=str(row["provider_source"]),
        provider_mode=str(row["provider_mode"]),
        remote_provider_url=row["remote_provider_url"],
        require_route_readiness=bool(row["require_route_readiness"]),
        readiness_gate_failure_mode=str(row["readiness_gate_failure_mode"]),
        readiness_gate_defer_seconds=int(row["readiness_gate_defer_seconds"]),
        limit_requested=int(row["limit_requested"]),
        result_count=int(row["result_count"]),
        processed_count=int(row["processed_count"]),
        succeeded_count=int(row["succeeded_count"]),
        failed_count=int(row["failed_count"]),
        deferred_count=int(row["deferred_count"]),
        idle_count=int(row["idle_count"]),
        stopped_reason=str(row["stopped_reason"]),
        job_ids=tuple(int(job_id) for job_id in (row["job_ids"] or [])),
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        elapsed_ms=int(row["elapsed_ms"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        created_at=row["created_at"],
    )
