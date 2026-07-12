"""Dashboard throughput and latency read-model helpers."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.core.database import connect

DEFAULT_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS = 24
MAX_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS = 720
PIPELINE_STAGE_ORDER = (
    "upload_saved",
    "text_extraction",
    "parsing",
    "chunking",
    "embedding",
    "vector_indexing",
    "completed",
)


@dataclass(frozen=True)
class DashboardPipelineStageLatency:
    stage: str
    completed_count: int
    succeeded_count: int
    failed_count: int
    canceled_count: int
    average_duration_ms: float | None


@dataclass(frozen=True)
class DashboardPipelineThroughput:
    completed_count: int
    succeeded_count: int
    failed_count: int
    canceled_count: int
    skipped_count: int
    average_duration_ms: float | None
    latest_finished_at: datetime | None
    stages: tuple[DashboardPipelineStageLatency, ...]


@dataclass(frozen=True)
class DashboardEmbeddingProfileThroughput:
    profile_name: str
    completed_job_count: int
    succeeded_job_count: int
    failed_job_count: int
    skipped_job_count: int
    average_job_duration_ms: float | None
    batch_run_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    deferred_count: int
    average_batch_elapsed_ms: float | None
    throughput_per_second: float
    success_rate_percent: float


@dataclass(frozen=True)
class DashboardEmbeddingThroughput:
    completed_job_count: int
    succeeded_job_count: int
    failed_job_count: int
    skipped_job_count: int
    average_job_duration_ms: float | None
    batch_run_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    deferred_count: int
    average_batch_elapsed_ms: float | None
    throughput_per_second: float
    latest_completed_at: datetime | None
    profiles: tuple[DashboardEmbeddingProfileThroughput, ...]


@dataclass(frozen=True)
class DashboardSearchProfileLatency:
    profile_name: str
    search_log_count: int
    result_count: int
    average_profile_elapsed_ms: float | None


@dataclass(frozen=True)
class DashboardSearchLatency:
    search_log_count: int
    result_count: int
    average_total_elapsed_ms: float | None
    average_profile_elapsed_ms: float | None
    latest_search_at: datetime | None
    profiles: tuple[DashboardSearchProfileLatency, ...]


@dataclass(frozen=True)
class DashboardThroughputLatencySnapshot:
    lookback_hours: int
    pipeline: DashboardPipelineThroughput
    embedding: DashboardEmbeddingThroughput
    search: DashboardSearchLatency


class InvalidDashboardThroughputError(ValueError):
    """Raised when throughput dashboard inputs are invalid."""


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _rounded_float(value: Any, *, digits: int = 1) -> float | None:
    numeric = _optional_float(value)
    return round(numeric, digits) if numeric is not None else None


def _rate_per_second(processed_count: int, elapsed_ms: int) -> float:
    if processed_count <= 0 or elapsed_ms <= 0:
        return 0.0
    return round(processed_count / (elapsed_ms / 1000), 2)


def _success_rate_percent(succeeded_count: int, processed_count: int) -> float:
    if processed_count <= 0:
        return 0.0
    return round((succeeded_count / processed_count) * 100, 1)


def validate_lookback_hours(lookback_hours: int) -> int:
    if lookback_hours <= 0:
        raise InvalidDashboardThroughputError("lookback_hours must be greater than 0")
    if lookback_hours > MAX_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS:
        raise InvalidDashboardThroughputError(
            "lookback_hours must be less than or equal to "
            f"{MAX_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS}"
        )
    return lookback_hours


def _pipeline_throughput_from_rows(
    total_row: dict[str, Any],
    stage_rows: list[dict[str, Any]],
) -> DashboardPipelineThroughput:
    return DashboardPipelineThroughput(
        completed_count=int(total_row["completed_count"] or 0),
        succeeded_count=int(total_row["succeeded_count"] or 0),
        failed_count=int(total_row["failed_count"] or 0),
        canceled_count=int(total_row["canceled_count"] or 0),
        skipped_count=int(total_row["skipped_count"] or 0),
        average_duration_ms=_rounded_float(total_row["average_duration_ms"]),
        latest_finished_at=total_row["latest_finished_at"],
        stages=tuple(
            DashboardPipelineStageLatency(
                stage=str(row["stage"]),
                completed_count=int(row["completed_count"] or 0),
                succeeded_count=int(row["succeeded_count"] or 0),
                failed_count=int(row["failed_count"] or 0),
                canceled_count=int(row["canceled_count"] or 0),
                average_duration_ms=_rounded_float(row["average_duration_ms"]),
            )
            for row in stage_rows
        ),
    )


def _embedding_throughput_from_rows(
    total_row: dict[str, Any],
    profile_rows: list[dict[str, Any]],
) -> DashboardEmbeddingThroughput:
    processed_count = int(total_row["processed_count"] or 0)
    succeeded_count = int(total_row["batch_succeeded_count"] or 0)
    elapsed_ms = int(total_row["batch_elapsed_ms"] or 0)
    return DashboardEmbeddingThroughput(
        completed_job_count=int(total_row["completed_job_count"] or 0),
        succeeded_job_count=int(total_row["succeeded_job_count"] or 0),
        failed_job_count=int(total_row["failed_job_count"] or 0),
        skipped_job_count=int(total_row["skipped_job_count"] or 0),
        average_job_duration_ms=_rounded_float(total_row["average_job_duration_ms"]),
        batch_run_count=int(total_row["batch_run_count"] or 0),
        processed_count=processed_count,
        succeeded_count=succeeded_count,
        failed_count=int(total_row["batch_failed_count"] or 0),
        deferred_count=int(total_row["deferred_count"] or 0),
        average_batch_elapsed_ms=_rounded_float(total_row["average_batch_elapsed_ms"]),
        throughput_per_second=_rate_per_second(processed_count, elapsed_ms),
        latest_completed_at=total_row["latest_completed_at"],
        profiles=tuple(
            _embedding_profile_throughput_from_row(row) for row in profile_rows
        ),
    )


def _embedding_profile_throughput_from_row(
    row: dict[str, Any],
) -> DashboardEmbeddingProfileThroughput:
    processed_count = int(row["processed_count"] or 0)
    succeeded_count = int(row["batch_succeeded_count"] or 0)
    elapsed_ms = int(row["batch_elapsed_ms"] or 0)
    return DashboardEmbeddingProfileThroughput(
        profile_name=str(row["profile_name"]),
        completed_job_count=int(row["completed_job_count"] or 0),
        succeeded_job_count=int(row["succeeded_job_count"] or 0),
        failed_job_count=int(row["failed_job_count"] or 0),
        skipped_job_count=int(row["skipped_job_count"] or 0),
        average_job_duration_ms=_rounded_float(row["average_job_duration_ms"]),
        batch_run_count=int(row["batch_run_count"] or 0),
        processed_count=processed_count,
        succeeded_count=succeeded_count,
        failed_count=int(row["batch_failed_count"] or 0),
        deferred_count=int(row["deferred_count"] or 0),
        average_batch_elapsed_ms=_rounded_float(row["average_batch_elapsed_ms"]),
        throughput_per_second=_rate_per_second(processed_count, elapsed_ms),
        success_rate_percent=_success_rate_percent(succeeded_count, processed_count),
    )


def _search_latency_from_rows(
    total_row: dict[str, Any],
    profile_rows: list[dict[str, Any]],
) -> DashboardSearchLatency:
    return DashboardSearchLatency(
        search_log_count=int(total_row["search_log_count"] or 0),
        result_count=int(total_row["result_count"] or 0),
        average_total_elapsed_ms=_rounded_float(total_row["average_total_elapsed_ms"]),
        average_profile_elapsed_ms=_rounded_float(
            total_row["average_profile_elapsed_ms"]
        ),
        latest_search_at=total_row["latest_search_at"],
        profiles=tuple(
            DashboardSearchProfileLatency(
                profile_name=str(row["profile_name"]),
                search_log_count=int(row["search_log_count"] or 0),
                result_count=int(row["result_count"] or 0),
                average_profile_elapsed_ms=_rounded_float(
                    row["average_profile_elapsed_ms"]
                ),
            )
            for row in profile_rows
        ),
    )


def get_dashboard_throughput_latency_snapshot(
    database_url: str,
    *,
    lookback_hours: int = DEFAULT_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS,
) -> DashboardThroughputLatencySnapshot:
    validated_lookback = validate_lookback_hours(lookback_hours)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH recent_pipeline AS (
                    SELECT
                        status,
                        stage,
                        finished_at,
                        EXTRACT(
                            EPOCH FROM (finished_at - COALESCE(started_at, queued_at))
                        ) * 1000 AS duration_ms
                    FROM pipeline_jobs
                    WHERE status IN ('succeeded', 'failed', 'canceled', 'skipped')
                      AND finished_at IS NOT NULL
                      AND finished_at >= now() - (%s::int * interval '1 hour')
                )
                SELECT
                    COUNT(*)::int AS completed_count,
                    COUNT(*) FILTER (WHERE status = 'succeeded')::int
                        AS succeeded_count,
                    COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count,
                    COUNT(*) FILTER (WHERE status = 'canceled')::int
                        AS canceled_count,
                    COUNT(*) FILTER (WHERE status = 'skipped')::int AS skipped_count,
                    AVG(duration_ms) AS average_duration_ms,
                    MAX(finished_at) AS latest_finished_at
                FROM recent_pipeline
                """,
                (validated_lookback,),
            )
            pipeline_total_row = dict(cursor.fetchone() or {})

            stage_values = ", ".join(
                f"({index}, '{stage}')"
                for index, stage in enumerate(PIPELINE_STAGE_ORDER, start=1)
            )
            cursor.execute(
                f"""
                WITH stages(stage_order, stage) AS (
                    VALUES {stage_values}
                ),
                recent_pipeline AS (
                    SELECT
                        status,
                        stage,
                        EXTRACT(
                            EPOCH FROM (finished_at - COALESCE(started_at, queued_at))
                        ) * 1000 AS duration_ms
                    FROM pipeline_jobs
                    WHERE status IN ('succeeded', 'failed', 'canceled', 'skipped')
                      AND finished_at IS NOT NULL
                      AND finished_at >= now() - (%s::int * interval '1 hour')
                )
                SELECT
                    stages.stage,
                    COUNT(recent_pipeline.stage)::int AS completed_count,
                    COUNT(recent_pipeline.stage) FILTER (
                        WHERE recent_pipeline.status = 'succeeded'
                    )::int AS succeeded_count,
                    COUNT(recent_pipeline.stage) FILTER (
                        WHERE recent_pipeline.status = 'failed'
                    )::int AS failed_count,
                    COUNT(recent_pipeline.stage) FILTER (
                        WHERE recent_pipeline.status = 'canceled'
                    )::int AS canceled_count,
                    AVG(recent_pipeline.duration_ms) AS average_duration_ms
                FROM stages
                LEFT JOIN recent_pipeline ON recent_pipeline.stage = stages.stage
                GROUP BY stages.stage_order, stages.stage
                ORDER BY stages.stage_order
                """,
                (validated_lookback,),
            )
            pipeline_stage_rows = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                WITH recent_jobs AS (
                    SELECT
                        status,
                        EXTRACT(
                            EPOCH FROM (finished_at - COALESCE(started_at, created_at))
                        ) * 1000 AS duration_ms
                    FROM embedding_jobs
                    WHERE status IN ('succeeded', 'failed', 'skipped')
                      AND finished_at IS NOT NULL
                      AND finished_at >= now() - (%s::int * interval '1 hour')
                ),
                recent_batches AS (
                    SELECT *
                    FROM embedding_worker_batch_runs
                    WHERE completed_at >= now() - (%s::int * interval '1 hour')
                )
                SELECT
                    (SELECT COUNT(*)::int FROM recent_jobs) AS completed_job_count,
                    (
                        SELECT COUNT(*)::int FROM recent_jobs WHERE status = 'succeeded'
                    ) AS succeeded_job_count,
                    (
                        SELECT COUNT(*)::int FROM recent_jobs WHERE status = 'failed'
                    ) AS failed_job_count,
                    (
                        SELECT COUNT(*)::int FROM recent_jobs WHERE status = 'skipped'
                    ) AS skipped_job_count,
                    (SELECT AVG(duration_ms) FROM recent_jobs)
                        AS average_job_duration_ms,
                    (SELECT COUNT(*)::int FROM recent_batches) AS batch_run_count,
                    COALESCE((SELECT SUM(processed_count) FROM recent_batches), 0)::int
                        AS processed_count,
                    COALESCE((SELECT SUM(succeeded_count) FROM recent_batches), 0)::int
                        AS batch_succeeded_count,
                    COALESCE((SELECT SUM(failed_count) FROM recent_batches), 0)::int
                        AS batch_failed_count,
                    COALESCE((SELECT SUM(deferred_count) FROM recent_batches), 0)::int
                        AS deferred_count,
                    COALESCE((SELECT SUM(elapsed_ms) FROM recent_batches), 0)::int
                        AS batch_elapsed_ms,
                    (SELECT AVG(elapsed_ms) FROM recent_batches)
                        AS average_batch_elapsed_ms,
                    (SELECT MAX(completed_at) FROM recent_batches)
                        AS latest_completed_at
                """,
                (validated_lookback, validated_lookback),
            )
            embedding_total_row = dict(cursor.fetchone() or {})

            cursor.execute(
                """
                WITH recent_jobs AS (
                    SELECT
                        profile_name,
                        status,
                        EXTRACT(
                            EPOCH FROM (finished_at - COALESCE(started_at, created_at))
                        ) * 1000 AS duration_ms
                    FROM embedding_jobs
                    WHERE status IN ('succeeded', 'failed', 'skipped')
                      AND finished_at IS NOT NULL
                      AND finished_at >= now() - (%s::int * interval '1 hour')
                ),
                job_groups AS (
                    SELECT
                        profile_name,
                        COUNT(*)::int AS completed_job_count,
                        COUNT(*) FILTER (WHERE status = 'succeeded')::int
                            AS succeeded_job_count,
                        COUNT(*) FILTER (WHERE status = 'failed')::int
                            AS failed_job_count,
                        COUNT(*) FILTER (WHERE status = 'skipped')::int
                            AS skipped_job_count,
                        AVG(duration_ms) AS average_job_duration_ms
                    FROM recent_jobs
                    GROUP BY profile_name
                ),
                batch_groups AS (
                    SELECT
                        profile_name,
                        COUNT(*)::int AS batch_run_count,
                        COALESCE(SUM(processed_count), 0)::int AS processed_count,
                        COALESCE(SUM(succeeded_count), 0)::int
                            AS batch_succeeded_count,
                        COALESCE(SUM(failed_count), 0)::int AS batch_failed_count,
                        COALESCE(SUM(deferred_count), 0)::int AS deferred_count,
                        COALESCE(SUM(elapsed_ms), 0)::int AS batch_elapsed_ms,
                        AVG(elapsed_ms) AS average_batch_elapsed_ms
                    FROM embedding_worker_batch_runs
                    WHERE completed_at >= now() - (%s::int * interval '1 hour')
                      AND profile_name IS NOT NULL
                    GROUP BY profile_name
                )
                SELECT
                    ep.profile_name,
                    COALESCE(job_groups.completed_job_count, 0)::int
                        AS completed_job_count,
                    COALESCE(job_groups.succeeded_job_count, 0)::int
                        AS succeeded_job_count,
                    COALESCE(job_groups.failed_job_count, 0)::int
                        AS failed_job_count,
                    COALESCE(job_groups.skipped_job_count, 0)::int
                        AS skipped_job_count,
                    job_groups.average_job_duration_ms,
                    COALESCE(batch_groups.batch_run_count, 0)::int AS batch_run_count,
                    COALESCE(batch_groups.processed_count, 0)::int AS processed_count,
                    COALESCE(batch_groups.batch_succeeded_count, 0)::int
                        AS batch_succeeded_count,
                    COALESCE(batch_groups.batch_failed_count, 0)::int
                        AS batch_failed_count,
                    COALESCE(batch_groups.deferred_count, 0)::int AS deferred_count,
                    COALESCE(batch_groups.batch_elapsed_ms, 0)::int AS batch_elapsed_ms,
                    batch_groups.average_batch_elapsed_ms
                FROM embedding_profiles ep
                LEFT JOIN job_groups ON job_groups.profile_name = ep.profile_name
                LEFT JOIN batch_groups ON batch_groups.profile_name = ep.profile_name
                WHERE ep.is_active
                ORDER BY ep.profile_name ASC
                """,
                (validated_lookback, validated_lookback),
            )
            embedding_profile_rows = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                WITH recent_logs AS (
                    SELECT search_log_id, total_elapsed_ms, created_at
                    FROM search_logs
                    WHERE created_at >= now() - (%s::int * interval '1 hour')
                ),
                recent_results AS (
                    SELECT slr.profile_elapsed_ms
                    FROM search_log_results slr
                    JOIN recent_logs rl ON rl.search_log_id = slr.search_log_id
                )
                SELECT
                    (SELECT COUNT(*)::int FROM recent_logs) AS search_log_count,
                    (SELECT COUNT(*)::int FROM recent_results) AS result_count,
                    (SELECT AVG(total_elapsed_ms) FROM recent_logs)
                        AS average_total_elapsed_ms,
                    (SELECT AVG(profile_elapsed_ms) FROM recent_results)
                        AS average_profile_elapsed_ms,
                    (SELECT MAX(created_at) FROM recent_logs) AS latest_search_at
                """,
                (validated_lookback,),
            )
            search_total_row = dict(cursor.fetchone() or {})

            cursor.execute(
                """
                WITH recent_logs AS (
                    SELECT search_log_id
                    FROM search_logs
                    WHERE created_at >= now() - (%s::int * interval '1 hour')
                ),
                result_groups AS (
                    SELECT
                        slr.profile_name,
                        COUNT(DISTINCT slr.search_log_id)::int AS search_log_count,
                        COUNT(*)::int AS result_count,
                        AVG(slr.profile_elapsed_ms) AS average_profile_elapsed_ms
                    FROM search_log_results slr
                    JOIN recent_logs rl ON rl.search_log_id = slr.search_log_id
                    GROUP BY slr.profile_name
                )
                SELECT
                    ep.profile_name,
                    COALESCE(result_groups.search_log_count, 0)::int
                        AS search_log_count,
                    COALESCE(result_groups.result_count, 0)::int AS result_count,
                    result_groups.average_profile_elapsed_ms
                FROM embedding_profiles ep
                LEFT JOIN result_groups ON result_groups.profile_name = ep.profile_name
                WHERE ep.is_active
                ORDER BY ep.profile_name ASC
                """,
                (validated_lookback,),
            )
            search_profile_rows = [dict(row) for row in cursor.fetchall()]

    return DashboardThroughputLatencySnapshot(
        lookback_hours=validated_lookback,
        pipeline=_pipeline_throughput_from_rows(
            pipeline_total_row,
            pipeline_stage_rows,
        ),
        embedding=_embedding_throughput_from_rows(
            embedding_total_row,
            embedding_profile_rows,
        ),
        search=_search_latency_from_rows(search_total_row, search_profile_rows),
    )
