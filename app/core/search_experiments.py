"""Search experiment run repository helpers."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect
from app.core.search_logs import SEARCH_SCOPES, SIMILARITY_METRICS

SEARCH_EXPERIMENT_RUN_STATUSES = {"pending", "running", "succeeded", "failed", "canceled"}
SEARCH_EXPERIMENT_PROFILE_STATUSES = {
    "pending",
    "running",
    "succeeded",
    "failed",
    "skipped",
}


@dataclass(frozen=True)
class SearchExperimentRunInput:
    run_name: str
    query_text: str
    profile_names: tuple[str, ...]
    normalized_query_text: str | None = None
    actor_user_id: int | None = None
    requested_search_scope: str | None = None
    effective_search_scope: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    chunk_policy_name: str | None = None
    strategy_name: str = "vector_cosine"
    similarity_metric: str = "cosine"
    top_k: int = 5
    score_threshold: float | None = None
    status: str = "pending"
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    created_by: str | None = None
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class SearchExperimentRunRecord:
    experiment_run_id: int
    run_name: str
    query_text: str
    normalized_query_text: str | None
    actor_user_id: int | None
    requested_search_scope: str | None
    effective_search_scope: str | None
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    strategy_name: str
    similarity_metric: str
    top_k: int
    score_threshold: float | None
    profile_names: tuple[str, ...]
    status: str
    total_profile_count: int
    completed_profile_count: int
    result_count: int
    failure_count: int
    total_elapsed_ms: int | None
    runtime_metadata: dict[str, Any]
    error_message: str | None
    created_by: str | None
    created_by_user_id: int | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SearchExperimentProfileRunInput:
    experiment_run_id: int
    profile_name: str
    status: str = "pending"
    search_log_id: int | None = None
    result_count: int = 0
    top_score: float | None = None
    average_score: float | None = None
    elapsed_ms: int | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass(frozen=True)
class SearchExperimentProfileRunRecord:
    experiment_profile_run_id: int
    experiment_run_id: int
    profile_name: str
    search_log_id: int | None
    status: str
    result_count: int
    top_score: float | None
    average_score: float | None
    elapsed_ms: int | None
    runtime_metadata: dict[str, Any]
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SearchExperimentRunDetail:
    run: SearchExperimentRunRecord
    profiles: tuple[SearchExperimentProfileRunRecord, ...]


class InvalidSearchExperimentError(ValueError):
    """Raised when search experiment inputs are invalid before reaching the DB."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidSearchExperimentError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidSearchExperimentError(f"{field_name} must not be blank")
    return normalized


def _validate_scope(scope: str | None, field_name: str) -> str | None:
    normalized = _validate_nonblank(scope, field_name)
    if normalized is not None and normalized not in SEARCH_SCOPES:
        raise InvalidSearchExperimentError(f"Unsupported {field_name}: {normalized}")
    return normalized


def _validate_similarity_metric(similarity_metric: str) -> str:
    normalized = similarity_metric.strip()
    if normalized not in SIMILARITY_METRICS:
        raise InvalidSearchExperimentError(f"Unsupported similarity_metric: {similarity_metric}")
    return normalized


def _validate_run_status(status: str) -> str:
    normalized = status.strip()
    if normalized not in SEARCH_EXPERIMENT_RUN_STATUSES:
        raise InvalidSearchExperimentError(f"Unsupported experiment status: {status}")
    return normalized


def _validate_profile_status(status: str) -> str:
    normalized = status.strip()
    if normalized not in SEARCH_EXPERIMENT_PROFILE_STATUSES:
        raise InvalidSearchExperimentError(f"Unsupported profile status: {status}")
    return normalized


def _validate_limit(limit: int, *, max_limit: int = 500) -> int:
    if limit <= 0:
        raise InvalidSearchExperimentError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidSearchExperimentError(f"limit must be less than or equal to {max_limit}")
    return limit


def _validate_metadata(metadata: dict[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise InvalidSearchExperimentError(f"{field_name} must be a JSON object")
    return dict(metadata)


def _validate_optional_nonnegative_int(value: int | None, field_name: str) -> int | None:
    if value is not None and value < 0:
        raise InvalidSearchExperimentError(f"{field_name} must be greater than or equal to 0")
    return value


def _validate_optional_finite_float(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise InvalidSearchExperimentError(f"{field_name} must be finite")
    return parsed


def _validate_profile_names(profile_names: tuple[str, ...]) -> tuple[str, ...]:
    if not profile_names:
        raise InvalidSearchExperimentError("profile_names must not be empty")
    normalized: list[str] = []
    seen: set[str] = set()
    for profile_name in profile_names:
        profile = _validate_nonblank(profile_name, "profile_name")
        if profile is None:
            raise InvalidSearchExperimentError("profile_name must not be blank")
        if profile not in seen:
            normalized.append(profile)
            seen.add(profile)
    return tuple(normalized)


def validate_search_experiment_run_input(
    run_input: SearchExperimentRunInput,
) -> SearchExperimentRunInput:
    _require_positive_id(run_input.actor_user_id, "actor_user_id")
    _require_positive_id(run_input.created_by_user_id, "created_by_user_id")
    if run_input.top_k <= 0:
        raise InvalidSearchExperimentError("top_k must be greater than 0")
    return SearchExperimentRunInput(
        run_name=_validate_nonblank(run_input.run_name, "run_name") or run_input.run_name,
        query_text=_validate_nonblank(run_input.query_text, "query_text") or run_input.query_text,
        profile_names=_validate_profile_names(run_input.profile_names),
        normalized_query_text=_validate_nonblank(
            run_input.normalized_query_text,
            "normalized_query_text",
        ),
        actor_user_id=run_input.actor_user_id,
        requested_search_scope=_validate_scope(
            run_input.requested_search_scope,
            "requested_search_scope",
        ),
        effective_search_scope=_validate_scope(
            run_input.effective_search_scope,
            "effective_search_scope",
        ),
        document_group=_validate_nonblank(run_input.document_group, "document_group"),
        file_type=_validate_nonblank(run_input.file_type, "file_type"),
        chunk_policy_name=_validate_nonblank(run_input.chunk_policy_name, "chunk_policy_name"),
        strategy_name=_validate_nonblank(run_input.strategy_name, "strategy_name")
        or run_input.strategy_name,
        similarity_metric=_validate_similarity_metric(run_input.similarity_metric),
        top_k=run_input.top_k,
        score_threshold=_validate_optional_finite_float(
            run_input.score_threshold,
            "score_threshold",
        ),
        status=_validate_run_status(run_input.status),
        runtime_metadata=_validate_metadata(run_input.runtime_metadata, "runtime_metadata"),
        created_by=_validate_nonblank(run_input.created_by, "created_by"),
        created_by_user_id=run_input.created_by_user_id,
    )


def validate_search_experiment_profile_run_input(
    profile_input: SearchExperimentProfileRunInput,
) -> SearchExperimentProfileRunInput:
    _require_positive_id(profile_input.experiment_run_id, "experiment_run_id")
    _require_positive_id(profile_input.search_log_id, "search_log_id")
    return SearchExperimentProfileRunInput(
        experiment_run_id=profile_input.experiment_run_id,
        profile_name=_validate_nonblank(profile_input.profile_name, "profile_name")
        or profile_input.profile_name,
        status=_validate_profile_status(profile_input.status),
        search_log_id=profile_input.search_log_id,
        result_count=_validate_optional_nonnegative_int(
            profile_input.result_count,
            "result_count",
        )
        or 0,
        top_score=_validate_optional_finite_float(profile_input.top_score, "top_score"),
        average_score=_validate_optional_finite_float(
            profile_input.average_score,
            "average_score",
        ),
        elapsed_ms=_validate_optional_nonnegative_int(profile_input.elapsed_ms, "elapsed_ms"),
        runtime_metadata=_validate_metadata(profile_input.runtime_metadata, "runtime_metadata"),
        error_message=_validate_nonblank(profile_input.error_message, "error_message"),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _row_to_run_record(row: dict[str, Any]) -> SearchExperimentRunRecord:
    return SearchExperimentRunRecord(
        experiment_run_id=int(row["experiment_run_id"]),
        run_name=str(row["run_name"]),
        query_text=str(row["query_text"]),
        normalized_query_text=row["normalized_query_text"],
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        requested_search_scope=row["requested_search_scope"],
        effective_search_scope=row["effective_search_scope"],
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        strategy_name=str(row["strategy_name"]),
        similarity_metric=str(row["similarity_metric"]),
        top_k=int(row["top_k"]),
        score_threshold=_optional_float(row["score_threshold"]),
        profile_names=tuple(row["profile_names"] or ()),
        status=str(row["status"]),
        total_profile_count=int(row["total_profile_count"]),
        completed_profile_count=int(row["completed_profile_count"]),
        result_count=int(row["result_count"]),
        failure_count=int(row["failure_count"]),
        total_elapsed_ms=(
            int(row["total_elapsed_ms"]) if row["total_elapsed_ms"] is not None else None
        ),
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        error_message=row["error_message"],
        created_by=row["created_by"],
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_profile_run_record(row: dict[str, Any]) -> SearchExperimentProfileRunRecord:
    return SearchExperimentProfileRunRecord(
        experiment_profile_run_id=int(row["experiment_profile_run_id"]),
        experiment_run_id=int(row["experiment_run_id"]),
        profile_name=str(row["profile_name"]),
        search_log_id=int(row["search_log_id"]) if row["search_log_id"] is not None else None,
        status=str(row["status"]),
        result_count=int(row["result_count"]),
        top_score=_optional_float(row["top_score"]),
        average_score=_optional_float(row["average_score"]),
        elapsed_ms=int(row["elapsed_ms"]) if row["elapsed_ms"] is not None else None,
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        error_message=row["error_message"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_search_experiment_run(
    database_url: str,
    run_input: SearchExperimentRunInput,
) -> SearchExperimentRunRecord:
    validated = validate_search_experiment_run_input(run_input)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO search_experiment_runs (
                    run_name,
                    query_text,
                    normalized_query_text,
                    actor_user_id,
                    requested_search_scope,
                    effective_search_scope,
                    document_group,
                    file_type,
                    chunk_policy_name,
                    strategy_name,
                    similarity_metric,
                    top_k,
                    score_threshold,
                    profile_names,
                    status,
                    total_profile_count,
                    runtime_metadata,
                    created_by,
                    created_by_user_id,
                    started_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s = 'running' THEN now() ELSE NULL END
                )
                RETURNING *
                """,
                (
                    validated.run_name,
                    validated.query_text,
                    validated.normalized_query_text,
                    validated.actor_user_id,
                    validated.requested_search_scope,
                    validated.effective_search_scope,
                    validated.document_group,
                    validated.file_type,
                    validated.chunk_policy_name,
                    validated.strategy_name,
                    validated.similarity_metric,
                    validated.top_k,
                    validated.score_threshold,
                    Json(list(validated.profile_names)),
                    validated.status,
                    len(validated.profile_names),
                    Json(validated.runtime_metadata),
                    validated.created_by,
                    validated.created_by_user_id,
                    validated.status,
                ),
            )
            return _row_to_run_record(dict(cursor.fetchone()))


def get_search_experiment_run(
    database_url: str,
    experiment_run_id: int,
) -> SearchExperimentRunRecord | None:
    _require_positive_id(experiment_run_id, "experiment_run_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM search_experiment_runs
                WHERE experiment_run_id = %s
                """,
                (experiment_run_id,),
            )
            row = cursor.fetchone()
    return _row_to_run_record(dict(row)) if row else None


def list_search_experiment_runs(
    database_url: str,
    *,
    status: str | None = None,
    strategy_name: str | None = None,
    limit: int = 100,
) -> list[SearchExperimentRunRecord]:
    validated_limit = _validate_limit(limit)
    validated_status = _validate_run_status(status) if status is not None else None
    validated_strategy = _validate_nonblank(strategy_name, "strategy_name")

    filters: list[str] = []
    params: list[object] = []
    if validated_status is not None:
        filters.append("status = %s")
        params.append(validated_status)
    if validated_strategy is not None:
        filters.append("strategy_name = %s")
        params.append(validated_strategy)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(validated_limit)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM search_experiment_runs
                {where_clause}
                ORDER BY created_at DESC, experiment_run_id DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
    return [_row_to_run_record(dict(row)) for row in rows]


def upsert_search_experiment_profile_run(
    database_url: str,
    profile_input: SearchExperimentProfileRunInput,
) -> SearchExperimentProfileRunRecord:
    validated = validate_search_experiment_profile_run_input(profile_input)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO search_experiment_profile_runs (
                    experiment_run_id,
                    profile_name,
                    search_log_id,
                    status,
                    result_count,
                    top_score,
                    average_score,
                    elapsed_ms,
                    runtime_metadata,
                    error_message,
                    started_at,
                    finished_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s = 'running' THEN now() ELSE NULL END,
                    CASE
                        WHEN %s IN ('succeeded', 'failed', 'skipped') THEN now()
                        ELSE NULL
                    END
                )
                ON CONFLICT (experiment_run_id, profile_name) DO UPDATE
                SET search_log_id = EXCLUDED.search_log_id,
                    status = EXCLUDED.status,
                    result_count = EXCLUDED.result_count,
                    top_score = EXCLUDED.top_score,
                    average_score = EXCLUDED.average_score,
                    elapsed_ms = EXCLUDED.elapsed_ms,
                    runtime_metadata = EXCLUDED.runtime_metadata,
                    error_message = EXCLUDED.error_message,
                    started_at = COALESCE(
                        search_experiment_profile_runs.started_at,
                        EXCLUDED.started_at
                    ),
                    finished_at = EXCLUDED.finished_at,
                    updated_at = now()
                RETURNING *
                """,
                (
                    validated.experiment_run_id,
                    validated.profile_name,
                    validated.search_log_id,
                    validated.status,
                    validated.result_count,
                    validated.top_score,
                    validated.average_score,
                    validated.elapsed_ms,
                    Json(validated.runtime_metadata),
                    validated.error_message,
                    validated.status,
                    validated.status,
                ),
            )
            return _row_to_profile_run_record(dict(cursor.fetchone()))


def update_search_experiment_run_status(
    database_url: str,
    experiment_run_id: int,
    *,
    status: str,
    total_elapsed_ms: int | None = None,
    error_message: str | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> SearchExperimentRunRecord | None:
    _require_positive_id(experiment_run_id, "experiment_run_id")
    validated_status = _validate_run_status(status)
    validated_elapsed_ms = _validate_optional_nonnegative_int(
        total_elapsed_ms,
        "total_elapsed_ms",
    )
    validated_metadata = _validate_metadata(runtime_metadata or {}, "runtime_metadata")

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH profile_summary AS (
                    SELECT
                        experiment_run_id,
                        count(*) FILTER (
                            WHERE status IN ('succeeded', 'failed', 'skipped')
                        )::int AS completed_profile_count,
                        count(*) FILTER (WHERE status = 'failed')::int AS failure_count,
                        COALESCE(sum(result_count), 0)::int AS result_count
                    FROM search_experiment_profile_runs
                    WHERE experiment_run_id = %s
                    GROUP BY experiment_run_id
                )
                UPDATE search_experiment_runs ser
                SET status = %s,
                    completed_profile_count = COALESCE(
                        profile_summary.completed_profile_count,
                        ser.completed_profile_count
                    ),
                    failure_count = COALESCE(profile_summary.failure_count, ser.failure_count),
                    result_count = COALESCE(profile_summary.result_count, ser.result_count),
                    total_elapsed_ms = %s,
                    error_message = %s,
                    runtime_metadata = ser.runtime_metadata || %s::jsonb,
                    started_at = CASE
                        WHEN %s = 'running' THEN COALESCE(ser.started_at, now())
                        ELSE ser.started_at
                    END,
                    finished_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'canceled') THEN now()
                        ELSE ser.finished_at
                    END,
                    updated_at = now()
                FROM profile_summary
                WHERE ser.experiment_run_id = %s
                  AND profile_summary.experiment_run_id = ser.experiment_run_id
                RETURNING ser.*
                """,
                (
                    experiment_run_id,
                    validated_status,
                    validated_elapsed_ms,
                    _validate_nonblank(error_message, "error_message"),
                    Json(validated_metadata),
                    validated_status,
                    validated_status,
                    experiment_run_id,
                ),
            )
            row = cursor.fetchone()
            if row is not None:
                return _row_to_run_record(dict(row))

            cursor.execute(
                """
                UPDATE search_experiment_runs
                SET status = %s,
                    total_elapsed_ms = %s,
                    error_message = %s,
                    runtime_metadata = runtime_metadata || %s::jsonb,
                    started_at = CASE
                        WHEN %s = 'running' THEN COALESCE(started_at, now())
                        ELSE started_at
                    END,
                    finished_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'canceled') THEN now()
                        ELSE finished_at
                    END,
                    updated_at = now()
                WHERE experiment_run_id = %s
                RETURNING *
                """,
                (
                    validated_status,
                    validated_elapsed_ms,
                    _validate_nonblank(error_message, "error_message"),
                    Json(validated_metadata),
                    validated_status,
                    validated_status,
                    experiment_run_id,
                ),
            )
            fallback_row = cursor.fetchone()
    return _row_to_run_record(dict(fallback_row)) if fallback_row else None


def get_search_experiment_run_detail(
    database_url: str,
    experiment_run_id: int,
) -> SearchExperimentRunDetail | None:
    run = get_search_experiment_run(database_url, experiment_run_id)
    if run is None:
        return None
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM search_experiment_profile_runs
                WHERE experiment_run_id = %s
                ORDER BY profile_name ASC
                """,
                (experiment_run_id,),
            )
            rows = cursor.fetchall()
    return SearchExperimentRunDetail(
        run=run,
        profiles=tuple(_row_to_profile_run_record(dict(row)) for row in rows),
    )
