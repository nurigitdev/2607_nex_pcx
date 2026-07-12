"""Search experiment run repository helpers."""

import base64
import binascii
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect
from app.core.evaluation_metrics import (
    EvaluationSummaryRecord,
    ExpectedTarget,
    InvalidEvaluationMetricError,
    QuestionEvaluationInput,
    QuestionMetricRecord,
    RankedSearchResult,
    evaluate_question,
    summarize_question_metrics,
)
from app.core.golden_questions import list_expected_targets
from app.core.search_logs import SEARCH_SCOPES, SIMILARITY_METRICS, list_search_log_results
from app.core.search_strategies import (
    InvalidSearchStrategyError,
    validate_search_strategy_selection,
)

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


@dataclass(frozen=True)
class GoldenSearchExperimentBatchIdentity:
    question_set_id: int
    batch_prefix: str
    strategy_name: str
    top_k: int
    score_threshold: float | None
    chunk_policy_name: str | None
    profile_names: tuple[str, ...]


@dataclass(frozen=True)
class GoldenSearchExperimentBatchSummary:
    batch_key: str
    question_set_id: int
    question_set_name: str
    batch_prefix: str
    strategy_name: str
    top_k: int
    score_threshold: float | None
    chunk_policy_name: str | None
    profile_names: tuple[str, ...]
    status: str
    question_count: int
    succeeded_count: int
    failed_count: int
    running_count: int
    total_result_count: int
    average_result_count: float
    total_elapsed_ms: int
    average_elapsed_ms: float | None
    first_experiment_run_id: int
    last_experiment_run_id: int
    first_created_at: datetime
    last_updated_at: datetime


@dataclass(frozen=True)
class GoldenSearchExperimentBatchQuestionSummary:
    question_id: int | None
    question_text: str
    experiment_run: SearchExperimentRunRecord


@dataclass(frozen=True)
class GoldenSearchExperimentBatchDetail:
    summary: GoldenSearchExperimentBatchSummary
    questions: tuple[GoldenSearchExperimentBatchQuestionSummary, ...]


@dataclass(frozen=True)
class GoldenSearchExperimentBatchProfileMetricSummary:
    profile_name: str
    question_count: int
    recall_question_count: int
    ndcg_question_count: int
    no_answer_question_count: int
    hidden_violation_count: int
    mean_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg: float | None
    no_answer_success_rate: float | None
    total_result_count: int
    average_result_count: float | None
    average_elapsed_ms: float | None


@dataclass(frozen=True)
class GoldenSearchExperimentBatchQuestionMetricSummary:
    question_id: int
    question_text: str
    profile_name: str
    experiment_run_id: int
    search_log_id: int
    top_k: int
    result_count: int
    elapsed_ms: int | None
    metric: QuestionMetricRecord


@dataclass(frozen=True)
class GoldenSearchExperimentBatchMetricSummary:
    summary: GoldenSearchExperimentBatchSummary
    overall: EvaluationSummaryRecord
    profiles: tuple[GoldenSearchExperimentBatchProfileMetricSummary, ...]
    questions: tuple[GoldenSearchExperimentBatchQuestionMetricSummary, ...]


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
    strategy_name = _validate_nonblank(run_input.strategy_name, "strategy_name")
    try:
        strategy_selection = validate_search_strategy_selection(
            strategy_name or run_input.strategy_name,
            top_k=run_input.top_k,
            score_threshold=run_input.score_threshold,
        )
    except InvalidSearchStrategyError as exc:
        raise InvalidSearchExperimentError(str(exc)) from exc
    similarity_metric = _validate_similarity_metric(run_input.similarity_metric)
    if similarity_metric != strategy_selection.strategy.similarity_metric:
        raise InvalidSearchExperimentError(
            f"{strategy_selection.strategy.strategy_name} requires "
            f"{strategy_selection.strategy.similarity_metric} similarity_metric"
        )
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
        strategy_name=strategy_selection.strategy.strategy_name,
        similarity_metric=similarity_metric,
        top_k=strategy_selection.top_k,
        score_threshold=strategy_selection.score_threshold,
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


def _metadata_int(metadata: dict[str, Any], field_name: str) -> int | None:
    raw_value = metadata.get(field_name)
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _golden_question_set_id(run: SearchExperimentRunRecord) -> int | None:
    return _metadata_int(run.runtime_metadata, "question_set_id")


def _golden_question_id(run: SearchExperimentRunRecord) -> int | None:
    return _metadata_int(run.runtime_metadata, "question_id")


def _golden_batch_prefix(run: SearchExperimentRunRecord) -> str:
    question_id = _golden_question_id(run)
    if question_id is not None:
        suffix = f" / Q{question_id}"
        if run.run_name.endswith(suffix):
            return run.run_name[: -len(suffix)]
    if " / Q" in run.run_name:
        return run.run_name.rsplit(" / Q", 1)[0]
    return run.run_name


def _golden_batch_identity_from_run(
    run: SearchExperimentRunRecord,
) -> GoldenSearchExperimentBatchIdentity | None:
    question_set_id = _golden_question_set_id(run)
    if question_set_id is None:
        return None
    return GoldenSearchExperimentBatchIdentity(
        question_set_id=question_set_id,
        batch_prefix=_golden_batch_prefix(run),
        strategy_name=run.strategy_name,
        top_k=run.top_k,
        score_threshold=run.score_threshold,
        chunk_policy_name=run.chunk_policy_name,
        profile_names=run.profile_names,
    )


def _golden_batch_identity_payload(
    identity: GoldenSearchExperimentBatchIdentity,
) -> dict[str, object]:
    return {
        "question_set_id": identity.question_set_id,
        "batch_prefix": identity.batch_prefix,
        "strategy_name": identity.strategy_name,
        "top_k": identity.top_k,
        "score_threshold": identity.score_threshold,
        "chunk_policy_name": identity.chunk_policy_name,
        "profile_names": list(identity.profile_names),
    }


def encode_golden_search_experiment_batch_key(
    identity: GoldenSearchExperimentBatchIdentity,
) -> str:
    raw_payload = json.dumps(
        _golden_batch_identity_payload(identity),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw_payload).decode("ascii").rstrip("=")


def golden_search_experiment_batch_key_from_run(
    run: SearchExperimentRunRecord,
) -> str | None:
    identity = _golden_batch_identity_from_run(run)
    if identity is None:
        return None
    return encode_golden_search_experiment_batch_key(identity)


def decode_golden_search_experiment_batch_key(
    batch_key: str,
) -> GoldenSearchExperimentBatchIdentity:
    normalized_key = _validate_nonblank(batch_key, "batch_key")
    if normalized_key is None:
        raise InvalidSearchExperimentError("batch_key must not be blank")
    padding = "=" * (-len(normalized_key) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(f"{normalized_key}{padding}").decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise InvalidSearchExperimentError("Invalid golden search experiment batch key") from exc
    if not isinstance(payload, dict):
        raise InvalidSearchExperimentError("Invalid golden search experiment batch key")

    raw_profiles = payload.get("profile_names")
    if not isinstance(raw_profiles, list):
        raise InvalidSearchExperimentError("Invalid golden search experiment batch key")
    try:
        question_set_id = int(payload["question_set_id"])
        top_k = int(payload["top_k"])
        score_threshold = payload.get("score_threshold")
        parsed_threshold = (
            _validate_optional_finite_float(score_threshold, "score_threshold")
            if score_threshold is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidSearchExperimentError("Invalid golden search experiment batch key") from exc

    _require_positive_id(question_set_id, "question_set_id")
    if top_k <= 0:
        raise InvalidSearchExperimentError("Invalid golden search experiment batch key")
    return GoldenSearchExperimentBatchIdentity(
        question_set_id=question_set_id,
        batch_prefix=_validate_nonblank(
            str(payload.get("batch_prefix", "")),
            "batch_prefix",
        )
        or "",
        strategy_name=_validate_nonblank(
            str(payload.get("strategy_name", "")),
            "strategy_name",
        )
        or "",
        top_k=top_k,
        score_threshold=parsed_threshold,
        chunk_policy_name=_validate_nonblank(
            payload.get("chunk_policy_name"),
            "chunk_policy_name",
        ),
        profile_names=_validate_profile_names(tuple(str(profile) for profile in raw_profiles)),
    )


def _summary_status(runs: list[SearchExperimentRunRecord]) -> str:
    statuses = {run.status for run in runs}
    if statuses == {"succeeded"}:
        return "succeeded"
    if "failed" in statuses:
        return "failed"
    if statuses & {"running", "pending"}:
        return "running"
    if "canceled" in statuses:
        return "canceled"
    return "failed"


def _batch_summary_from_runs(
    identity: GoldenSearchExperimentBatchIdentity,
    runs: list[SearchExperimentRunRecord],
) -> GoldenSearchExperimentBatchSummary:
    ordered_runs = sorted(runs, key=lambda run: run.experiment_run_id)
    elapsed_values = [
        run.total_elapsed_ms for run in ordered_runs if run.total_elapsed_ms is not None
    ]
    total_elapsed_ms = sum(elapsed_values)
    question_set_name = ""
    for run in ordered_runs:
        raw_name = run.runtime_metadata.get("question_set_name")
        if isinstance(raw_name, str) and raw_name:
            question_set_name = raw_name
            break
    return GoldenSearchExperimentBatchSummary(
        batch_key=encode_golden_search_experiment_batch_key(identity),
        question_set_id=identity.question_set_id,
        question_set_name=question_set_name,
        batch_prefix=identity.batch_prefix,
        strategy_name=identity.strategy_name,
        top_k=identity.top_k,
        score_threshold=identity.score_threshold,
        chunk_policy_name=identity.chunk_policy_name,
        profile_names=identity.profile_names,
        status=_summary_status(ordered_runs),
        question_count=len(ordered_runs),
        succeeded_count=sum(1 for run in ordered_runs if run.status == "succeeded"),
        failed_count=sum(1 for run in ordered_runs if run.status == "failed"),
        running_count=sum(1 for run in ordered_runs if run.status in {"pending", "running"}),
        total_result_count=sum(run.result_count for run in ordered_runs),
        average_result_count=sum(run.result_count for run in ordered_runs) / len(ordered_runs),
        total_elapsed_ms=total_elapsed_ms,
        average_elapsed_ms=(total_elapsed_ms / len(elapsed_values) if elapsed_values else None),
        first_experiment_run_id=ordered_runs[0].experiment_run_id,
        last_experiment_run_id=ordered_runs[-1].experiment_run_id,
        first_created_at=min(run.created_at for run in ordered_runs),
        last_updated_at=max(run.updated_at for run in ordered_runs),
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


def list_golden_search_experiment_batch_summaries(
    database_url: str,
    *,
    question_set_id: int | None = None,
    limit: int = 20,
) -> list[GoldenSearchExperimentBatchSummary]:
    validated_limit = _validate_limit(limit, max_limit=100)
    _require_positive_id(question_set_id, "question_set_id")
    scan_limit = max(validated_limit * 100, 500)

    filters = ["runtime_metadata->>'golden_question_batch' = 'true'"]
    params: list[object] = []
    if question_set_id is not None:
        filters.append("runtime_metadata->>'question_set_id' = %s")
        params.append(str(question_set_id))
    params.append(scan_limit)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM search_experiment_runs
                WHERE {" AND ".join(filters)}
                ORDER BY created_at DESC, experiment_run_id DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()

    groups: dict[GoldenSearchExperimentBatchIdentity, list[SearchExperimentRunRecord]] = {}
    for row in rows:
        run = _row_to_run_record(dict(row))
        identity = _golden_batch_identity_from_run(run)
        if identity is None:
            continue
        groups.setdefault(identity, []).append(run)

    summaries = [
        _batch_summary_from_runs(identity, runs) for identity, runs in groups.items() if runs
    ]
    summaries.sort(
        key=lambda summary: (summary.last_updated_at, summary.last_experiment_run_id),
        reverse=True,
    )
    return summaries[:validated_limit]


def get_golden_search_experiment_batch_detail(
    database_url: str,
    batch_key: str,
) -> GoldenSearchExperimentBatchDetail | None:
    identity = decode_golden_search_experiment_batch_key(batch_key)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM search_experiment_runs
                WHERE runtime_metadata->>'golden_question_batch' = 'true'
                  AND runtime_metadata->>'question_set_id' = %s
                  AND regexp_replace(run_name, '\\s*/\\s*Q[0-9]+$', '') = %s
                  AND strategy_name = %s
                  AND top_k = %s
                  AND (
                    (%s::double precision IS NULL AND score_threshold IS NULL)
                    OR score_threshold = %s::double precision
                  )
                  AND (
                    (%s::text IS NULL AND chunk_policy_name IS NULL)
                    OR chunk_policy_name = %s::text
                  )
                  AND profile_names = %s::jsonb
                ORDER BY runtime_metadata->>'question_id' ASC, experiment_run_id ASC
                """,
                (
                    str(identity.question_set_id),
                    identity.batch_prefix,
                    identity.strategy_name,
                    identity.top_k,
                    identity.score_threshold,
                    identity.score_threshold,
                    identity.chunk_policy_name,
                    identity.chunk_policy_name,
                    Json(list(identity.profile_names)),
                ),
            )
            rows = cursor.fetchall()

    runs = [_row_to_run_record(dict(row)) for row in rows]
    if not runs:
        return None
    summary = _batch_summary_from_runs(identity, runs)
    questions = tuple(
        GoldenSearchExperimentBatchQuestionSummary(
            question_id=_golden_question_id(run),
            question_text=run.query_text,
            experiment_run=run,
        )
        for run in sorted(
            runs,
            key=lambda item: (_golden_question_id(item) or 0, item.experiment_run_id),
        )
    )
    return GoldenSearchExperimentBatchDetail(summary=summary, questions=questions)


def _expected_targets_for_question(
    database_url: str,
    question_id: int,
) -> tuple[ExpectedTarget, ...]:
    return tuple(
        ExpectedTarget(
            chunk_id=target.chunk_id,
            expected_heading_path=target.expected_heading_path,
            expectation_type=target.expectation_type,
            relevance_grade=target.relevance_grade,
        )
        for target in list_expected_targets(database_url, question_id)
    )


def _ranked_results_for_profile(
    database_url: str,
    search_log_id: int,
    profile_name: str,
    top_k: int,
    score_threshold: float | None,
    result_cache: dict[int, tuple[Any, ...]],
) -> tuple[RankedSearchResult, ...]:
    if search_log_id not in result_cache:
        result_cache[search_log_id] = tuple(list_search_log_results(database_url, search_log_id))
    filtered_results = (
        result
        for result in result_cache[search_log_id]
        if result.profile_name == profile_name
        and result.rank <= top_k
        and (
            score_threshold is None
            or (result.score is not None and result.score >= score_threshold)
        )
    )
    return tuple(
        RankedSearchResult(
            rank=result.rank,
            chunk_id=result.chunk_id,
            score=result.score,
        )
        for result in filtered_results
    )


def _profile_metric_summary(
    profile_name: str,
    metrics: tuple[GoldenSearchExperimentBatchQuestionMetricSummary, ...],
) -> GoldenSearchExperimentBatchProfileMetricSummary:
    summary = summarize_question_metrics(tuple(item.metric for item in metrics))
    elapsed_values = [item.elapsed_ms for item in metrics if item.elapsed_ms is not None]
    result_counts = [item.result_count for item in metrics]
    return GoldenSearchExperimentBatchProfileMetricSummary(
        profile_name=profile_name,
        question_count=summary.question_count,
        recall_question_count=summary.recall_question_count,
        ndcg_question_count=summary.ndcg_question_count,
        no_answer_question_count=summary.no_answer_question_count,
        hidden_violation_count=summary.hidden_violation_count,
        mean_recall_at_k=summary.mean_recall_at_k,
        mean_reciprocal_rank=summary.mean_reciprocal_rank,
        mean_ndcg=summary.mean_ndcg,
        no_answer_success_rate=summary.no_answer_success_rate,
        total_result_count=sum(result_counts),
        average_result_count=(sum(result_counts) / len(result_counts) if result_counts else None),
        average_elapsed_ms=(sum(elapsed_values) / len(elapsed_values) if elapsed_values else None),
    )


def get_golden_search_experiment_batch_metric_summary(
    database_url: str,
    batch_key: str,
) -> GoldenSearchExperimentBatchMetricSummary | None:
    detail = get_golden_search_experiment_batch_detail(database_url, batch_key)
    if detail is None:
        return None

    expected_cache: dict[int, tuple[ExpectedTarget, ...]] = {}
    result_cache: dict[int, tuple[Any, ...]] = {}
    question_metrics: list[GoldenSearchExperimentBatchQuestionMetricSummary] = []
    for question in detail.questions:
        if question.question_id is None:
            continue
        expected_targets = expected_cache.setdefault(
            question.question_id,
            _expected_targets_for_question(database_url, question.question_id),
        )
        run_detail = get_search_experiment_run_detail(
            database_url,
            question.experiment_run.experiment_run_id,
        )
        if run_detail is None:
            continue
        for profile_run in run_detail.profiles:
            if profile_run.search_log_id is None:
                continue
            try:
                metric = evaluate_question(
                    QuestionEvaluationInput(
                        question_id=question.question_id,
                        top_k=question.experiment_run.top_k,
                        expected_targets=expected_targets,
                        ranked_results=_ranked_results_for_profile(
                            database_url,
                            profile_run.search_log_id,
                            profile_run.profile_name,
                            question.experiment_run.top_k,
                            question.experiment_run.score_threshold,
                            result_cache,
                        ),
                    )
                )
            except InvalidEvaluationMetricError as exc:
                raise InvalidSearchExperimentError(str(exc)) from exc
            question_metrics.append(
                GoldenSearchExperimentBatchQuestionMetricSummary(
                    question_id=question.question_id,
                    question_text=question.question_text,
                    profile_name=profile_run.profile_name,
                    experiment_run_id=question.experiment_run.experiment_run_id,
                    search_log_id=profile_run.search_log_id,
                    top_k=question.experiment_run.top_k,
                    result_count=profile_run.result_count,
                    elapsed_ms=profile_run.elapsed_ms,
                    metric=metric,
                )
            )

    grouped: dict[str, list[GoldenSearchExperimentBatchQuestionMetricSummary]] = {}
    for question_metric in question_metrics:
        grouped.setdefault(question_metric.profile_name, []).append(question_metric)

    return GoldenSearchExperimentBatchMetricSummary(
        summary=detail.summary,
        overall=summarize_question_metrics(tuple(item.metric for item in question_metrics)),
        profiles=tuple(
            _profile_metric_summary(profile_name, tuple(items))
            for profile_name, items in sorted(grouped.items())
        ),
        questions=tuple(
            sorted(
                question_metrics,
                key=lambda item: (item.question_id, item.profile_name, item.experiment_run_id),
            )
        ),
    )


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
