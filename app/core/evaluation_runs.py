"""Golden evaluation run repository and runner helpers."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect
from app.core.evaluation_metrics import (
    EvaluationSummaryRecord,
    ExpectedTarget,
    QuestionEvaluationInput,
    QuestionMetricRecord,
    RankedSearchResult,
    evaluate_question,
    summarize_question_metrics,
)

EVALUATION_RUN_STATUSES = {"pending", "running", "succeeded", "failed"}
SIMILARITY_METRICS = {"cosine", "l2", "inner_product"}


@dataclass(frozen=True)
class EvaluationRunInput:
    question_set_id: int
    run_name: str
    profile_name: str
    chunk_policy_name: str | None = None
    similarity_metric: str = "cosine"
    top_k: int = 5
    status: str = "pending"
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRunRecord:
    evaluation_run_id: int
    question_set_id: int
    run_name: str
    profile_name: str
    chunk_policy_name: str | None
    similarity_metric: str
    top_k: int
    status: str
    question_count: int
    recall_question_count: int
    ndcg_question_count: int
    no_answer_question_count: int
    hidden_violation_count: int
    mean_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg: float | None
    no_answer_success_rate: float | None
    runtime_metadata: dict[str, Any]
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class EvaluationResultInput:
    evaluation_run_id: int
    metric: QuestionMetricRecord
    search_log_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResultRecord:
    evaluation_result_id: int
    evaluation_run_id: int
    question_id: int
    search_log_id: int | None
    top_k: int
    visible_expected_count: int
    retrieved_count: int
    matched_visible_count: int
    hidden_violation_count: int
    matched_chunk_ids: tuple[int, ...]
    hidden_violation_chunk_ids: tuple[int, ...]
    recall_at_k: float | None
    reciprocal_rank: float | None
    dcg: float
    ideal_dcg: float
    ndcg: float | None
    no_answer_success: bool | None
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class EvaluationRunReport:
    run: EvaluationRunRecord
    results: tuple[EvaluationResultRecord, ...]
    summary: EvaluationSummaryRecord


class InvalidEvaluationRunError(ValueError):
    """Raised when evaluation run inputs are invalid before reaching the DB."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidEvaluationRunError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidEvaluationRunError(f"{field_name} must not be blank")
    return normalized


def _validate_metadata(metadata: dict[str, Any], field_name: str = "metadata") -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise InvalidEvaluationRunError(f"{field_name} must be a JSON object")
    return dict(metadata)


def _validate_similarity_metric(similarity_metric: str) -> str:
    metric = similarity_metric.strip()
    if metric not in SIMILARITY_METRICS:
        raise InvalidEvaluationRunError(f"Unsupported similarity_metric: {similarity_metric}")
    return metric


def _validate_status(status: str) -> str:
    normalized = status.strip()
    if normalized not in EVALUATION_RUN_STATUSES:
        raise InvalidEvaluationRunError(f"Unsupported evaluation run status: {status}")
    return normalized


def _validate_limit(limit: int, *, max_limit: int = 500) -> int:
    if limit <= 0:
        raise InvalidEvaluationRunError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidEvaluationRunError(f"limit must be less than or equal to {max_limit}")
    return limit


def validate_evaluation_run_input(run_input: EvaluationRunInput) -> EvaluationRunInput:
    _require_positive_id(run_input.question_set_id, "question_set_id")
    if run_input.top_k <= 0:
        raise InvalidEvaluationRunError("top_k must be greater than 0")
    return EvaluationRunInput(
        question_set_id=run_input.question_set_id,
        run_name=_validate_nonblank(run_input.run_name, "run_name") or run_input.run_name,
        profile_name=_validate_nonblank(run_input.profile_name, "profile_name")
        or run_input.profile_name,
        chunk_policy_name=_validate_nonblank(run_input.chunk_policy_name, "chunk_policy_name"),
        similarity_metric=_validate_similarity_metric(run_input.similarity_metric),
        top_k=run_input.top_k,
        status=_validate_status(run_input.status),
        runtime_metadata=_validate_metadata(run_input.runtime_metadata, "runtime_metadata"),
    )


def validate_evaluation_result_input(result_input: EvaluationResultInput) -> EvaluationResultInput:
    _require_positive_id(result_input.evaluation_run_id, "evaluation_run_id")
    _require_positive_id(result_input.search_log_id, "search_log_id")
    return EvaluationResultInput(
        evaluation_run_id=result_input.evaluation_run_id,
        metric=result_input.metric,
        search_log_id=result_input.search_log_id,
        metadata=_validate_metadata(result_input.metadata),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _row_to_run_record(row: dict[str, Any]) -> EvaluationRunRecord:
    return EvaluationRunRecord(
        evaluation_run_id=int(row["evaluation_run_id"]),
        question_set_id=int(row["question_set_id"]),
        run_name=str(row["run_name"]),
        profile_name=str(row["profile_name"]),
        chunk_policy_name=row["chunk_policy_name"],
        similarity_metric=str(row["similarity_metric"]),
        top_k=int(row["top_k"]),
        status=str(row["status"]),
        question_count=int(row["question_count"]),
        recall_question_count=int(row["recall_question_count"]),
        ndcg_question_count=int(row["ndcg_question_count"]),
        no_answer_question_count=int(row["no_answer_question_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        mean_recall_at_k=_optional_float(row["mean_recall_at_k"]),
        mean_reciprocal_rank=_optional_float(row["mean_reciprocal_rank"]),
        mean_ndcg=_optional_float(row["mean_ndcg"]),
        no_answer_success_rate=_optional_float(row["no_answer_success_rate"]),
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        error_message=row["error_message"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_result_record(row: dict[str, Any]) -> EvaluationResultRecord:
    return EvaluationResultRecord(
        evaluation_result_id=int(row["evaluation_result_id"]),
        evaluation_run_id=int(row["evaluation_run_id"]),
        question_id=int(row["question_id"]),
        search_log_id=int(row["search_log_id"]) if row["search_log_id"] is not None else None,
        top_k=int(row["top_k"]),
        visible_expected_count=int(row["visible_expected_count"]),
        retrieved_count=int(row["retrieved_count"]),
        matched_visible_count=int(row["matched_visible_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        matched_chunk_ids=tuple(row["matched_chunk_ids"] or ()),
        hidden_violation_chunk_ids=tuple(row["hidden_violation_chunk_ids"] or ()),
        recall_at_k=_optional_float(row["recall_at_k"]),
        reciprocal_rank=_optional_float(row["reciprocal_rank"]),
        dcg=float(row["dcg"]),
        ideal_dcg=float(row["ideal_dcg"]),
        ndcg=_optional_float(row["ndcg"]),
        no_answer_success=row["no_answer_success"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def create_evaluation_run_in_connection(
    connection: Connection,
    run_input: EvaluationRunInput,
) -> EvaluationRunRecord:
    validated = validate_evaluation_run_input(run_input)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO golden_evaluation_runs (
                question_set_id,
                run_name,
                profile_name,
                chunk_policy_name,
                similarity_metric,
                top_k,
                status,
                runtime_metadata,
                started_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                CASE WHEN %s = 'running' THEN now() ELSE NULL END
            )
            RETURNING *
            """,
            (
                validated.question_set_id,
                validated.run_name,
                validated.profile_name,
                validated.chunk_policy_name,
                validated.similarity_metric,
                validated.top_k,
                validated.status,
                Json(validated.runtime_metadata),
                validated.status,
            ),
        )
        return _row_to_run_record(dict(cursor.fetchone()))


def create_evaluation_run(
    database_url: str,
    run_input: EvaluationRunInput,
) -> EvaluationRunRecord:
    with connect(database_url) as connection:
        return create_evaluation_run_in_connection(connection, run_input)


def get_evaluation_run(database_url: str, evaluation_run_id: int) -> EvaluationRunRecord | None:
    _require_positive_id(evaluation_run_id, "evaluation_run_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM golden_evaluation_runs WHERE evaluation_run_id = %s",
                (evaluation_run_id,),
            )
            row = cursor.fetchone()
    return _row_to_run_record(dict(row)) if row else None


def list_evaluation_runs(
    database_url: str,
    *,
    question_set_id: int | None = None,
    profile_name: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[EvaluationRunRecord]:
    _require_positive_id(question_set_id, "question_set_id")
    validated_limit = _validate_limit(limit)
    filters: list[str] = []
    params: list[object] = []
    if question_set_id is not None:
        filters.append("question_set_id = %s")
        params.append(question_set_id)
    if profile_name is not None:
        filters.append("profile_name = %s")
        params.append(_validate_nonblank(profile_name, "profile_name") or profile_name)
    if status is not None:
        filters.append("status = %s")
        params.append(_validate_status(status))
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM golden_evaluation_runs
                {where_clause}
                ORDER BY created_at DESC, evaluation_run_id DESC
                LIMIT %s
                """,
                [*params, validated_limit],
            )
            rows = cursor.fetchall()
    return [_row_to_run_record(dict(row)) for row in rows]


def create_evaluation_result_in_connection(
    connection: Connection,
    result_input: EvaluationResultInput,
) -> EvaluationResultRecord:
    validated = validate_evaluation_result_input(result_input)
    metric = validated.metric
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO golden_evaluation_results (
                evaluation_run_id,
                question_id,
                search_log_id,
                top_k,
                visible_expected_count,
                retrieved_count,
                matched_visible_count,
                hidden_violation_count,
                matched_chunk_ids,
                hidden_violation_chunk_ids,
                recall_at_k,
                reciprocal_rank,
                dcg,
                ideal_dcg,
                ndcg,
                no_answer_success,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING *
            """,
            (
                validated.evaluation_run_id,
                metric.question_id,
                validated.search_log_id,
                metric.top_k,
                metric.visible_expected_count,
                metric.retrieved_count,
                metric.matched_visible_count,
                metric.hidden_violation_count,
                list(metric.matched_chunk_ids),
                list(metric.hidden_violation_chunk_ids),
                metric.recall_at_k,
                metric.reciprocal_rank,
                metric.dcg,
                metric.ideal_dcg,
                metric.ndcg,
                metric.no_answer_success,
                Json(validated.metadata),
            ),
        )
        return _row_to_result_record(dict(cursor.fetchone()))


def create_evaluation_result(
    database_url: str,
    result_input: EvaluationResultInput,
) -> EvaluationResultRecord:
    with connect(database_url) as connection:
        return create_evaluation_result_in_connection(connection, result_input)


def list_evaluation_results(
    database_url: str,
    evaluation_run_id: int,
) -> list[EvaluationResultRecord]:
    _require_positive_id(evaluation_run_id, "evaluation_run_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM golden_evaluation_results
                WHERE evaluation_run_id = %s
                ORDER BY question_id ASC
                """,
                (evaluation_run_id,),
            )
            rows = cursor.fetchall()
    return [_row_to_result_record(dict(row)) for row in rows]


def complete_evaluation_run_in_connection(
    connection: Connection,
    evaluation_run_id: int,
    summary: EvaluationSummaryRecord,
) -> EvaluationRunRecord:
    _require_positive_id(evaluation_run_id, "evaluation_run_id")
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE golden_evaluation_runs
            SET status = 'succeeded',
                question_count = %s,
                recall_question_count = %s,
                ndcg_question_count = %s,
                no_answer_question_count = %s,
                hidden_violation_count = %s,
                mean_recall_at_k = %s,
                mean_reciprocal_rank = %s,
                mean_ndcg = %s,
                no_answer_success_rate = %s,
                error_message = NULL,
                finished_at = now(),
                updated_at = now()
            WHERE evaluation_run_id = %s
            RETURNING *
            """,
            (
                summary.question_count,
                summary.recall_question_count,
                summary.ndcg_question_count,
                summary.no_answer_question_count,
                summary.hidden_violation_count,
                summary.mean_recall_at_k,
                summary.mean_reciprocal_rank,
                summary.mean_ndcg,
                summary.no_answer_success_rate,
                evaluation_run_id,
            ),
        )
        row = cursor.fetchone()
    if row is None:
        raise InvalidEvaluationRunError("evaluation_run_id was not found")
    return _row_to_run_record(dict(row))


def complete_evaluation_run(
    database_url: str,
    evaluation_run_id: int,
    summary: EvaluationSummaryRecord,
) -> EvaluationRunRecord:
    with connect(database_url) as connection:
        return complete_evaluation_run_in_connection(connection, evaluation_run_id, summary)


def _load_question_ids_in_connection(
    connection: Connection,
    question_set_id: int,
) -> tuple[int, ...]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT question_id
            FROM golden_questions
            WHERE question_set_id = %s
            ORDER BY question_id ASC
            """,
            (question_set_id,),
        )
        rows = cursor.fetchall()
    return tuple(int(row["question_id"]) for row in rows)


def _load_expected_targets_in_connection(
    connection: Connection,
    question_id: int,
) -> tuple[ExpectedTarget, ...]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT chunk_id, expected_heading_path, expectation_type, relevance_grade
            FROM golden_question_expected_targets
            WHERE question_id = %s
            ORDER BY expected_target_id ASC
            """,
            (question_id,),
        )
        rows = cursor.fetchall()
    return tuple(
        ExpectedTarget(
            chunk_id=int(row["chunk_id"]) if row["chunk_id"] is not None else None,
            expected_heading_path=tuple(row["expected_heading_path"] or ()),
            expectation_type=str(row["expectation_type"]),
            relevance_grade=int(row["relevance_grade"]),
        )
        for row in rows
    )


def run_golden_evaluation(
    database_url: str,
    run_input: EvaluationRunInput,
    *,
    ranked_results_by_question: dict[int, tuple[RankedSearchResult, ...]] | None = None,
    search_log_ids_by_question: dict[int, int] | None = None,
) -> EvaluationRunReport:
    validated = validate_evaluation_run_input(run_input)
    ranked_results_by_question = ranked_results_by_question or {}
    search_log_ids_by_question = search_log_ids_by_question or {}
    with connect(database_url) as connection:
        run = create_evaluation_run_in_connection(
            connection,
            replace(validated, status="running"),
        )
        question_ids = _load_question_ids_in_connection(connection, validated.question_set_id)
        result_records: list[EvaluationResultRecord] = []
        metric_records: list[QuestionMetricRecord] = []
        for question_id in question_ids:
            metric = evaluate_question(
                QuestionEvaluationInput(
                    question_id=question_id,
                    top_k=validated.top_k,
                    expected_targets=_load_expected_targets_in_connection(connection, question_id),
                    ranked_results=ranked_results_by_question.get(question_id, ()),
                )
            )
            metric_records.append(metric)
            result_records.append(
                create_evaluation_result_in_connection(
                    connection,
                    EvaluationResultInput(
                        evaluation_run_id=run.evaluation_run_id,
                        metric=metric,
                        search_log_id=search_log_ids_by_question.get(question_id),
                    ),
                )
            )

        summary = summarize_question_metrics(tuple(metric_records))
        completed_run = complete_evaluation_run_in_connection(
            connection,
            run.evaluation_run_id,
            summary,
        )

    return EvaluationRunReport(
        run=completed_run,
        results=tuple(result_records),
        summary=summary,
    )
