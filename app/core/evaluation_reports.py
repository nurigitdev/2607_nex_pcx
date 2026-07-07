"""Read-model helpers for golden evaluation reports."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect


@dataclass(frozen=True)
class ProfileComparisonRecord:
    evaluation_run_id: int
    question_set_id: int
    run_name: str
    profile_name: str
    chunk_policy_name: str | None
    top_k: int
    question_count: int
    recall_question_count: int
    ndcg_question_count: int
    no_answer_question_count: int
    hidden_violation_count: int
    mean_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg: float | None
    no_answer_success_rate: float | None
    finished_at: datetime | None
    created_at: datetime


class InvalidEvaluationReportError(ValueError):
    """Raised when evaluation report input is invalid before reaching the DB."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is None or value <= 0:
        raise InvalidEvaluationReportError(f"{field_name} must be greater than 0")


def _validate_limit(limit: int, *, max_limit: int = 100) -> int:
    if limit <= 0:
        raise InvalidEvaluationReportError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidEvaluationReportError(f"limit must be less than or equal to {max_limit}")
    return limit


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _row_to_profile_comparison_record(row: dict[str, Any]) -> ProfileComparisonRecord:
    return ProfileComparisonRecord(
        evaluation_run_id=int(row["evaluation_run_id"]),
        question_set_id=int(row["question_set_id"]),
        run_name=str(row["run_name"]),
        profile_name=str(row["profile_name"]),
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        question_count=int(row["question_count"]),
        recall_question_count=int(row["recall_question_count"]),
        ndcg_question_count=int(row["ndcg_question_count"]),
        no_answer_question_count=int(row["no_answer_question_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        mean_recall_at_k=_optional_float(row["mean_recall_at_k"]),
        mean_reciprocal_rank=_optional_float(row["mean_reciprocal_rank"]),
        mean_ndcg=_optional_float(row["mean_ndcg"]),
        no_answer_success_rate=_optional_float(row["no_answer_success_rate"]),
        finished_at=row["finished_at"],
        created_at=row["created_at"],
    )


def get_latest_profile_comparison(
    database_url: str,
    question_set_id: int,
    *,
    limit: int = 20,
) -> list[ProfileComparisonRecord]:
    _require_positive_id(question_set_id, "question_set_id")
    validated_limit = _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH ranked_runs AS (
                    SELECT
                        ger.*,
                        row_number() OVER (
                            PARTITION BY ger.profile_name
                            ORDER BY ger.finished_at DESC NULLS LAST,
                                     ger.evaluation_run_id DESC
                        ) AS profile_rank
                    FROM golden_evaluation_runs ger
                    WHERE ger.question_set_id = %s
                      AND ger.status = 'succeeded'
                )
                SELECT *
                FROM ranked_runs
                WHERE profile_rank = 1
                ORDER BY mean_recall_at_k DESC NULLS LAST,
                         mean_ndcg DESC NULLS LAST,
                         profile_name ASC
                LIMIT %s
                """,
                (question_set_id, validated_limit),
            )
            rows = cursor.fetchall()
    return [_row_to_profile_comparison_record(dict(row)) for row in rows]
