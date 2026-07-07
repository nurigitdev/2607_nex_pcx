"""Dashboard read-model helpers for golden evaluation activity."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect
from app.core.evaluation_runs import EVALUATION_RUN_STATUSES

EVALUATION_DASHBOARD_STATUSES = ("pending", "running", "succeeded", "failed")


@dataclass(frozen=True)
class EvaluationDashboardStatusCount:
    status: str
    count: int


@dataclass(frozen=True)
class EvaluationDashboardRecentRun:
    evaluation_run_id: int
    question_set_id: int
    question_set_name: str
    run_name: str
    profile_name: str
    status: str
    question_count: int
    hidden_violation_count: int
    mean_recall_at_k: float | None
    mean_ndcg: float | None
    no_answer_success_rate: float | None
    created_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class EvaluationDashboardSummary:
    question_set_count: int
    active_question_set_count: int
    question_count: int
    expected_target_count: int
    evaluation_run_count: int
    status_counts: tuple[EvaluationDashboardStatusCount, ...]
    recent_runs: tuple[EvaluationDashboardRecentRun, ...]


class InvalidEvaluationDashboardError(ValueError):
    """Raised when dashboard inputs are invalid before reaching the DB."""


def _validate_recent_limit(recent_limit: int, *, max_limit: int = 20) -> int:
    if recent_limit <= 0:
        raise InvalidEvaluationDashboardError("recent_limit must be greater than 0")
    if recent_limit > max_limit:
        raise InvalidEvaluationDashboardError(
            f"recent_limit must be less than or equal to {max_limit}",
        )
    return recent_limit


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _status_counts_by_name(
    rows: list[dict[str, Any]],
) -> tuple[EvaluationDashboardStatusCount, ...]:
    counts = {status: 0 for status in EVALUATION_DASHBOARD_STATUSES}
    for row in rows:
        status = str(row["status"])
        if status in EVALUATION_RUN_STATUSES:
            counts[status] = int(row["run_count"])
    return tuple(
        EvaluationDashboardStatusCount(status=status, count=counts[status])
        for status in EVALUATION_DASHBOARD_STATUSES
    )


def _row_to_recent_run(row: dict[str, Any]) -> EvaluationDashboardRecentRun:
    return EvaluationDashboardRecentRun(
        evaluation_run_id=int(row["evaluation_run_id"]),
        question_set_id=int(row["question_set_id"]),
        question_set_name=str(row["question_set_name"]),
        run_name=str(row["run_name"]),
        profile_name=str(row["profile_name"]),
        status=str(row["status"]),
        question_count=int(row["question_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        mean_recall_at_k=_optional_float(row["mean_recall_at_k"]),
        mean_ndcg=_optional_float(row["mean_ndcg"]),
        no_answer_success_rate=_optional_float(row["no_answer_success_rate"]),
        created_at=row["created_at"],
        finished_at=row["finished_at"],
    )


def get_evaluation_dashboard_summary(
    database_url: str,
    *,
    recent_limit: int = 5,
) -> EvaluationDashboardSummary:
    validated_limit = _validate_recent_limit(recent_limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT count(*) FROM golden_question_sets) AS question_set_count,
                    (
                        SELECT count(*)
                        FROM golden_question_sets
                        WHERE is_active
                    ) AS active_question_set_count,
                    (SELECT count(*) FROM golden_questions) AS question_count,
                    (
                        SELECT count(*)
                        FROM golden_question_expected_targets
                    ) AS expected_target_count,
                    (SELECT count(*) FROM golden_evaluation_runs) AS evaluation_run_count
                """,
            )
            summary_row = dict(cursor.fetchone())

            cursor.execute(
                """
                SELECT status, count(*) AS run_count
                FROM golden_evaluation_runs
                GROUP BY status
                """,
            )
            status_rows = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT
                    ger.evaluation_run_id,
                    ger.question_set_id,
                    gqs.set_name AS question_set_name,
                    ger.run_name,
                    ger.profile_name,
                    ger.status,
                    ger.question_count,
                    ger.hidden_violation_count,
                    ger.mean_recall_at_k,
                    ger.mean_ndcg,
                    ger.no_answer_success_rate,
                    ger.created_at,
                    ger.finished_at
                FROM golden_evaluation_runs ger
                JOIN golden_question_sets gqs
                  ON gqs.question_set_id = ger.question_set_id
                ORDER BY ger.created_at DESC, ger.evaluation_run_id DESC
                LIMIT %s
                """,
                (validated_limit,),
            )
            recent_rows = [dict(row) for row in cursor.fetchall()]

    return EvaluationDashboardSummary(
        question_set_count=int(summary_row["question_set_count"]),
        active_question_set_count=int(summary_row["active_question_set_count"]),
        question_count=int(summary_row["question_count"]),
        expected_target_count=int(summary_row["expected_target_count"]),
        evaluation_run_count=int(summary_row["evaluation_run_count"]),
        status_counts=_status_counts_by_name(status_rows),
        recent_runs=tuple(_row_to_recent_run(row) for row in recent_rows),
    )
