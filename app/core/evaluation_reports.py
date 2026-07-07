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


@dataclass(frozen=True)
class EvaluationPermissionAuditRecord:
    evaluation_result_id: int
    evaluation_run_id: int
    question_id: int
    question_text: str
    actor_user_id: int | None
    actor_login_id: str | None
    actor_display_name: str | None
    requested_search_scope: str | None
    effective_search_scope: str | None
    permission_filter_metadata: dict[str, Any]
    search_log_id: int | None
    top_k: int
    retrieved_count: int
    visible_expected_count: int
    matched_visible_count: int
    hidden_violation_count: int
    matched_chunk_ids: tuple[int, ...]
    hidden_violation_chunk_ids: tuple[int, ...]
    no_answer_success: bool | None


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


def _row_to_permission_audit_record(row: dict[str, Any]) -> EvaluationPermissionAuditRecord:
    requested_search_scope = (
        row["search_requested_search_scope"] or row["question_requested_search_scope"]
    )
    return EvaluationPermissionAuditRecord(
        evaluation_result_id=int(row["evaluation_result_id"]),
        evaluation_run_id=int(row["evaluation_run_id"]),
        question_id=int(row["question_id"]),
        question_text=str(row["question_text"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        requested_search_scope=requested_search_scope,
        effective_search_scope=row["effective_search_scope"] or requested_search_scope,
        permission_filter_metadata=dict(row["permission_filter_metadata"] or {}),
        search_log_id=int(row["search_log_id"]) if row["search_log_id"] is not None else None,
        top_k=int(row["top_k"]),
        retrieved_count=int(row["retrieved_count"]),
        visible_expected_count=int(row["visible_expected_count"]),
        matched_visible_count=int(row["matched_visible_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        matched_chunk_ids=tuple(row["matched_chunk_ids"] or ()),
        hidden_violation_chunk_ids=tuple(row["hidden_violation_chunk_ids"] or ()),
        no_answer_success=row["no_answer_success"],
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


def get_evaluation_permission_audit(
    database_url: str,
    evaluation_run_id: int,
    *,
    limit: int = 500,
) -> list[EvaluationPermissionAuditRecord]:
    _require_positive_id(evaluation_run_id, "evaluation_run_id")
    validated_limit = _validate_limit(limit, max_limit=500)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    ger.evaluation_result_id,
                    ger.evaluation_run_id,
                    ger.question_id,
                    gq.question_text,
                    gq.actor_user_id,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name,
                    gq.requested_search_scope AS question_requested_search_scope,
                    sl.requested_search_scope AS search_requested_search_scope,
                    sl.effective_search_scope,
                    sl.permission_filter_metadata,
                    ger.search_log_id,
                    ger.top_k,
                    ger.retrieved_count,
                    ger.visible_expected_count,
                    ger.matched_visible_count,
                    ger.hidden_violation_count,
                    ger.matched_chunk_ids,
                    ger.hidden_violation_chunk_ids,
                    ger.no_answer_success
                FROM golden_evaluation_results ger
                JOIN golden_questions gq ON gq.question_id = ger.question_id
                LEFT JOIN app_users au ON au.user_id = gq.actor_user_id
                LEFT JOIN search_logs sl ON sl.search_log_id = ger.search_log_id
                WHERE ger.evaluation_run_id = %s
                ORDER BY ger.hidden_violation_count DESC,
                         ger.question_id ASC
                LIMIT %s
                """,
                (evaluation_run_id, validated_limit),
            )
            rows = cursor.fetchall()
    return [_row_to_permission_audit_record(dict(row)) for row in rows]
