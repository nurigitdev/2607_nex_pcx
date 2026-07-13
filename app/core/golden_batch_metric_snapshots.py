"""Persistence helpers for golden search experiment batch metric snapshots."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect
from app.core.search_experiments import (
    GoldenSearchExperimentBatchMetricSummary,
    GoldenSearchExperimentBatchProfileMetricSummary,
    GoldenSearchExperimentBatchQuestionMetricSummary,
    decode_golden_search_experiment_batch_key,
    get_golden_search_experiment_batch_metric_summary,
)


@dataclass(frozen=True)
class GoldenBatchMetricSnapshotRecord:
    snapshot_id: int
    batch_key: str
    question_set_id: int
    question_set_name: str
    batch_prefix: str
    strategy_name: str
    top_k: int
    score_threshold: float | None
    chunk_policy_name: str | None
    profile_names: tuple[str, ...]
    batch_status: str
    batch_question_count: int
    batch_succeeded_count: int
    batch_failed_count: int
    batch_running_count: int
    total_result_count: int
    average_result_count: float
    total_elapsed_ms: int
    average_elapsed_ms: float | None
    evaluated_row_count: int
    recall_question_count: int
    ndcg_question_count: int
    no_answer_question_count: int
    hidden_violation_count: int
    mean_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg: float | None
    no_answer_success_rate: float | None
    source_first_experiment_run_id: int
    source_last_experiment_run_id: int
    source_first_created_at: datetime
    source_last_updated_at: datetime
    metric_payload: dict[str, Any]
    created_by: str | None
    created_by_user_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class GoldenBatchProfileMetricSnapshotRecord:
    snapshot_profile_metric_id: int
    snapshot_id: int
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
class GoldenBatchQuestionMetricSnapshotRecord:
    snapshot_question_metric_id: int
    snapshot_id: int
    question_id: int
    question_text: str
    profile_name: str
    experiment_run_id: int
    search_log_id: int
    top_k: int
    result_count: int
    elapsed_ms: int | None
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


@dataclass(frozen=True)
class GoldenBatchMetricSnapshotDetail:
    snapshot: GoldenBatchMetricSnapshotRecord
    profiles: tuple[GoldenBatchProfileMetricSnapshotRecord, ...]
    questions: tuple[GoldenBatchQuestionMetricSnapshotRecord, ...]


@dataclass(frozen=True)
class GoldenBatchSnapshotOverallComparison:
    evaluated_row_count_delta: int
    total_result_count_delta: int
    average_result_count_delta: float
    total_elapsed_ms_delta: int
    average_elapsed_ms_delta: float | None
    hidden_violation_count_delta: int
    mean_recall_at_k_delta: float | None
    mean_reciprocal_rank_delta: float | None
    mean_ndcg_delta: float | None
    no_answer_success_rate_delta: float | None


@dataclass(frozen=True)
class GoldenBatchProfileMetricSnapshotComparison:
    profile_name: str
    comparison_status: str
    base: GoldenBatchProfileMetricSnapshotRecord | None
    target: GoldenBatchProfileMetricSnapshotRecord | None
    question_count_delta: int | None
    hidden_violation_count_delta: int | None
    mean_recall_at_k_delta: float | None
    mean_reciprocal_rank_delta: float | None
    mean_ndcg_delta: float | None
    no_answer_success_rate_delta: float | None
    average_result_count_delta: float | None
    average_elapsed_ms_delta: float | None


@dataclass(frozen=True)
class GoldenBatchQuestionMetricSnapshotComparison:
    question_id: int
    question_text: str
    profile_name: str
    comparison_status: str
    base: GoldenBatchQuestionMetricSnapshotRecord | None
    target: GoldenBatchQuestionMetricSnapshotRecord | None
    result_count_delta: int | None
    elapsed_ms_delta: int | None
    matched_visible_count_delta: int | None
    hidden_violation_count_delta: int | None
    recall_at_k_delta: float | None
    reciprocal_rank_delta: float | None
    ndcg_delta: float | None


@dataclass(frozen=True)
class GoldenBatchMetricSnapshotComparison:
    base: GoldenBatchMetricSnapshotDetail
    target: GoldenBatchMetricSnapshotDetail
    overall: GoldenBatchSnapshotOverallComparison
    profiles: tuple[GoldenBatchProfileMetricSnapshotComparison, ...]
    questions: tuple[GoldenBatchQuestionMetricSnapshotComparison, ...]
    compatibility_warnings: tuple[str, ...]


@dataclass(frozen=True)
class GoldenBatchMetricSnapshotTrendPoint:
    snapshot: GoldenBatchMetricSnapshotRecord
    sequence_number: int
    previous_snapshot_id: int | None
    evaluated_row_count_delta: int | None
    total_result_count_delta: int | None
    average_result_count_delta: float | None
    total_elapsed_ms_delta: int | None
    average_elapsed_ms_delta: float | None
    hidden_violation_count_delta: int | None
    mean_recall_at_k_delta: float | None
    mean_reciprocal_rank_delta: float | None
    mean_ndcg_delta: float | None
    no_answer_success_rate_delta: float | None


@dataclass(frozen=True)
class GoldenBatchMetricSnapshotTrend:
    batch_key: str
    points: tuple[GoldenBatchMetricSnapshotTrendPoint, ...]
    first_snapshot: GoldenBatchMetricSnapshotRecord | None
    latest_snapshot: GoldenBatchMetricSnapshotRecord | None


class InvalidGoldenBatchMetricSnapshotError(ValueError):
    """Raised when golden batch metric snapshot inputs are invalid."""


def record_golden_batch_metric_snapshot(
    database_url: str,
    batch_key: str,
    *,
    created_by: str | None = None,
    created_by_user_id: int | None = None,
) -> GoldenBatchMetricSnapshotDetail | None:
    _validate_optional_positive_id(created_by_user_id, "created_by_user_id")
    validated_created_by = _validate_nonblank(created_by, "created_by")
    metric_summary = get_golden_search_experiment_batch_metric_summary(database_url, batch_key)
    if metric_summary is None:
        return None
    with connect(database_url) as connection:
        return _record_golden_batch_metric_snapshot_in_connection(
            connection,
            metric_summary,
            created_by=validated_created_by,
            created_by_user_id=created_by_user_id,
        )


def list_golden_batch_metric_snapshots(
    database_url: str,
    batch_key: str,
    *,
    limit: int = 10,
) -> list[GoldenBatchMetricSnapshotRecord]:
    decode_golden_search_experiment_batch_key(batch_key)
    validated_limit = _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM golden_search_experiment_batch_metric_snapshots
                WHERE batch_key = %s
                ORDER BY created_at DESC, snapshot_id DESC
                LIMIT %s
                """,
                (batch_key, validated_limit),
            )
            rows = cursor.fetchall()
    return [_row_to_snapshot_record(dict(row)) for row in rows]


def get_latest_golden_batch_metric_snapshot(
    database_url: str,
    batch_key: str,
) -> GoldenBatchMetricSnapshotRecord | None:
    snapshots = list_golden_batch_metric_snapshots(database_url, batch_key, limit=1)
    return snapshots[0] if snapshots else None


def get_golden_batch_metric_snapshot_trend(
    database_url: str,
    batch_key: str,
    *,
    limit: int = 20,
) -> GoldenBatchMetricSnapshotTrend:
    snapshots = list_golden_batch_metric_snapshots(database_url, batch_key, limit=limit)
    return _snapshot_trend_from_records(batch_key, snapshots)


def get_golden_batch_metric_snapshot_detail(
    database_url: str,
    snapshot_id: int,
) -> GoldenBatchMetricSnapshotDetail | None:
    _validate_positive_id(snapshot_id, "snapshot_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM golden_search_experiment_batch_metric_snapshots
                WHERE snapshot_id = %s
                """,
                (snapshot_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            snapshot = _row_to_snapshot_record(dict(row))
        return _snapshot_detail_in_connection(connection, snapshot)


def compare_golden_batch_metric_snapshots(
    database_url: str,
    *,
    base_snapshot_id: int,
    target_snapshot_id: int,
) -> GoldenBatchMetricSnapshotComparison | None:
    _validate_positive_id(base_snapshot_id, "base_snapshot_id")
    _validate_positive_id(target_snapshot_id, "target_snapshot_id")
    if base_snapshot_id == target_snapshot_id:
        raise InvalidGoldenBatchMetricSnapshotError("snapshot ids must be different")

    with connect(database_url) as connection:
        base = _snapshot_detail_by_id_in_connection(connection, base_snapshot_id)
        target = _snapshot_detail_by_id_in_connection(connection, target_snapshot_id)
    if base is None or target is None:
        return None
    return _compare_snapshot_details(base, target)


def _record_golden_batch_metric_snapshot_in_connection(
    connection: Connection,
    metric_summary: GoldenSearchExperimentBatchMetricSummary,
    *,
    created_by: str | None,
    created_by_user_id: int | None,
) -> GoldenBatchMetricSnapshotDetail:
    batch = metric_summary.summary
    overall = metric_summary.overall
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO golden_search_experiment_batch_metric_snapshots (
                batch_key,
                question_set_id,
                question_set_name,
                batch_prefix,
                strategy_name,
                top_k,
                score_threshold,
                chunk_policy_name,
                profile_names,
                batch_status,
                batch_question_count,
                batch_succeeded_count,
                batch_failed_count,
                batch_running_count,
                total_result_count,
                average_result_count,
                total_elapsed_ms,
                average_elapsed_ms,
                evaluated_row_count,
                recall_question_count,
                ndcg_question_count,
                no_answer_question_count,
                hidden_violation_count,
                mean_recall_at_k,
                mean_reciprocal_rank,
                mean_ndcg,
                no_answer_success_rate,
                source_first_experiment_run_id,
                source_last_experiment_run_id,
                source_first_created_at,
                source_last_updated_at,
                metric_payload,
                created_by,
                created_by_user_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            RETURNING *
            """,
            (
                batch.batch_key,
                batch.question_set_id,
                batch.question_set_name,
                batch.batch_prefix,
                batch.strategy_name,
                batch.top_k,
                batch.score_threshold,
                batch.chunk_policy_name,
                Json(list(batch.profile_names)),
                batch.status,
                batch.question_count,
                batch.succeeded_count,
                batch.failed_count,
                batch.running_count,
                batch.total_result_count,
                batch.average_result_count,
                batch.total_elapsed_ms,
                batch.average_elapsed_ms,
                overall.question_count,
                overall.recall_question_count,
                overall.ndcg_question_count,
                overall.no_answer_question_count,
                overall.hidden_violation_count,
                overall.mean_recall_at_k,
                overall.mean_reciprocal_rank,
                overall.mean_ndcg,
                overall.no_answer_success_rate,
                batch.first_experiment_run_id,
                batch.last_experiment_run_id,
                batch.first_created_at,
                batch.last_updated_at,
                Json(_metric_summary_payload(metric_summary)),
                created_by,
                created_by_user_id,
            ),
        )
        snapshot = _row_to_snapshot_record(dict(cursor.fetchone()))
        for profile in metric_summary.profiles:
            _insert_profile_metric_snapshot(cursor, snapshot.snapshot_id, profile)
        for question in metric_summary.questions:
            _insert_question_metric_snapshot(cursor, snapshot.snapshot_id, question)
    return _snapshot_detail_in_connection(connection, snapshot)


def _snapshot_detail_by_id_in_connection(
    connection: Connection,
    snapshot_id: int,
) -> GoldenBatchMetricSnapshotDetail | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT *
            FROM golden_search_experiment_batch_metric_snapshots
            WHERE snapshot_id = %s
            """,
            (snapshot_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return _snapshot_detail_in_connection(connection, _row_to_snapshot_record(dict(row)))


def _compare_snapshot_details(
    base: GoldenBatchMetricSnapshotDetail,
    target: GoldenBatchMetricSnapshotDetail,
) -> GoldenBatchMetricSnapshotComparison:
    return GoldenBatchMetricSnapshotComparison(
        base=base,
        target=target,
        overall=_compare_overall(base.snapshot, target.snapshot),
        profiles=_compare_profile_metrics(base.profiles, target.profiles),
        questions=_compare_question_metrics(base.questions, target.questions),
        compatibility_warnings=_snapshot_compatibility_warnings(base.snapshot, target.snapshot),
    )


def _snapshot_trend_from_records(
    batch_key: str,
    snapshots: list[GoldenBatchMetricSnapshotRecord],
) -> GoldenBatchMetricSnapshotTrend:
    chronological_snapshots = tuple(
        sorted(snapshots, key=lambda snapshot: (snapshot.created_at, snapshot.snapshot_id))
    )
    points: list[GoldenBatchMetricSnapshotTrendPoint] = []
    previous: GoldenBatchMetricSnapshotRecord | None = None
    for sequence_number, snapshot in enumerate(chronological_snapshots, start=1):
        overall = _compare_overall(previous, snapshot) if previous is not None else None
        points.append(
            GoldenBatchMetricSnapshotTrendPoint(
                snapshot=snapshot,
                sequence_number=sequence_number,
                previous_snapshot_id=previous.snapshot_id if previous is not None else None,
                evaluated_row_count_delta=(
                    overall.evaluated_row_count_delta if overall is not None else None
                ),
                total_result_count_delta=(
                    overall.total_result_count_delta if overall is not None else None
                ),
                average_result_count_delta=(
                    overall.average_result_count_delta if overall is not None else None
                ),
                total_elapsed_ms_delta=(
                    overall.total_elapsed_ms_delta if overall is not None else None
                ),
                average_elapsed_ms_delta=(
                    overall.average_elapsed_ms_delta if overall is not None else None
                ),
                hidden_violation_count_delta=(
                    overall.hidden_violation_count_delta if overall is not None else None
                ),
                mean_recall_at_k_delta=(
                    overall.mean_recall_at_k_delta if overall is not None else None
                ),
                mean_reciprocal_rank_delta=(
                    overall.mean_reciprocal_rank_delta if overall is not None else None
                ),
                mean_ndcg_delta=overall.mean_ndcg_delta if overall is not None else None,
                no_answer_success_rate_delta=(
                    overall.no_answer_success_rate_delta if overall is not None else None
                ),
            )
        )
        previous = snapshot
    return GoldenBatchMetricSnapshotTrend(
        batch_key=batch_key,
        points=tuple(points),
        first_snapshot=chronological_snapshots[0] if chronological_snapshots else None,
        latest_snapshot=chronological_snapshots[-1] if chronological_snapshots else None,
    )


def _compare_overall(
    base: GoldenBatchMetricSnapshotRecord,
    target: GoldenBatchMetricSnapshotRecord,
) -> GoldenBatchSnapshotOverallComparison:
    return GoldenBatchSnapshotOverallComparison(
        evaluated_row_count_delta=target.evaluated_row_count - base.evaluated_row_count,
        total_result_count_delta=target.total_result_count - base.total_result_count,
        average_result_count_delta=target.average_result_count - base.average_result_count,
        total_elapsed_ms_delta=target.total_elapsed_ms - base.total_elapsed_ms,
        average_elapsed_ms_delta=_optional_delta(
            base.average_elapsed_ms,
            target.average_elapsed_ms,
        ),
        hidden_violation_count_delta=(target.hidden_violation_count - base.hidden_violation_count),
        mean_recall_at_k_delta=_optional_delta(
            base.mean_recall_at_k,
            target.mean_recall_at_k,
        ),
        mean_reciprocal_rank_delta=_optional_delta(
            base.mean_reciprocal_rank,
            target.mean_reciprocal_rank,
        ),
        mean_ndcg_delta=_optional_delta(base.mean_ndcg, target.mean_ndcg),
        no_answer_success_rate_delta=_optional_delta(
            base.no_answer_success_rate,
            target.no_answer_success_rate,
        ),
    )


def _compare_profile_metrics(
    base_profiles: tuple[GoldenBatchProfileMetricSnapshotRecord, ...],
    target_profiles: tuple[GoldenBatchProfileMetricSnapshotRecord, ...],
) -> tuple[GoldenBatchProfileMetricSnapshotComparison, ...]:
    base_by_profile = {profile.profile_name: profile for profile in base_profiles}
    target_by_profile = {profile.profile_name: profile for profile in target_profiles}
    comparisons: list[GoldenBatchProfileMetricSnapshotComparison] = []
    for profile_name in sorted(base_by_profile.keys() | target_by_profile.keys()):
        base = base_by_profile.get(profile_name)
        target = target_by_profile.get(profile_name)
        comparisons.append(
            GoldenBatchProfileMetricSnapshotComparison(
                profile_name=profile_name,
                comparison_status=_comparison_status(base, target),
                base=base,
                target=target,
                question_count_delta=_required_record_delta(
                    base,
                    target,
                    "question_count",
                ),
                hidden_violation_count_delta=_required_record_delta(
                    base,
                    target,
                    "hidden_violation_count",
                ),
                mean_recall_at_k_delta=_optional_record_delta(
                    base,
                    target,
                    "mean_recall_at_k",
                ),
                mean_reciprocal_rank_delta=_optional_record_delta(
                    base,
                    target,
                    "mean_reciprocal_rank",
                ),
                mean_ndcg_delta=_optional_record_delta(base, target, "mean_ndcg"),
                no_answer_success_rate_delta=_optional_record_delta(
                    base,
                    target,
                    "no_answer_success_rate",
                ),
                average_result_count_delta=_optional_record_delta(
                    base,
                    target,
                    "average_result_count",
                ),
                average_elapsed_ms_delta=_optional_record_delta(
                    base,
                    target,
                    "average_elapsed_ms",
                ),
            )
        )
    return tuple(comparisons)


def _compare_question_metrics(
    base_questions: tuple[GoldenBatchQuestionMetricSnapshotRecord, ...],
    target_questions: tuple[GoldenBatchQuestionMetricSnapshotRecord, ...],
) -> tuple[GoldenBatchQuestionMetricSnapshotComparison, ...]:
    base_by_key = {
        (question.question_id, question.profile_name): question for question in base_questions
    }
    target_by_key = {
        (question.question_id, question.profile_name): question for question in target_questions
    }
    comparisons: list[GoldenBatchQuestionMetricSnapshotComparison] = []
    for question_id, profile_name in sorted(base_by_key.keys() | target_by_key.keys()):
        base = base_by_key.get((question_id, profile_name))
        target = target_by_key.get((question_id, profile_name))
        source = target or base
        if source is None:
            continue
        comparisons.append(
            GoldenBatchQuestionMetricSnapshotComparison(
                question_id=question_id,
                question_text=source.question_text,
                profile_name=profile_name,
                comparison_status=_comparison_status(base, target),
                base=base,
                target=target,
                result_count_delta=_required_record_delta(base, target, "result_count"),
                elapsed_ms_delta=_optional_int_record_delta(base, target, "elapsed_ms"),
                matched_visible_count_delta=_required_record_delta(
                    base,
                    target,
                    "matched_visible_count",
                ),
                hidden_violation_count_delta=_required_record_delta(
                    base,
                    target,
                    "hidden_violation_count",
                ),
                recall_at_k_delta=_optional_record_delta(base, target, "recall_at_k"),
                reciprocal_rank_delta=_optional_record_delta(
                    base,
                    target,
                    "reciprocal_rank",
                ),
                ndcg_delta=_optional_record_delta(base, target, "ndcg"),
            )
        )
    return tuple(comparisons)


def _snapshot_compatibility_warnings(
    base: GoldenBatchMetricSnapshotRecord,
    target: GoldenBatchMetricSnapshotRecord,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if base.batch_key != target.batch_key:
        warnings.append("batch_key_differs")
    if base.question_set_id != target.question_set_id:
        warnings.append("question_set_differs")
    if base.strategy_name != target.strategy_name:
        warnings.append("strategy_differs")
    if base.top_k != target.top_k:
        warnings.append("top_k_differs")
    if base.score_threshold != target.score_threshold:
        warnings.append("score_threshold_differs")
    if base.chunk_policy_name != target.chunk_policy_name:
        warnings.append("chunk_policy_differs")
    if set(base.profile_names) != set(target.profile_names):
        warnings.append("profile_set_differs")
    if target.created_at < base.created_at:
        warnings.append("target_older_than_base")
    return tuple(warnings)


def _comparison_status(base: object | None, target: object | None) -> str:
    if base is None:
        return "added"
    if target is None:
        return "removed"
    return "common"


def _required_record_delta(base: object | None, target: object | None, field_name: str) -> Any:
    if base is None or target is None:
        return None
    return getattr(target, field_name) - getattr(base, field_name)


def _optional_int_record_delta(
    base: object | None,
    target: object | None,
    field_name: str,
) -> int | None:
    delta = _optional_record_delta(base, target, field_name)
    return int(delta) if delta is not None else None


def _optional_record_delta(
    base: object | None,
    target: object | None,
    field_name: str,
) -> float | None:
    if base is None or target is None:
        return None
    return _optional_delta(getattr(base, field_name), getattr(target, field_name))


def _optional_delta(
    base_value: float | int | None, target_value: float | int | None
) -> float | None:
    if base_value is None or target_value is None:
        return None
    return float(target_value) - float(base_value)


def _insert_profile_metric_snapshot(
    cursor: Any,
    snapshot_id: int,
    profile: GoldenSearchExperimentBatchProfileMetricSummary,
) -> None:
    cursor.execute(
        """
        INSERT INTO golden_search_experiment_batch_profile_metric_snapshots (
            snapshot_id,
            profile_name,
            question_count,
            recall_question_count,
            ndcg_question_count,
            no_answer_question_count,
            hidden_violation_count,
            mean_recall_at_k,
            mean_reciprocal_rank,
            mean_ndcg,
            no_answer_success_rate,
            total_result_count,
            average_result_count,
            average_elapsed_ms
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            snapshot_id,
            profile.profile_name,
            profile.question_count,
            profile.recall_question_count,
            profile.ndcg_question_count,
            profile.no_answer_question_count,
            profile.hidden_violation_count,
            profile.mean_recall_at_k,
            profile.mean_reciprocal_rank,
            profile.mean_ndcg,
            profile.no_answer_success_rate,
            profile.total_result_count,
            profile.average_result_count,
            profile.average_elapsed_ms,
        ),
    )


def _insert_question_metric_snapshot(
    cursor: Any,
    snapshot_id: int,
    question: GoldenSearchExperimentBatchQuestionMetricSummary,
) -> None:
    metric = question.metric
    cursor.execute(
        """
        INSERT INTO golden_search_experiment_batch_question_metric_snapshots (
            snapshot_id,
            question_id,
            question_text,
            profile_name,
            experiment_run_id,
            search_log_id,
            top_k,
            result_count,
            elapsed_ms,
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
            no_answer_success
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            snapshot_id,
            question.question_id,
            question.question_text,
            question.profile_name,
            question.experiment_run_id,
            question.search_log_id,
            question.top_k,
            question.result_count,
            question.elapsed_ms,
            metric.visible_expected_count,
            metric.retrieved_count,
            metric.matched_visible_count,
            metric.hidden_violation_count,
            Json(list(metric.matched_chunk_ids)),
            Json(list(metric.hidden_violation_chunk_ids)),
            metric.recall_at_k,
            metric.reciprocal_rank,
            metric.dcg,
            metric.ideal_dcg,
            metric.ndcg,
            metric.no_answer_success,
        ),
    )


def _snapshot_detail_in_connection(
    connection: Connection,
    snapshot: GoldenBatchMetricSnapshotRecord,
) -> GoldenBatchMetricSnapshotDetail:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT *
            FROM golden_search_experiment_batch_profile_metric_snapshots
            WHERE snapshot_id = %s
            ORDER BY profile_name ASC
            """,
            (snapshot.snapshot_id,),
        )
        profile_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT *
            FROM golden_search_experiment_batch_question_metric_snapshots
            WHERE snapshot_id = %s
            ORDER BY question_id ASC, profile_name ASC, experiment_run_id ASC
            """,
            (snapshot.snapshot_id,),
        )
        question_rows = cursor.fetchall()
    return GoldenBatchMetricSnapshotDetail(
        snapshot=snapshot,
        profiles=tuple(_row_to_profile_record(dict(row)) for row in profile_rows),
        questions=tuple(_row_to_question_record(dict(row)) for row in question_rows),
    )


def _metric_summary_payload(
    metric_summary: GoldenSearchExperimentBatchMetricSummary,
) -> dict[str, Any]:
    return {
        "batch_key": metric_summary.summary.batch_key,
        "overall": {
            "question_count": metric_summary.overall.question_count,
            "recall_question_count": metric_summary.overall.recall_question_count,
            "ndcg_question_count": metric_summary.overall.ndcg_question_count,
            "no_answer_question_count": metric_summary.overall.no_answer_question_count,
            "hidden_violation_count": metric_summary.overall.hidden_violation_count,
            "mean_recall_at_k": metric_summary.overall.mean_recall_at_k,
            "mean_reciprocal_rank": metric_summary.overall.mean_reciprocal_rank,
            "mean_ndcg": metric_summary.overall.mean_ndcg,
            "no_answer_success_rate": metric_summary.overall.no_answer_success_rate,
        },
        "profiles": [
            {
                "profile_name": profile.profile_name,
                "question_count": profile.question_count,
                "mean_recall_at_k": profile.mean_recall_at_k,
                "mean_reciprocal_rank": profile.mean_reciprocal_rank,
                "mean_ndcg": profile.mean_ndcg,
                "hidden_violation_count": profile.hidden_violation_count,
            }
            for profile in metric_summary.profiles
        ],
    }


def _row_to_snapshot_record(row: dict[str, Any]) -> GoldenBatchMetricSnapshotRecord:
    return GoldenBatchMetricSnapshotRecord(
        snapshot_id=int(row["snapshot_id"]),
        batch_key=str(row["batch_key"]),
        question_set_id=int(row["question_set_id"]),
        question_set_name=str(row["question_set_name"] or ""),
        batch_prefix=str(row["batch_prefix"]),
        strategy_name=str(row["strategy_name"]),
        top_k=int(row["top_k"]),
        score_threshold=_optional_float(row["score_threshold"]),
        chunk_policy_name=row["chunk_policy_name"],
        profile_names=tuple(str(profile) for profile in row["profile_names"]),
        batch_status=str(row["batch_status"]),
        batch_question_count=int(row["batch_question_count"]),
        batch_succeeded_count=int(row["batch_succeeded_count"]),
        batch_failed_count=int(row["batch_failed_count"]),
        batch_running_count=int(row["batch_running_count"]),
        total_result_count=int(row["total_result_count"]),
        average_result_count=float(row["average_result_count"]),
        total_elapsed_ms=int(row["total_elapsed_ms"]),
        average_elapsed_ms=_optional_float(row["average_elapsed_ms"]),
        evaluated_row_count=int(row["evaluated_row_count"]),
        recall_question_count=int(row["recall_question_count"]),
        ndcg_question_count=int(row["ndcg_question_count"]),
        no_answer_question_count=int(row["no_answer_question_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        mean_recall_at_k=_optional_float(row["mean_recall_at_k"]),
        mean_reciprocal_rank=_optional_float(row["mean_reciprocal_rank"]),
        mean_ndcg=_optional_float(row["mean_ndcg"]),
        no_answer_success_rate=_optional_float(row["no_answer_success_rate"]),
        source_first_experiment_run_id=int(row["source_first_experiment_run_id"]),
        source_last_experiment_run_id=int(row["source_last_experiment_run_id"]),
        source_first_created_at=row["source_first_created_at"],
        source_last_updated_at=row["source_last_updated_at"],
        metric_payload=dict(row["metric_payload"] or {}),
        created_by=row["created_by"],
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        created_at=row["created_at"],
    )


def _row_to_profile_record(row: dict[str, Any]) -> GoldenBatchProfileMetricSnapshotRecord:
    return GoldenBatchProfileMetricSnapshotRecord(
        snapshot_profile_metric_id=int(row["snapshot_profile_metric_id"]),
        snapshot_id=int(row["snapshot_id"]),
        profile_name=str(row["profile_name"]),
        question_count=int(row["question_count"]),
        recall_question_count=int(row["recall_question_count"]),
        ndcg_question_count=int(row["ndcg_question_count"]),
        no_answer_question_count=int(row["no_answer_question_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        mean_recall_at_k=_optional_float(row["mean_recall_at_k"]),
        mean_reciprocal_rank=_optional_float(row["mean_reciprocal_rank"]),
        mean_ndcg=_optional_float(row["mean_ndcg"]),
        no_answer_success_rate=_optional_float(row["no_answer_success_rate"]),
        total_result_count=int(row["total_result_count"]),
        average_result_count=_optional_float(row["average_result_count"]),
        average_elapsed_ms=_optional_float(row["average_elapsed_ms"]),
    )


def _row_to_question_record(row: dict[str, Any]) -> GoldenBatchQuestionMetricSnapshotRecord:
    return GoldenBatchQuestionMetricSnapshotRecord(
        snapshot_question_metric_id=int(row["snapshot_question_metric_id"]),
        snapshot_id=int(row["snapshot_id"]),
        question_id=int(row["question_id"]),
        question_text=str(row["question_text"]),
        profile_name=str(row["profile_name"]),
        experiment_run_id=int(row["experiment_run_id"]),
        search_log_id=int(row["search_log_id"]),
        top_k=int(row["top_k"]),
        result_count=int(row["result_count"]),
        elapsed_ms=int(row["elapsed_ms"]) if row["elapsed_ms"] is not None else None,
        visible_expected_count=int(row["visible_expected_count"]),
        retrieved_count=int(row["retrieved_count"]),
        matched_visible_count=int(row["matched_visible_count"]),
        hidden_violation_count=int(row["hidden_violation_count"]),
        matched_chunk_ids=tuple(int(chunk_id) for chunk_id in row["matched_chunk_ids"]),
        hidden_violation_chunk_ids=tuple(
            int(chunk_id) for chunk_id in row["hidden_violation_chunk_ids"]
        ),
        recall_at_k=_optional_float(row["recall_at_k"]),
        reciprocal_rank=_optional_float(row["reciprocal_rank"]),
        dcg=float(row["dcg"]),
        ideal_dcg=float(row["ideal_dcg"]),
        ndcg=_optional_float(row["ndcg"]),
        no_answer_success=(
            bool(row["no_answer_success"]) if row["no_answer_success"] is not None else None
        ),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _validate_limit(limit: int) -> int:
    if limit <= 0:
        raise InvalidGoldenBatchMetricSnapshotError("limit must be greater than 0")
    if limit > 100:
        raise InvalidGoldenBatchMetricSnapshotError("limit must be less than or equal to 100")
    return limit


def _validate_positive_id(value: int | None, field_name: str) -> None:
    if value is None or value <= 0:
        raise InvalidGoldenBatchMetricSnapshotError(f"{field_name} must be greater than 0")


def _validate_optional_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidGoldenBatchMetricSnapshotError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidGoldenBatchMetricSnapshotError(f"{field_name} must not be blank")
    return normalized
