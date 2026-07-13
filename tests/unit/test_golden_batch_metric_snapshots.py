from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.core.golden_batch_metric_snapshots import (
    GoldenBatchMetricSnapshotDetail,
    GoldenBatchMetricSnapshotRecord,
    GoldenBatchProfileMetricSnapshotRecord,
    GoldenBatchQuestionMetricSnapshotRecord,
    _compare_snapshot_details,
    _snapshot_trend_from_records,
)


def _snapshot_record(**overrides: object) -> GoldenBatchMetricSnapshotRecord:
    now = datetime(2026, 7, 13, 9, 0, tzinfo=UTC)
    record = GoldenBatchMetricSnapshotRecord(
        snapshot_id=1,
        batch_key="batch-a",
        question_set_id=1,
        question_set_name="Golden Set",
        batch_prefix="batch",
        strategy_name="vector_cosine_threshold",
        top_k=5,
        score_threshold=0.1,
        chunk_policy_name="heading_512_64",
        profile_names=("common_profile", "removed_profile"),
        batch_status="succeeded",
        batch_question_count=2,
        batch_succeeded_count=2,
        batch_failed_count=0,
        batch_running_count=0,
        total_result_count=4,
        average_result_count=2.0,
        total_elapsed_ms=40,
        average_elapsed_ms=20.0,
        evaluated_row_count=2,
        recall_question_count=2,
        ndcg_question_count=2,
        no_answer_question_count=0,
        hidden_violation_count=1,
        mean_recall_at_k=0.5,
        mean_reciprocal_rank=0.4,
        mean_ndcg=0.3,
        no_answer_success_rate=1.0,
        source_first_experiment_run_id=10,
        source_last_experiment_run_id=11,
        source_first_created_at=now,
        source_last_updated_at=now,
        metric_payload={"version": 1},
        created_by="unit-test",
        created_by_user_id=None,
        created_at=now,
    )
    return replace(record, **overrides)


def _profile_record(
    profile_name: str = "common_profile",
    **overrides: object,
) -> GoldenBatchProfileMetricSnapshotRecord:
    record = GoldenBatchProfileMetricSnapshotRecord(
        snapshot_profile_metric_id=1,
        snapshot_id=1,
        profile_name=profile_name,
        question_count=2,
        recall_question_count=2,
        ndcg_question_count=2,
        no_answer_question_count=0,
        hidden_violation_count=1,
        mean_recall_at_k=0.5,
        mean_reciprocal_rank=0.4,
        mean_ndcg=0.3,
        no_answer_success_rate=1.0,
        total_result_count=4,
        average_result_count=2.0,
        average_elapsed_ms=20.0,
    )
    return replace(record, **overrides)


def _question_record(
    question_id: int = 1,
    profile_name: str = "common_profile",
    **overrides: object,
) -> GoldenBatchQuestionMetricSnapshotRecord:
    record = GoldenBatchQuestionMetricSnapshotRecord(
        snapshot_question_metric_id=1,
        snapshot_id=1,
        question_id=question_id,
        question_text=f"Question {question_id}",
        profile_name=profile_name,
        experiment_run_id=10,
        search_log_id=20,
        top_k=5,
        result_count=2,
        elapsed_ms=20,
        visible_expected_count=2,
        retrieved_count=2,
        matched_visible_count=1,
        hidden_violation_count=1,
        matched_chunk_ids=(100,),
        hidden_violation_chunk_ids=(200,),
        recall_at_k=0.5,
        reciprocal_rank=0.4,
        dcg=0.3,
        ideal_dcg=1.0,
        ndcg=0.3,
        no_answer_success=None,
    )
    return replace(record, **overrides)


def test_compare_snapshot_details_reports_deltas_and_compatibility_warnings() -> None:
    base_snapshot = _snapshot_record()
    target_snapshot = _snapshot_record(
        snapshot_id=2,
        batch_key="batch-b",
        question_set_id=2,
        strategy_name="hybrid",
        top_k=10,
        score_threshold=0.2,
        chunk_policy_name="heading_1000_200",
        profile_names=("common_profile", "added_profile"),
        total_result_count=7,
        average_result_count=3.5,
        total_elapsed_ms=70,
        average_elapsed_ms=None,
        evaluated_row_count=3,
        hidden_violation_count=0,
        mean_recall_at_k=0.75,
        mean_reciprocal_rank=0.5,
        mean_ndcg=0.6,
        no_answer_success_rate=None,
        created_at=base_snapshot.created_at - timedelta(minutes=5),
    )
    base = GoldenBatchMetricSnapshotDetail(
        snapshot=base_snapshot,
        profiles=(
            _profile_record("common_profile"),
            _profile_record("removed_profile"),
        ),
        questions=(
            _question_record(1, "common_profile"),
            _question_record(2, "removed_profile"),
        ),
    )
    target = GoldenBatchMetricSnapshotDetail(
        snapshot=target_snapshot,
        profiles=(
            _profile_record(
                "common_profile",
                snapshot_id=2,
                mean_recall_at_k=0.75,
                mean_reciprocal_rank=0.6,
                mean_ndcg=0.65,
                average_elapsed_ms=None,
            ),
            _profile_record("added_profile", snapshot_id=2),
        ),
        questions=(
            _question_record(
                1,
                "common_profile",
                snapshot_id=2,
                result_count=3,
                elapsed_ms=None,
                matched_visible_count=2,
                hidden_violation_count=0,
                recall_at_k=1.0,
                reciprocal_rank=0.8,
                ndcg=None,
            ),
            _question_record(3, "added_profile", snapshot_id=2),
        ),
    )

    comparison = _compare_snapshot_details(base, target)

    assert comparison.overall.evaluated_row_count_delta == 1
    assert comparison.overall.total_result_count_delta == 3
    assert comparison.overall.average_elapsed_ms_delta is None
    assert comparison.overall.mean_recall_at_k_delta == 0.25
    assert comparison.overall.hidden_violation_count_delta == -1
    assert comparison.overall.no_answer_success_rate_delta is None
    assert comparison.compatibility_warnings == (
        "batch_key_differs",
        "question_set_differs",
        "strategy_differs",
        "top_k_differs",
        "score_threshold_differs",
        "chunk_policy_differs",
        "profile_set_differs",
        "target_older_than_base",
    )

    profiles_by_name = {profile.profile_name: profile for profile in comparison.profiles}
    assert profiles_by_name["common_profile"].comparison_status == "common"
    assert profiles_by_name["common_profile"].mean_ndcg_delta == pytest.approx(0.35)
    assert profiles_by_name["common_profile"].average_elapsed_ms_delta is None
    assert profiles_by_name["added_profile"].comparison_status == "added"
    assert profiles_by_name["added_profile"].question_count_delta is None
    assert profiles_by_name["removed_profile"].comparison_status == "removed"
    assert profiles_by_name["removed_profile"].mean_recall_at_k_delta is None

    questions_by_key = {
        (question.question_id, question.profile_name): question for question in comparison.questions
    }
    common_question = questions_by_key[(1, "common_profile")]
    assert common_question.comparison_status == "common"
    assert common_question.result_count_delta == 1
    assert common_question.elapsed_ms_delta is None
    assert common_question.matched_visible_count_delta == 1
    assert common_question.hidden_violation_count_delta == -1
    assert common_question.recall_at_k_delta == 0.5
    assert common_question.ndcg_delta is None
    assert questions_by_key[(2, "removed_profile")].comparison_status == "removed"
    assert questions_by_key[(3, "added_profile")].comparison_status == "added"


def test_snapshot_trend_orders_points_and_reports_previous_deltas() -> None:
    first = _snapshot_record(
        snapshot_id=1,
        created_at=datetime(2026, 7, 13, 9, 0, tzinfo=UTC),
        mean_recall_at_k=0.5,
        mean_reciprocal_rank=0.4,
        mean_ndcg=0.3,
        hidden_violation_count=2,
        average_elapsed_ms=30,
    )
    second = _snapshot_record(
        snapshot_id=2,
        created_at=datetime(2026, 7, 13, 9, 5, tzinfo=UTC),
        mean_recall_at_k=0.75,
        mean_reciprocal_rank=0.6,
        mean_ndcg=0.5,
        hidden_violation_count=1,
        average_elapsed_ms=25,
        evaluated_row_count=3,
        total_result_count=7,
        average_result_count=2.33,
        total_elapsed_ms=75,
    )
    third = _snapshot_record(
        snapshot_id=3,
        created_at=datetime(2026, 7, 13, 9, 10, tzinfo=UTC),
        mean_recall_at_k=None,
        mean_reciprocal_rank=0.7,
        mean_ndcg=0.65,
        hidden_violation_count=0,
        average_elapsed_ms=None,
    )

    trend = _snapshot_trend_from_records("batch-a", [third, first, second])

    assert trend.batch_key == "batch-a"
    assert trend.first_snapshot == first
    assert trend.latest_snapshot == third
    assert [point.snapshot.snapshot_id for point in trend.points] == [1, 2, 3]
    assert trend.points[0].sequence_number == 1
    assert trend.points[0].previous_snapshot_id is None
    assert trend.points[0].mean_recall_at_k_delta is None
    assert trend.points[1].previous_snapshot_id == 1
    assert trend.points[1].evaluated_row_count_delta == 1
    assert trend.points[1].total_result_count_delta == 3
    assert trend.points[1].average_result_count_delta == pytest.approx(0.33)
    assert trend.points[1].total_elapsed_ms_delta == 35
    assert trend.points[1].average_elapsed_ms_delta == -5
    assert trend.points[1].mean_recall_at_k_delta == 0.25
    assert trend.points[1].hidden_violation_count_delta == -1
    assert trend.points[2].previous_snapshot_id == 2
    assert trend.points[2].mean_recall_at_k_delta is None
    assert trend.points[2].average_elapsed_ms_delta is None
    assert trend.points[2].hidden_violation_count_delta == -1


def test_snapshot_trend_handles_empty_snapshot_list() -> None:
    trend = _snapshot_trend_from_records("batch-a", [])

    assert trend.batch_key == "batch-a"
    assert trend.points == ()
    assert trend.first_snapshot is None
    assert trend.latest_snapshot is None
