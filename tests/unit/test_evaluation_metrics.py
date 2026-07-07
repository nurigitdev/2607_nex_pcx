import math

import pytest

from app.core.evaluation_metrics import (
    ExpectedTarget,
    InvalidEvaluationMetricError,
    QuestionEvaluationInput,
    RankedSearchResult,
    dcg_at_k,
    evaluate_question,
    summarize_question_metrics,
)


def test_dcg_at_k_uses_graded_relevance_discount() -> None:
    assert dcg_at_k((3, 2, 0), 3) == pytest.approx(7 + (3 / math.log2(3)))


def test_evaluate_question_calculates_recall_mrr_ndcg_and_hidden_violations() -> None:
    metric = evaluate_question(
        QuestionEvaluationInput(
            question_id=1,
            top_k=3,
            expected_targets=(
                ExpectedTarget(chunk_id=1, relevance_grade=3),
                ExpectedTarget(chunk_id=2, relevance_grade=2),
                ExpectedTarget(chunk_id=5, expectation_type="hidden", relevance_grade=0),
            ),
            ranked_results=(
                RankedSearchResult(rank=1, chunk_id=9, score=0.91),
                RankedSearchResult(rank=2, chunk_id=2, score=0.88),
                RankedSearchResult(rank=3, chunk_id=5, score=0.80),
                RankedSearchResult(rank=4, chunk_id=1, score=0.77),
            ),
        )
    )

    ideal_dcg = 7 + (3 / math.log2(3))
    assert metric.visible_expected_count == 2
    assert metric.retrieved_count == 3
    assert metric.matched_visible_count == 1
    assert metric.hidden_violation_count == 1
    assert metric.matched_chunk_ids == (2,)
    assert metric.hidden_violation_chunk_ids == (5,)
    assert metric.recall_at_k == pytest.approx(0.5)
    assert metric.reciprocal_rank == pytest.approx(0.5)
    assert metric.dcg == pytest.approx(3 / math.log2(3))
    assert metric.ideal_dcg == pytest.approx(ideal_dcg)
    assert metric.ndcg == pytest.approx((3 / math.log2(3)) / ideal_dcg)
    assert metric.no_answer_success is None


def test_evaluate_question_counts_duplicate_visible_targets_and_results_once() -> None:
    metric = evaluate_question(
        QuestionEvaluationInput(
            question_id=2,
            top_k=3,
            expected_targets=(
                ExpectedTarget(chunk_id=10, relevance_grade=1),
                ExpectedTarget(chunk_id=10, relevance_grade=3),
            ),
            ranked_results=(
                RankedSearchResult(rank=1, chunk_id=10),
                RankedSearchResult(rank=2, chunk_id=10),
            ),
        )
    )

    assert metric.visible_expected_count == 1
    assert metric.matched_visible_count == 1
    assert metric.recall_at_k == pytest.approx(1)
    assert metric.dcg == pytest.approx(7)
    assert metric.ndcg == pytest.approx(1)


def test_evaluate_question_records_zero_when_visible_target_is_not_found() -> None:
    metric = evaluate_question(
        QuestionEvaluationInput(
            question_id=3,
            top_k=5,
            expected_targets=(ExpectedTarget(chunk_id=10, relevance_grade=3),),
            ranked_results=(RankedSearchResult(rank=1, chunk_id=11),),
        )
    )

    assert metric.recall_at_k == pytest.approx(0)
    assert metric.reciprocal_rank == pytest.approx(0)
    assert metric.ndcg == pytest.approx(0)
    assert metric.no_answer_success is None


def test_evaluate_question_tracks_no_answer_success_when_no_visible_target_exists() -> None:
    success = evaluate_question(
        QuestionEvaluationInput(
            question_id=4,
            top_k=3,
            expected_targets=(ExpectedTarget(expected_heading_path=("No Answer",)),),
            ranked_results=(),
        )
    )
    failure = evaluate_question(
        QuestionEvaluationInput(
            question_id=5,
            top_k=3,
            expected_targets=(ExpectedTarget(expected_heading_path=("No Answer",)),),
            ranked_results=(RankedSearchResult(rank=1, chunk_id=42),),
        )
    )

    assert success.recall_at_k is None
    assert success.ndcg is None
    assert success.no_answer_success is True
    assert failure.no_answer_success is False


def test_summarize_question_metrics_averages_evaluable_questions() -> None:
    hit = evaluate_question(
        QuestionEvaluationInput(
            question_id=6,
            top_k=3,
            expected_targets=(ExpectedTarget(chunk_id=1, relevance_grade=3),),
            ranked_results=(RankedSearchResult(rank=1, chunk_id=1),),
        )
    )
    miss = evaluate_question(
        QuestionEvaluationInput(
            question_id=7,
            top_k=3,
            expected_targets=(ExpectedTarget(chunk_id=2, relevance_grade=3),),
            ranked_results=(RankedSearchResult(rank=1, chunk_id=3),),
        )
    )
    no_answer = evaluate_question(
        QuestionEvaluationInput(
            question_id=8,
            top_k=3,
            expected_targets=(ExpectedTarget(expected_heading_path=("No Answer",)),),
            ranked_results=(),
        )
    )

    summary = summarize_question_metrics((hit, miss, no_answer))

    assert summary.question_count == 3
    assert summary.recall_question_count == 2
    assert summary.ndcg_question_count == 2
    assert summary.no_answer_question_count == 1
    assert summary.mean_recall_at_k == pytest.approx(0.5)
    assert summary.mean_reciprocal_rank == pytest.approx(0.5)
    assert summary.mean_ndcg == pytest.approx(0.5)
    assert summary.no_answer_success_rate == pytest.approx(1)


def test_summarize_question_metrics_returns_none_for_empty_metrics() -> None:
    summary = summarize_question_metrics(())

    assert summary.question_count == 0
    assert summary.mean_recall_at_k is None
    assert summary.mean_reciprocal_rank is None
    assert summary.mean_ndcg is None
    assert summary.no_answer_success_rate is None


@pytest.mark.parametrize(
    ("evaluation_input", "message"),
    [
        (
            QuestionEvaluationInput(
                question_id=0,
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=1),),
                ranked_results=(),
            ),
            "question_id",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=0,
                expected_targets=(ExpectedTarget(chunk_id=1),),
                ranked_results=(),
            ),
            "top_k",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=0),),
                ranked_results=(),
            ),
            "chunk_id",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=1, expectation_type="maybe"),),
                ranked_results=(),
            ),
            "expectation_type",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=1, relevance_grade=4),),
                ranked_results=(),
            ),
            "relevance_grade",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=3,
                expected_targets=(ExpectedTarget(),),
                ranked_results=(),
            ),
            "chunk_id or expected_heading_path",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=1),),
                ranked_results=(RankedSearchResult(rank=0, chunk_id=1),),
            ),
            "rank",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=1),),
                ranked_results=(RankedSearchResult(rank=1, chunk_id=0),),
            ),
            "chunk_id",
        ),
        (
            QuestionEvaluationInput(
                question_id=1,
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=1),),
                ranked_results=(
                    RankedSearchResult(rank=1, chunk_id=1),
                    RankedSearchResult(rank=1, chunk_id=2),
                ),
            ),
            "rank values",
        ),
    ],
)
def test_evaluate_question_rejects_invalid_values(
    evaluation_input: QuestionEvaluationInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEvaluationMetricError, match=message):
        evaluate_question(evaluation_input)


def test_dcg_at_k_rejects_invalid_top_k() -> None:
    with pytest.raises(InvalidEvaluationMetricError, match="top_k"):
        dcg_at_k((3,), 0)
