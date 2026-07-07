"""Evaluation metric helpers for golden question search results."""

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class ExpectedTarget:
    chunk_id: int | None = None
    expected_heading_path: tuple[str, ...] = ()
    expectation_type: str = "visible"
    relevance_grade: int = 3


@dataclass(frozen=True)
class RankedSearchResult:
    rank: int
    chunk_id: int
    score: float | None = None


@dataclass(frozen=True)
class QuestionEvaluationInput:
    question_id: int
    top_k: int
    expected_targets: tuple[ExpectedTarget, ...]
    ranked_results: tuple[RankedSearchResult, ...]


@dataclass(frozen=True)
class QuestionMetricRecord:
    question_id: int
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


@dataclass(frozen=True)
class EvaluationSummaryRecord:
    question_count: int
    recall_question_count: int
    ndcg_question_count: int
    no_answer_question_count: int
    hidden_violation_count: int
    mean_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg: float | None
    no_answer_success_rate: float | None


class InvalidEvaluationMetricError(ValueError):
    """Raised when evaluation metric inputs are invalid."""


def dcg_at_k(grades: tuple[int, ...], top_k: int) -> float:
    if top_k <= 0:
        raise InvalidEvaluationMetricError("top_k must be greater than 0")
    return sum((2**grade - 1) / log2(index + 2) for index, grade in enumerate(grades[:top_k]))


def _require_positive_id(value: int, field_name: str) -> None:
    if value <= 0:
        raise InvalidEvaluationMetricError(f"{field_name} must be greater than 0")


def _validate_expected_target(target: ExpectedTarget) -> None:
    if target.chunk_id is not None:
        _require_positive_id(target.chunk_id, "chunk_id")
    if target.expectation_type not in {"visible", "hidden"}:
        raise InvalidEvaluationMetricError(
            f"Unsupported expectation_type: {target.expectation_type}"
        )
    if target.relevance_grade < 0 or target.relevance_grade > 3:
        raise InvalidEvaluationMetricError("relevance_grade must be between 0 and 3")
    if target.chunk_id is None and not target.expected_heading_path:
        raise InvalidEvaluationMetricError("chunk_id or expected_heading_path is required")


def _validate_ranked_results(ranked_results: tuple[RankedSearchResult, ...]) -> None:
    seen_ranks: set[int] = set()
    for result in ranked_results:
        _require_positive_id(result.rank, "rank")
        _require_positive_id(result.chunk_id, "chunk_id")
        if result.rank in seen_ranks:
            raise InvalidEvaluationMetricError("rank values must be unique")
        seen_ranks.add(result.rank)


def _visible_chunk_grades(expected_targets: tuple[ExpectedTarget, ...]) -> dict[int, int]:
    grades: dict[int, int] = {}
    for target in expected_targets:
        _validate_expected_target(target)
        if target.chunk_id is None or target.expectation_type != "visible":
            continue
        grades[target.chunk_id] = max(grades.get(target.chunk_id, 0), target.relevance_grade)
    return {chunk_id: grade for chunk_id, grade in grades.items() if grade > 0}


def _hidden_chunk_ids(expected_targets: tuple[ExpectedTarget, ...]) -> set[int]:
    hidden: set[int] = set()
    for target in expected_targets:
        _validate_expected_target(target)
        if target.chunk_id is not None and target.expectation_type == "hidden":
            hidden.add(target.chunk_id)
    return hidden


def evaluate_question(evaluation_input: QuestionEvaluationInput) -> QuestionMetricRecord:
    _require_positive_id(evaluation_input.question_id, "question_id")
    if evaluation_input.top_k <= 0:
        raise InvalidEvaluationMetricError("top_k must be greater than 0")
    _validate_ranked_results(evaluation_input.ranked_results)

    visible_grades = _visible_chunk_grades(evaluation_input.expected_targets)
    hidden_chunk_ids = _hidden_chunk_ids(evaluation_input.expected_targets)
    ranked_results = tuple(
        sorted(
            (
                result
                for result in evaluation_input.ranked_results
                if result.rank <= evaluation_input.top_k
            ),
            key=lambda result: result.rank,
        )
    )

    seen_chunks: set[int] = set()
    result_grades: list[int] = []
    matched_chunks: set[int] = set()
    hidden_violations: set[int] = set()
    first_match_rank: int | None = None
    for result in ranked_results:
        if result.chunk_id in hidden_chunk_ids:
            hidden_violations.add(result.chunk_id)

        grade = 0
        if result.chunk_id not in seen_chunks:
            grade = visible_grades.get(result.chunk_id, 0)
            seen_chunks.add(result.chunk_id)
        result_grades.append(grade)

        if grade > 0:
            matched_chunks.add(result.chunk_id)
            if first_match_rank is None:
                first_match_rank = result.rank

    ideal_grades = tuple(sorted(visible_grades.values(), reverse=True))
    dcg = dcg_at_k(tuple(result_grades), evaluation_input.top_k)
    ideal_dcg = dcg_at_k(ideal_grades, evaluation_input.top_k) if ideal_grades else 0.0
    has_visible_expectation = bool(visible_grades)

    recall_at_k = len(matched_chunks) / len(visible_grades) if has_visible_expectation else None
    reciprocal_rank = (
        (1 / first_match_rank if first_match_rank is not None else 0.0)
        if has_visible_expectation
        else None
    )
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else None
    no_answer_success = (
        len(ranked_results) == 0 and not hidden_violations if not has_visible_expectation else None
    )

    return QuestionMetricRecord(
        question_id=evaluation_input.question_id,
        top_k=evaluation_input.top_k,
        visible_expected_count=len(visible_grades),
        retrieved_count=len(ranked_results),
        matched_visible_count=len(matched_chunks),
        hidden_violation_count=len(hidden_violations),
        matched_chunk_ids=tuple(sorted(matched_chunks)),
        hidden_violation_chunk_ids=tuple(sorted(hidden_violations)),
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        dcg=dcg,
        ideal_dcg=ideal_dcg,
        ndcg=ndcg,
        no_answer_success=no_answer_success,
    )


def _mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_question_metrics(
    question_metrics: tuple[QuestionMetricRecord, ...],
) -> EvaluationSummaryRecord:
    recall_values = tuple(
        metric.recall_at_k for metric in question_metrics if metric.recall_at_k is not None
    )
    reciprocal_rank_values = tuple(
        metric.reciprocal_rank for metric in question_metrics if metric.reciprocal_rank is not None
    )
    ndcg_values = tuple(metric.ndcg for metric in question_metrics if metric.ndcg is not None)
    no_answer_values = tuple(
        metric.no_answer_success
        for metric in question_metrics
        if metric.no_answer_success is not None
    )
    return EvaluationSummaryRecord(
        question_count=len(question_metrics),
        recall_question_count=len(recall_values),
        ndcg_question_count=len(ndcg_values),
        no_answer_question_count=len(no_answer_values),
        hidden_violation_count=sum(metric.hidden_violation_count for metric in question_metrics),
        mean_recall_at_k=_mean(recall_values),
        mean_reciprocal_rank=_mean(reciprocal_rank_values),
        mean_ndcg=_mean(ndcg_values),
        no_answer_success_rate=(
            sum(1 for value in no_answer_values if value) / len(no_answer_values)
            if no_answer_values
            else None
        ),
    )
