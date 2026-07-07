import pytest

from app.core.evaluation_metrics import (
    ExpectedTarget,
    QuestionEvaluationInput,
    RankedSearchResult,
    evaluate_question,
)
from app.core.evaluation_runs import (
    EvaluationResultInput,
    EvaluationRunInput,
    InvalidEvaluationRunError,
    list_evaluation_runs,
    run_golden_evaluation_from_search_logs,
    validate_evaluation_result_input,
    validate_evaluation_run_input,
)


def _metric():
    return evaluate_question(
        QuestionEvaluationInput(
            question_id=1,
            top_k=3,
            expected_targets=(ExpectedTarget(chunk_id=1),),
            ranked_results=(RankedSearchResult(rank=1, chunk_id=1),),
        )
    )


def test_validate_evaluation_run_input_normalizes_values() -> None:
    validated = validate_evaluation_run_input(
        EvaluationRunInput(
            question_set_id=1,
            run_name="  baseline run  ",
            profile_name=" kure_v1_1024 ",
            chunk_policy_name=" heading_512_64 ",
            similarity_metric="cosine",
            top_k=3,
            status="pending",
            runtime_metadata={"source": "unit"},
        )
    )

    assert validated.run_name == "baseline run"
    assert validated.profile_name == "kure_v1_1024"
    assert validated.chunk_policy_name == "heading_512_64"
    assert validated.runtime_metadata == {"source": "unit"}


@pytest.mark.parametrize(
    ("run_input", "message"),
    [
        (
            EvaluationRunInput(question_set_id=0, run_name="run", profile_name="p"),
            "question_set_id",
        ),
        (EvaluationRunInput(question_set_id=1, run_name=" ", profile_name="p"), "run_name"),
        (EvaluationRunInput(question_set_id=1, run_name="run", profile_name=" "), "profile_name"),
        (
            EvaluationRunInput(question_set_id=1, run_name="run", profile_name="p", top_k=0),
            "top_k",
        ),
        (
            EvaluationRunInput(
                question_set_id=1,
                run_name="run",
                profile_name="p",
                similarity_metric="jaccard",
            ),
            "similarity_metric",
        ),
        (
            EvaluationRunInput(
                question_set_id=1,
                run_name="run",
                profile_name="p",
                status="paused",
            ),
            "status",
        ),
        (
            EvaluationRunInput(
                question_set_id=1,
                run_name="run",
                profile_name="p",
                runtime_metadata=[],
            ),
            "runtime_metadata",
        ),
    ],
)
def test_validate_evaluation_run_input_rejects_invalid_values(
    run_input: EvaluationRunInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEvaluationRunError, match=message):
        validate_evaluation_run_input(run_input)


def test_validate_evaluation_result_input_normalizes_metadata() -> None:
    validated = validate_evaluation_result_input(
        EvaluationResultInput(
            evaluation_run_id=1,
            metric=_metric(),
            search_log_id=2,
            metadata={"profile_elapsed_ms": 4},
        )
    )

    assert validated.metadata == {"profile_elapsed_ms": 4}


@pytest.mark.parametrize(
    ("result_input", "message"),
    [
        (EvaluationResultInput(evaluation_run_id=0, metric=_metric()), "evaluation_run_id"),
        (
            EvaluationResultInput(evaluation_run_id=1, metric=_metric(), search_log_id=0),
            "search_log_id",
        ),
        (
            EvaluationResultInput(evaluation_run_id=1, metric=_metric(), metadata=[]),
            "metadata",
        ),
    ],
)
def test_validate_evaluation_result_input_rejects_invalid_values(
    result_input: EvaluationResultInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEvaluationRunError, match=message):
        validate_evaluation_result_input(result_input)


def test_list_evaluation_runs_rejects_invalid_limit_before_connecting() -> None:
    with pytest.raises(InvalidEvaluationRunError, match="limit"):
        list_evaluation_runs("postgresql://unused", limit=0)

    with pytest.raises(InvalidEvaluationRunError, match="less than or equal"):
        list_evaluation_runs("postgresql://unused", limit=501)


def test_search_log_adapter_rejects_invalid_mapping_before_connecting() -> None:
    run_input = EvaluationRunInput(question_set_id=1, run_name="run", profile_name="profile")

    with pytest.raises(InvalidEvaluationRunError, match="must not be empty"):
        run_golden_evaluation_from_search_logs(
            "postgresql://unused",
            run_input,
            search_log_ids_by_question={},
        )

    with pytest.raises(InvalidEvaluationRunError, match="question_id"):
        run_golden_evaluation_from_search_logs(
            "postgresql://unused",
            run_input,
            search_log_ids_by_question={0: 1},
        )

    with pytest.raises(InvalidEvaluationRunError, match="search_log_id"):
        run_golden_evaluation_from_search_logs(
            "postgresql://unused",
            run_input,
            search_log_ids_by_question={1: 0},
        )
