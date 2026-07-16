from datetime import datetime

import pytest

from app.core.golden_questions import GoldenQuestionRecord, GoldenQuestionSetRecord
from app.core.golden_search_experiments import (
    GoldenSearchExperimentBatchInput,
    InvalidGoldenSearchExperimentError,
    _question_execution_input,
    validate_golden_search_experiment_batch_input,
)


def _question_set() -> GoldenQuestionSetRecord:
    now = datetime(2026, 7, 12, 12, 0, 0)
    return GoldenQuestionSetRecord(
        question_set_id=7,
        set_name="Baseline",
        description=None,
        is_active=True,
        metadata={},
        created_by_user_id=1,
        created_at=now,
        updated_at=now,
    )


def _question(*, actor_user_id: int | None = 3) -> GoldenQuestionRecord:
    now = datetime(2026, 7, 12, 12, 0, 0)
    return GoldenQuestionRecord(
        question_id=11,
        question_set_id=7,
        question_text="What is the reset procedure?",
        normalized_question_text="what is the reset procedure?",
        question_type="single_fact",
        actor_user_id=actor_user_id,
        requested_search_scope="company",
        document_group="manual",
        file_type=".md",
        chunk_policy_name="heading_512_64",
        top_k=4,
        metadata={},
        created_by_user_id=3,
        created_at=now,
        updated_at=now,
    )


def test_golden_search_experiment_batch_validation_normalizes_values() -> None:
    validated = validate_golden_search_experiment_batch_input(
        GoldenSearchExperimentBatchInput(
            question_set_id=1,
            run_name_prefix=" Batch ",
            profiles=("kure_v1_1024",),
            strategy_name=" vector_cosine_threshold ",
            top_k=3,
            chunk_policy_name=" heading_512_64 ",
            runtime_metadata={"slice": 170},
            created_by=" api ",
            created_by_user_id=2,
            allow_mock_fallback=False,
        )
    )

    assert validated.run_name_prefix == "Batch"
    assert validated.strategy_name == "vector_cosine_threshold"
    assert validated.chunk_policy_name == "heading_512_64"
    assert validated.created_by == "api"
    assert validated.allow_mock_fallback is False


@pytest.mark.parametrize(
    ("batch_input", "message"),
    [
        (GoldenSearchExperimentBatchInput(question_set_id=0), "question_set_id"),
        (GoldenSearchExperimentBatchInput(question_set_id=1, top_k=0), "top_k"),
        (
            GoldenSearchExperimentBatchInput(
                question_set_id=1,
                runtime_metadata=[],
            ),
            "runtime_metadata",
        ),
    ],
)
def test_golden_search_experiment_batch_validation_rejects_invalid_values(
    batch_input: GoldenSearchExperimentBatchInput,
    message: str,
) -> None:
    with pytest.raises(InvalidGoldenSearchExperimentError, match=message):
        validate_golden_search_experiment_batch_input(batch_input)


def test_golden_search_experiment_question_input_uses_question_defaults() -> None:
    execution_input = _question_execution_input(
        question_set=_question_set(),
        question=_question(),
        batch_input=GoldenSearchExperimentBatchInput(
            question_set_id=7,
            profiles=("kure_v1_1024",),
            strategy_name="vector_cosine",
            runtime_metadata={"batch": True},
            allow_mock_fallback=False,
        ),
        run_name_prefix="Batch",
    )

    assert execution_input.run_name == "Batch / Q11"
    assert execution_input.top_k == 4
    assert execution_input.chunk_policy_name == "heading_512_64"
    assert execution_input.allow_mock_fallback is False
    assert execution_input.runtime_metadata["question_id"] == 11
    assert execution_input.runtime_metadata["golden_question_batch"] is True


def test_golden_search_experiment_question_input_requires_actor() -> None:
    with pytest.raises(InvalidGoldenSearchExperimentError, match="actor_user_id"):
        _question_execution_input(
            question_set=_question_set(),
            question=_question(actor_user_id=None),
            batch_input=GoldenSearchExperimentBatchInput(question_set_id=7),
            run_name_prefix="Batch",
        )
