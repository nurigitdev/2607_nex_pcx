from datetime import datetime

import pytest

from app.core.evaluation_executor import (
    GoldenEvaluationExecutionInput,
    InvalidGoldenEvaluationExecutionError,
    _default_run_name,
    _question_search_input,
    validate_golden_evaluation_execution_input,
)
from app.core.golden_questions import GoldenQuestionRecord, GoldenQuestionSetRecord


def test_validate_golden_evaluation_execution_input_normalizes_optional_fields() -> None:
    validated = validate_golden_evaluation_execution_input(
        GoldenEvaluationExecutionInput(
            question_set_id=1,
            profile_name=" kure_v1_1024 ",
            run_name=" nightly check ",
            chunk_policy_name=" heading_512_64 ",
            top_k=7,
            runtime_metadata={"slice": "044"},
        )
    )

    assert validated.profile_name == "kure_v1_1024"
    assert validated.run_name == "nightly check"
    assert validated.chunk_policy_name == "heading_512_64"
    assert validated.runtime_metadata == {"slice": "044"}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"question_set_id": 0}, "question_set_id"),
        ({"profile_name": " "}, "profile_name"),
        ({"run_name": " "}, "run_name"),
        ({"top_k": 0}, "top_k"),
        ({"runtime_metadata": []}, "runtime_metadata"),
    ],
)
def test_validate_golden_evaluation_execution_input_rejects_invalid_values(
    values: dict[str, object],
    message: str,
) -> None:
    payload = {
        "question_set_id": 1,
        "profile_name": "kure_v1_1024",
        **values,
    }

    with pytest.raises(InvalidGoldenEvaluationExecutionError, match=message):
        validate_golden_evaluation_execution_input(GoldenEvaluationExecutionInput(**payload))


def test_default_run_name_includes_question_set_and_profile() -> None:
    now = datetime.now()
    question_set = GoldenQuestionSetRecord(
        question_set_id=1,
        set_name="baseline",
        description=None,
        is_active=True,
        metadata={},
        created_by_user_id=None,
        created_at=now,
        updated_at=now,
    )

    run_name = _default_run_name(question_set, "kure_v1_1024")

    assert run_name.startswith("baseline / kure_v1_1024 / ")


def test_question_search_input_uses_question_defaults_and_execution_override() -> None:
    now = datetime.now()
    question = GoldenQuestionRecord(
        question_id=10,
        question_set_id=1,
        question_text="What is expected?",
        normalized_question_text="what is expected?",
        question_type="single_fact",
        actor_user_id=3,
        requested_search_scope="team",
        document_group="policy",
        file_type=".md",
        chunk_policy_name="heading_512_64",
        top_k=5,
        metadata={},
        created_by_user_id=3,
        created_at=now,
        updated_at=now,
    )

    default_search = _question_search_input(
        question,
        GoldenEvaluationExecutionInput(question_set_id=1, profile_name="kure_v1_1024", top_k=9),
    )
    override_search = _question_search_input(
        question,
        GoldenEvaluationExecutionInput(
            question_set_id=1,
            profile_name="kure_v1_1024",
            chunk_policy_name="override_policy",
            top_k=9,
        ),
    )

    assert default_search.query_text == "What is expected?"
    assert default_search.actor_user_id == 3
    assert default_search.requested_search_scope == "team"
    assert default_search.top_k == 9
    assert default_search.profiles == ("kure_v1_1024",)
    assert default_search.chunk_policy_name == "heading_512_64"
    assert override_search.chunk_policy_name == "override_policy"


def test_question_search_input_rejects_question_without_actor() -> None:
    now = datetime.now()
    question = GoldenQuestionRecord(
        question_id=10,
        question_set_id=1,
        question_text="What is expected?",
        normalized_question_text="what is expected?",
        question_type="single_fact",
        actor_user_id=None,
        requested_search_scope="team",
        document_group="policy",
        file_type=".md",
        chunk_policy_name="heading_512_64",
        top_k=5,
        metadata={},
        created_by_user_id=None,
        created_at=now,
        updated_at=now,
    )

    with pytest.raises(InvalidGoldenEvaluationExecutionError, match="actor_user_id"):
        _question_search_input(
            question,
            GoldenEvaluationExecutionInput(question_set_id=1, profile_name="kure_v1_1024"),
        )
