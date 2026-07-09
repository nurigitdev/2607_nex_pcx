import pytest

from app.core.golden_question_promotions import (
    InvalidGoldenQuestionPromotionError,
    list_golden_question_candidates,
)
from app.core.golden_questions import (
    GoldenQuestionExpectedTargetInput,
    GoldenQuestionInput,
    GoldenQuestionSetInput,
    InvalidGoldenQuestionError,
    list_golden_question_sets,
    list_golden_questions,
    normalize_question_text,
    validate_expected_target_input,
    validate_golden_question_input,
    validate_golden_question_set_input,
)


def test_normalize_question_text_collapses_case_and_whitespace() -> None:
    assert normalize_question_text("  What   IS\nThis?  ") == "what is this?"


def test_validate_golden_question_set_input_normalizes_values() -> None:
    validated = validate_golden_question_set_input(
        GoldenQuestionSetInput(
            set_name="  Baseline Set  ",
            description="  regression fixture  ",
            metadata={"version": 1},
            created_by_user_id=1,
        )
    )

    assert validated.set_name == "Baseline Set"
    assert validated.description == "regression fixture"
    assert validated.metadata == {"version": 1}


def test_validate_golden_question_input_normalizes_values() -> None:
    validated = validate_golden_question_input(
        GoldenQuestionInput(
            question_set_id=1,
            question_text="  Which policy applies?  ",
            question_type="section",
            actor_user_id=2,
            requested_search_scope="team",
            document_group="  policies  ",
            file_type=" .md ",
            chunk_policy_name=" heading_512_64 ",
            top_k=3,
            metadata={"difficulty": "easy"},
            created_by_user_id=1,
        )
    )

    assert validated.question_text == "Which policy applies?"
    assert validated.normalized_question_text == "which policy applies?"
    assert validated.document_group == "policies"
    assert validated.file_type == ".md"
    assert validated.chunk_policy_name == "heading_512_64"


def test_validate_expected_target_input_normalizes_heading_path() -> None:
    validated = validate_expected_target_input(
        GoldenQuestionExpectedTargetInput(
            question_id=1,
            expected_heading_path=(" Policy ", " Scope "),
            expectation_type="hidden",
            relevance_grade=0,
            notes="  should not be visible  ",
        )
    )

    assert validated.expected_heading_path == ("Policy", "Scope")
    assert validated.notes == "should not be visible"


@pytest.mark.parametrize(
    ("question_set_input", "message"),
    [
        (GoldenQuestionSetInput(set_name=" "), "set_name"),
        (GoldenQuestionSetInput(set_name="valid", description=" "), "description"),
        (GoldenQuestionSetInput(set_name="valid", created_by_user_id=0), "created_by_user_id"),
        (GoldenQuestionSetInput(set_name="valid", metadata=[]), "metadata"),
    ],
)
def test_validate_golden_question_set_input_rejects_invalid_values(
    question_set_input: GoldenQuestionSetInput,
    message: str,
) -> None:
    with pytest.raises(InvalidGoldenQuestionError, match=message):
        validate_golden_question_set_input(question_set_input)


@pytest.mark.parametrize(
    ("question_input", "message"),
    [
        (GoldenQuestionInput(question_set_id=0, question_text="valid"), "question_set_id"),
        (GoldenQuestionInput(question_set_id=1, question_text=" "), "question_text"),
        (
            GoldenQuestionInput(
                question_set_id=1,
                question_text="valid",
                normalized_question_text=" ",
            ),
            "normalized_question_text",
        ),
        (
            GoldenQuestionInput(
                question_set_id=1,
                question_text="valid",
                question_type="unsupported",
            ),
            "question_type",
        ),
        (
            GoldenQuestionInput(
                question_set_id=1,
                question_text="valid",
                requested_search_scope="all",
            ),
            "requested_search_scope",
        ),
        (GoldenQuestionInput(question_set_id=1, question_text="valid", top_k=0), "top_k"),
        (
            GoldenQuestionInput(
                question_set_id=1,
                question_text="valid",
                metadata=[],
            ),
            "metadata",
        ),
    ],
)
def test_validate_golden_question_input_rejects_invalid_values(
    question_input: GoldenQuestionInput,
    message: str,
) -> None:
    with pytest.raises(InvalidGoldenQuestionError, match=message):
        validate_golden_question_input(question_input)


@pytest.mark.parametrize(
    ("target_input", "message"),
    [
        (GoldenQuestionExpectedTargetInput(question_id=0, chunk_id=1), "question_id"),
        (GoldenQuestionExpectedTargetInput(question_id=1, chunk_id=0), "chunk_id"),
        (GoldenQuestionExpectedTargetInput(question_id=1), "chunk_id or expected_heading_path"),
        (
            GoldenQuestionExpectedTargetInput(question_id=1, expected_heading_path=(" ",)),
            "expected_heading_path",
        ),
        (
            GoldenQuestionExpectedTargetInput(
                question_id=1,
                chunk_id=1,
                expectation_type="maybe",
            ),
            "expectation_type",
        ),
        (
            GoldenQuestionExpectedTargetInput(question_id=1, chunk_id=1, relevance_grade=4),
            "relevance_grade",
        ),
        (
            GoldenQuestionExpectedTargetInput(question_id=1, chunk_id=1, notes=" "),
            "notes",
        ),
        (
            GoldenQuestionExpectedTargetInput(question_id=1, chunk_id=1, metadata=[]),
            "metadata",
        ),
    ],
)
def test_validate_expected_target_input_rejects_invalid_values(
    target_input: GoldenQuestionExpectedTargetInput,
    message: str,
) -> None:
    with pytest.raises(InvalidGoldenQuestionError, match=message):
        validate_expected_target_input(target_input)


def test_list_helpers_reject_invalid_limits_before_connecting() -> None:
    with pytest.raises(InvalidGoldenQuestionError, match="limit"):
        list_golden_question_sets("postgresql://unused", limit=0)

    with pytest.raises(InvalidGoldenQuestionError, match="less than or equal"):
        list_golden_questions("postgresql://unused", 1, limit=501)

    with pytest.raises(InvalidGoldenQuestionPromotionError, match="limit"):
        list_golden_question_candidates("postgresql://unused", limit=0)


def test_candidate_helper_rejects_blank_document_group_before_connecting() -> None:
    with pytest.raises(InvalidGoldenQuestionPromotionError, match="document_group"):
        list_golden_question_candidates("postgresql://unused", document_group=" ")
