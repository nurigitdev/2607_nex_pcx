import pytest

from app.core.generation_answer_quality import (
    ANSWER_QUALITY_REASON_GUARDRAIL_NO_ANSWER_NOT_REFLECTED,
    ANSWER_QUALITY_REASON_MISSING_REQUIRED_CITATION,
    ANSWER_QUALITY_REASON_PARTIAL_CITATION_COVERAGE,
    ANSWER_QUALITY_REASON_PROVIDER_ERROR,
    ANSWER_QUALITY_REASON_UNEXPECTED_NO_ANSWER,
    ANSWER_QUALITY_REASON_UNRECOGNIZED_CITATION_KEY,
    GENERATION_ANSWER_QUALITY_FAILED,
    GENERATION_ANSWER_QUALITY_NOT_EVALUATED,
    GENERATION_ANSWER_QUALITY_PASSED,
    GENERATION_ANSWER_QUALITY_WARNING,
    assess_generation_answer_quality,
    generation_answer_quality_payload,
)


def test_generation_answer_quality_passes_when_expected_citation_is_used() -> None:
    assessment = assess_generation_answer_quality(
        answer_text="사내 보안 규정은 계정 공유를 금지합니다. [RCP-001]",
        expected_citation_keys=("RCP-001",),
        guardrail_status="allowed",
    )

    assert assessment.status == GENERATION_ANSWER_QUALITY_PASSED
    assert assessment.answer_present is True
    assert assessment.requires_citation is True
    assert assessment.cited_citation_keys == ("RCP-001",)
    assert assessment.recognized_citation_keys == ("RCP-001",)
    assert assessment.missing_citation_keys == ()
    assert assessment.citation_coverage_percent == 100.0
    assert assessment.reason_codes == ()


def test_generation_answer_quality_fails_when_no_required_citation_is_used() -> None:
    assessment = assess_generation_answer_quality(
        answer_text="사내 보안 규정은 계정 공유를 금지합니다.",
        expected_citation_keys=("RCP-001",),
        guardrail_status="allowed",
    )

    assert assessment.status == GENERATION_ANSWER_QUALITY_FAILED
    assert assessment.recognized_citation_keys == ()
    assert assessment.missing_citation_keys == ("RCP-001",)
    assert assessment.reason_codes == (ANSWER_QUALITY_REASON_MISSING_REQUIRED_CITATION,)


def test_generation_answer_quality_warns_for_partial_and_unknown_citations() -> None:
    assessment = assess_generation_answer_quality(
        answer_text="보안 규정은 계정 공유를 금지하고 [RCP-001], 다른 근거도 참고합니다 [RCP-999].",
        expected_citation_keys=("RCP-001", "RCP-002"),
        guardrail_status="allowed",
    )

    assert assessment.status == GENERATION_ANSWER_QUALITY_WARNING
    assert assessment.recognized_citation_keys == ("RCP-001",)
    assert assessment.missing_citation_keys == ("RCP-002",)
    assert assessment.unrecognized_citation_keys == ("RCP-999",)
    assert assessment.citation_coverage_percent == 50.0
    assert assessment.reason_codes == (
        ANSWER_QUALITY_REASON_PARTIAL_CITATION_COVERAGE,
        ANSWER_QUALITY_REASON_UNRECOGNIZED_CITATION_KEY,
    )


def test_generation_answer_quality_passes_expected_guardrail_no_answer() -> None:
    assessment = assess_generation_answer_quality(
        answer_text="제공된 문서 근거만으로는 답변할 수 없습니다.",
        expected_citation_keys=("RCP-001",),
        guardrail_status="no_answer",
    )

    assert assessment.status == GENERATION_ANSWER_QUALITY_PASSED
    assert assessment.no_answer_detected is True
    assert assessment.requires_citation is False
    assert assessment.reason_codes == ()


def test_generation_answer_quality_fails_when_allowed_answer_returns_no_answer() -> None:
    assessment = assess_generation_answer_quality(
        answer_text="제공된 문서 근거만으로는 답변할 수 없습니다.",
        expected_citation_keys=("RCP-001",),
        guardrail_status="allowed",
    )

    assert assessment.status == GENERATION_ANSWER_QUALITY_FAILED
    assert assessment.reason_codes == (
        ANSWER_QUALITY_REASON_UNEXPECTED_NO_ANSWER,
        ANSWER_QUALITY_REASON_MISSING_REQUIRED_CITATION,
    )


def test_generation_answer_quality_fails_when_no_answer_guardrail_is_not_reflected() -> None:
    assessment = assess_generation_answer_quality(
        answer_text="근거가 없지만 임의 답변입니다.",
        expected_citation_keys=("RCP-001",),
        guardrail_status="no_answer",
    )

    assert assessment.status == GENERATION_ANSWER_QUALITY_FAILED
    assert assessment.reason_codes == (ANSWER_QUALITY_REASON_GUARDRAIL_NO_ANSWER_NOT_REFLECTED,)


def test_generation_answer_quality_marks_provider_error_not_evaluated() -> None:
    assessment = assess_generation_answer_quality(
        answer_text=None,
        expected_citation_keys=("RCP-001",),
        guardrail_status="allowed",
        provider_error=True,
    )
    payload = generation_answer_quality_payload(assessment)

    assert assessment.status == GENERATION_ANSWER_QUALITY_NOT_EVALUATED
    assert assessment.reason_codes == (ANSWER_QUALITY_REASON_PROVIDER_ERROR,)
    assert payload["contract_version"] == "generation_answer_quality_v1"
    assert payload["status"] == "not_evaluated"


@pytest.mark.parametrize(
    "answer_text",
    (
        "[RCP-001] [RCP-001]",
        "중복 citation [RCP-001] 그리고 다시 [RCP-001]",
    ),
)
def test_generation_answer_quality_deduplicates_citations(answer_text: str) -> None:
    assessment = assess_generation_answer_quality(
        answer_text=answer_text,
        expected_citation_keys=("RCP-001",),
        guardrail_status="allowed",
    )

    assert assessment.cited_citation_keys == ("RCP-001",)
    assert assessment.status == GENERATION_ANSWER_QUALITY_PASSED
