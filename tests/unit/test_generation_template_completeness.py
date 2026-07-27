from datetime import datetime

from app.core.generation_runs import GenerationRunRecord
from app.core.generation_template_completeness import (
    GENERATION_TEMPLATE_COMPLETENESS_FAILED,
    GENERATION_TEMPLATE_COMPLETENESS_NOT_EVALUATED,
    GENERATION_TEMPLATE_COMPLETENESS_PASSED,
    GENERATION_TEMPLATE_COMPLETENESS_WARNING,
    TEMPLATE_COMPLETENESS_REASON_EMPTY_ANSWER,
    TEMPLATE_COMPLETENESS_REASON_MISSING_REQUIRED_SECTION,
    TEMPLATE_COMPLETENESS_REASON_MISSING_SECTION_CITATION,
    TEMPLATE_COMPLETENESS_REASON_NO_ANSWER_GUARDRAIL,
    TEMPLATE_COMPLETENESS_REASON_NO_REQUIRED_SECTIONS,
    TEMPLATE_COMPLETENESS_REASON_NO_TEMPLATE_METADATA,
    TEMPLATE_COMPLETENESS_REASON_UNSUPPORTED_OUTPUT_FORMAT,
    assess_generation_template_completeness,
    generation_template_completeness_payload,
)


def _report_template_snapshot() -> dict[str, object]:
    return {
        "template_key": "report",
        "template_name": "보고서 초안",
        "template_version": "v1",
        "document_type": "report",
        "output_format": "markdown",
        "section_schema": [
            {"key": "title", "heading": "제목", "required": True},
            {"key": "overview", "heading": "요약", "required": True},
            {"key": "findings", "heading": "주요 내용", "required": True},
            {"key": "evidence", "heading": "근거", "required": True},
            {"key": "risks", "heading": "리스크", "required": False},
        ],
        "citation_policy": {"required": True, "placement": "per_section"},
    }


def _run(
    *,
    answer_text: str | None,
    request_metadata: dict[str, object] | None = None,
    response_metadata: dict[str, object] | None = None,
    status: str = "succeeded",
    guardrail_status: str = "allowed",
) -> GenerationRunRecord:
    now = datetime(2026, 7, 27, 9, 0, 0)
    return GenerationRunRecord(
        generation_run_id=1,
        search_log_id=2,
        retrieval_package_key="pytest-package",
        generation_template_id=3,
        provider_config_id=4,
        provider_name="mock",
        provider_mode="mock",
        model_id="pytest-model",
        prompt_version="grounded_answer_v1",
        prompt_hash="prompt-hash",
        context_hash="context-hash",
        status=status,
        guardrail_status=guardrail_status,
        retrieval_confidence_status="answerable",
        citation_readiness_status="ready",
        query_text="테스트 질의",
        answer_text=answer_text,
        finish_reason="stop",
        input_token_count=10,
        output_token_count=20,
        total_token_count=30,
        elapsed_ms=40,
        request_metadata=request_metadata or {},
        response_metadata=response_metadata or {},
        guardrail_metadata={},
        error_message=None,
        created_by="pytest",
        created_by_user_id=5,
        started_at=now,
        finished_at=now,
        created_at=now,
        updated_at=now,
    )


def test_template_completeness_passes_report_with_required_sections_and_citations() -> None:
    assessment = assess_generation_template_completeness(
        _run(
            answer_text=(
                "# 보고서 초안\n\n"
                "## 요약\n요약입니다. [RCP-001]\n\n"
                "## 주요 내용\n- 주요 내용입니다. [RCP-001]\n\n"
                "## 근거\n- 근거입니다. [RCP-001]\n\n"
                "## 한계\n- 추가 확인 필요"
            ),
            request_metadata={"generation_template": _report_template_snapshot()},
        )
    )
    payload = generation_template_completeness_payload(assessment)

    assert assessment.status == GENERATION_TEMPLATE_COMPLETENESS_PASSED
    assert assessment.required_section_count == 4
    assert assessment.present_required_section_count == 4
    assert assessment.citation_required_section_count == 3
    assert assessment.cited_required_section_count == 3
    assert assessment.required_section_coverage_percent == 100.0
    assert assessment.required_section_citation_coverage_percent == 100.0
    assert assessment.reason_codes == ()
    assert payload["contract_version"] == "generation_template_completeness_v1"
    assert payload["section_checks"][0]["key"] == "title"


def test_template_completeness_fails_for_missing_required_section() -> None:
    assessment = assess_generation_template_completeness(
        _run(
            answer_text=(
                "# 보고서 초안\n\n"
                "## 요약\n요약입니다. [RCP-001]\n\n"
                "## 주요 내용\n- 주요 내용입니다. [RCP-001]"
            ),
            request_metadata={"generation_template": _report_template_snapshot()},
        )
    )

    assert assessment.status == GENERATION_TEMPLATE_COMPLETENESS_FAILED
    assert assessment.required_section_coverage_percent == 75.0
    assert assessment.required_section_citation_coverage_percent == 66.67
    assert assessment.reason_codes == (
        TEMPLATE_COMPLETENESS_REASON_MISSING_REQUIRED_SECTION,
        TEMPLATE_COMPLETENESS_REASON_MISSING_SECTION_CITATION,
    )


def test_template_completeness_warns_for_present_required_section_without_citation() -> None:
    assessment = assess_generation_template_completeness(
        _run(
            answer_text=(
                "# 보고서 초안\n\n"
                "## 요약\n요약입니다. [RCP-001]\n\n"
                "## 주요 내용\n- 주요 내용입니다.\n\n"
                "## 근거\n- 근거입니다. [RCP-001]"
            ),
            request_metadata={"generation_template": _report_template_snapshot()},
        )
    )

    assert assessment.status == GENERATION_TEMPLATE_COMPLETENESS_WARNING
    assert assessment.required_section_coverage_percent == 100.0
    assert assessment.required_section_citation_coverage_percent == 66.67
    assert assessment.section_checks[2].reason_codes == (
        TEMPLATE_COMPLETENESS_REASON_MISSING_SECTION_CITATION,
    )


def test_template_completeness_marks_no_answer_guardrail_not_evaluated() -> None:
    assessment = assess_generation_template_completeness(
        _run(
            answer_text="제공된 문서 근거만으로는 답변할 수 없습니다.",
            request_metadata={"generation_template": _report_template_snapshot()},
            status="no_answer",
            guardrail_status="no_answer",
        )
    )

    assert assessment.status == GENERATION_TEMPLATE_COMPLETENESS_NOT_EVALUATED
    assert assessment.required_section_coverage_percent == 0.0
    assert assessment.reason_codes == (TEMPLATE_COMPLETENESS_REASON_NO_ANSWER_GUARDRAIL,)


def test_template_completeness_handles_missing_and_unsupported_template_metadata() -> None:
    missing = assess_generation_template_completeness(_run(answer_text="# 답변"))
    unsupported = assess_generation_template_completeness(
        _run(
            answer_text="plain output",
            request_metadata={
                "generation_template": {
                    **_report_template_snapshot(),
                    "output_format": "plain_text",
                }
            },
        )
    )

    assert missing.status == GENERATION_TEMPLATE_COMPLETENESS_NOT_EVALUATED
    assert missing.reason_codes == (TEMPLATE_COMPLETENESS_REASON_NO_TEMPLATE_METADATA,)
    assert unsupported.status == GENERATION_TEMPLATE_COMPLETENESS_NOT_EVALUATED
    assert unsupported.output_format == "plain_text"
    assert unsupported.reason_codes == (TEMPLATE_COMPLETENESS_REASON_UNSUPPORTED_OUTPUT_FORMAT,)


def test_template_completeness_uses_response_metadata_fallback_sections() -> None:
    assessment = assess_generation_template_completeness(
        _run(
            answer_text="# 응답\n\n## answer\n답변\n\n## evidence\n근거",
            response_metadata={
                "template": {
                    "template_key": "fallback",
                    "template_name": "Fallback",
                    "template_version": "v1",
                    "document_type": "summary",
                    "output_format": "markdown",
                    "template_section_keys": ["answer", "evidence", "limits"],
                    "required_template_section_keys": ["answer", "evidence"],
                }
            },
        )
    )

    assert assessment.status == GENERATION_TEMPLATE_COMPLETENESS_PASSED
    assert assessment.template_key == "fallback"
    assert assessment.required_section_count == 2
    assert assessment.required_section_citation_coverage_percent is None


def test_template_completeness_warns_when_template_has_no_required_sections() -> None:
    assessment = assess_generation_template_completeness(
        _run(
            answer_text="# 임의 문서",
            request_metadata={
                "generation_template": {
                    **_report_template_snapshot(),
                    "section_schema": [{"key": "optional", "heading": "선택", "required": False}],
                }
            },
        )
    )

    assert assessment.status == GENERATION_TEMPLATE_COMPLETENESS_WARNING
    assert assessment.reason_codes == (TEMPLATE_COMPLETENESS_REASON_NO_REQUIRED_SECTIONS,)


def test_template_completeness_fails_empty_answer() -> None:
    assessment = assess_generation_template_completeness(
        _run(
            answer_text="",
            request_metadata={"generation_template": _report_template_snapshot()},
        )
    )

    assert assessment.status == GENERATION_TEMPLATE_COMPLETENESS_FAILED
    assert assessment.reason_codes[0] == TEMPLATE_COMPLETENESS_REASON_EMPTY_ANSWER
