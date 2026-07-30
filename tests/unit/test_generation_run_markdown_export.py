from datetime import UTC, datetime

from app.core.generation_runs import GenerationRunCitationRecord, GenerationRunRecord
from app.main import (
    _generation_run_export_datetime,
    _generation_run_export_template,
    _generation_run_markdown_export,
)

NOW = datetime(2026, 7, 27, 12, 30, tzinfo=UTC)


def _run(
    *,
    request_metadata: dict[str, object] | None = None,
    response_metadata: dict[str, object] | None = None,
    guardrail_metadata: dict[str, object] | None = None,
) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_run_id=42,
        search_log_id=24,
        retrieval_package_key="pkg-generation-42",
        generation_template_id=None,
        provider_config_id=7,
        provider_name="mock_qwen35_122b_a10b_nvfp4",
        provider_mode="mock",
        model_id="nvidia/Qwen3.5-122B-A10B-NVFP4",
        prompt_version="grounded_answer_v1",
        prompt_hash="prompt-hash",
        context_hash="context-hash",
        status="succeeded",
        guardrail_status="allowed",
        retrieval_confidence_status="answerable",
        citation_readiness_status="ready",
        query_text="내부 규정 요약",
        answer_text="요약 답변입니다. [RCP-001]",
        finish_reason="mock_completed",
        input_token_count=100,
        output_token_count=20,
        total_token_count=120,
        elapsed_ms=12,
        request_metadata=request_metadata or {},
        response_metadata=response_metadata or {},
        guardrail_metadata=guardrail_metadata or {},
        error_message=None,
        created_by="pytest",
        created_by_user_id=1,
        started_at=NOW,
        finished_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _citation() -> GenerationRunCitationRecord:
    return GenerationRunCitationRecord(
        generation_run_citation_id=9,
        generation_run_id=42,
        citation_key="RCP-001",
        citation_index=1,
        search_log_result_id=None,
        chunk_id=None,
        document_id=None,
        file_id=None,
        source_label="",
        source_anchor={},
        citation_payload={
            "document_title": "업무 규정",
            "chunk_id": 77,
            "document_id": 88,
            "file_id": 99,
        },
        was_cited=False,
        created_at=NOW,
    )


def test_generation_markdown_export_uses_request_template_fallback_without_citations() -> None:
    run = _run(
        request_metadata={
            "generation_template": {"template_name": "요약문"},
            "template_key": "summary",
            "template_version": "v2",
            "document_type": "summary",
            "output_format": "markdown",
        },
        guardrail_metadata={"answer_quality_status": "warning"},
    )

    template = _generation_run_export_template(run)
    markdown = _generation_run_markdown_export(run, ())

    assert template == {
        "template_key": "summary",
        "template_name": "요약문",
        "template_version": "v2",
        "document_type": "summary",
        "output_format": "markdown",
    }
    assert _generation_run_export_datetime(None) == "-"
    assert "Template: 요약문 (summary) v2" in markdown
    assert "Answer Quality: warning" in markdown
    assert "- No citation trace was stored." in markdown


def test_generation_markdown_export_uses_default_template_and_citation_fallbacks() -> None:
    run = _run(
        response_metadata={"answer_quality": {"status": "passed"}},
    )

    template = _generation_run_export_template(run)
    markdown = _generation_run_markdown_export(run, (_citation(),))

    assert template == {
        "template_key": "-",
        "template_version": "-",
        "document_type": "-",
        "output_format": "-",
    }
    assert _generation_run_export_datetime(NOW) == "2026-07-27 12:30:00"
    assert "- [RCP-001] 업무 규정" in markdown
    assert "Used In Answer: no" in markdown
    assert "Search Result ID: -" in markdown
    assert "Chunk ID: 77" in markdown
    assert "Source Anchor: `-`" in markdown
