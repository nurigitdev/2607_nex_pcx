from types import SimpleNamespace

import pytest

from app.core.chat_document_generation import (
    CHAT_DOCUMENT_GENERATION_EXECUTION_MODE,
    CHAT_DOCUMENT_GENERATION_PROMPT_VERSION,
    ChatDocumentGenerationInput,
    InvalidChatDocumentGenerationError,
    execute_chat_document_generation,
    select_chat_document_generation_template_key,
)
from app.core.direct_generation import DirectGenerationInput
from app.core.generation_runs import (
    GENERATION_GUARDRAIL_ALLOWED,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_STATUS_SUCCEEDED,
)


def _direct_generation_result():
    return SimpleNamespace(
        search_result=SimpleNamespace(search_log_id=101),
        retrieval_package=SimpleNamespace(
            package_key="RCP-DOC-101",
            summary=SimpleNamespace(included_count=1),
        ),
        generation_report=SimpleNamespace(
            run=SimpleNamespace(
                generation_run_id=202,
                answer_text="# 보고서 초안\n\n근거 기반 문서입니다. [RCP-001]",
                status=GENERATION_STATUS_SUCCEEDED,
                guardrail_status=GENERATION_GUARDRAIL_ALLOWED,
            )
        ),
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    (
        ("신규 서비스 제안서를 작성해줘", "proposal"),
        ("회의록을 생성해줘", "meeting_minutes"),
        ("요약서를 작성해줘", "summary"),
        ("보고서를 작성해줘", "report"),
        ("Please draft a proposal", "proposal"),
        ("Please write meeting minutes", "meeting_minutes"),
    ),
)
def test_select_chat_document_generation_template_key_uses_content_keywords(
    content: str,
    expected: str,
) -> None:
    assert select_chat_document_generation_template_key(content) == expected


def test_select_chat_document_generation_template_key_prefers_explicit_request() -> None:
    assert (
        select_chat_document_generation_template_key(
            "보고서를 작성해줘",
            requested_template_key=" proposal ",
        )
        == "proposal"
    )


def test_select_chat_document_generation_template_key_rejects_invalid_values() -> None:
    with pytest.raises(InvalidChatDocumentGenerationError, match="content"):
        select_chat_document_generation_template_key(" ")
    with pytest.raises(InvalidChatDocumentGenerationError, match="generation_template_key"):
        select_chat_document_generation_template_key("보고서", requested_template_key=" ")
    with pytest.raises(InvalidChatDocumentGenerationError, match="generation_template_key"):
        select_chat_document_generation_template_key("보고서", requested_template_key="memo")


def test_execute_chat_document_generation_passes_template_to_direct_generation() -> None:
    captured: dict[str, object] = {}

    def runner(database_url: str, direct_input: DirectGenerationInput, **kwargs):
        captured["database_url"] = database_url
        captured["direct_input"] = direct_input
        return _direct_generation_result()

    result = execute_chat_document_generation(
        "postgresql://db",
        ChatDocumentGenerationInput(
            content="신규 제품 제안서를 작성해줘",
            actor_user_id=3,
            requested_search_scope="company",
            provider_mode=GENERATION_PROVIDER_MODE_MOCK,
            requested_template_key=None,
            profiles=None,
        ),
        direct_generation_runner=runner,
    )

    direct_input = captured["direct_input"]
    assert isinstance(direct_input, DirectGenerationInput)
    assert direct_input.generation_template_key == "proposal"
    assert direct_input.actor_user_id == 3
    assert result.template_key == "proposal"
    assert result.execution_mode == CHAT_DOCUMENT_GENERATION_EXECUTION_MODE
    assert result.prompt_version == CHAT_DOCUMENT_GENERATION_PROMPT_VERSION
    assert result.search_log_id == 101
    assert result.generation_run_id == 202
    assert result.response_metadata["template_key"] == "proposal"
    assert "# 보고서 초안" in result.answer_text
