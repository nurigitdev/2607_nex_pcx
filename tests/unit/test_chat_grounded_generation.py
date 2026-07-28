from types import SimpleNamespace

import pytest

from app.core.bm25_search import BM25_SEARCH_PROFILE_NAME
from app.core.chat_grounded_generation import (
    CHAT_GROUNDED_ANSWER_EXECUTION_MODE,
    CHAT_GROUNDED_ANSWER_PROMPT_VERSION,
    ChatGroundedAnswerInput,
    InvalidChatGroundedGenerationError,
    execute_chat_grounded_answer,
    validate_chat_grounded_answer_input,
)
from app.core.direct_generation import DirectGenerationInput
from app.core.generation_runs import (
    GENERATION_GUARDRAIL_ALLOWED,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GENERATION_STATUS_SUCCEEDED,
)


def _direct_generation_result(answer_text: str | None = "근거 기반 답변입니다. [RCP-001]"):
    return SimpleNamespace(
        search_result=SimpleNamespace(search_log_id=77),
        retrieval_package=SimpleNamespace(
            package_key="RCP-77",
            summary=SimpleNamespace(included_count=2),
        ),
        generation_report=SimpleNamespace(
            run=SimpleNamespace(
                generation_run_id=88,
                answer_text=answer_text,
                status=GENERATION_STATUS_SUCCEEDED,
                guardrail_status=GENERATION_GUARDRAIL_ALLOWED,
            )
        ),
    )


def test_execute_chat_grounded_answer_runs_direct_generation_with_bm25_default() -> None:
    captured: dict[str, object] = {}

    def runner(database_url: str, direct_input: DirectGenerationInput, **kwargs):
        captured["database_url"] = database_url
        captured["direct_input"] = direct_input
        captured["api_key"] = kwargs.get("api_key")
        return _direct_generation_result()

    result = execute_chat_grounded_answer(
        "postgresql://db",
        ChatGroundedAnswerInput(
            content="사규 근거를 찾아서 답변해줘",
            actor_user_id=5,
            requested_search_scope="team",
            provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            profiles=None,
        ),
        api_key="secret",
        direct_generation_runner=runner,
    )

    direct_input = captured["direct_input"]
    assert isinstance(direct_input, DirectGenerationInput)
    assert direct_input.query_text == "사규 근거를 찾아서 답변해줘"
    assert direct_input.actor_user_id == 5
    assert direct_input.requested_search_scope == "team"
    assert direct_input.provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    assert direct_input.profiles == (BM25_SEARCH_PROFILE_NAME,)
    assert captured["api_key"] == "secret"
    assert result.answer_text == "근거 기반 답변입니다. [RCP-001]"
    assert result.search_log_id == 77
    assert result.generation_run_id == 88
    assert result.retrieval_context_included_count == 2
    assert result.execution_mode == CHAT_GROUNDED_ANSWER_EXECUTION_MODE
    assert result.prompt_version == CHAT_GROUNDED_ANSWER_PROMPT_VERSION
    assert result.response_metadata["retrieval_package_key"] == "RCP-77"


def test_execute_chat_grounded_answer_uses_fallback_text_when_answer_is_empty() -> None:
    result = execute_chat_grounded_answer(
        "postgresql://db",
        ChatGroundedAnswerInput(content="근거 답변", actor_user_id=1),
        direct_generation_runner=lambda *args, **kwargs: _direct_generation_result(" "),
    )

    assert "표시할 답변 본문이 없습니다" in result.answer_text


@pytest.mark.parametrize(
    ("grounded_input", "message"),
    (
        (ChatGroundedAnswerInput(content="", actor_user_id=1), "content"),
        (ChatGroundedAnswerInput(content="근거", actor_user_id=None), "actor_user_id"),
        (ChatGroundedAnswerInput(content="근거", actor_user_id=0), "actor_user_id"),
        (
            ChatGroundedAnswerInput(
                content="근거",
                actor_user_id=1,
                provider_mode="local",
            ),
            "provider_mode",
        ),
        (ChatGroundedAnswerInput(content="근거", actor_user_id=1, top_k=0), "top_k"),
        (
            ChatGroundedAnswerInput(content="근거", actor_user_id=1, profiles=()),
            "profiles",
        ),
        (
            ChatGroundedAnswerInput(
                content="근거",
                actor_user_id=1,
                profiles=("bm25_keyword", "bm25_keyword"),
            ),
            "profiles",
        ),
        (
            ChatGroundedAnswerInput(
                content="근거",
                actor_user_id=1,
                generation_template_key=" ",
            ),
            "generation_template_key",
        ),
        (
            ChatGroundedAnswerInput(content="근거", actor_user_id=1, max_context_chars=0),
            "max_context_chars",
        ),
        (ChatGroundedAnswerInput(content="근거", actor_user_id=1, max_items=0), "max_items"),
    ),
)
def test_validate_chat_grounded_answer_input_rejects_invalid_values(
    grounded_input: ChatGroundedAnswerInput,
    message: str,
) -> None:
    with pytest.raises(InvalidChatGroundedGenerationError, match=message):
        validate_chat_grounded_answer_input(grounded_input)


def test_validate_chat_grounded_answer_input_normalizes_mock_values() -> None:
    validated = validate_chat_grounded_answer_input(
        ChatGroundedAnswerInput(
            content=" 근거 답변 ",
            actor_user_id=1,
            provider_mode=" MOCK ",
            profiles=(" bm25_keyword ",),
        )
    )

    assert validated.content == "근거 답변"
    assert validated.provider_mode == GENERATION_PROVIDER_MODE_MOCK
    assert validated.profiles == (BM25_SEARCH_PROFILE_NAME,)
