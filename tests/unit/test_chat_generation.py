from datetime import UTC, datetime

import pytest

from app.core.chat_generation import (
    CHAT_GENERAL_ANSWER_PROMPT_VERSION,
    ChatGeneralAnswerInput,
    InvalidChatGenerationError,
    execute_chat_general_answer,
    validate_chat_general_answer_input,
)
from app.core.generation_provider_metrics import parse_openai_chat_completion_metrics
from app.core.generation_providers import (
    DEFAULT_GENERATION_MODEL_ID,
    GenerationChatCompletionRequest,
    GenerationChatCompletionResponse,
    GenerationProviderRequestError,
)
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GenerationProviderConfigRecord,
)

NOW = datetime(2026, 7, 28, 9, 0, tzinfo=UTC)


def _provider_record(**overrides) -> GenerationProviderConfigRecord:
    values = {
        "provider_config_id": 10,
        "provider_name": "dgx-vllm",
        "provider_mode": GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
        "provider_base_url": "http://dgx.local:12000",
        "model_id": DEFAULT_GENERATION_MODEL_ID,
        "is_default": True,
        "is_active": True,
        "request_timeout_seconds": 180,
        "max_tokens": 4096,
        "temperature": 0.15,
        "top_p": 0.88,
        "runtime_options": {
            "headers": {"X-Route": "chat"},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "created_by": "pytest",
        "created_by_user_id": 1,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return GenerationProviderConfigRecord(**values)


class _SuccessfulChatProvider:
    def __init__(self) -> None:
        self.requests: list[GenerationChatCompletionRequest] = []
        self.closed = False

    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        self.requests.append(request)
        answer_text = "remote general chat answer"
        metrics = parse_openai_chat_completion_metrics(
            {
                "id": "chatcmpl-general",
                "object": "chat.completion",
                "model": request.model_id,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": answer_text},
                    }
                ],
                "usage": {
                    "prompt_tokens": 33,
                    "completion_tokens": 7,
                    "total_tokens": 40,
                },
            },
            provider_name="dgx-vllm",
            provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            requested_model_id=request.model_id,
            http_status_code=200,
            elapsed_ms=22,
            provider_elapsed_ms=20,
        )
        return GenerationChatCompletionResponse(
            answer_text=answer_text,
            finish_reason="stop",
            provider_model_id=request.model_id,
            response_id="chatcmpl-general",
            input_token_count=33,
            output_token_count=7,
            total_token_count=40,
            elapsed_ms=22,
            provider_metrics=metrics,
            response_metadata={"provider_name": "dgx-vllm"},
            raw_response={},
        )

    def close(self) -> None:
        self.closed = True


class _FailingChatProvider:
    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        raise GenerationProviderRequestError("provider timeout")


def test_execute_chat_general_answer_mock_returns_deterministic_answer() -> None:
    result = execute_chat_general_answer(
        ChatGeneralAnswerInput(
            content="오늘 회의를 잘 준비하는 방법은?",
            runtime_metadata={"chat_session_id": 1},
        )
    )

    assert "일반 답변 초안입니다." in result.answer_text
    assert result.provider_mode == GENERATION_PROVIDER_MODE_MOCK
    assert result.prompt_version == CHAT_GENERAL_ANSWER_PROMPT_VERSION
    assert result.finish_reason == "mock_completed"
    assert result.input_token_count is not None and result.input_token_count > 0
    assert result.output_token_count is not None and result.output_token_count > 0
    assert result.total_token_count == result.input_token_count + result.output_token_count
    assert result.request_metadata["runtime_metadata"] == {"chat_session_id": 1}
    assert result.response_metadata["execution_mode"] == "general_llm_mock"


def test_execute_chat_general_answer_remote_uses_provider_config_and_request_shape() -> None:
    provider = _SuccessfulChatProvider()
    provider_config = _provider_record()

    result = execute_chat_general_answer(
        ChatGeneralAnswerInput(
            content="일반적인 회의 준비 체크리스트를 알려줘",
            provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            trace_id=" chat-session-7 ",
            runtime_metadata={"chat_session_id": 7},
        ),
        provider_config=provider_config,
        api_key="secret",
        provider_client=provider,
    )

    request = provider.requests[0]
    assert result.answer_text == "remote general chat answer"
    assert result.provider_name == "dgx-vllm"
    assert result.provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    assert result.model_id == DEFAULT_GENERATION_MODEL_ID
    assert result.input_token_count == 33
    assert result.total_token_count == 40
    assert result.request_metadata["max_tokens"] == 4096
    assert result.response_metadata["execution_mode"] == "general_llm_remote"
    assert result.response_metadata["response_id"] == "chatcmpl-general"
    assert request.messages[0].role == "system"
    assert request.messages[1].content == "일반적인 회의 준비 체크리스트를 알려줘"
    assert request.trace_id == "chat-session-7"
    assert request.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert provider.closed is False


def test_execute_chat_general_answer_remote_requires_provider_config() -> None:
    with pytest.raises(InvalidChatGenerationError, match="provider_config"):
        execute_chat_general_answer(
            ChatGeneralAnswerInput(
                content="remote 설정이 필요합니다",
                provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            )
        )


def test_execute_chat_general_answer_wraps_provider_request_error() -> None:
    with pytest.raises(InvalidChatGenerationError, match="provider timeout"):
        execute_chat_general_answer(
            ChatGeneralAnswerInput(
                content="remote provider 실패",
                provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            ),
            provider_config=_provider_record(),
            provider_client=_FailingChatProvider(),
        )


@pytest.mark.parametrize(
    ("answer_input", "message"),
    (
        (ChatGeneralAnswerInput(content=" "), "content"),
        (ChatGeneralAnswerInput(content="질문", provider_mode="local"), "provider_mode"),
        (ChatGeneralAnswerInput(content="질문", provider_name=" "), "provider_name"),
        (ChatGeneralAnswerInput(content="질문", model_id=" "), "model_id"),
        (ChatGeneralAnswerInput(content="질문", max_tokens=0), "max_tokens"),
        (ChatGeneralAnswerInput(content="질문", temperature=-0.1), "temperature"),
        (ChatGeneralAnswerInput(content="질문", temperature=float("nan")), "temperature"),
        (ChatGeneralAnswerInput(content="질문", top_p=0), "top_p"),
        (ChatGeneralAnswerInput(content="질문", top_p=float("inf")), "top_p"),
        (ChatGeneralAnswerInput(content="질문", trace_id=" "), "trace_id"),
        (ChatGeneralAnswerInput(content="질문", runtime_metadata=[]), "runtime_metadata"),
    ),
)
def test_validate_chat_general_answer_input_rejects_invalid_values(
    answer_input: ChatGeneralAnswerInput,
    message: str,
) -> None:
    with pytest.raises(InvalidChatGenerationError, match=message):
        validate_chat_general_answer_input(answer_input)
