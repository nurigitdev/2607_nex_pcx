"""General chat answer execution helpers for conversational UX."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from app.core.generation_provider_metrics import generation_provider_metrics_payload
from app.core.generation_providers import (
    DEFAULT_GENERATION_MAX_TOKENS,
    DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_GENERATION_TOP_P,
    GenerationProvider,
    GenerationProviderRequestError,
    build_generation_provider_from_runtime_config,
    generation_chat_request_from_openai_messages,
    generation_provider_runtime_config_from_record,
)
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GenerationProviderConfigRecord,
)

CHAT_GENERAL_ANSWER_PROMPT_VERSION = "chat_general_answer_v1"
CHAT_GENERAL_ANSWER_SYSTEM_INSTRUCTION = (
    "You are NeX-PCX chat. Answer directly without using document retrieval. "
    "If company document evidence is required, say that a grounded search flow should be used."
)
CHAT_GENERAL_MOCK_PROVIDER_NAME = "mock_chat_llm"
CHAT_GENERAL_MOCK_MODEL_ID = "mock-general-chat-v1"
CHAT_GENERAL_MOCK_FINISH_REASON = "mock_completed"
CHAT_GENERAL_PROVIDER_MODES = {
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
}


@dataclass(frozen=True)
class ChatGeneralAnswerInput:
    content: str
    provider_mode: str = GENERATION_PROVIDER_MODE_MOCK
    provider_name: str = CHAT_GENERAL_MOCK_PROVIDER_NAME
    model_id: str = CHAT_GENERAL_MOCK_MODEL_ID
    max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS
    temperature: float = DEFAULT_GENERATION_TEMPERATURE
    top_p: float = DEFAULT_GENERATION_TOP_P
    trace_id: str | None = None
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatGeneralAnswerResult:
    answer_text: str
    provider_name: str
    provider_mode: str
    model_id: str
    prompt_version: str
    finish_reason: str | None
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    elapsed_ms: int
    request_metadata: dict[str, Any]
    response_metadata: dict[str, Any]


class InvalidChatGenerationError(ValueError):
    """Raised when a general chat generation request is invalid."""


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidChatGenerationError(f"{field_name} must not be blank")
    return normalized


def _validate_positive(value: int, field_name: str) -> int:
    if value <= 0:
        raise InvalidChatGenerationError(f"{field_name} must be greater than 0")
    return value


def _validate_temperature(value: float) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0 or normalized > 2:
        raise InvalidChatGenerationError("temperature must be between 0 and 2")
    return normalized


def _validate_top_p(value: float) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0 or normalized > 1:
        raise InvalidChatGenerationError("top_p must be greater than 0 and less than or equal to 1")
    return normalized


def _estimate_token_count(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return len(normalized.split())


def validate_chat_general_answer_input(
    answer_input: ChatGeneralAnswerInput,
) -> ChatGeneralAnswerInput:
    provider_mode = answer_input.provider_mode.strip().lower()
    if provider_mode not in CHAT_GENERAL_PROVIDER_MODES:
        raise InvalidChatGenerationError("provider_mode is not supported")
    if not isinstance(answer_input.runtime_metadata, Mapping):
        raise InvalidChatGenerationError("runtime_metadata must be a JSON object")
    return ChatGeneralAnswerInput(
        content=_validate_nonblank(answer_input.content, "content"),
        provider_mode=provider_mode,
        provider_name=_validate_nonblank(answer_input.provider_name, "provider_name"),
        model_id=_validate_nonblank(answer_input.model_id, "model_id"),
        max_tokens=_validate_positive(answer_input.max_tokens, "max_tokens"),
        temperature=_validate_temperature(answer_input.temperature),
        top_p=_validate_top_p(answer_input.top_p),
        trace_id=(
            _validate_nonblank(answer_input.trace_id, "trace_id")
            if answer_input.trace_id is not None
            else None
        ),
        runtime_metadata=dict(answer_input.runtime_metadata),
    )


def _openai_messages(content: str) -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": CHAT_GENERAL_ANSWER_SYSTEM_INSTRUCTION},
        {"role": "user", "content": content},
    )


def _request_metadata(answer_input: ChatGeneralAnswerInput) -> dict[str, Any]:
    return {
        "prompt_version": CHAT_GENERAL_ANSWER_PROMPT_VERSION,
        "messages": list(_openai_messages(answer_input.content)),
        "provider_mode": answer_input.provider_mode,
        "provider_name": answer_input.provider_name,
        "model_id": answer_input.model_id,
        "max_tokens": answer_input.max_tokens,
        "temperature": answer_input.temperature,
        "top_p": answer_input.top_p,
        "runtime_metadata": dict(answer_input.runtime_metadata),
    }


def _execute_mock_chat_general_answer(
    answer_input: ChatGeneralAnswerInput,
) -> ChatGeneralAnswerResult:
    answer_text = (
        "일반 답변 초안입니다.\n\n"
        f"'{answer_input.content}' 요청은 문서 검색 없이 일반 LLM 응답 경로로 처리했습니다. "
        "회사 내부 문서 근거가 필요한 질문이라면 근거 기반 답변 흐름으로 다시 실행하세요."
    )
    input_token_count = _estimate_token_count(CHAT_GENERAL_ANSWER_SYSTEM_INSTRUCTION) + (
        _estimate_token_count(answer_input.content)
    )
    output_token_count = _estimate_token_count(answer_text)
    return ChatGeneralAnswerResult(
        answer_text=answer_text,
        provider_name=answer_input.provider_name,
        provider_mode=GENERATION_PROVIDER_MODE_MOCK,
        model_id=answer_input.model_id,
        prompt_version=CHAT_GENERAL_ANSWER_PROMPT_VERSION,
        finish_reason=CHAT_GENERAL_MOCK_FINISH_REASON,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        total_token_count=input_token_count + output_token_count,
        elapsed_ms=0,
        request_metadata=_request_metadata(answer_input),
        response_metadata={
            "deterministic": True,
            "execution_mode": "general_llm_mock",
        },
    )


def _provider_answer_input(
    answer_input: ChatGeneralAnswerInput,
    provider_config: GenerationProviderConfigRecord,
) -> ChatGeneralAnswerInput:
    return ChatGeneralAnswerInput(
        content=answer_input.content,
        provider_mode=provider_config.provider_mode,
        provider_name=provider_config.provider_name,
        model_id=provider_config.model_id,
        max_tokens=provider_config.max_tokens,
        temperature=provider_config.temperature,
        top_p=provider_config.top_p,
        trace_id=answer_input.trace_id,
        runtime_metadata=answer_input.runtime_metadata,
    )


def execute_chat_general_answer(
    answer_input: ChatGeneralAnswerInput,
    *,
    provider_config: GenerationProviderConfigRecord | None = None,
    api_key: str | None = None,
    provider_client: GenerationProvider | None = None,
) -> ChatGeneralAnswerResult:
    validated = validate_chat_general_answer_input(answer_input)
    if validated.provider_mode == GENERATION_PROVIDER_MODE_MOCK:
        return _execute_mock_chat_general_answer(validated)
    if provider_config is None:
        raise InvalidChatGenerationError("provider_config is required for remote provider mode")

    provider_answer_input = validate_chat_general_answer_input(
        _provider_answer_input(validated, provider_config)
    )
    runtime_config = generation_provider_runtime_config_from_record(
        provider_config,
        api_key=api_key,
    )
    owns_provider_client = provider_client is None
    generation_provider = provider_client or build_generation_provider_from_runtime_config(
        runtime_config,
        provider_name=provider_config.provider_name,
    )
    chat_request = generation_chat_request_from_openai_messages(
        _openai_messages(provider_answer_input.content),
        model_id=provider_answer_input.model_id,
        max_tokens=provider_answer_input.max_tokens,
        temperature=provider_answer_input.temperature,
        top_p=provider_answer_input.top_p,
        trace_id=provider_answer_input.trace_id,
        extra_body=provider_config.runtime_options.get("extra_body", {}),
        runtime_metadata=provider_answer_input.runtime_metadata,
    )
    try:
        response = generation_provider.complete(chat_request)
    except GenerationProviderRequestError as exc:
        raise InvalidChatGenerationError(str(exc)) from exc
    finally:
        if owns_provider_client and hasattr(generation_provider, "close"):
            generation_provider.close()  # type: ignore[attr-defined]

    return ChatGeneralAnswerResult(
        answer_text=response.answer_text,
        provider_name=provider_config.provider_name,
        provider_mode=provider_config.provider_mode,
        model_id=response.provider_model_id or provider_answer_input.model_id,
        prompt_version=CHAT_GENERAL_ANSWER_PROMPT_VERSION,
        finish_reason=response.finish_reason,
        input_token_count=response.input_token_count,
        output_token_count=response.output_token_count,
        total_token_count=response.total_token_count,
        elapsed_ms=response.elapsed_ms,
        request_metadata=_request_metadata(provider_answer_input),
        response_metadata={
            "execution_mode": "general_llm_remote",
            "response_id": response.response_id,
            "provider_metrics": generation_provider_metrics_payload(response.provider_metrics),
            **dict(response.response_metadata),
        },
    )
