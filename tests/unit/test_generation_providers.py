import json
from datetime import UTC, datetime

import httpx
import pytest

from app.core.generation_providers import (
    DEFAULT_GENERATION_MODEL_ID,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GenerationChatCompletionRequest,
    GenerationChatMessage,
    GenerationProviderRequestError,
    GenerationProviderRuntimeConfig,
    InvalidGenerationProviderError,
    OpenAICompatibleGenerationProviderClient,
    build_generation_provider_from_runtime_config,
    extract_openai_chat_completion_answer_text,
    generation_chat_request_from_openai_messages,
    generation_provider_runtime_config_from_record,
    generation_provider_runtime_config_from_settings,
    normalize_generation_provider_runtime_config,
    openai_chat_completion_request_payload,
    validate_generation_chat_completion_request,
)
from app.core.generation_runs import GenerationProviderConfigRecord

NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def _request(**overrides) -> GenerationChatCompletionRequest:
    values = {
        "messages": (
            GenerationChatMessage(role="system", content="Ground answers only."),
            GenerationChatMessage(role="user", content="사내 보안 규정은?"),
        ),
        "model_id": DEFAULT_GENERATION_MODEL_ID,
        "max_tokens": 256,
        "temperature": 0.1,
        "top_p": 0.8,
        "stop_sequences": ("</answer>",),
        "trace_id": " trace-001 ",
        "runtime_metadata": {"search_log_id": 24},
    }
    values.update(overrides)
    return GenerationChatCompletionRequest(**values)


def _provider_record(**overrides) -> GenerationProviderConfigRecord:
    values = {
        "provider_config_id": 1,
        "provider_name": "dgx-vllm",
        "provider_mode": GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
        "provider_base_url": "http://dgx.local:8000/",
        "model_id": DEFAULT_GENERATION_MODEL_ID,
        "is_default": True,
        "is_active": True,
        "request_timeout_seconds": 45,
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "runtime_options": {"headers": {"X-Trace-Scope": "generation"}},
        "created_by": "pytest",
        "created_by_user_id": 1,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return GenerationProviderConfigRecord(**values)


def test_generation_runtime_config_from_settings_builds_remote_client() -> None:
    class SettingsStub:
        generation_provider_mode = " REMOTE_OPENAI_COMPATIBLE "
        remote_generation_provider_url = "http://dgx.local:8000/"
        remote_generation_provider_timeout_seconds = 77.5
        remote_generation_provider_api_key = "secret-token"
        generation_model_id = "nvidia/Qwen3.5-122B-A10B-NVFP4"
        generation_max_tokens = 2048
        generation_temperature = 0.3
        generation_top_p = 0.95

    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    config = generation_provider_runtime_config_from_settings(SettingsStub())
    provider = build_generation_provider_from_runtime_config(
        config,
        provider_name="dgx-vllm",
        http_client=http_client,
    )

    assert config.remote_base_url == "http://dgx.local:8000"
    assert config.remote_timeout_seconds == 77.5
    assert config.api_key == "secret-token"
    assert isinstance(provider, OpenAICompatibleGenerationProviderClient)
    assert provider.base_url == "http://dgx.local:8000"
    assert provider.headers["Authorization"] == "Bearer secret-token"


def test_generation_runtime_config_from_record_uses_runtime_headers() -> None:
    config = generation_provider_runtime_config_from_record(_provider_record(), api_key="api-key")

    assert config.mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    assert config.remote_base_url == "http://dgx.local:8000"
    assert config.remote_timeout_seconds == 45
    assert config.remote_headers == {"X-Trace-Scope": "generation"}
    assert config.api_key == "api-key"
    assert config.max_tokens == 512


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (GenerationProviderRuntimeConfig(mode="local"), "Unsupported"),
        (
            GenerationProviderRuntimeConfig(mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE),
            "remote_generation_provider_url",
        ),
        (
            GenerationProviderRuntimeConfig(
                mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                remote_base_url="http://dgx",
                remote_timeout_seconds=0,
            ),
            "timeout",
        ),
        (GenerationProviderRuntimeConfig(model_id=" "), "model_id"),
        (GenerationProviderRuntimeConfig(max_tokens=0), "max_tokens"),
        (GenerationProviderRuntimeConfig(temperature=2.1), "temperature"),
        (GenerationProviderRuntimeConfig(top_p=0), "top_p"),
        (GenerationProviderRuntimeConfig(remote_headers={"Bad:Header": "value"}), "header"),
        (
            GenerationProviderRuntimeConfig(remote_headers={"X-Good": "bad\nvalue"}),
            "header value",
        ),
        (GenerationProviderRuntimeConfig(api_key="bad\nkey"), "api_key"),
    ),
)
def test_normalize_generation_runtime_config_rejects_invalid_values(
    config: GenerationProviderRuntimeConfig,
    message: str,
) -> None:
    with pytest.raises(InvalidGenerationProviderError, match=message):
        normalize_generation_provider_runtime_config(config)


def test_generation_runtime_config_from_record_rejects_bad_header_option() -> None:
    with pytest.raises(InvalidGenerationProviderError, match="headers"):
        generation_provider_runtime_config_from_record(
            _provider_record(runtime_options={"headers": ["bad"]})
        )


def test_build_generation_provider_rejects_mock_mode() -> None:
    with pytest.raises(InvalidGenerationProviderError, match="mock generation provider"):
        build_generation_provider_from_runtime_config(GenerationProviderRuntimeConfig())


def test_chat_completion_request_validation_and_payload() -> None:
    request = validate_generation_chat_completion_request(
        _request(
            messages=(GenerationChatMessage(role=" USER ", content=" hello "),),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )
    payload = openai_chat_completion_request_payload(request)

    assert request.messages[0] == GenerationChatMessage(role="user", content="hello")
    assert request.trace_id == "trace-001"
    assert request.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert payload == {
        "model": DEFAULT_GENERATION_MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 256,
        "temperature": 0.1,
        "top_p": 0.8,
        "stream": False,
        "stop": ["</answer>"],
        "user": "trace-001",
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize(
    ("provider_request", "message"),
    (
        (_request(messages=()), "messages"),
        (_request(messages=(GenerationChatMessage(role="bad", content="x"),)), "role"),
        (_request(messages=(GenerationChatMessage(role="user", content=" "),)), "content"),
        (_request(stop_sequences=(" ",)), "stop sequence"),
        (_request(trace_id="bad\ntrace"), "trace_id"),
        (_request(max_tokens=0), "max_tokens"),
        (_request(temperature=-0.1), "temperature"),
        (_request(top_p=1.1), "top_p"),
        (_request(extra_body={"model": "override"}), "reserved"),
        (_request(extra_body={" ": "value"}), "extra_body key"),
    ),
)
def test_chat_completion_request_validation_rejects_bad_values(
    provider_request: GenerationChatCompletionRequest,
    message: str,
) -> None:
    with pytest.raises(InvalidGenerationProviderError, match=message):
        validate_generation_chat_completion_request(provider_request)


def test_generation_chat_request_from_openai_messages_normalizes_prompt_preview_payload() -> None:
    request = generation_chat_request_from_openai_messages(
        [
            {"role": "system", "content": "Use citations."},
            {"role": "user", "content": "질문"},
        ],
        model_id="qwen",
        max_tokens=128,
        temperature=0,
        top_p=1,
        trace_id="run-24",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert request.model_id == "qwen"
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.trace_id == "run-24"
    assert request.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_openai_compatible_client_posts_chat_completion_and_parses_metrics() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer token"
        assert request.headers["x-provider"] == "dgx"
        assert payload["model"] == DEFAULT_GENERATION_MODEL_ID
        assert payload["stream"] is False
        assert payload["user"] == "trace-001"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-001",
                "object": "chat.completion",
                "created": 1785000000,
                "model": DEFAULT_GENERATION_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "답변입니다. [RCP-001]"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": 20,
                    "total_tokens": 320,
                },
            },
        )

    client = OpenAICompatibleGenerationProviderClient(
        "http://dgx.local:8000/",
        provider_name="dgx-vllm",
        headers={"X-Provider": "dgx"},
        api_key="token",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.complete(_request())
    client.close()

    assert len(seen_requests) == 1
    assert response.answer_text == "답변입니다. [RCP-001]"
    assert response.finish_reason == "stop"
    assert response.provider_model_id == DEFAULT_GENERATION_MODEL_ID
    assert response.response_id == "chatcmpl-001"
    assert response.input_token_count == 300
    assert response.output_token_count == 20
    assert response.total_token_count == 320
    assert response.response_metadata["provider_name"] == "dgx-vllm"
    assert response.response_metadata["metrics"]["succeeded"] is True


def test_openai_answer_extraction_supports_content_parts() -> None:
    answer = extract_openai_chat_completion_answer_text(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "첫 문장. "},
                            {"type": "text", "text": "둘째 문장."},
                        ]
                    }
                }
            ]
        }
    )

    assert answer == "첫 문장. 둘째 문장."


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": []}}]},
    ),
)
def test_openai_answer_extraction_rejects_missing_text(payload: dict[str, object]) -> None:
    with pytest.raises(InvalidGenerationProviderError):
        extract_openai_chat_completion_answer_text(payload)


def test_openai_compatible_client_exposes_http_error_metrics() -> None:
    client = OpenAICompatibleGenerationProviderClient(
        "http://dgx.local:8000",
        provider_name="dgx-vllm",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    503,
                    json={
                        "error": {
                            "message": "model is warming up",
                            "type": "server_error",
                            "code": "overloaded",
                        }
                    },
                )
            )
        ),
    )

    with pytest.raises(GenerationProviderRequestError, match="HTTP 503") as exc_info:
        client.complete(_request())

    assert exc_info.value.metrics is not None
    assert exc_info.value.metrics.succeeded is False
    assert exc_info.value.metrics.http_status_code == 503
    assert exc_info.value.metrics.error_code == "overloaded"
    assert exc_info.value.payload["error"]["message"] == "model is warming up"


def test_openai_compatible_client_wraps_non_object_and_invalid_json_responses() -> None:
    non_object_client = OpenAICompatibleGenerationProviderClient(
        "http://dgx.local:8000",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
        ),
    )
    invalid_json_client = OpenAICompatibleGenerationProviderClient(
        "http://dgx.local:8000",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not-json"))
        ),
    )

    with pytest.raises(GenerationProviderRequestError, match="JSON object") as non_object_error:
        non_object_client.complete(_request())
    with pytest.raises(GenerationProviderRequestError, match="not valid JSON") as invalid_error:
        invalid_json_client.complete(_request())

    assert non_object_error.value.metrics is not None
    assert non_object_error.value.metrics.error_code == "invalid_response"
    assert invalid_error.value.metrics is not None
    assert invalid_error.value.metrics.error_code == "invalid_json"


def test_openai_compatible_client_wraps_transport_errors_and_invalid_success_payload() -> None:
    transport_error_client = OpenAICompatibleGenerationProviderClient(
        "http://dgx.local:8000",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(httpx.ConnectError("offline"))
            )
        ),
    )
    bad_success_client = OpenAICompatibleGenerationProviderClient(
        "http://dgx.local:8000",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "model": DEFAULT_GENERATION_MODEL_ID,
                        "choices": [{"message": {}}],
                    },
                )
            )
        ),
    )

    with pytest.raises(GenerationProviderRequestError, match="request failed"):
        transport_error_client.complete(_request())
    with pytest.raises(GenerationProviderRequestError, match="Invalid generation provider"):
        bad_success_client.complete(_request())
