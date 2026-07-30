import pytest

from app.core.generation_provider_metrics import (
    GENERATION_PROVIDER_METRICS_CONTRACT_VERSION,
    InvalidGenerationProviderMetricsError,
    generation_provider_metrics_payload,
    parse_openai_chat_completion_metrics,
)


def test_parse_openai_chat_completion_metrics_reads_vllm_usage_and_finish_reason() -> None:
    metrics = parse_openai_chat_completion_metrics(
        {
            "id": "chatcmpl-001",
            "object": "chat.completion",
            "created": 1785000000,
            "model": "nvidia/Qwen3.5-122B-A10B-NVFP4",
            "system_fingerprint": "fp-qwen",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "답변입니다. [RCP-001]"},
                }
            ],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 84,
                "total_tokens": 1284,
            },
        },
        provider_name="dgx-vllm",
        requested_model_id="nvidia/Qwen3.5-122B-A10B-NVFP4",
        http_status_code=200,
        elapsed_ms=3400,
        provider_elapsed_ms=3200,
        retry_count=1,
    )
    payload = generation_provider_metrics_payload(metrics)

    assert metrics.contract_version == GENERATION_PROVIDER_METRICS_CONTRACT_VERSION
    assert metrics.succeeded is True
    assert payload["provider_name"] == "dgx-vllm"
    assert payload["response_model_id"] == "nvidia/Qwen3.5-122B-A10B-NVFP4"
    assert payload["response_id"] == "chatcmpl-001"
    assert payload["finish_reason"] == "stop"
    assert payload["input_token_count"] == 1200
    assert payload["output_token_count"] == 84
    assert payload["total_token_count"] == 1284
    assert payload["provider_elapsed_ms"] == 3200
    assert payload["response_metadata"]["choice_count"] == 1
    assert payload["response_metadata"]["system_fingerprint"] == "fp-qwen"


def test_parse_openai_chat_completion_metrics_derives_total_tokens_when_missing() -> None:
    metrics = parse_openai_chat_completion_metrics(
        {
            "model": "qwen",
            "choices": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 7,
            },
        },
        provider_name="mock-vllm",
        http_status_code=204,
    )

    assert metrics.finish_reason is None
    assert metrics.total_token_count == 17
    assert metrics.raw_usage == {"input_tokens": 10, "output_tokens": 7}
    assert metrics.response_metadata["choice_count"] == 0


def test_parse_openai_chat_completion_metrics_reads_error_payload() -> None:
    metrics = parse_openai_chat_completion_metrics(
        {
            "error": {
                "message": "model overloaded",
                "type": "server_error",
                "code": "overloaded",
            }
        },
        provider_name="dgx-vllm",
        requested_model_id="qwen",
        http_status_code=503,
        elapsed_ms=900,
        retry_count=2,
    )

    assert metrics.succeeded is False
    assert metrics.error_code == "overloaded"
    assert metrics.error_message == "model overloaded"
    assert generation_provider_metrics_payload(metrics)["succeeded"] is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"provider_name": ""}, "provider_name"),
        ({"http_status_code": -1}, "http_status_code"),
        ({"elapsed_ms": -1}, "elapsed_ms"),
        ({"provider_elapsed_ms": True}, "provider_elapsed_ms"),
        ({"retry_count": -1}, "retry_count"),
    ),
)
def test_parse_openai_chat_completion_metrics_rejects_invalid_contract_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    options = {"provider_name": "dgx-vllm"} | kwargs

    with pytest.raises(InvalidGenerationProviderMetricsError, match=match):
        parse_openai_chat_completion_metrics({}, **options)


def test_parse_openai_chat_completion_metrics_rejects_invalid_payload_type() -> None:
    with pytest.raises(InvalidGenerationProviderMetricsError, match="payload"):
        parse_openai_chat_completion_metrics(
            [],  # type: ignore[arg-type]
            provider_name="dgx-vllm",
        )
