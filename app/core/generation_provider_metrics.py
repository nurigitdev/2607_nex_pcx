"""Metrics parser for OpenAI-compatible generation provider responses."""

from dataclasses import dataclass, field
from typing import Any

GENERATION_PROVIDER_METRICS_CONTRACT_VERSION = "generation_provider_metrics_v1"
GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE = "remote_openai_compatible"


@dataclass(frozen=True)
class GenerationProviderMetrics:
    contract_version: str
    provider_name: str
    provider_mode: str
    requested_model_id: str | None
    response_model_id: str | None
    response_id: str | None
    http_status_code: int | None
    finish_reason: str | None
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    elapsed_ms: int | None
    provider_elapsed_ms: int | None
    retry_count: int
    error_code: str | None
    error_message: str | None
    raw_usage: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        http_ok = self.http_status_code is None or 200 <= self.http_status_code < 300
        return http_ok and self.error_message is None


class InvalidGenerationProviderMetricsError(ValueError):
    """Raised when provider metrics cannot be parsed safely."""


def _validate_optional_non_negative_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value:
        raise InvalidGenerationProviderMetricsError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise InvalidGenerationProviderMetricsError(
            f"{field_name} must be greater than or equal to 0"
        )
    return normalized


def _validate_non_negative_int(value: int, field_name: str) -> int:
    normalized = _validate_optional_non_negative_int(value, field_name)
    assert normalized is not None
    return normalized


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if value is None:
            continue
        return _validate_optional_non_negative_int(value, key)
    return None


def _finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    return _optional_text(first_choice.get("finish_reason"))


def _error_payload(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None, None
    error_code = _optional_text(error.get("code")) or _optional_text(error.get("type"))
    error_message = _optional_text(error.get("message"))
    return error_code, error_message


def _response_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    return {
        "object": payload.get("object"),
        "created": payload.get("created"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "service_tier": payload.get("service_tier"),
        "choice_count": len(choices) if isinstance(choices, list) else 0,
    }


def parse_openai_chat_completion_metrics(
    payload: dict[str, Any],
    *,
    provider_name: str,
    requested_model_id: str | None = None,
    provider_mode: str = GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    http_status_code: int | None = None,
    elapsed_ms: int | None = None,
    provider_elapsed_ms: int | None = None,
    retry_count: int = 0,
    error_code: str | None = None,
    error_message: str | None = None,
) -> GenerationProviderMetrics:
    """Parse vLLM/OpenAI-compatible chat completion response metrics."""

    if not isinstance(payload, dict):
        raise InvalidGenerationProviderMetricsError("payload must be a dictionary")
    normalized_provider_name = provider_name.strip()
    if not normalized_provider_name:
        raise InvalidGenerationProviderMetricsError("provider_name must not be empty")
    normalized_provider_mode = provider_mode.strip()
    if not normalized_provider_mode:
        raise InvalidGenerationProviderMetricsError("provider_mode must not be empty")

    normalized_http_status = _validate_optional_non_negative_int(
        http_status_code,
        "http_status_code",
    )
    normalized_elapsed_ms = _validate_optional_non_negative_int(elapsed_ms, "elapsed_ms")
    normalized_provider_elapsed_ms = _validate_optional_non_negative_int(
        provider_elapsed_ms,
        "provider_elapsed_ms",
    )
    normalized_retry_count = _validate_non_negative_int(retry_count, "retry_count")

    usage = payload.get("usage")
    usage_payload = dict(usage) if isinstance(usage, dict) else {}
    input_token_count = _usage_int(usage_payload, "prompt_tokens", "input_tokens")
    output_token_count = _usage_int(usage_payload, "completion_tokens", "output_tokens")
    total_token_count = _usage_int(usage_payload, "total_tokens")
    if (
        total_token_count is None
        and input_token_count is not None
        and output_token_count is not None
    ):
        total_token_count = input_token_count + output_token_count

    payload_error_code, payload_error_message = _error_payload(payload)
    return GenerationProviderMetrics(
        contract_version=GENERATION_PROVIDER_METRICS_CONTRACT_VERSION,
        provider_name=normalized_provider_name,
        provider_mode=normalized_provider_mode,
        requested_model_id=_optional_text(requested_model_id),
        response_model_id=_optional_text(payload.get("model")),
        response_id=_optional_text(payload.get("id")),
        http_status_code=normalized_http_status,
        finish_reason=_finish_reason(payload),
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        total_token_count=total_token_count,
        elapsed_ms=normalized_elapsed_ms,
        provider_elapsed_ms=normalized_provider_elapsed_ms,
        retry_count=normalized_retry_count,
        error_code=_optional_text(error_code) or payload_error_code,
        error_message=_optional_text(error_message) or payload_error_message,
        raw_usage=usage_payload,
        response_metadata=_response_metadata(payload),
    )


def generation_provider_metrics_payload(
    metrics: GenerationProviderMetrics,
) -> dict[str, Any]:
    """Serialize provider metrics for API responses and generation run metadata."""

    return {
        "contract_version": metrics.contract_version,
        "provider_name": metrics.provider_name,
        "provider_mode": metrics.provider_mode,
        "requested_model_id": metrics.requested_model_id,
        "response_model_id": metrics.response_model_id,
        "response_id": metrics.response_id,
        "http_status_code": metrics.http_status_code,
        "finish_reason": metrics.finish_reason,
        "input_token_count": metrics.input_token_count,
        "output_token_count": metrics.output_token_count,
        "total_token_count": metrics.total_token_count,
        "elapsed_ms": metrics.elapsed_ms,
        "provider_elapsed_ms": metrics.provider_elapsed_ms,
        "retry_count": metrics.retry_count,
        "succeeded": metrics.succeeded,
        "error_code": metrics.error_code,
        "error_message": metrics.error_message,
        "raw_usage": metrics.raw_usage,
        "response_metadata": metrics.response_metadata,
    }
