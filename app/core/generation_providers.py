"""OpenAI-compatible generation provider client contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urljoin

from app.core.generation_provider_metrics import (
    GenerationProviderMetrics,
    generation_provider_metrics_payload,
    parse_openai_chat_completion_metrics,
)
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GenerationProviderConfigRecord,
)

DEFAULT_GENERATION_MODEL_ID = "nvidia/Qwen3.5-122B-A10B-NVFP4"
DEFAULT_GENERATION_MAX_TOKENS = 1024
DEFAULT_GENERATION_TEMPERATURE = 0.2
DEFAULT_GENERATION_TOP_P = 0.9
DEFAULT_REMOTE_GENERATION_TIMEOUT_SECONDS = 120.0
OPENAI_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
GENERATION_PROVIDER_MODES = (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
)
OPENAI_CHAT_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class GenerationProviderRuntimeConfig:
    mode: str = GENERATION_PROVIDER_MODE_MOCK
    remote_base_url: str | None = None
    remote_timeout_seconds: float = DEFAULT_REMOTE_GENERATION_TIMEOUT_SECONDS
    remote_headers: Mapping[str, str] = field(default_factory=dict)
    api_key: str | None = None
    model_id: str = DEFAULT_GENERATION_MODEL_ID
    max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS
    temperature: float = DEFAULT_GENERATION_TEMPERATURE
    top_p: float = DEFAULT_GENERATION_TOP_P
    runtime_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationChatCompletionRequest:
    messages: tuple[GenerationChatMessage, ...]
    model_id: str = DEFAULT_GENERATION_MODEL_ID
    max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS
    temperature: float = DEFAULT_GENERATION_TEMPERATURE
    top_p: float = DEFAULT_GENERATION_TOP_P
    stop_sequences: tuple[str, ...] = ()
    trace_id: str | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationChatCompletionResponse:
    answer_text: str
    finish_reason: str | None
    provider_model_id: str | None
    response_id: str | None
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    elapsed_ms: int
    provider_metrics: GenerationProviderMetrics
    response_metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_response: Mapping[str, Any] = field(default_factory=dict)


class GenerationProvider(Protocol):
    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        """Generate an answer for an OpenAI-compatible chat request."""


class InvalidGenerationProviderError(ValueError):
    """Raised when generation provider input, config, or output is invalid."""


class GenerationProviderRequestError(InvalidGenerationProviderError):
    """Raised when a remote generation provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        metrics: GenerationProviderMetrics | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.metrics = metrics
        self.payload = dict(payload or {})


class OpenAICompatibleGenerationProviderClient:
    """HTTP client for vLLM/OpenAI-compatible chat completion endpoints."""

    provider_mode = GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE

    def __init__(
        self,
        base_url: str,
        *,
        provider_name: str = "vllm",
        model_id: str = DEFAULT_GENERATION_MODEL_ID,
        timeout_seconds: float = DEFAULT_REMOTE_GENERATION_TIMEOUT_SECONDS,
        headers: Mapping[str, str] | None = None,
        api_key: str | None = None,
        http_client: object | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise InvalidGenerationProviderError("base_url is required")
        if timeout_seconds <= 0:
            raise InvalidGenerationProviderError("timeout_seconds must be greater than 0")
        self.base_url = normalized_base_url
        self.provider_name = _validate_nonblank(provider_name, "provider_name")
        self.model_id = _validate_nonblank(model_id, "model_id")
        self.timeout_seconds = timeout_seconds
        self.headers = _headers_with_api_key(headers or {}, api_key)
        self._owns_client = http_client is None
        self._client = http_client or _create_httpx_client(timeout_seconds=timeout_seconds)

    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        validated = validate_generation_chat_completion_request(request)
        payload = openai_chat_completion_request_payload(validated)
        response_payload, metrics = self._request_chat_completion(payload, validated)
        try:
            answer_text = extract_openai_chat_completion_answer_text(response_payload)
        except InvalidGenerationProviderError as exc:
            raise GenerationProviderRequestError(
                "Invalid generation provider chat completion response",
                metrics=metrics,
                payload=response_payload,
            ) from exc
        return GenerationChatCompletionResponse(
            answer_text=answer_text,
            finish_reason=metrics.finish_reason,
            provider_model_id=metrics.response_model_id,
            response_id=metrics.response_id,
            input_token_count=metrics.input_token_count,
            output_token_count=metrics.output_token_count,
            total_token_count=metrics.total_token_count,
            elapsed_ms=metrics.elapsed_ms or 0,
            provider_metrics=metrics,
            response_metadata={
                "provider_name": self.provider_name,
                "provider_mode": self.provider_mode,
                "base_url": self.base_url,
                "metrics": generation_provider_metrics_payload(metrics),
                **metrics.response_metadata,
            },
            raw_response=response_payload,
        )

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def _request_chat_completion(
        self,
        payload: Mapping[str, Any],
        request: GenerationChatCompletionRequest,
    ) -> tuple[dict[str, Any], GenerationProviderMetrics]:
        started = perf_counter()
        try:
            response = self._client.request(  # type: ignore[attr-defined]
                "POST",
                urljoin(f"{self.base_url}/", OPENAI_CHAT_COMPLETIONS_PATH.lstrip("/")),
                timeout=self.timeout_seconds,
                headers={"Accept": "application/json", **dict(self.headers)},
                json=payload,
            )
        except Exception as exc:
            raise GenerationProviderRequestError(
                f"Remote generation provider request failed: {exc}"
            ) from exc

        elapsed_ms = int((perf_counter() - started) * 1000)
        response_payload = self._response_json(response, elapsed_ms=elapsed_ms)
        metrics = parse_openai_chat_completion_metrics(
            response_payload,
            provider_name=self.provider_name,
            provider_mode=self.provider_mode,
            requested_model_id=request.model_id,
            http_status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            provider_elapsed_ms=elapsed_ms,
        )
        if response.is_error:
            reason = metrics.error_message or getattr(response, "reason_phrase", "")
            raise GenerationProviderRequestError(
                f"Remote generation provider returned HTTP {response.status_code}: {reason}",
                metrics=metrics,
                payload=response_payload,
            )
        return response_payload, metrics

    def _response_json(self, response: object, *, elapsed_ms: int) -> dict[str, Any]:
        try:
            payload = response.json()  # type: ignore[attr-defined]
        except Exception as exc:
            metrics = parse_openai_chat_completion_metrics(
                {},
                provider_name=self.provider_name,
                provider_mode=self.provider_mode,
                requested_model_id=self.model_id,
                http_status_code=getattr(response, "status_code", None),
                elapsed_ms=elapsed_ms,
                provider_elapsed_ms=elapsed_ms,
                error_code="invalid_json",
                error_message="Remote generation provider response was not valid JSON.",
            )
            raise GenerationProviderRequestError(
                "Remote generation provider response was not valid JSON",
                metrics=metrics,
            ) from exc
        if not isinstance(payload, dict):
            metrics = parse_openai_chat_completion_metrics(
                {},
                provider_name=self.provider_name,
                provider_mode=self.provider_mode,
                requested_model_id=self.model_id,
                http_status_code=getattr(response, "status_code", None),
                elapsed_ms=elapsed_ms,
                provider_elapsed_ms=elapsed_ms,
                error_code="invalid_response",
                error_message="Remote generation provider response must be a JSON object.",
            )
            raise GenerationProviderRequestError(
                "Remote generation provider response must be a JSON object",
                metrics=metrics,
            )
        return payload


def generation_provider_runtime_config_from_settings(
    settings: object,
) -> GenerationProviderRuntimeConfig:
    return normalize_generation_provider_runtime_config(
        GenerationProviderRuntimeConfig(
            mode=str(getattr(settings, "generation_provider_mode", GENERATION_PROVIDER_MODE_MOCK)),
            remote_base_url=getattr(settings, "remote_generation_provider_url", None),
            remote_timeout_seconds=float(
                getattr(
                    settings,
                    "remote_generation_provider_timeout_seconds",
                    DEFAULT_REMOTE_GENERATION_TIMEOUT_SECONDS,
                )
            ),
            api_key=getattr(settings, "remote_generation_provider_api_key", None),
            model_id=str(getattr(settings, "generation_model_id", DEFAULT_GENERATION_MODEL_ID)),
            max_tokens=int(
                getattr(settings, "generation_max_tokens", DEFAULT_GENERATION_MAX_TOKENS)
            ),
            temperature=float(
                getattr(settings, "generation_temperature", DEFAULT_GENERATION_TEMPERATURE)
            ),
            top_p=float(getattr(settings, "generation_top_p", DEFAULT_GENERATION_TOP_P)),
        )
    )


def generation_provider_runtime_config_from_record(
    provider_config: GenerationProviderConfigRecord,
    *,
    api_key: str | None = None,
) -> GenerationProviderRuntimeConfig:
    headers = provider_config.runtime_options.get("headers", {})
    if headers is None:
        headers = {}
    if not isinstance(headers, Mapping):
        raise InvalidGenerationProviderError("runtime_options.headers must be a mapping")
    return normalize_generation_provider_runtime_config(
        GenerationProviderRuntimeConfig(
            mode=provider_config.provider_mode,
            remote_base_url=provider_config.provider_base_url,
            remote_timeout_seconds=float(provider_config.request_timeout_seconds),
            remote_headers=headers,
            api_key=api_key,
            model_id=provider_config.model_id,
            max_tokens=provider_config.max_tokens,
            temperature=provider_config.temperature,
            top_p=provider_config.top_p,
            runtime_options=dict(provider_config.runtime_options),
        )
    )


def normalize_generation_provider_runtime_config(
    config: GenerationProviderRuntimeConfig,
) -> GenerationProviderRuntimeConfig:
    mode = config.mode.strip().lower()
    if mode not in GENERATION_PROVIDER_MODES:
        raise InvalidGenerationProviderError(f"Unsupported generation provider mode: {config.mode}")
    if config.remote_timeout_seconds <= 0:
        raise InvalidGenerationProviderError(
            "remote_generation_provider_timeout_seconds must be greater than 0"
        )
    model_id = _validate_nonblank(config.model_id, "model_id")
    _validate_max_tokens(config.max_tokens)
    _validate_temperature(config.temperature)
    _validate_top_p(config.top_p)

    remote_base_url = config.remote_base_url.strip().rstrip("/") if config.remote_base_url else None
    if mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE and not remote_base_url:
        raise InvalidGenerationProviderError(
            "remote_generation_provider_url is required for remote provider mode"
        )

    return GenerationProviderRuntimeConfig(
        mode=mode,
        remote_base_url=remote_base_url,
        remote_timeout_seconds=config.remote_timeout_seconds,
        remote_headers=_normalize_remote_headers(config.remote_headers),
        api_key=_optional_nonblank(config.api_key, "api_key"),
        model_id=model_id,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        runtime_options=dict(config.runtime_options),
    )


def build_generation_provider_from_runtime_config(
    config: GenerationProviderRuntimeConfig,
    *,
    provider_name: str = "vllm",
    http_client: object | None = None,
) -> GenerationProvider:
    normalized = normalize_generation_provider_runtime_config(config)
    if normalized.mode == GENERATION_PROVIDER_MODE_MOCK:
        raise InvalidGenerationProviderError(
            "mock generation provider is handled by execute_mock_generation_run"
        )
    if normalized.remote_base_url is None:
        raise InvalidGenerationProviderError("remote_generation_provider_url is required")
    return OpenAICompatibleGenerationProviderClient(
        normalized.remote_base_url,
        provider_name=provider_name,
        model_id=normalized.model_id,
        timeout_seconds=normalized.remote_timeout_seconds,
        headers=normalized.remote_headers,
        api_key=normalized.api_key,
        http_client=http_client,
    )


def validate_generation_chat_completion_request(
    request: GenerationChatCompletionRequest,
) -> GenerationChatCompletionRequest:
    if not request.messages:
        raise InvalidGenerationProviderError("messages are required")
    messages = tuple(validate_generation_chat_message(message) for message in request.messages)
    model_id = _validate_nonblank(request.model_id, "model_id")
    _validate_max_tokens(request.max_tokens)
    _validate_temperature(request.temperature)
    _validate_top_p(request.top_p)
    stop_sequences = tuple(
        _validate_nonblank(sequence, "stop sequence") for sequence in request.stop_sequences
    )
    trace_id = _optional_nonblank(request.trace_id, "trace_id")
    extra_body = _normalize_extra_body(request.extra_body)
    return GenerationChatCompletionRequest(
        messages=messages,
        model_id=model_id,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=stop_sequences,
        trace_id=trace_id,
        extra_body=extra_body,
        runtime_metadata=dict(request.runtime_metadata),
    )


def validate_generation_chat_message(message: GenerationChatMessage) -> GenerationChatMessage:
    role = _validate_nonblank(message.role, "message role").lower()
    if role not in OPENAI_CHAT_MESSAGE_ROLES:
        raise InvalidGenerationProviderError(f"Unsupported message role: {message.role}")
    return GenerationChatMessage(
        role=role,
        content=_validate_nonblank(message.content, "message content"),
    )


def openai_chat_completion_request_payload(
    request: GenerationChatCompletionRequest,
) -> dict[str, Any]:
    validated = validate_generation_chat_completion_request(request)
    payload: dict[str, Any] = {
        "model": validated.model_id,
        "messages": [
            {"role": message.role, "content": message.content} for message in validated.messages
        ],
        "max_tokens": validated.max_tokens,
        "temperature": validated.temperature,
        "top_p": validated.top_p,
        "stream": False,
    }
    if validated.stop_sequences:
        payload["stop"] = list(validated.stop_sequences)
    if validated.trace_id:
        payload["user"] = validated.trace_id
    payload.update(validated.extra_body)
    return payload


def generation_chat_request_from_openai_messages(
    messages: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    model_id: str = DEFAULT_GENERATION_MODEL_ID,
    max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS,
    temperature: float = DEFAULT_GENERATION_TEMPERATURE,
    top_p: float = DEFAULT_GENERATION_TOP_P,
    stop_sequences: tuple[str, ...] = (),
    trace_id: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> GenerationChatCompletionRequest:
    return validate_generation_chat_completion_request(
        GenerationChatCompletionRequest(
            messages=tuple(
                GenerationChatMessage(
                    role=str(message.get("role", "")),
                    content=str(message.get("content", "")),
                )
                for message in messages
            ),
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            trace_id=trace_id,
            extra_body=dict(extra_body or {}),
            runtime_metadata=dict(runtime_metadata or {}),
        )
    )


def extract_openai_chat_completion_answer_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InvalidGenerationProviderError("choices are required")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise InvalidGenerationProviderError("choice must be a JSON object")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise InvalidGenerationProviderError("choice.message must be a JSON object")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise InvalidGenerationProviderError("choice.message.content must contain text")


def _create_httpx_client(*, timeout_seconds: float):
    try:
        import httpx
    except ImportError as exc:
        raise InvalidGenerationProviderError(
            "httpx is required for remote generation providers."
        ) from exc
    return httpx.Client(timeout=timeout_seconds)


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise InvalidGenerationProviderError(f"{field_name} is required")
    return normalized


def _optional_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if any(char in normalized for char in ("\r", "\n")):
        raise InvalidGenerationProviderError(f"{field_name} contains invalid characters")
    return normalized


def _validate_max_tokens(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise InvalidGenerationProviderError("max_tokens must be greater than 0")
    return int(value)


def _validate_temperature(value: float) -> float:
    if not 0 <= float(value) <= 2:
        raise InvalidGenerationProviderError("temperature must be between 0 and 2")
    return float(value)


def _validate_top_p(value: float) -> float:
    if not 0 < float(value) <= 1:
        raise InvalidGenerationProviderError(
            "top_p must be greater than 0 and less than or equal to 1"
        )
    return float(value)


def _normalize_remote_headers(headers: Mapping[str, str]) -> dict[str, str]:
    normalized_headers: dict[str, str] = {}
    for key, value in headers.items():
        header_name = _validate_nonblank(str(key), "header")
        if any(char in header_name for char in ("\r", "\n", ":")):
            raise InvalidGenerationProviderError("header contains invalid characters")
        header_value = str(value).strip()
        if any(char in header_value for char in ("\r", "\n")):
            raise InvalidGenerationProviderError("header value contains invalid characters")
        normalized_headers[header_name] = header_value
    return normalized_headers


def _normalize_extra_body(extra_body: Mapping[str, Any]) -> dict[str, Any]:
    reserved_keys = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "user",
    }
    normalized: dict[str, Any] = {}
    for key, value in extra_body.items():
        normalized_key = _validate_nonblank(str(key), "extra_body key")
        if normalized_key in reserved_keys:
            raise InvalidGenerationProviderError(
                f"extra_body must not override reserved request field: {normalized_key}"
            )
        normalized[normalized_key] = value
    return normalized


def _headers_with_api_key(
    headers: Mapping[str, str],
    api_key: str | None,
) -> dict[str, str]:
    normalized = _normalize_remote_headers(headers)
    normalized_api_key = _optional_nonblank(api_key, "api_key")
    if normalized_api_key and "Authorization" not in normalized:
        normalized["Authorization"] = f"Bearer {normalized_api_key}"
    return normalized
