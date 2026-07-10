"""Embedding provider request/response contracts and mock implementation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol
from urllib.parse import urljoin

from app.core.embedding_vectors import generate_mock_embedding

EMBEDDING_PROVIDER_INPUT_TYPES = ("query", "document")
MOCK_EMBEDDING_PROVIDER_TYPE = "mock"
REMOTE_EMBEDDING_PROVIDER_TYPE = "remote"
REMOTE_EMBEDDING_PROVIDER_HEALTH_PATH = "/healthz"
REMOTE_EMBEDDING_PROVIDER_EMBEDDINGS_PATH = "/v1/embeddings"


@dataclass(frozen=True)
class EmbeddingProviderRequest:
    profile_name: str
    model_key: str
    input_type: str
    texts: tuple[str, ...]
    output_dimension: int
    normalize_embeddings: bool = True
    trace_id: str | None = None
    runtime_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingProviderResponse:
    embeddings: tuple[tuple[float, ...], ...]
    dimension: int
    provider_model_id: str
    provider_type: str
    elapsed_ms: int
    input_count: int
    runtime_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingProviderHealth:
    ready: bool
    provider_type: str
    provider_model_id: str
    model_key: str
    profile_names: tuple[str, ...]
    dimension: int | None = None
    device: str | None = None
    runtime_metadata: Mapping[str, object] = field(default_factory=dict)


class EmbeddingProvider(Protocol):
    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        """Return embeddings for the requested profile/text batch."""


class InvalidEmbeddingProviderError(ValueError):
    """Raised when provider input or output is invalid."""


def validate_embedding_provider_request(
    request: EmbeddingProviderRequest,
) -> EmbeddingProviderRequest:
    profile_name = _validate_nonblank(request.profile_name, "profile_name")
    model_key = _validate_nonblank(request.model_key, "model_key")
    input_type = _validate_nonblank(request.input_type, "input_type")
    if input_type not in EMBEDDING_PROVIDER_INPUT_TYPES:
        raise InvalidEmbeddingProviderError(f"Unsupported input_type: {input_type}")
    if request.output_dimension <= 0:
        raise InvalidEmbeddingProviderError("output_dimension must be greater than 0")
    if not request.texts:
        raise InvalidEmbeddingProviderError("texts must not be empty")

    texts = tuple(_validate_nonblank(text, "text") for text in request.texts)
    trace_id = request.trace_id.strip() if request.trace_id else None
    return EmbeddingProviderRequest(
        profile_name=profile_name,
        model_key=model_key,
        input_type=input_type,
        texts=texts,
        output_dimension=request.output_dimension,
        normalize_embeddings=request.normalize_embeddings,
        trace_id=trace_id,
        runtime_metadata=dict(request.runtime_metadata),
    )


def validate_embedding_provider_response(
    response: EmbeddingProviderResponse,
    request: EmbeddingProviderRequest,
) -> EmbeddingProviderResponse:
    if response.dimension != request.output_dimension:
        raise InvalidEmbeddingProviderError(
            f"embedding dimension mismatch: expected {request.output_dimension}, "
            f"got {response.dimension}"
        )
    if response.input_count != len(request.texts):
        raise InvalidEmbeddingProviderError(
            f"embedding count mismatch: expected {len(request.texts)}, got {response.input_count}"
        )
    if len(response.embeddings) != len(request.texts):
        raise InvalidEmbeddingProviderError(
            f"embedding row count mismatch: expected {len(request.texts)}, "
            f"got {len(response.embeddings)}"
        )
    for embedding in response.embeddings:
        _validate_embedding_values(embedding, request.output_dimension)
    if response.elapsed_ms < 0:
        raise InvalidEmbeddingProviderError("elapsed_ms must be greater than or equal to 0")
    _validate_nonblank(response.provider_model_id, "provider_model_id")
    _validate_nonblank(response.provider_type, "provider_type")
    return response


@dataclass(frozen=True)
class MockEmbeddingProvider:
    provider_model_id: str = "mock-provider"

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        validated = validate_embedding_provider_request(request)
        embeddings = tuple(
            generate_mock_embedding(
                text,
                profile_name=validated.profile_name,
                dimension=validated.output_dimension,
            )
            for text in validated.texts
        )
        response = EmbeddingProviderResponse(
            embeddings=embeddings,
            dimension=validated.output_dimension,
            provider_model_id=self.provider_model_id,
            provider_type=MOCK_EMBEDDING_PROVIDER_TYPE,
            elapsed_ms=0,
            input_count=len(validated.texts),
            runtime_metadata={
                "provider": MOCK_EMBEDDING_PROVIDER_TYPE,
                "profile_name": validated.profile_name,
                "model_key": validated.model_key,
                "input_type": validated.input_type,
                "normalize_embeddings": validated.normalize_embeddings,
                "trace_id": validated.trace_id,
            },
        )
        return validate_embedding_provider_response(response, validated)


class RemoteEmbeddingProviderClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        http_client: object | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise InvalidEmbeddingProviderError("base_url is required")
        if timeout_seconds <= 0:
            raise InvalidEmbeddingProviderError("timeout_seconds must be greater than 0")
        self.base_url = normalized_base_url
        self.timeout_seconds = timeout_seconds
        self._owns_client = http_client is None
        self._client = http_client or _create_httpx_client(timeout_seconds=timeout_seconds)

    def health(self) -> EmbeddingProviderHealth:
        payload = self._request_json("GET", REMOTE_EMBEDDING_PROVIDER_HEALTH_PATH)
        try:
            return EmbeddingProviderHealth(
                ready=bool(payload["ready"]),
                provider_type=_validate_nonblank(str(payload["provider_type"]), "provider_type"),
                provider_model_id=_validate_nonblank(
                    str(payload["provider_model_id"]),
                    "provider_model_id",
                ),
                model_key=_validate_nonblank(str(payload["model_key"]), "model_key"),
                profile_names=tuple(
                    _validate_nonblank(str(profile_name), "profile_name")
                    for profile_name in payload.get("profile_names", ())
                ),
                dimension=(
                    int(payload["dimension"]) if payload.get("dimension") is not None else None
                ),
                device=str(payload["device"]) if payload.get("device") is not None else None,
                runtime_metadata=dict(payload.get("runtime_metadata") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidEmbeddingProviderError("Invalid provider health response") from exc

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        validated = validate_embedding_provider_request(request)
        payload = {
            "profile_name": validated.profile_name,
            "model_key": validated.model_key,
            "input_type": validated.input_type,
            "texts": list(validated.texts),
            "output_dimension": validated.output_dimension,
            "normalize_embeddings": validated.normalize_embeddings,
            "trace_id": validated.trace_id,
            "runtime_metadata": dict(validated.runtime_metadata),
        }
        response_payload = self._request_json(
            "POST",
            REMOTE_EMBEDDING_PROVIDER_EMBEDDINGS_PATH,
            json=payload,
        )
        try:
            response = EmbeddingProviderResponse(
                embeddings=tuple(
                    tuple(float(value) for value in embedding)
                    for embedding in response_payload["embeddings"]
                ),
                dimension=int(response_payload["dimension"]),
                provider_model_id=str(response_payload["provider_model_id"]),
                provider_type=str(
                    response_payload.get("provider_type") or REMOTE_EMBEDDING_PROVIDER_TYPE
                ),
                elapsed_ms=int(response_payload["elapsed_ms"]),
                input_count=int(response_payload.get("input_count") or len(validated.texts)),
                runtime_metadata=dict(response_payload.get("runtime_metadata") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidEmbeddingProviderError("Invalid provider embedding response") from exc

        return validate_embedding_provider_response(response, validated)

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def _request_json(self, method: str, path: str, **kwargs) -> dict[str, object]:
        try:
            response = self._client.request(  # type: ignore[attr-defined]
                method,
                urljoin(f"{self.base_url}/", path.lstrip("/")),
                timeout=self.timeout_seconds,
                **kwargs,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise InvalidEmbeddingProviderError(f"Remote provider request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidEmbeddingProviderError("Remote provider response must be a JSON object")
        return payload


def _create_httpx_client(*, timeout_seconds: float):
    try:
        import httpx
    except ImportError as exc:
        raise InvalidEmbeddingProviderError(
            "httpx is required for remote embedding providers."
        ) from exc
    return httpx.Client(timeout=timeout_seconds)


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderError(f"{field_name} is required")
    return normalized


def _validate_embedding_values(values: Sequence[float], expected_dimension: int) -> None:
    if len(values) != expected_dimension:
        raise InvalidEmbeddingProviderError(
            f"embedding dimension mismatch: expected {expected_dimension}, got {len(values)}"
        )
    if not all(isfinite(float(value)) for value in values):
        raise InvalidEmbeddingProviderError("embedding values must be finite")
