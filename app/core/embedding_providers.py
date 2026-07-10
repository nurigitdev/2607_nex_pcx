"""Embedding provider request/response contracts and mock implementation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

from app.core.embedding_vectors import generate_mock_embedding

EMBEDDING_PROVIDER_INPUT_TYPES = ("query", "document")
MOCK_EMBEDDING_PROVIDER_TYPE = "mock"


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
