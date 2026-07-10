import math

import pytest

from app.core.embedding_providers import (
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    InvalidEmbeddingProviderError,
    MockEmbeddingProvider,
    validate_embedding_provider_request,
    validate_embedding_provider_response,
)


def make_request(**overrides) -> EmbeddingProviderRequest:
    values = {
        "profile_name": "kure_v1_1024",
        "model_key": "kure_v1",
        "input_type": "document",
        "texts": ("hello", "world"),
        "output_dimension": 8,
        "normalize_embeddings": True,
        "trace_id": " trace-001 ",
    }
    values.update(overrides)
    return EmbeddingProviderRequest(**values)


def test_validate_embedding_provider_request_normalizes_fields() -> None:
    request = validate_embedding_provider_request(
        make_request(profile_name=" kure_v1_1024 ", texts=(" hello ",))
    )

    assert request.profile_name == "kure_v1_1024"
    assert request.texts == ("hello",)
    assert request.trace_id == "trace-001"


@pytest.mark.parametrize(
    ("provider_request", "message"),
    [
        (make_request(profile_name=" "), "profile_name"),
        (make_request(model_key=" "), "model_key"),
        (make_request(input_type="image"), "Unsupported input_type"),
        (make_request(texts=()), "texts"),
        (make_request(texts=(" ",)), "text"),
        (make_request(output_dimension=0), "output_dimension"),
    ],
)
def test_validate_embedding_provider_request_rejects_invalid_input(
    provider_request: EmbeddingProviderRequest,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderError, match=message):
        validate_embedding_provider_request(provider_request)


def test_mock_embedding_provider_returns_deterministic_vectors() -> None:
    provider = MockEmbeddingProvider()
    request = make_request(output_dimension=8)

    first = provider.embed(request)
    second = provider.embed(request)

    assert first == second
    assert first.dimension == 8
    assert first.input_count == 2
    assert first.provider_type == "mock"
    assert len(first.embeddings) == 2
    assert all(len(embedding) == 8 for embedding in first.embeddings)
    assert first.runtime_metadata["profile_name"] == "kure_v1_1024"
    assert first.runtime_metadata["trace_id"] == "trace-001"


def test_validate_embedding_provider_response_rejects_bad_shape_and_values() -> None:
    request = make_request(output_dimension=3, texts=("hello",))

    with pytest.raises(InvalidEmbeddingProviderError, match="dimension mismatch"):
        validate_embedding_provider_response(
            EmbeddingProviderResponse(
                embeddings=((1.0, 2.0, 3.0),),
                dimension=4,
                provider_model_id="mock",
                provider_type="mock",
                elapsed_ms=0,
                input_count=1,
            ),
            request,
        )

    with pytest.raises(InvalidEmbeddingProviderError, match="row count"):
        validate_embedding_provider_response(
            EmbeddingProviderResponse(
                embeddings=(),
                dimension=3,
                provider_model_id="mock",
                provider_type="mock",
                elapsed_ms=0,
                input_count=1,
            ),
            request,
        )

    with pytest.raises(InvalidEmbeddingProviderError, match="finite"):
        validate_embedding_provider_response(
            EmbeddingProviderResponse(
                embeddings=((1.0, math.nan, 3.0),),
                dimension=3,
                provider_model_id="mock",
                provider_type="mock",
                elapsed_ms=0,
                input_count=1,
            ),
            request,
        )
