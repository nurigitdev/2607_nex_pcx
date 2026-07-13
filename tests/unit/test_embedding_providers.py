import math

import httpx
import pytest

from app.core.embedding_providers import (
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    EmbeddingProviderRuntimeConfig,
    InvalidEmbeddingProviderError,
    MockEmbeddingProvider,
    RemoteEmbeddingProviderClient,
    build_embedding_provider_from_runtime_config,
    embedding_provider_runtime_config_from_settings,
    normalize_embedding_provider_runtime_config,
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


def test_embedding_provider_runtime_config_defaults_to_mock() -> None:
    config = embedding_provider_runtime_config_from_settings(object())
    provider = build_embedding_provider_from_runtime_config(config)

    assert config == EmbeddingProviderRuntimeConfig(mode="mock")
    assert isinstance(provider, MockEmbeddingProvider)


def test_embedding_provider_runtime_config_builds_remote_client() -> None:
    class SettingsStub:
        embedding_provider_mode = " REMOTE "
        remote_embedding_provider_url = "http://embedding-provider.local/"
        remote_embedding_provider_timeout_seconds = 7.5

    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    config = embedding_provider_runtime_config_from_settings(SettingsStub())
    provider = build_embedding_provider_from_runtime_config(config, http_client=http_client)

    assert config == EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://embedding-provider.local",
        remote_timeout_seconds=7.5,
    )
    assert isinstance(provider, RemoteEmbeddingProviderClient)
    assert provider.base_url == "http://embedding-provider.local"
    assert provider.timeout_seconds == 7.5


def test_embedding_provider_runtime_config_rejects_invalid_remote_settings() -> None:
    with pytest.raises(InvalidEmbeddingProviderError, match="Unsupported"):
        normalize_embedding_provider_runtime_config(EmbeddingProviderRuntimeConfig(mode="local"))

    with pytest.raises(InvalidEmbeddingProviderError, match="remote_embedding_provider_url"):
        normalize_embedding_provider_runtime_config(EmbeddingProviderRuntimeConfig(mode="remote"))

    with pytest.raises(InvalidEmbeddingProviderError, match="timeout"):
        normalize_embedding_provider_runtime_config(
            EmbeddingProviderRuntimeConfig(
                mode="remote", remote_base_url="http://p", remote_timeout_seconds=0
            )
        )


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


def test_remote_embedding_provider_client_reads_health_and_embeddings() -> None:
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        assert request.headers["x-provider-route"] == "gpu-a"
        if request.url.path == "/healthz":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "provider_type": "remote",
                    "provider_model_id": "gpu-kure-v1",
                    "model_key": "kure_v1",
                    "profile_names": ["kure_v1_1024"],
                    "dimension": 8,
                    "device": "cuda:0",
                    "runtime_metadata": {"batching": "dynamic"},
                },
            )
        if request.url.path == "/v1/embeddings":
            payload = json_from_request(request)
            assert payload["profile_name"] == "kure_v1_1024"
            assert payload["texts"] == ["hello", "world"]
            return httpx.Response(
                200,
                json={
                    "embeddings": [[1.0] * 8, [2.0] * 8],
                    "dimension": 8,
                    "provider_model_id": "gpu-kure-v1",
                    "provider_type": "remote",
                    "elapsed_ms": 12,
                    "input_count": 2,
                    "runtime_metadata": {"device": "cuda:0"},
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    client = RemoteEmbeddingProviderClient(
        "http://embedding-provider.local/",
        headers={"X-Provider-Route": "gpu-a"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    health = client.health()
    response = client.embed(make_request(output_dimension=8))
    client.close()

    assert health.ready is True
    assert health.provider_model_id == "gpu-kure-v1"
    assert health.device == "cuda:0"
    assert response.embeddings == ((1.0,) * 8, (2.0,) * 8)
    assert response.elapsed_ms == 12
    assert [request.url.path for request in seen_requests] == ["/healthz", "/v1/embeddings"]


def test_remote_embedding_provider_client_rejects_invalid_settings_and_responses() -> None:
    with pytest.raises(InvalidEmbeddingProviderError, match="base_url"):
        RemoteEmbeddingProviderClient(" ")

    with pytest.raises(InvalidEmbeddingProviderError, match="timeout_seconds"):
        RemoteEmbeddingProviderClient("http://provider", timeout_seconds=0)

    with pytest.raises(InvalidEmbeddingProviderError, match="header"):
        RemoteEmbeddingProviderClient("http://provider", headers={"Bad:Header": "value"})

    with pytest.raises(InvalidEmbeddingProviderError, match="header value"):
        RemoteEmbeddingProviderClient("http://provider", headers={"X-Good": "bad\nvalue"})

    bad_client = RemoteEmbeddingProviderClient(
        "http://provider",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
        ),
    )

    with pytest.raises(InvalidEmbeddingProviderError, match="JSON object"):
        bad_client.health()


def test_remote_embedding_provider_client_wraps_http_errors() -> None:
    client = RemoteEmbeddingProviderClient(
        "http://provider",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, json={"detail": "warming"})
            )
        ),
    )

    with pytest.raises(InvalidEmbeddingProviderError, match="Remote provider request failed"):
        client.health()


def json_from_request(request: httpx.Request) -> dict[str, object]:
    import json

    payload = json.loads(request.content.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload
