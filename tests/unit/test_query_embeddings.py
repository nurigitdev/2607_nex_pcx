from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord
from app.core.embedding_providers import (
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    EmbeddingProviderRuntimeConfig,
    InvalidEmbeddingProviderError,
)
from app.core.embedding_vectors import generate_mock_embedding
from app.core.query_embeddings import (
    InvalidQueryEmbeddingError,
    embed_query_for_profile,
    query_embedding_runtime_metadata,
)


def make_route(**overrides) -> EmbeddingProviderRouteRecord:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    values = {
        "route_id": 1,
        "profile_name": "kure_v1_1024",
        "provider_name": "gpu-a",
        "provider_mode": "remote",
        "provider_base_url": "http://gpu-a.local",
        "timeout_seconds": 5.0,
        "priority": 1,
        "is_active": True,
        "health_check_enabled": True,
        "runtime_metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingProviderRouteRecord(**values)


class CapturingProvider:
    def __init__(self, *, provider_type: str = "remote") -> None:
        self.provider_type = provider_type
        self.requests: list[EmbeddingProviderRequest] = []
        self.closed = False

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        self.requests.append(request)
        embedding = generate_mock_embedding(
            request.texts[0],
            profile_name=request.profile_name,
            dimension=request.output_dimension,
        )
        return EmbeddingProviderResponse(
            embeddings=(embedding,),
            dimension=request.output_dimension,
            provider_model_id="unit-provider",
            provider_type=self.provider_type,
            elapsed_ms=7,
            input_count=1,
            runtime_metadata={"request_trace_id": request.trace_id},
        )

    def close(self) -> None:
        self.closed = True


def test_embed_query_for_profile_uses_mock_fallback_without_routes() -> None:
    result = embed_query_for_profile(
        "postgresql://unused",
        query_text=" reimbursement policy ",
        profile_name="kure_v1_1024",
        route_candidates_selector=lambda _database_url, _profile_name: [],
        trace_id="unit-trace",
    )

    assert result.profile_name == "kure_v1_1024"
    assert result.dimension == 1024
    assert len(result.embedding) == 1024
    assert result.provider_type == "mock"
    assert result.runtime_source == "fallback_runtime_config"
    assert result.runtime_metadata["query_embedding_bridge"] is True
    assert result.runtime_metadata["trace_id"] == "unit-trace"
    payload = query_embedding_runtime_metadata(result)
    assert "embedding" not in payload
    assert payload["runtime_metadata"]["embedding_table_name"] == "chunk_embeddings_kure_v1_1024"


def test_embed_query_for_profile_uses_active_route_headers_and_query_contract(monkeypatch) -> None:
    route = make_route(
        runtime_metadata={
            "request_headers": {"X-Provider-Route": "gpu-a"},
            "auth": {"type": "bearer", "token_env": "NEX_PCX_ROUTE_TOKEN"},
        },
    )
    provider = CapturingProvider()
    captured_configs: list[EmbeddingProviderRuntimeConfig] = []
    monkeypatch.setenv("NEX_PCX_ROUTE_TOKEN", "secret-token")

    def provider_builder(config: EmbeddingProviderRuntimeConfig):
        captured_configs.append(config)
        return provider

    result = embed_query_for_profile(
        "postgresql://unused",
        query_text="query text",
        profile_name="kure_v1_1024",
        provider_builder=provider_builder,
        route_candidates_selector=lambda _database_url, _profile_name: [route],
        trace_id="route-trace",
    )

    assert captured_configs == [
        EmbeddingProviderRuntimeConfig(
            mode="remote",
            remote_base_url="http://gpu-a.local",
            remote_timeout_seconds=5.0,
            remote_headers={
                "X-Provider-Route": "gpu-a",
                "Authorization": "Bearer secret-token",
            },
        )
    ]
    assert provider.closed is True
    assert provider.requests[0].input_type == "query"
    assert provider.requests[0].model_key == "kure_v1"
    assert provider.requests[0].trace_id == "route-trace"
    assert result.runtime_source == "route"
    assert result.runtime_metadata["provider_route_id"] == route.route_id
    assert result.runtime_metadata["provider_response_metadata"] == {
        "request_trace_id": "route-trace"
    }


def test_embed_query_for_profile_fails_over_to_next_route() -> None:
    first_route = make_route(
        route_id=10,
        provider_name="gpu-down",
        provider_base_url="http://gpu-down.local",
        priority=0,
    )
    second_route = replace(
        first_route,
        route_id=11,
        provider_name="gpu-ready",
        provider_base_url="http://gpu-ready.local",
        priority=1,
    )
    built_urls: list[str | None] = []

    def provider_builder(config: EmbeddingProviderRuntimeConfig):
        built_urls.append(config.remote_base_url)
        if config.remote_base_url == "http://gpu-down.local":
            raise InvalidEmbeddingProviderError("route is unavailable")
        return CapturingProvider()

    result = embed_query_for_profile(
        "postgresql://unused",
        query_text="query text",
        profile_name="kure_v1_1024",
        provider_builder=provider_builder,
        route_candidates_selector=lambda _database_url, _profile_name: [
            first_route,
            second_route,
        ],
    )

    assert built_urls == ["http://gpu-down.local", "http://gpu-ready.local"]
    assert result.runtime_metadata["provider_route_id"] == second_route.route_id
    assert result.runtime_metadata["provider_route_failover_attempt"] == 2
    assert result.runtime_metadata["provider_route_failover_candidate_count"] == 2
    assert result.runtime_metadata["provider_route_failed_attempts"] == [
        {
            "route_id": first_route.route_id,
            "provider_name": "gpu-down",
            "provider_mode": "remote",
            "priority": 0,
            "error_message": "route is unavailable",
        }
    ]


@pytest.mark.parametrize(
    ("query_text", "profile_name", "message"),
    [
        (" ", "kure_v1_1024", "query_text"),
        ("query", " ", "profile_name"),
        ("query", "unknown_profile", "Unsupported embedding profile"),
    ],
)
def test_embed_query_for_profile_rejects_invalid_input_before_provider(
    query_text: str,
    profile_name: str,
    message: str,
) -> None:
    with pytest.raises(InvalidQueryEmbeddingError, match=message):
        embed_query_for_profile(
            "postgresql://unused",
            query_text=query_text,
            profile_name=profile_name,
            route_candidates_selector=lambda _database_url, _profile_name: [],
        )


def test_embed_query_for_profile_reports_all_candidate_failures() -> None:
    with pytest.raises(InvalidQueryEmbeddingError, match="all provider candidates") as exc_info:
        embed_query_for_profile(
            "postgresql://unused",
            query_text="query",
            profile_name="kure_v1_1024",
            fallback_runtime_config=EmbeddingProviderRuntimeConfig(mode="remote"),
            route_candidates_selector=lambda _database_url, _profile_name: [],
        )

    assert "remote_embedding_provider_url is required" in str(exc_info.value)
