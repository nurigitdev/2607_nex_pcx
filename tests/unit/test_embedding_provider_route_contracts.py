from datetime import UTC, datetime

import httpx

from app.core.embedding_provider_route_contracts import (
    check_embedding_provider_route_contract,
)
from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord


def make_route(**overrides) -> EmbeddingProviderRouteRecord:
    now = datetime(2026, 7, 10, tzinfo=UTC)
    values = {
        "route_id": 1,
        "profile_name": "kure_v1_1024",
        "provider_name": "gpu-primary",
        "provider_mode": "remote",
        "provider_base_url": "http://provider.local",
        "timeout_seconds": 5.0,
        "priority": 10,
        "is_active": True,
        "health_check_enabled": True,
        "runtime_metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingProviderRouteRecord(**values)


def test_route_contract_passes_remote_health_and_embedding_request() -> None:
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/healthz":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "provider_type": "remote",
                    "provider_model_id": "gpu-kure-v1",
                    "model_key": "kure_v1",
                    "profile_names": ["kure_v1_1024"],
                    "dimension": 1024,
                    "device": "cuda:0",
                    "runtime_metadata": {"pool": "primary"},
                },
            )
        if request.url.path == "/v1/embeddings":
            payload = json_from_request(request)
            assert payload["profile_name"] == "kure_v1_1024"
            assert payload["model_key"] == "kure_v1"
            assert payload["input_type"] == "document"
            assert payload["output_dimension"] == 1024
            assert payload["texts"] == ["unit contract sample"]
            assert payload["runtime_metadata"]["contract_sample_set_name"] == "unit_samples"
            return httpx.Response(
                200,
                json={
                    "embeddings": [[0.25] * 1024],
                    "dimension": 1024,
                    "provider_model_id": "gpu-kure-v1",
                    "provider_type": "remote",
                    "elapsed_ms": 15,
                    "input_count": 1,
                    "runtime_metadata": {"device": "cuda:0"},
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    result = check_embedding_provider_route_contract(
        make_route(),
        sample_texts=("unit contract sample",),
        sample_set_name="unit_samples",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.passed is True
    assert result.status == "passed"
    assert result.expected_dimension == 1024
    assert result.dimension == 1024
    assert result.input_count == 1
    assert result.provider_model_id == "gpu-kure-v1"
    assert result.runtime_metadata == {
        "device": "cuda:0",
        "contract_sample_set_name": "unit_samples",
    }
    assert result.validation_errors == ()
    assert seen_paths == ["/healthz", "/v1/embeddings"]


def test_route_contract_blocks_embedding_when_health_mismatches() -> None:
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "ready": True,
                "provider_type": "remote",
                "provider_model_id": "gpu-bge",
                "model_key": "bge_m3",
                "profile_names": ["bge_m3_1024"],
                "dimension": 1024,
                "device": "cuda:1",
                "runtime_metadata": {},
            },
        )

    result = check_embedding_provider_route_contract(
        make_route(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.passed is False
    assert result.status == "health_mismatch"
    assert any("model_key mismatch" in error for error in result.validation_errors)
    assert seen_paths == ["/healthz"]


def test_route_contract_passes_mock_route() -> None:
    result = check_embedding_provider_route_contract(
        make_route(provider_mode="mock", provider_base_url=None)
    )

    assert result.passed is True
    assert result.status == "passed"
    assert result.provider_type == "mock"
    assert result.provider_model_id == "mock-provider"
    assert result.expected_dimension == 1024
    assert result.dimension == 1024


def test_route_contract_reports_invalid_profile() -> None:
    result = check_embedding_provider_route_contract(
        make_route(profile_name="unsupported_profile", provider_mode="mock", provider_base_url=None)
    )

    assert result.passed is False
    assert result.status == "invalid_route"
    assert "Unsupported embedding profile" in str(result.error_message)


def json_from_request(request: httpx.Request) -> dict[str, object]:
    import json

    payload = json.loads(request.content.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload
