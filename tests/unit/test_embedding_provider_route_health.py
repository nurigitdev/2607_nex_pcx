from datetime import UTC, datetime

import httpx

from app.core.embedding_provider_route_health import (
    check_embedding_provider_route_health,
    summarize_embedding_provider_route_health,
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


def test_route_health_reports_remote_provider_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        assert request.headers["x-provider-route"] == "gpu-primary"
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

    health = check_embedding_provider_route_health(
        make_route(runtime_metadata={"request_headers": {"X-Provider-Route": "gpu-primary"}}),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert health.checked is True
    assert health.ready is True
    assert health.status == "ready"
    assert health.provider_model_id == "gpu-kure-v1"
    assert health.model_key == "kure_v1"
    assert health.profile_names == ("kure_v1_1024",)
    assert health.dimension == 1024
    assert health.device == "cuda:0"
    assert health.runtime_metadata == {"pool": "primary"}
    assert health.validation_errors == ()


def test_route_health_marks_remote_provider_mismatch() -> None:
    health = check_embedding_provider_route_health(
        make_route(),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
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
            )
        ),
    )

    assert health.checked is True
    assert health.ready is False
    assert health.status == "mismatch"
    assert any("profile_name missing" in error for error in health.validation_errors)
    assert any("model_key mismatch" in error for error in health.validation_errors)


def test_route_health_accepts_profile_dimension_metadata_for_shared_qwen_provider() -> None:
    health = check_embedding_provider_route_health(
        make_route(profile_name="qwen3_4b_2560", provider_name="qwen-primary"),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "provider_type": "remote",
                        "provider_model_id": "gpu-qwen3-4b",
                        "model_key": "qwen3_embedding_4b",
                        "profile_names": ["qwen3_4b_1000", "qwen3_4b_2560"],
                        "dimension": None,
                        "device": "cuda:0",
                        "runtime_metadata": {
                            "profile_dimensions": {
                                "qwen3_4b_1000": 1000,
                                "qwen3_4b_2560": 2560,
                            }
                        },
                    },
                )
            )
        ),
    )

    assert health.checked is True
    assert health.ready is True
    assert health.status == "ready"
    assert health.dimension is None
    assert health.validation_errors == ()


def test_route_health_rejects_mismatched_profile_dimension_metadata() -> None:
    health = check_embedding_provider_route_health(
        make_route(profile_name="qwen3_4b_2560", provider_name="qwen-primary"),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "provider_type": "remote",
                        "provider_model_id": "gpu-qwen3-4b",
                        "model_key": "qwen3_embedding_4b",
                        "profile_names": ["qwen3_4b_1000", "qwen3_4b_2560"],
                        "dimension": None,
                        "device": "cuda:0",
                        "runtime_metadata": {
                            "profile_dimensions": {
                                "qwen3_4b_1000": 1000,
                                "qwen3_4b_2560": 1000,
                            }
                        },
                    },
                )
            )
        ),
    )

    assert health.ready is False
    assert health.status == "mismatch"
    assert any("dimension mismatch" in error for error in health.validation_errors)


def test_route_health_reports_unreachable_remote_provider() -> None:
    health = check_embedding_provider_route_health(
        make_route(),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, json={"detail": "warming"})
            )
        ),
    )

    assert health.checked is True
    assert health.ready is False
    assert health.status == "unreachable"
    assert "Remote provider request failed" in str(health.error_message)


def test_route_health_skips_disabled_health_check_and_counts_mock_route() -> None:
    summary = summarize_embedding_provider_route_health(
        [
            make_route(route_id=1, provider_name="disabled", health_check_enabled=False),
            make_route(
                route_id=2,
                provider_name="mock",
                provider_mode="mock",
                provider_base_url=None,
            ),
        ]
    )

    assert summary.route_count == 2
    assert summary.checked_count == 1
    assert summary.ready_count == 1
    assert summary.routes[0].status == "skipped"
    assert summary.routes[0].checked is False
    assert summary.routes[1].status == "ready"
    assert summary.routes[1].provider_model_id == "mock-provider"
