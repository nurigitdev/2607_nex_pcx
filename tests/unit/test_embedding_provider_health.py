import httpx

from app.core.config import Settings
from app.core.embedding_provider_health import get_embedding_provider_health_status


def test_embedding_provider_health_reports_mock_provider_ready() -> None:
    status = get_embedding_provider_health_status(Settings())

    assert status.status_code == 200
    assert status.payload["provider_mode"] == "mock"
    assert status.payload["ready"] is True
    assert status.payload["provider_type"] == "mock"
    assert status.payload["provider_model_id"] == "mock-provider"


def test_embedding_provider_health_reads_remote_provider_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        return httpx.Response(
            200,
            json={
                "ready": True,
                "provider_type": "remote",
                "provider_model_id": "gpu-bge-m3",
                "model_key": "bge_m3",
                "profile_names": ["bge_m3_1024"],
                "dimension": 1024,
                "device": "cuda:0",
                "runtime_metadata": {"batching": "static"},
            },
        )

    status = get_embedding_provider_health_status(
        Settings(
            embedding_provider_mode="remote",
            remote_embedding_provider_url="http://provider.local",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert status.status_code == 200
    assert status.payload["provider_mode"] == "remote"
    assert status.payload["ready"] is True
    assert status.payload["provider_model_id"] == "gpu-bge-m3"
    assert status.payload["profile_names"] == ["bge_m3_1024"]
    assert status.payload["runtime_metadata"] == {"batching": "static"}


def test_embedding_provider_health_reports_remote_unreachable() -> None:
    status = get_embedding_provider_health_status(
        Settings(
            embedding_provider_mode="remote",
            remote_embedding_provider_url="http://provider.local",
        ),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, json={"detail": "warming"})
            )
        ),
    )

    assert status.status_code == 502
    assert status.payload["provider_mode"] == "remote"
    assert status.payload["ready"] is False
    assert status.payload["status"] == "unreachable"
    assert "Remote provider request failed" in str(status.payload["error_message"])


def test_embedding_provider_health_reports_misconfigured_remote_provider() -> None:
    status = get_embedding_provider_health_status(Settings(embedding_provider_mode="remote"))

    assert status.status_code == 503
    assert status.payload["configured"] is False
    assert status.payload["status"] == "misconfigured"
    assert "remote_embedding_provider_url" in str(status.payload["error_message"])
