"""Embedding provider runtime health checks."""

from dataclasses import dataclass

from app.core.embedding_providers import (
    MOCK_EMBEDDING_PROVIDER_TYPE,
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    InvalidEmbeddingProviderError,
    RemoteEmbeddingProviderClient,
    embedding_provider_runtime_config_from_settings,
)


@dataclass(frozen=True)
class EmbeddingProviderHealthStatus:
    status_code: int
    payload: dict[str, object]


def get_embedding_provider_health_status(
    settings: object,
    *,
    http_client: object | None = None,
) -> EmbeddingProviderHealthStatus:
    try:
        runtime_config = embedding_provider_runtime_config_from_settings(settings)
    except InvalidEmbeddingProviderError as exc:
        return EmbeddingProviderHealthStatus(
            status_code=503,
            payload={
                "provider_mode": "invalid",
                "remote_provider_url": None,
                "ready": False,
                "configured": False,
                "status": "misconfigured",
                "error_message": str(exc),
            },
        )

    base_payload = {
        "provider_mode": runtime_config.mode,
        "remote_provider_url": runtime_config.remote_base_url,
        "configured": True,
    }
    if runtime_config.mode == MOCK_EMBEDDING_PROVIDER_TYPE:
        return EmbeddingProviderHealthStatus(
            status_code=200,
            payload={
                **base_payload,
                "ready": True,
                "status": "ready",
                "provider_type": MOCK_EMBEDDING_PROVIDER_TYPE,
                "provider_model_id": "mock-provider",
                "model_key": None,
                "profile_names": [],
                "dimension": None,
                "device": None,
                "runtime_metadata": {},
                "message": "Mock embedding provider is available.",
            },
        )

    client = RemoteEmbeddingProviderClient(
        runtime_config.remote_base_url or "",
        timeout_seconds=runtime_config.remote_timeout_seconds,
        http_client=http_client,
    )
    try:
        health = client.health()
    except InvalidEmbeddingProviderError as exc:
        return EmbeddingProviderHealthStatus(
            status_code=502,
            payload={
                **base_payload,
                "ready": False,
                "status": "unreachable",
                "provider_type": REMOTE_EMBEDDING_PROVIDER_TYPE,
                "error_message": str(exc),
            },
        )
    finally:
        client.close()

    return EmbeddingProviderHealthStatus(
        status_code=200 if health.ready else 503,
        payload={
            **base_payload,
            "ready": health.ready,
            "status": "ready" if health.ready else "not_ready",
            "provider_type": health.provider_type,
            "provider_model_id": health.provider_model_id,
            "model_key": health.model_key,
            "profile_names": list(health.profile_names),
            "dimension": health.dimension,
            "device": health.device,
            "runtime_metadata": dict(health.runtime_metadata),
        },
    )
