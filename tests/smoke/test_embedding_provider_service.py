from fastapi.testclient import TestClient

from app.embedding_provider_service import EmbeddingProviderServiceSettings, create_app


def test_embedding_provider_service_healthz_reports_skeleton_contract() -> None:
    app = create_app(
        EmbeddingProviderServiceSettings(
            provider_model_id="gpu-skeleton-kure",
            model_key="kure_v1",
            profile_names=("kure_v1_1024",),
            dimension=1024,
            device="cuda:0",
        )
    )

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["provider_type"] == "remote"
    assert body["provider_model_id"] == "gpu-skeleton-kure"
    assert body["profile_names"] == ["kure_v1_1024"]
    assert body["device"] == "cuda:0"


def test_embedding_provider_service_returns_embedding_contract() -> None:
    app = create_app(
        EmbeddingProviderServiceSettings(
            provider_model_id="gpu-skeleton-kure",
            model_key="kure_v1",
            profile_names=("kure_v1_1024",),
            dimension=8,
            device="cuda:0",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "profile_name": "kure_v1_1024",
                "model_key": "kure_v1",
                "input_type": "document",
                "texts": ["hello", "world"],
                "output_dimension": 8,
                "trace_id": "trace-provider-001",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["dimension"] == 8
    assert body["input_count"] == 2
    assert body["provider_type"] == "remote"
    assert body["provider_model_id"] == "gpu-skeleton-kure"
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 8
    assert body["runtime_metadata"]["trace_id"] == "trace-provider-001"


def test_embedding_provider_service_rejects_unsupported_profile() -> None:
    app = create_app(EmbeddingProviderServiceSettings(dimension=8))

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "profile_name": "bge_m3_1024",
                "model_key": "kure_v1",
                "input_type": "document",
                "texts": ["hello"],
                "output_dimension": 8,
            },
        )

    assert response.status_code == 400
    assert "Unsupported profile_name" in response.json()["detail"]


def test_embedding_provider_service_rejects_embeddings_when_not_ready() -> None:
    app = create_app(EmbeddingProviderServiceSettings(dimension=8, ready=False))

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "profile_name": "kure_v1_1024",
                "model_key": "kure_v1",
                "input_type": "document",
                "texts": ["hello"],
                "output_dimension": 8,
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding provider is not ready."
