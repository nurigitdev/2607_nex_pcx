import sys
from types import SimpleNamespace

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


def test_embedding_provider_service_can_use_local_sentence_transformers_backend(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_source, **kwargs) -> None:
            calls["model_source"] = model_source
            calls["kwargs"] = kwargs

        def encode(self, texts, **kwargs):
            calls["texts"] = texts
            calls["encode_kwargs"] = kwargs
            return [[float(index + 1) for _ in range(1024)] for index, _ in enumerate(texts)]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    model_dir = tmp_path / "kure_v1"
    model_dir.mkdir()
    app = create_app(
        EmbeddingProviderServiceSettings(
            backend="sentence_transformers",
            provider_model_id="local-kure-v1",
            model_key="kure_v1",
            profile_names=("kure_v1_1024",),
            dimension=1024,
            device="cpu",
            models_dir=tmp_path,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "profile_name": "kure_v1_1024",
                "model_key": "kure_v1",
                "input_type": "document",
                "texts": ["alpha", "beta"],
                "output_dimension": 1024,
                "trace_id": "local-provider-trace",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["provider_model_id"] == "local-kure-v1"
    assert body["input_count"] == 2
    assert body["embeddings"][0][0] == 1.0
    assert body["embeddings"][1][0] == 2.0
    assert body["runtime_metadata"]["backend"] == "sentence_transformers"
    assert body["runtime_metadata"]["model_source"] == str(model_dir)
    assert body["runtime_metadata"]["trace_id"] == "local-provider-trace"
    assert calls["model_source"] == str(model_dir)
    assert calls["kwargs"] == {"device": "cpu"}
    assert calls["texts"] == ["alpha", "beta"]
