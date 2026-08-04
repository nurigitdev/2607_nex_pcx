import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.rerankers import DEFAULT_RERANKER_MODEL_ID, DEFAULT_RERANKER_PROFILE_NAME
from app.reranker_provider_service import (
    RERANKER_PROVIDER_BACKEND_QWEN,
    RerankerProviderServiceSettings,
    create_app,
    get_reranker_provider_service_settings,
)


def _request_payload() -> dict[str, object]:
    return {
        "query_text": "policy",
        "top_k": 2,
        "reranker_profile_name": DEFAULT_RERANKER_PROFILE_NAME,
        "reranker_model_id": DEFAULT_RERANKER_MODEL_ID,
        "candidates": [
            {
                "candidate_key": "c1",
                "rank": 1,
                "text": "unrelated document text",
                "source_profile_name": "qwen3_4b_2560",
                "source_retrieval_strategy": "vector",
                "source_score": 0.91,
                "chunk_id": 10,
                "metadata": {"file_name": "a.md"},
            },
            {
                "candidate_key": "c2",
                "rank": 2,
                "text": "policy document text",
                "source_profile_name": "qwen3_4b_2560",
                "source_retrieval_strategy": "vector",
                "source_score": 0.89,
                "chunk_id": 20,
                "metadata": {"file_name": "b.md"},
            },
        ],
    }


def test_reranker_provider_service_healthz_reports_runtime_contract() -> None:
    app = create_app(
        RerankerProviderServiceSettings(
            provider_model_id="gpu-qwen-reranker",
            reranker_profile_name="qwen3_reranker_0_6b",
            device="cuda:0",
        )
    )

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["provider_type"] == "remote"
    assert body["provider_model_id"] == "gpu-qwen-reranker"
    assert body["reranker_profile_name"] == "qwen3_reranker_0_6b"
    assert body["device"] == "cuda:0"
    assert body["runtime_metadata"]["service"] == "nex_pcx_reranker_provider_service"
    assert body["runtime_metadata"]["backend"] == "mock"


def test_reranker_provider_service_returns_remote_rerank_contract() -> None:
    app = create_app(RerankerProviderServiceSettings(device="cuda:0"))

    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=_request_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["query_text"] == "policy"
    assert body["provider_type"] == "remote"
    assert body["retrieval_strategy"] == "reranked"
    assert body["candidate_count"] == 2
    assert body["returned_count"] == 2
    assert body["results"][0]["candidate_key"] == "c2"
    assert body["results"][0]["rank"] == 1
    assert body["runtime_metadata"]["backend"] == "mock"
    assert body["runtime_metadata"]["device"] == "cuda:0"


def test_reranker_provider_service_rejects_when_not_ready() -> None:
    app = create_app(RerankerProviderServiceSettings(ready=False))

    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=_request_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Reranker provider is not ready."


def test_reranker_provider_service_rejects_unsupported_profile() -> None:
    payload = _request_payload()
    payload["reranker_profile_name"] = "other"
    app = create_app(RerankerProviderServiceSettings())

    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=payload)

    assert response.status_code == 400
    assert "Unsupported reranker_profile_name" in response.json()["detail"]


def test_reranker_provider_service_rejects_unsupported_model_id() -> None:
    payload = _request_payload()
    payload["reranker_model_id"] = "other-model"
    app = create_app(RerankerProviderServiceSettings())

    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=payload)

    assert response.status_code == 400
    assert "Unsupported reranker_model_id" in response.json()["detail"]


def test_reranker_provider_service_settings_read_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_BACKEND", " QWEN_RERANKER ")
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_MODEL_ID", "local-qwen-reranker")
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_PROFILE_NAME", "qwen3_reranker_0_6b_local")
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_DEVICE", "cuda:1")
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_TORCH_DTYPE", "bf16")
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_READY", "false")
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME", "reranker-model")

    settings = get_reranker_provider_service_settings()

    assert settings.backend == "qwen_reranker"
    assert settings.provider_model_id == "local-qwen-reranker"
    assert settings.reranker_profile_name == "qwen3_reranker_0_6b_local"
    assert settings.device == "cuda:1"
    assert settings.torch_dtype == "bf16"
    assert settings.ready is False
    assert settings.local_model_dir == tmp_path / "reranker-model"


def test_reranker_provider_service_rejects_unsupported_backend() -> None:
    settings = RerankerProviderServiceSettings(backend="missing")

    try:
        create_app(settings)
    except ValueError as exc:
        assert "Unsupported reranker provider backend" in str(exc)
    else:
        raise AssertionError("Expected create_app to reject unsupported backend")


def test_qwen_reranker_backend_requires_local_model_directory(tmp_path) -> None:
    settings = RerankerProviderServiceSettings(
        backend=RERANKER_PROVIDER_BACKEND_QWEN,
        models_dir=tmp_path,
        model_dir_name="missing-reranker",
    )

    try:
        create_app(settings)
    except ValueError as exc:
        assert "Reranker model directory does not exist" in str(exc)
    else:
        raise AssertionError("Expected create_app to reject missing model directory")


def test_reranker_provider_service_can_use_qwen_cross_encoder_backend(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {}
    bfloat16_dtype = object()

    class FakeParameter:
        dtype = "torch.bfloat16"

    class FakeCrossEncoder:
        def __init__(self, model_source, **kwargs) -> None:
            calls["model_source"] = model_source
            calls["kwargs"] = kwargs

        def parameters(self):
            return iter([FakeParameter()])

        def predict(self, pairs):
            calls["pairs"] = list(pairs)
            return [0.2, 5.5]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=bfloat16_dtype))
    model_dir = tmp_path / "qwen3_reranker_0_6b"
    model_dir.mkdir()
    app = create_app(
        RerankerProviderServiceSettings(
            backend=RERANKER_PROVIDER_BACKEND_QWEN,
            device="cuda:0",
            torch_dtype="bfloat16",
            models_dir=tmp_path,
        )
    )

    with TestClient(app) as client:
        health_response = client.get("/healthz")
        response = client.post("/v1/rerank", json=_request_payload())

    assert health_response.status_code == 200
    health = health_response.json()
    assert health["runtime_metadata"]["requested_torch_dtype"] == "bfloat16"
    assert health["runtime_metadata"]["loaded_parameter_dtype"] == "bfloat16"
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["candidate_key"] == "c2"
    assert body["results"][0]["score"] == 5.5
    assert body["results"][0]["score_components"]["raw_cross_encoder_score"] == 5.5
    assert body["runtime_metadata"]["backend"] == "qwen_reranker"
    assert body["runtime_metadata"]["model_source"] == str(model_dir)
    assert body["runtime_metadata"]["requested_torch_dtype"] == "bfloat16"
    assert body["runtime_metadata"]["loaded_parameter_dtype"] == "bfloat16"
    assert calls["model_source"] == str(model_dir)
    assert calls["kwargs"] == {
        "device": "cuda:0",
        "model_kwargs": {"torch_dtype": bfloat16_dtype},
    }
    assert calls["pairs"] == [
        ("policy", "unrelated document text"),
        ("policy", "policy document text"),
    ]


def test_qwen_reranker_backend_rejects_invalid_score_count(tmp_path, monkeypatch) -> None:
    class FakeCrossEncoder:
        def __init__(self, model_source, **kwargs) -> None:
            pass

        def predict(self, pairs):
            return [0.2]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    model_dir = tmp_path / "qwen3_reranker_0_6b"
    model_dir.mkdir()
    app = create_app(
        RerankerProviderServiceSettings(
            backend=RERANKER_PROVIDER_BACKEND_QWEN,
            models_dir=tmp_path,
        )
    )

    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=_request_payload())

    assert response.status_code == 400
    assert "score count mismatch" in response.json()["detail"]


def test_qwen_reranker_backend_rejects_nonfinite_scores(tmp_path, monkeypatch) -> None:
    class FakeCrossEncoder:
        def __init__(self, model_source, **kwargs) -> None:
            pass

        def predict(self, pairs):
            return [float("nan"), 1.0]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    model_dir = tmp_path / "qwen3_reranker_0_6b"
    model_dir.mkdir()
    app = create_app(
        RerankerProviderServiceSettings(
            backend=RERANKER_PROVIDER_BACKEND_QWEN,
            models_dir=tmp_path,
        )
    )

    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=_request_payload())

    assert response.status_code == 400
    assert "scores must be finite" in response.json()["detail"]
