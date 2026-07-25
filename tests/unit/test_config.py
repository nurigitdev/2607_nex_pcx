from app.core.config import Settings, get_settings


def test_default_settings(monkeypatch) -> None:
    monkeypatch.delenv("NEX_PCX_APP_NAME", raising=False)
    monkeypatch.delenv("NEX_PCX_APP_VERSION", raising=False)
    monkeypatch.delenv("NEX_PCX_ENV", raising=False)
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_UPLOAD_STORAGE_DIR", raising=False)
    monkeypatch.delenv("NEX_PCX_MODELS_DIR", raising=False)
    monkeypatch.delenv("NEX_PCX_EMBEDDING_PROVIDER_MODE", raising=False)
    monkeypatch.delenv("NEX_PCX_REMOTE_EMBEDDING_PROVIDER_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_REMOTE_EMBEDDING_PROVIDER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("NEX_PCX_RERANKER_PROVIDER_MODE", raising=False)
    monkeypatch.delenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS", raising=False)
    monkeypatch.delenv("NEX_PCX_EMBEDDING_ROUTE_READINESS_FAILURE_MODE", raising=False)
    monkeypatch.delenv("NEX_PCX_EMBEDDING_ROUTE_READINESS_DEFER_SECONDS", raising=False)

    settings = get_settings()

    assert settings == Settings(
        app_name="NeX_PCX",
        app_version="0.1.0",
        environment="local",
        database_url=None,
        test_database_url=None,
    )
    assert settings.upload_storage_dir.as_posix() == "storage/uploads"
    assert settings.embedding_models_dir.as_posix() == "models"
    assert settings.embedding_provider_mode == "mock"
    assert settings.remote_embedding_provider_url is None
    assert settings.remote_embedding_provider_timeout_seconds == 30.0
    assert settings.reranker_provider_mode == "mock"
    assert settings.remote_reranker_provider_url is None
    assert settings.remote_reranker_provider_timeout_seconds == 60.0
    assert settings.embedding_require_route_readiness is False
    assert settings.embedding_route_readiness_failure_mode == "fail"
    assert settings.embedding_route_readiness_defer_seconds == 300


def test_database_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NEX_PCX_DATABASE_URL", "postgresql://app@example/db")
    monkeypatch.setenv("NEX_PCX_TEST_DATABASE_URL", "postgresql://test@example/db")
    monkeypatch.setenv("NEX_PCX_UPLOAD_STORAGE_DIR", "/tmp/nex_pcx_uploads")
    monkeypatch.setenv("NEX_PCX_MODELS_DIR", "/tmp/nex_pcx_models")
    monkeypatch.setenv("NEX_PCX_EMBEDDING_PROVIDER_MODE", "remote")
    monkeypatch.setenv(
        "NEX_PCX_REMOTE_EMBEDDING_PROVIDER_URL",
        "http://embedding-provider.local",
    )
    monkeypatch.setenv("NEX_PCX_REMOTE_EMBEDDING_PROVIDER_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("NEX_PCX_RERANKER_PROVIDER_MODE", "remote")
    monkeypatch.setenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_URL", "http://reranker.local")
    monkeypatch.setenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_TIMEOUT_SECONDS", "90.5")
    monkeypatch.setenv("NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS", "true")
    monkeypatch.setenv("NEX_PCX_EMBEDDING_ROUTE_READINESS_FAILURE_MODE", "defer")
    monkeypatch.setenv("NEX_PCX_EMBEDDING_ROUTE_READINESS_DEFER_SECONDS", "900")

    settings = get_settings()

    assert settings.database_url == "postgresql://app@example/db"
    assert settings.test_database_url == "postgresql://test@example/db"
    assert settings.upload_storage_dir.as_posix() == "/tmp/nex_pcx_uploads"
    assert settings.embedding_models_dir.as_posix() == "/tmp/nex_pcx_models"
    assert settings.embedding_provider_mode == "remote"
    assert settings.remote_embedding_provider_url == "http://embedding-provider.local"
    assert settings.remote_embedding_provider_timeout_seconds == 12.5
    assert settings.reranker_provider_mode == "remote"
    assert settings.remote_reranker_provider_url == "http://reranker.local"
    assert settings.remote_reranker_provider_timeout_seconds == 90.5
    assert settings.embedding_require_route_readiness is True
    assert settings.embedding_route_readiness_failure_mode == "defer"
    assert settings.embedding_route_readiness_defer_seconds == 900
