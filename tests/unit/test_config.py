from app.core.config import Settings, get_settings


def test_default_settings(monkeypatch) -> None:
    monkeypatch.delenv("NEX_PCX_APP_NAME", raising=False)
    monkeypatch.delenv("NEX_PCX_APP_VERSION", raising=False)
    monkeypatch.delenv("NEX_PCX_ENV", raising=False)
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_UPLOAD_STORAGE_DIR", raising=False)

    settings = get_settings()

    assert settings == Settings(
        app_name="NeX_PCX",
        app_version="0.1.0",
        environment="local",
        database_url=None,
        test_database_url=None,
    )
    assert settings.upload_storage_dir.as_posix() == "storage/uploads"


def test_database_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NEX_PCX_DATABASE_URL", "postgresql://app@example/db")
    monkeypatch.setenv("NEX_PCX_TEST_DATABASE_URL", "postgresql://test@example/db")
    monkeypatch.setenv("NEX_PCX_UPLOAD_STORAGE_DIR", "/tmp/nex_pcx_uploads")

    settings = get_settings()

    assert settings.database_url == "postgresql://app@example/db"
    assert settings.test_database_url == "postgresql://test@example/db"
    assert settings.upload_storage_dir.as_posix() == "/tmp/nex_pcx_uploads"
