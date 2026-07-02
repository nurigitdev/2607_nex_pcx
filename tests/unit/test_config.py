from app.core.config import Settings, get_settings


def test_default_settings() -> None:
    settings = get_settings()

    assert settings == Settings(
        app_name="NeX_PCX",
        app_version="0.1.0",
        environment="local",
    )
