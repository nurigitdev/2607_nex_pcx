"""Application configuration."""

from dataclasses import dataclass
from os import getenv
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_name: str = "NeX_PCX"
    app_version: str = "0.1.0"
    environment: str = "local"
    database_url: str | None = None
    test_database_url: str | None = None
    upload_storage_dir: Path = Path("storage/uploads")
    embedding_models_dir: Path = Path("models")
    embedding_provider_mode: str = "mock"
    remote_embedding_provider_url: str | None = None
    remote_embedding_provider_timeout_seconds: float = 30.0
    embedding_require_route_readiness: bool = False


def get_settings() -> Settings:
    return Settings(
        app_name=getenv("NEX_PCX_APP_NAME", "NeX_PCX"),
        app_version=getenv("NEX_PCX_APP_VERSION", "0.1.0"),
        environment=getenv("NEX_PCX_ENV", "local"),
        database_url=getenv("NEX_PCX_DATABASE_URL"),
        test_database_url=getenv("NEX_PCX_TEST_DATABASE_URL"),
        upload_storage_dir=Path(getenv("NEX_PCX_UPLOAD_STORAGE_DIR", "storage/uploads")),
        embedding_models_dir=Path(getenv("NEX_PCX_MODELS_DIR", "models")),
        embedding_provider_mode=getenv("NEX_PCX_EMBEDDING_PROVIDER_MODE", "mock"),
        remote_embedding_provider_url=getenv("NEX_PCX_REMOTE_EMBEDDING_PROVIDER_URL"),
        remote_embedding_provider_timeout_seconds=float(
            getenv("NEX_PCX_REMOTE_EMBEDDING_PROVIDER_TIMEOUT_SECONDS", "30.0")
        ),
        embedding_require_route_readiness=_env_bool(
            "NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS",
            False,
        ),
    )
