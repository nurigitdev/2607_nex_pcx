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
    reranker_provider_mode: str = "mock"
    remote_reranker_provider_url: str | None = None
    remote_reranker_provider_timeout_seconds: float = 60.0
    generation_provider_mode: str = "mock"
    remote_generation_provider_url: str | None = None
    remote_generation_provider_timeout_seconds: float = 120.0
    remote_generation_provider_api_key: str | None = None
    generation_model_id: str = "nvidia/Qwen3.6-27B-NVFP4"
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.2
    generation_top_p: float = 0.9
    embedding_require_route_readiness: bool = False
    embedding_route_readiness_failure_mode: str = "fail"
    embedding_route_readiness_defer_seconds: int = 300


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
        reranker_provider_mode=getenv("NEX_PCX_RERANKER_PROVIDER_MODE", "mock"),
        remote_reranker_provider_url=getenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_URL"),
        remote_reranker_provider_timeout_seconds=float(
            getenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_TIMEOUT_SECONDS", "60.0")
        ),
        generation_provider_mode=getenv("NEX_PCX_GENERATION_PROVIDER_MODE", "mock"),
        remote_generation_provider_url=getenv("NEX_PCX_REMOTE_GENERATION_PROVIDER_URL"),
        remote_generation_provider_timeout_seconds=float(
            getenv("NEX_PCX_REMOTE_GENERATION_PROVIDER_TIMEOUT_SECONDS", "120.0")
        ),
        remote_generation_provider_api_key=getenv("NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY"),
        generation_model_id=getenv(
            "NEX_PCX_GENERATION_MODEL_ID",
            "nvidia/Qwen3.6-27B-NVFP4",
        ),
        generation_max_tokens=int(getenv("NEX_PCX_GENERATION_MAX_TOKENS", "1024")),
        generation_temperature=float(getenv("NEX_PCX_GENERATION_TEMPERATURE", "0.2")),
        generation_top_p=float(getenv("NEX_PCX_GENERATION_TOP_P", "0.9")),
        embedding_require_route_readiness=_env_bool(
            "NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS",
            False,
        ),
        embedding_route_readiness_failure_mode=getenv(
            "NEX_PCX_EMBEDDING_ROUTE_READINESS_FAILURE_MODE",
            "fail",
        ),
        embedding_route_readiness_defer_seconds=int(
            getenv("NEX_PCX_EMBEDDING_ROUTE_READINESS_DEFER_SECONDS", "300")
        ),
    )
