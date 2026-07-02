"""Application configuration."""

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_name: str = "NeX_PCX"
    app_version: str = "0.1.0"
    environment: str = "local"


def get_settings() -> Settings:
    return Settings(
        app_name=getenv("NEX_PCX_APP_NAME", "NeX_PCX"),
        app_version=getenv("NEX_PCX_APP_VERSION", "0.1.0"),
        environment=getenv("NEX_PCX_ENV", "local"),
    )
