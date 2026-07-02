"""Alembic migration helpers."""

from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def get_migration_database_url() -> str:
    """Return the database URL used by Alembic migrations."""

    settings = get_settings()
    database_url = settings.database_url or settings.test_database_url
    if not database_url:
        msg = "NEX_PCX_DATABASE_URL or NEX_PCX_TEST_DATABASE_URL is required"
        raise RuntimeError(msg)
    return database_url


def to_sqlalchemy_url(database_url: str) -> str:
    """Convert a psycopg connection URL into an explicit SQLAlchemy psycopg URL."""

    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def make_alembic_config(database_url: str | None = None) -> Config:
    """Build an Alembic config with the resolved database URL injected."""

    resolved_database_url = database_url or get_migration_database_url()
    config = Config(str(ALEMBIC_INI))
    config.attributes["database_url"] = resolved_database_url
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", to_sqlalchemy_url(resolved_database_url))
    return config


def upgrade(revision: str = "head", database_url: str | None = None) -> None:
    """Upgrade the configured database to a revision."""

    command.upgrade(make_alembic_config(database_url), revision)


def downgrade(revision: str = "-1", database_url: str | None = None) -> None:
    """Downgrade the configured database to a revision."""

    command.downgrade(make_alembic_config(database_url), revision)
