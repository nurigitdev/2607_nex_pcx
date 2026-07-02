import pytest

from app.core import migrations


def test_get_migration_database_url_prefers_database_url(monkeypatch) -> None:
    monkeypatch.setenv("NEX_PCX_DATABASE_URL", "postgresql://app@example/db")
    monkeypatch.setenv("NEX_PCX_TEST_DATABASE_URL", "postgresql://test@example/db")

    assert migrations.get_migration_database_url() == "postgresql://app@example/db"


def test_get_migration_database_url_falls_back_to_test_database_url(monkeypatch) -> None:
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)
    monkeypatch.setenv("NEX_PCX_TEST_DATABASE_URL", "postgresql://test@example/db")

    assert migrations.get_migration_database_url() == "postgresql://test@example/db"


def test_get_migration_database_url_requires_database_url(monkeypatch) -> None:
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_TEST_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="NEX_PCX_DATABASE_URL"):
        migrations.get_migration_database_url()


def test_make_alembic_config_injects_database_url() -> None:
    config = migrations.make_alembic_config("postgresql://example/db")

    assert config.attributes["database_url"] == "postgresql://example/db"
    assert config.get_main_option("sqlalchemy.url") == "postgresql+psycopg://example/db"
    assert config.get_main_option("script_location").endswith("migrations")


def test_to_sqlalchemy_url_preserves_explicit_driver() -> None:
    assert (
        migrations.to_sqlalchemy_url("postgresql+psycopg://example/db")
        == "postgresql+psycopg://example/db"
    )
