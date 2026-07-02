from collections.abc import Generator

import pytest

from app.core.config import get_settings
from app.core.database import connect, ensure_pgvector_extension
from app.core.migrations import upgrade


@pytest.fixture(scope="session")
def test_database_url() -> str:
    database_url = get_settings().test_database_url
    if not database_url:
        pytest.skip("NEX_PCX_TEST_DATABASE_URL is not set")
    return database_url


@pytest.fixture()
def db_connection(test_database_url: str) -> Generator:
    with connect(test_database_url) as connection:
        yield connection


@pytest.fixture()
def pgvector_connection(db_connection):
    ensure_pgvector_extension(db_connection)
    return db_connection


@pytest.fixture()
def migrated_database_url(test_database_url: str) -> str:
    upgrade("head", test_database_url)
    return test_database_url
