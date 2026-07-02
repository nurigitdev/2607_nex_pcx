import pytest

from app.core import database


def test_connect_uses_psycopg_with_dict_rows(monkeypatch) -> None:
    calls = {}

    class RawConnection:
        def __enter__(self):
            calls["entered"] = True
            return self

        def __exit__(self, *args):
            calls["exited"] = True
            return None

    def fake_connect(database_url, row_factory):
        calls["database_url"] = database_url
        calls["row_factory"] = row_factory
        return RawConnection()

    monkeypatch.setattr(database.psycopg, "connect", fake_connect)

    with database.connect("postgresql://example/test") as connection:
        assert isinstance(connection, RawConnection)

    assert calls == {
        "database_url": "postgresql://example/test",
        "row_factory": database.dict_row,
        "entered": True,
        "exited": True,
    }


def test_fetch_one_returns_row(monkeypatch) -> None:
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchone(self):
            return {"answer": 42}

    class Connection:
        def cursor(self):
            return Cursor()

    class ConnectionManager:
        def __enter__(self):
            return Connection()

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(database, "connect", lambda database_url: ConnectionManager())

    assert database.fetch_one("postgresql://example/test", "select %s", (42,)) == {"answer": 42}


def test_fetch_one_raises_when_query_returns_no_rows(monkeypatch) -> None:
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchone(self):
            return None

    class Connection:
        def cursor(self):
            return Cursor()

    class ConnectionManager:
        def __enter__(self):
            return Connection()

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(database, "connect", lambda database_url: ConnectionManager())

    with pytest.raises(RuntimeError, match="Query returned no rows"):
        database.fetch_one("postgresql://example/test", "select 1")


def test_ensure_pgvector_extension_returns_version() -> None:
    class Cursor:
        def __init__(self):
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, query):
            self.queries.append(query)

        def fetchone(self):
            return {"extversion": "0.8.0"}

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()

        def cursor(self):
            return self.cursor_instance

    connection = Connection()

    assert database.ensure_pgvector_extension(connection) == "0.8.0"
    assert connection.cursor_instance.queries == [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    ]


def test_ensure_pgvector_extension_raises_when_missing() -> None:
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, query):
            self.query = query

        def fetchone(self):
            return None

    class Connection:
        def cursor(self):
            return Cursor()

    with pytest.raises(RuntimeError, match="pgvector extension is not installed"):
        database.ensure_pgvector_extension(Connection())
