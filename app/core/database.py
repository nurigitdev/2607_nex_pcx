"""PostgreSQL helpers used by smoke and integration tests."""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row


@contextmanager
def connect(database_url: str) -> Iterator[Connection]:
    """Open a PostgreSQL connection using dict rows."""

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        yield connection


def fetch_one(database_url: str, query: str, params: tuple[object, ...] = ()) -> dict:
    """Execute a query and return a single row as a dictionary."""

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row is None:
                msg = "Query returned no rows"
                raise RuntimeError(msg)
            return dict(row)


def ensure_pgvector_extension(connection: Connection) -> str:
    """Create pgvector extension when possible and return its installed version."""

    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cursor.fetchone()
        if row is None:
            msg = "pgvector extension is not installed"
            raise RuntimeError(msg)
        return str(row["extversion"])
