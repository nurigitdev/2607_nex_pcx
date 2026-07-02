import pytest

from app.core.database import ensure_pgvector_extension

pytestmark = pytest.mark.integration


def test_pgvector_extension_can_be_enabled(db_connection) -> None:
    extension_version = ensure_pgvector_extension(db_connection)

    assert extension_version


def test_vector_type_accepts_embedding(pgvector_connection) -> None:
    with pgvector_connection.cursor() as cursor:
        cursor.execute("CREATE TEMP TABLE tmp_vector_check (embedding vector(3))")
        cursor.execute("INSERT INTO tmp_vector_check (embedding) VALUES ('[1,2,3]'::vector)")
        cursor.execute("SELECT embedding::text AS embedding FROM tmp_vector_check")
        row = cursor.fetchone()

    assert row["embedding"] == "[1,2,3]"
