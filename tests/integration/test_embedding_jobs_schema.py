from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration

EMBEDDING_TABLES = {
    "chunk_embeddings_kure_v1_1024": (1024, "vector"),
    "chunk_embeddings_bge_m3_1024": (1024, "vector"),
    "chunk_embeddings_qwen3_4b_1000": (1000, "vector"),
    "chunk_embeddings_qwen3_4b_2560": (2560, "halfvec"),
}


def _create_chunk(database_url: str) -> tuple[int, int]:
    checksum = f"embedding-schema-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path
                )
                VALUES (%s, %s, '.md', 1, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Embedding schema fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    char_count
                )
                VALUES (%s, 0, 'Embedding schema chunk', %s, 'heading_512_64', 22)
                RETURNING chunk_id
                """,
                (document_id, f"chunk-{checksum}"),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _insert_embedding(
    database_url: str,
    table_name: str,
    chunk_id: int,
    *,
    dimension: int,
    storage_type: str,
    elapsed_ms: int = 10,
) -> None:
    cast_type = "halfvec" if storage_type == "halfvec" else "vector"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {table_name} (chunk_id, embedding, elapsed_ms)
                SELECT
                    %s,
                    ('[' || string_agg('0', ',') || ']')::{cast_type},
                    %s
                FROM generate_series(1, %s)
                """,
                (chunk_id, elapsed_ms, dimension),
            )


def test_embedding_job_and_vector_tables_exist(migrated_database_url: str) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
              'embedding_jobs',
              'chunk_embeddings_kure_v1_1024',
              'chunk_embeddings_bge_m3_1024',
              'chunk_embeddings_qwen3_4b_1000',
              'chunk_embeddings_qwen3_4b_2560'
          )
        """,
    )

    assert table_count["count"] == 5


def test_embedding_jobs_constraints_and_chunk_cascade(migrated_database_url: str) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO embedding_jobs (chunk_id, profile_name)
                    VALUES (%s, 'kure_v1_1024')
                    RETURNING job_id, status, attempts, max_attempts, runtime_metadata
                    """,
                    (chunk_id,),
                )
                job = cursor.fetchone()

        assert job["status"] == "pending"
        assert job["attempts"] == 0
        assert job["max_attempts"] == 3
        assert job["runtime_metadata"] == {}

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.UniqueViolation):
                    cursor.execute(
                        """
                        INSERT INTO embedding_jobs (chunk_id, profile_name)
                        VALUES (%s, 'kure_v1_1024')
                        """,
                        (chunk_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO embedding_jobs (chunk_id, profile_name, status)
                        VALUES (%s, 'bge_m3_1024', 'unknown')
                        """,
                        (chunk_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
                cursor.execute(
                    "SELECT count(*) AS count FROM embedding_jobs WHERE job_id = %s",
                    (job["job_id"],),
                )
                row = cursor.fetchone()

        assert row["count"] == 0
        file_id = 0
    finally:
        if file_id:
            _cleanup_file(migrated_database_url, file_id)


def test_profile_embedding_tables_store_expected_dimensions(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    try:
        for table_name, (dimension, storage_type) in EMBEDDING_TABLES.items():
            _insert_embedding(
                migrated_database_url,
                table_name,
                chunk_id,
                dimension=dimension,
                storage_type=storage_type,
            )
            dimension_expr = (
                "vector_dims(embedding::vector)"
                if storage_type == "halfvec"
                else "vector_dims(embedding)"
            )
            stored_dimension = fetch_one(
                migrated_database_url,
                f"SELECT {dimension_expr} AS dimensions FROM {table_name} WHERE chunk_id = %s",
                (chunk_id,),
            )

            assert stored_dimension["dimensions"] == dimension

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
                counts = {}
                for table_name in EMBEDDING_TABLES:
                    cursor.execute(
                        f"SELECT count(*) AS count FROM {table_name} WHERE chunk_id = %s",
                        (chunk_id,),
                    )
                    counts[table_name] = cursor.fetchone()["count"]

        assert counts == {table_name: 0 for table_name in EMBEDDING_TABLES}
        file_id = 0
    finally:
        if file_id:
            _cleanup_file(migrated_database_url, file_id)


def test_embedding_table_rejects_negative_elapsed_ms(migrated_database_url: str) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    try:
        with pytest.raises(errors.CheckViolation):
            _insert_embedding(
                migrated_database_url,
                "chunk_embeddings_kure_v1_1024",
                chunk_id,
                dimension=1024,
                storage_type="vector",
                elapsed_ms=-1,
            )
    finally:
        _cleanup_file(migrated_database_url, file_id)
