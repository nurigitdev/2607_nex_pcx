from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_document(database_url: str) -> tuple[int, int]:
    checksum = f"chunks-schema-{uuid4()}"
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
                (file_id, f"Chunks schema fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_chunks_table_and_experiment_policy_seed_rows(migrated_database_url: str) -> None:
    chunks_table = fetch_one(
        migrated_database_url,
        "SELECT to_regclass('public.chunks') AS table_name",
    )
    policy_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM chunk_policies
        WHERE chunk_policy_name IN (
            'heading_512_64',
            'heading_1000_200',
            'heading_1500_200'
        )
        """,
    )

    assert chunks_table["table_name"] == "chunks"
    assert policy_count["count"] == 3


def test_chunks_constraints_and_document_cascade(migrated_database_url: str) -> None:
    file_id, document_id = _create_document(migrated_database_url)
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
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
                    VALUES (%s, 0, 'First chunk', 'hash-one', 'heading_512_64', 11)
                    RETURNING chunk_id
                    """,
                    (document_id,),
                )
                chunk_id = cursor.fetchone()["chunk_id"]
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
                    VALUES (
                        %s,
                        0,
                        'Same seq alternate policy',
                        'hash-one-policy-two',
                        'heading_1000_200',
                        25
                    )
                    """,
                    (document_id,),
                )

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.UniqueViolation):
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
                        VALUES (%s, 0, 'Duplicate seq', 'hash-two', 'heading_512_64', 13)
                        """,
                        (document_id,),
                    )
                connection.rollback()

        policy_scoped_count = fetch_one(
            migrated_database_url,
            """
            SELECT count(*) AS count
            FROM chunks
            WHERE document_id = %s
              AND chunk_seq = 0
            """,
            (document_id,),
        )

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
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
                        VALUES (%s, -1, 'Bad seq', 'hash-three', 'heading_512_64', 7)
                        """,
                        (document_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
                cursor.execute(
                    "SELECT count(*) AS count FROM chunks WHERE chunk_id = %s",
                    (chunk_id,),
                )
                row = cursor.fetchone()

        assert policy_scoped_count["count"] == 2
        assert row["count"] == 0
        file_id = 0
    finally:
        if file_id:
            _cleanup_file(migrated_database_url, file_id)
