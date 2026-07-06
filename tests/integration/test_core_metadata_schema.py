from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def insert_file(connection, *, checksum: str, parse_status: str = "pending") -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO files (
                original_file_name,
                stored_file_name,
                file_ext,
                mime_type,
                file_size_bytes,
                sha256_checksum,
                storage_path,
                parse_status
            )
            VALUES (%s, %s, '.md', 'text/markdown', 12, %s, %s, %s)
            RETURNING file_id
            """,
            (
                f"{checksum}.md",
                f"{checksum}.stored.md",
                checksum,
                f"/tmp/{checksum}.md",
                parse_status,
            ),
        )
        row = cursor.fetchone()
        return int(row["file_id"])


def test_core_metadata_tables_and_seed_rows(migrated_database_url: str) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
              'files',
              'documents',
              'chunks',
              'chunk_policies',
              'embedding_profiles'
          )
        """,
    )
    chunk_policy = fetch_one(
        migrated_database_url,
        """
        SELECT target_token_size, overlap_token_size, split_strategy
        FROM chunk_policies
        WHERE chunk_policy_name = 'heading_512_64'
        """,
    )
    embedding_profiles = fetch_one(
        migrated_database_url,
        "SELECT count(*) AS count FROM embedding_profiles WHERE is_active",
    )
    chunk_policy_count = fetch_one(
        migrated_database_url,
        "SELECT count(*) AS count FROM chunk_policies",
    )
    qwen_profile = fetch_one(
        migrated_database_url,
        """
        SELECT dimension, storage_type
        FROM embedding_profiles
        WHERE profile_name = 'qwen3_4b_2560'
        """,
    )

    assert table_count["count"] == 5
    assert chunk_policy == {
        "target_token_size": 512,
        "overlap_token_size": 64,
        "split_strategy": "heading-aware",
    }
    assert chunk_policy_count["count"] >= 3
    assert embedding_profiles["count"] == 4
    assert qwen_profile == {"dimension": 2560, "storage_type": "halfvec"}


def test_files_checksum_unique_constraint(migrated_database_url: str) -> None:
    checksum = f"checksum-{uuid4()}"
    with connect(migrated_database_url) as connection:
        insert_file(connection, checksum=checksum)
        with pytest.raises(errors.UniqueViolation):
            insert_file(connection, checksum=checksum)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE sha256_checksum = %s", (checksum,))


def test_files_parse_status_check_constraint(migrated_database_url: str) -> None:
    checksum = f"parse-status-{uuid4()}"
    with connect(migrated_database_url) as connection:
        with pytest.raises(errors.CheckViolation):
            insert_file(connection, checksum=checksum, parse_status="unknown")
        connection.rollback()


def test_documents_status_check_and_file_cascade(migrated_database_url: str) -> None:
    checksum = f"document-status-{uuid4()}"
    with connect(migrated_database_url) as connection:
        file_id = insert_file(connection, checksum=checksum)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, 'Valid document')
                """,
                (file_id,),
            )
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO documents (
                        file_id,
                        document_title,
                        document_status
                    )
                    VALUES (%s, 'Invalid document', 'missing')
                    """,
                    (file_id,),
                )
            connection.rollback()
            file_id = insert_file(connection, checksum=f"{checksum}-cascade")
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, 'Cascade document')
                RETURNING document_id
                """,
                (file_id,),
            )
            document_id = cursor.fetchone()["document_id"]
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            cursor.execute(
                "SELECT count(*) AS count FROM documents WHERE document_id = %s",
                (document_id,),
            )
            row = cursor.fetchone()

    assert row["count"] == 0


def test_embedding_profile_storage_type_check_constraint(migrated_database_url: str) -> None:
    profile_name = f"profile_{uuid4().hex}"
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO embedding_profiles (
                        profile_name,
                        model_name,
                        dimension,
                        storage_type
                    )
                    VALUES (%s, 'example/model', 128, 'blob')
                    """,
                    (profile_name,),
                )
        connection.rollback()


def test_chunk_policy_overlap_must_be_smaller_than_target(
    migrated_database_url: str,
) -> None:
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO chunk_policies (
                        chunk_policy_name,
                        target_token_size,
                        overlap_token_size,
                        split_strategy
                    )
                    VALUES (%s, 128, 128, 'heading-aware')
                    """,
                    (f"bad_policy_{uuid4().hex}",),
                )
        connection.rollback()
