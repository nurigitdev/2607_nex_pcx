from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_search_fixture(database_url: str) -> tuple[int, int, int]:
    checksum = f"search-log-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    uploaded_by_user_id
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    user_id,
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    owner_user_id,
                    access_scope
                )
                VALUES (%s, %s, %s, 'personal')
                RETURNING document_id
                """,
                (file_id, f"Search log fixture {checksum}", user_id),
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
                VALUES (%s, 0, 'Searchable chunk text', %s, 'heading_512_64', 21)
                RETURNING chunk_id
                """,
                (document_id, f"chunk-{checksum}"),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id, user_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM search_logs
                WHERE search_log_id IN (
                    SELECT DISTINCT sl.search_log_id
                    FROM search_logs sl
                    JOIN search_log_results slr
                      ON slr.search_log_id = sl.search_log_id
                    JOIN chunks c ON c.chunk_id = slr.chunk_id
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE d.file_id = %s
                )
                """,
                (file_id,),
            )
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_search_log_tables_and_indexes_exist(migrated_database_url: str) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
              'search_logs',
              'search_log_results',
              'search_result_feedback'
          )
        """,
    )
    index_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname IN (
              'idx_search_logs_created_at',
              'idx_search_logs_actor',
              'idx_search_logs_scope',
              'idx_search_logs_chunk_policy',
              'idx_search_logs_review_tags',
              'idx_search_logs_reviewed_at',
              'idx_search_log_results_log_profile',
              'idx_search_log_results_chunk',
              'idx_search_result_feedback_result',
              'idx_search_result_feedback_user'
          )
        """,
    )
    column_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'search_logs'
          AND column_name IN (
              'review_tags',
              'review_memo',
              'reviewed_by_user_id',
              'reviewed_at'
          )
        """,
    )

    assert table_count["count"] == 3
    assert index_count["count"] == 10
    assert column_count["count"] == 4


def test_search_log_result_feedback_defaults_and_cascade(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id, user_id = _create_search_fixture(migrated_database_url)
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO search_logs (
                        query_text,
                        normalized_query_text,
                        actor_user_id,
                        requested_search_scope,
                        effective_search_scope,
                        chunk_policy_name,
                        top_k,
                        profiles,
                        created_by_user_id
                    )
                    VALUES (
                        'What is NeX PCX?',
                        'what is nex pcx?',
                        %s,
                        'mine',
                        'mine',
                        'heading_512_64',
                        5,
                        '["kure_v1_1024"]'::jsonb,
                        %s
                    )
                    RETURNING
                        search_log_id,
                        permission_filter_metadata,
                        similarity_metric,
                        query_runtime_metadata
                    """,
                    (user_id, user_id),
                )
                search_log = cursor.fetchone()
                cursor.execute(
                    """
                    INSERT INTO search_log_results (
                        search_log_id,
                        profile_name,
                        rank,
                        chunk_id,
                        distance,
                        score,
                        profile_elapsed_ms
                    )
                    VALUES (%s, 'kure_v1_1024', 1, %s, 0.123, 0.877, 12)
                    RETURNING search_log_result_id
                    """,
                    (search_log["search_log_id"], chunk_id),
                )
                result_id = cursor.fetchone()["search_log_result_id"]
                cursor.execute(
                    """
                    INSERT INTO search_result_feedback (
                        search_log_result_id,
                        relevance_label,
                        comment,
                        created_by_user_id
                    )
                    VALUES (%s, 'correct', 'expected chunk', %s)
                    RETURNING feedback_id
                    """,
                    (result_id, user_id),
                )
                feedback_id = cursor.fetchone()["feedback_id"]
                cursor.execute(
                    "DELETE FROM search_logs WHERE search_log_id = %s",
                    (search_log["search_log_id"],),
                )
                cursor.execute(
                    """
                    SELECT count(*) AS count
                    FROM search_log_results
                    WHERE search_log_result_id = %s
                    """,
                    (result_id,),
                )
                result_count = cursor.fetchone()["count"]
                cursor.execute(
                    """
                    SELECT count(*) AS count
                    FROM search_result_feedback
                    WHERE feedback_id = %s
                    """,
                    (feedback_id,),
                )
                feedback_count = cursor.fetchone()["count"]

        assert search_log["permission_filter_metadata"] == {}
        assert search_log["similarity_metric"] == "cosine"
        assert search_log["query_runtime_metadata"] == {}
        assert result_count == 0
        assert feedback_count == 0
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_search_log_schema_constraints(migrated_database_url: str) -> None:
    file_id, chunk_id, user_id = _create_search_fixture(migrated_database_url)
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute("""
                        INSERT INTO search_logs (
                            query_text,
                            requested_search_scope,
                            top_k,
                            profiles
                        )
                        VALUES ('invalid scope', 'all', 5, '[]'::jsonb)
                        """)
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute("""
                        INSERT INTO search_logs (query_text, top_k, profiles)
                        VALUES ('invalid top k', 0, '[]'::jsonb)
                        """)
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO search_logs (
                        query_text,
                        actor_user_id,
                        top_k,
                        profiles
                    )
                    VALUES ('duplicate rank fixture', %s, 5, '["kure_v1_1024"]'::jsonb)
                    RETURNING search_log_id
                    """,
                    (user_id,),
                )
                search_log_id = cursor.fetchone()["search_log_id"]
                cursor.execute(
                    """
                    INSERT INTO search_log_results (
                        search_log_id,
                        profile_name,
                        rank,
                        chunk_id
                    )
                    VALUES (%s, 'kure_v1_1024', 1, %s)
                    RETURNING search_log_result_id
                    """,
                    (search_log_id, chunk_id),
                )
                cursor.fetchone()
                with pytest.raises(errors.UniqueViolation):
                    cursor.execute(
                        """
                        INSERT INTO search_log_results (
                            search_log_id,
                            profile_name,
                            rank,
                            chunk_id
                        )
                        VALUES (%s, 'kure_v1_1024', 1, %s)
                        """,
                        (search_log_id, chunk_id),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO search_logs (
                        query_text,
                        actor_user_id,
                        top_k,
                        profiles
                    )
                    VALUES ('feedback label fixture', %s, 5, '["kure_v1_1024"]'::jsonb)
                    RETURNING search_log_id
                    """,
                    (user_id,),
                )
                search_log_id = cursor.fetchone()["search_log_id"]
                cursor.execute(
                    """
                    INSERT INTO search_log_results (
                        search_log_id,
                        profile_name,
                        rank,
                        chunk_id
                    )
                    VALUES (%s, 'kure_v1_1024', 1, %s)
                    RETURNING search_log_result_id
                    """,
                    (search_log_id, chunk_id),
                )
                result_id = cursor.fetchone()["search_log_result_id"]

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO search_result_feedback (
                            search_log_result_id,
                            relevance_label
                        )
                        VALUES (%s, 'maybe')
                        """,
                        (result_id,),
                    )
                connection.rollback()
    finally:
        _cleanup_file(migrated_database_url, file_id)
