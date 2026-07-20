from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_bm25_fixture(database_url: str) -> tuple[int, int, int]:
    checksum = f"bm25-schema-{uuid4()}"
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
                (file_id, f"BM25 schema fixture {checksum}", user_id),
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
                VALUES (
                    %s,
                    0,
                    'BM25 keyword baseline chunk',
                    %s,
                    'heading_512_64',
                    27
                )
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


def test_bm25_search_profile_tables_indexes_and_seed_rows(
    migrated_database_url: str,
) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
              'search_profiles',
              'chunk_keyword_terms',
              'chunk_keyword_statistics'
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
              'idx_search_profiles_kind_active',
              'idx_search_profiles_embedding_profile',
              'idx_chunk_keyword_terms_policy_term',
              'idx_chunk_keyword_statistics_updated',
              'idx_search_logs_strategy',
              'idx_search_log_results_search_profile',
              'idx_search_log_results_retrieval_strategy'
          )
        """,
    )
    profile_summary = fetch_one(
        migrated_database_url,
        """
        SELECT
            count(*) FILTER (
                WHERE profile_kind = 'embedding' AND is_active
            ) AS active_embedding_count,
            count(*) FILTER (
                WHERE search_profile_name = 'bm25_keyword'
                  AND profile_kind = 'keyword'
                  AND NOT is_active
                  AND strategy_name = 'bm25_keyword'
            ) AS bm25_count,
            count(*) FILTER (
                WHERE search_profile_name = 'hybrid_keyword_vector'
                  AND profile_kind = 'hybrid'
                  AND NOT is_active
                  AND strategy_name = 'hybrid_keyword_vector'
            ) AS hybrid_count
        FROM search_profiles
        """,
    )

    assert table_count["count"] == 3
    assert index_count["count"] == 7
    assert profile_summary == {
        "active_embedding_count": 4,
        "bm25_count": 1,
        "hybrid_count": 1,
    }


def test_bm25_keyword_terms_and_search_log_result_contract(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id, user_id = _create_bm25_fixture(migrated_database_url)
    search_log_id = None
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chunk_keyword_terms (
                        chunk_id,
                        chunk_policy_name,
                        tokenizer_name,
                        term,
                        term_frequency
                    )
                    VALUES
                        (%s, 'heading_512_64', 'unicode_word_v1', 'bm25', 1),
                        (%s, 'heading_512_64', 'unicode_word_v1', 'keyword', 2)
                    """,
                    (chunk_id, chunk_id),
                )
                cursor.execute(
                    """
                    INSERT INTO chunk_keyword_statistics (
                        chunk_policy_name,
                        tokenizer_name,
                        term,
                        document_frequency,
                        corpus_chunk_count,
                        average_document_length
                    )
                    VALUES (
                        'heading_512_64',
                        'unicode_word_v1',
                        'bm25',
                        1,
                        1,
                        3.0000
                    )
                    """,
                )
                cursor.execute(
                    """
                    INSERT INTO search_logs (
                        query_text,
                        normalized_query_text,
                        actor_user_id,
                        requested_search_scope,
                        effective_search_scope,
                        chunk_policy_name,
                        strategy_name,
                        top_k,
                        similarity_metric,
                        profiles,
                        query_runtime_metadata,
                        created_by_user_id
                    )
                    VALUES (
                        'BM25 keyword?',
                        'bm25 keyword?',
                        %s,
                        'mine',
                        'mine',
                        'heading_512_64',
                        'bm25_keyword',
                        5,
                        'bm25',
                        '["bm25_keyword"]'::jsonb,
                        '{"tokenizer": "unicode_word_v1", "k1": 1.2, "b": 0.75}'::jsonb,
                        %s
                    )
                    RETURNING search_log_id
                    """,
                    (user_id, user_id),
                )
                search_log_id = cursor.fetchone()["search_log_id"]
                cursor.execute(
                    """
                    INSERT INTO search_log_results (
                        search_log_id,
                        profile_name,
                        search_profile_name,
                        retrieval_strategy,
                        rank,
                        chunk_id,
                        score,
                        score_components,
                        profile_elapsed_ms
                    )
                    VALUES (
                        %s,
                        'bm25_keyword',
                        'bm25_keyword',
                        'bm25_keyword',
                        1,
                        %s,
                        3.5,
                        '{"bm25_score": 3.5, "term_count": 2}'::jsonb,
                        4
                    )
                    RETURNING search_log_result_id
                    """,
                    (search_log_id, chunk_id),
                )
                result_id = cursor.fetchone()["search_log_result_id"]
                cursor.execute(
                    """
                    SELECT
                        sl.strategy_name,
                        sl.similarity_metric,
                        slr.profile_name,
                        slr.search_profile_name,
                        slr.retrieval_strategy,
                        slr.score_components
                    FROM search_logs sl
                    JOIN search_log_results slr
                      ON slr.search_log_id = sl.search_log_id
                    WHERE slr.search_log_result_id = %s
                    """,
                    (result_id,),
                )
                stored = dict(cursor.fetchone())

        assert stored == {
            "strategy_name": "bm25_keyword",
            "similarity_metric": "bm25",
            "profile_name": "bm25_keyword",
            "search_profile_name": "bm25_keyword",
            "retrieval_strategy": "bm25_keyword",
            "score_components": {"bm25_score": 3.5, "term_count": 2},
        }
    finally:
        if search_log_id is not None:
            with connect(migrated_database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM search_logs WHERE search_log_id = %s",
                        (search_log_id,),
                    )
        _cleanup_file(migrated_database_url, file_id)

    term_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM chunk_keyword_terms
        WHERE chunk_id = %s
        """,
        (chunk_id,),
    )
    assert term_count["count"] == 0


def test_bm25_search_profile_schema_constraints(migrated_database_url: str) -> None:
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute("""
                    INSERT INTO search_profiles (
                        search_profile_name,
                        profile_kind,
                        strategy_name,
                        display_name
                    )
                    VALUES ('bad-kind', 'sparse', 'bm25_keyword', 'Bad Kind')
                    """)
            connection.rollback()

    file_id, chunk_id, user_id = _create_bm25_fixture(migrated_database_url)
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO chunk_keyword_terms (
                            chunk_id,
                            chunk_policy_name,
                            term,
                            term_frequency
                        )
                        VALUES (%s, 'heading_512_64', 'bad', 0)
                        """,
                        (chunk_id,),
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
                    VALUES ('invalid score components', %s, 5, '["bm25_keyword"]'::jsonb)
                    RETURNING search_log_id
                    """,
                    (user_id,),
                )
                search_log_id = cursor.fetchone()["search_log_id"]
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO search_log_results (
                            search_log_id,
                            profile_name,
                            rank,
                            chunk_id,
                            score_components
                        )
                        VALUES (%s, 'bm25_keyword', 1, %s, '[]'::jsonb)
                        """,
                        (search_log_id, chunk_id),
                    )
                connection.rollback()
    finally:
        _cleanup_file(migrated_database_url, file_id)
