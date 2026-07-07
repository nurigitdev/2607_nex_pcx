from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_expected_chunk_fixture(database_url: str) -> tuple[int, int, int]:
    checksum = f"golden-question-{uuid4()}"
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
                    uploaded_by_user_id,
                    document_group
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s, 'golden-schema')
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
                    document_group,
                    owner_user_id,
                    access_scope
                )
                VALUES (%s, %s, 'golden-schema', %s, 'company')
                RETURNING document_id
                """,
                (file_id, f"Golden question fixture {checksum}", user_id),
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
                    heading_path,
                    char_count
                )
                VALUES (
                    %s,
                    0,
                    'Golden question expected chunk',
                    %s,
                    'heading_512_64',
                    ARRAY['Policy', 'Scope'],
                    30
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
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _cleanup_question_set(database_url: str, set_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM golden_question_sets WHERE set_name = %s", (set_name,))


def test_golden_question_tables_and_indexes_exist(migrated_database_url: str) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
              'golden_question_sets',
              'golden_questions',
              'golden_question_expected_targets'
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
              'idx_golden_question_sets_active',
              'idx_golden_questions_set',
              'idx_golden_questions_actor_scope',
              'idx_golden_questions_chunk_policy',
              'idx_golden_question_expected_targets_question',
              'idx_golden_question_expected_targets_chunk',
              'idx_golden_question_expected_targets_type',
              'idx_golden_question_expected_targets_heading_path'
          )
        """,
    )

    assert table_count["count"] == 3
    assert index_count["count"] == 8


def test_golden_question_schema_defaults_and_cascade(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id, user_id = _create_expected_chunk_fixture(migrated_database_url)
    set_name = f"slice-033-{uuid4()}"
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO golden_question_sets (
                        set_name,
                        description,
                        created_by_user_id
                    )
                    VALUES (%s, 'Slice 033 golden schema fixture', %s)
                    RETURNING question_set_id, is_active, metadata
                    """,
                    (set_name, user_id),
                )
                question_set = cursor.fetchone()
                cursor.execute(
                    """
                    INSERT INTO golden_questions (
                        question_set_id,
                        question_text,
                        normalized_question_text,
                        question_type,
                        actor_user_id,
                        requested_search_scope,
                        document_group,
                        file_type,
                        chunk_policy_name,
                        created_by_user_id
                    )
                    VALUES (
                        %s,
                        'What does the golden fixture cover?',
                        'what does the golden fixture cover?',
                        'single_fact',
                        %s,
                        'company',
                        'golden-schema',
                        '.md',
                        'heading_512_64',
                        %s
                    )
                    RETURNING question_id, top_k, metadata
                    """,
                    (question_set["question_set_id"], user_id, user_id),
                )
                question = cursor.fetchone()
                cursor.execute(
                    """
                    INSERT INTO golden_question_expected_targets (
                        question_id,
                        chunk_id,
                        expectation_type,
                        relevance_grade,
                        notes
                    )
                    VALUES (%s, %s, 'visible', 3, 'primary answer chunk')
                    RETURNING expected_target_id, metadata
                    """,
                    (question["question_id"], chunk_id),
                )
                expected_target = cursor.fetchone()
                cursor.execute(
                    """
                    INSERT INTO golden_question_expected_targets (
                        question_id,
                        expected_heading_path,
                        expectation_type,
                        relevance_grade
                    )
                    VALUES (%s, ARRAY['Policy', 'Scope'], 'hidden', 0)
                    RETURNING expected_target_id
                    """,
                    (question["question_id"],),
                )
                heading_target_id = cursor.fetchone()["expected_target_id"]
                cursor.execute(
                    """
                    DELETE FROM golden_question_sets
                    WHERE question_set_id = %s
                    """,
                    (question_set["question_set_id"],),
                )
                cursor.execute(
                    """
                    SELECT count(*) AS count
                    FROM golden_questions
                    WHERE question_id = %s
                    """,
                    (question["question_id"],),
                )
                question_count = cursor.fetchone()["count"]
                cursor.execute(
                    """
                    SELECT count(*) AS count
                    FROM golden_question_expected_targets
                    WHERE expected_target_id IN (%s, %s)
                    """,
                    (expected_target["expected_target_id"], heading_target_id),
                )
                target_count = cursor.fetchone()["count"]

        assert question_set["is_active"] is True
        assert question_set["metadata"] == {}
        assert question["top_k"] == 5
        assert question["metadata"] == {}
        assert expected_target["metadata"] == {}
        assert question_count == 0
        assert target_count == 0
    finally:
        _cleanup_question_set(migrated_database_url, set_name)
        _cleanup_file(migrated_database_url, file_id)


def test_golden_question_schema_constraints(migrated_database_url: str) -> None:
    file_id, chunk_id, user_id = _create_expected_chunk_fixture(migrated_database_url)
    set_name = f"slice-033-constraints-{uuid4()}"
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO golden_question_sets (set_name, created_by_user_id)
                    VALUES (%s, %s)
                    RETURNING question_set_id
                    """,
                    (set_name, user_id),
                )
                question_set_id = cursor.fetchone()["question_set_id"]
                cursor.execute(
                    """
                    INSERT INTO golden_questions (
                        question_set_id,
                        question_text,
                        question_type,
                        requested_search_scope
                    )
                    VALUES (%s, 'Valid golden question?', 'single_fact', 'company')
                    RETURNING question_id
                    """,
                    (question_set_id,),
                )
                question_id = cursor.fetchone()["question_id"]

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_questions (
                            question_set_id,
                            question_text,
                            question_type
                        )
                        VALUES (%s, 'Bad type?', 'unsupported')
                        """,
                        (question_set_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_questions (
                            question_set_id,
                            question_text,
                            requested_search_scope
                        )
                        VALUES (%s, 'Bad scope?', 'all')
                        """,
                        (question_set_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_question_expected_targets (
                            question_id,
                            chunk_id,
                            expectation_type,
                            relevance_grade
                        )
                        VALUES (%s, %s, 'visible', 4)
                        """,
                        (question_id, chunk_id),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_question_expected_targets (
                            question_id,
                            expectation_type
                        )
                        VALUES (%s, 'visible')
                        """,
                        (question_id,),
                    )
                connection.rollback()
    finally:
        _cleanup_question_set(migrated_database_url, set_name)
        _cleanup_file(migrated_database_url, file_id)
