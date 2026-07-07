from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_question_fixture(database_url: str) -> tuple[int, int, int, str]:
    set_name = f"golden-evaluation-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                """
                INSERT INTO golden_question_sets (
                    set_name,
                    description,
                    created_by_user_id
                )
                VALUES (%s, 'Slice 036 golden evaluation fixture', %s)
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
                    normalized_question_text,
                    question_type,
                    actor_user_id,
                    requested_search_scope,
                    chunk_policy_name,
                    top_k,
                    created_by_user_id
                )
                VALUES (
                    %s,
                    'What does the evaluation fixture cover?',
                    'what does the evaluation fixture cover?',
                    'single_fact',
                    %s,
                    'company',
                    'heading_512_64',
                    5,
                    %s
                )
                RETURNING question_id
                """,
                (question_set_id, user_id, user_id),
            )
            question_id = cursor.fetchone()["question_id"]
    return question_set_id, question_id, user_id, set_name


def _cleanup_question_set(database_url: str, set_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM golden_question_sets WHERE set_name = %s", (set_name,))


def test_golden_evaluation_tables_and_indexes_exist(migrated_database_url: str) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
              'golden_evaluation_runs',
              'golden_evaluation_results'
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
              'idx_golden_evaluation_runs_set',
              'idx_golden_evaluation_runs_profile_status',
              'idx_golden_evaluation_runs_created_at',
              'idx_golden_evaluation_results_run',
              'idx_golden_evaluation_results_question',
              'idx_golden_evaluation_results_search_log'
          )
        """,
    )

    assert table_count["count"] == 2
    assert index_count["count"] == 6


def test_golden_evaluation_schema_defaults_and_cascade(
    migrated_database_url: str,
) -> None:
    question_set_id, question_id, _user_id, set_name = _create_question_fixture(
        migrated_database_url
    )
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO golden_evaluation_runs (
                        question_set_id,
                        run_name,
                        profile_name,
                        chunk_policy_name
                    )
                    VALUES (%s, %s, 'kure_v1_1024', 'heading_512_64')
                    RETURNING
                        evaluation_run_id,
                        similarity_metric,
                        top_k,
                        status,
                        question_count,
                        runtime_metadata
                    """,
                    (question_set_id, f"slice-036-{uuid4()}"),
                )
                run = cursor.fetchone()
                cursor.execute(
                    """
                    INSERT INTO golden_evaluation_results (
                        evaluation_run_id,
                        question_id,
                        top_k
                    )
                    VALUES (%s, %s, 5)
                    RETURNING
                        evaluation_result_id,
                        visible_expected_count,
                        retrieved_count,
                        matched_visible_count,
                        hidden_violation_count,
                        matched_chunk_ids,
                        hidden_violation_chunk_ids,
                        dcg,
                        ideal_dcg,
                        metadata
                    """,
                    (run["evaluation_run_id"], question_id),
                )
                result = cursor.fetchone()
                cursor.execute(
                    """
                    DELETE FROM golden_question_sets
                    WHERE question_set_id = %s
                    """,
                    (question_set_id,),
                )
                cursor.execute(
                    """
                    SELECT count(*) AS count
                    FROM golden_evaluation_runs
                    WHERE evaluation_run_id = %s
                    """,
                    (run["evaluation_run_id"],),
                )
                run_count = cursor.fetchone()["count"]
                cursor.execute(
                    """
                    SELECT count(*) AS count
                    FROM golden_evaluation_results
                    WHERE evaluation_result_id = %s
                    """,
                    (result["evaluation_result_id"],),
                )
                result_count = cursor.fetchone()["count"]

        assert run["similarity_metric"] == "cosine"
        assert run["top_k"] == 5
        assert run["status"] == "pending"
        assert run["question_count"] == 0
        assert run["runtime_metadata"] == {}
        assert result["visible_expected_count"] == 0
        assert result["retrieved_count"] == 0
        assert result["matched_visible_count"] == 0
        assert result["hidden_violation_count"] == 0
        assert result["matched_chunk_ids"] == []
        assert result["hidden_violation_chunk_ids"] == []
        assert result["dcg"] == 0
        assert result["ideal_dcg"] == 0
        assert result["metadata"] == {}
        assert run_count == 0
        assert result_count == 0
    finally:
        _cleanup_question_set(migrated_database_url, set_name)


def test_golden_evaluation_schema_constraints(migrated_database_url: str) -> None:
    question_set_id, question_id, _user_id, set_name = _create_question_fixture(
        migrated_database_url
    )
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO golden_evaluation_runs (
                        question_set_id,
                        run_name,
                        profile_name,
                        question_count
                    )
                    VALUES (%s, %s, 'kure_v1_1024', 1)
                    RETURNING evaluation_run_id
                    """,
                    (question_set_id, f"slice-036-valid-{uuid4()}"),
                )
                evaluation_run_id = cursor.fetchone()["evaluation_run_id"]

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_evaluation_runs (
                            question_set_id,
                            run_name,
                            profile_name,
                            status
                        )
                        VALUES (%s, 'bad-status', 'kure_v1_1024', 'paused')
                        """,
                        (question_set_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_evaluation_runs (
                            question_set_id,
                            run_name,
                            profile_name,
                            mean_recall_at_k
                        )
                        VALUES (%s, 'bad-metric', 'kure_v1_1024', 1.5)
                        """,
                        (question_set_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_evaluation_runs (
                            question_set_id,
                            run_name,
                            profile_name,
                            runtime_metadata
                        )
                        VALUES (%s, 'bad-metadata', 'kure_v1_1024', '[]'::jsonb)
                        """,
                        (question_set_id,),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_evaluation_results (
                            evaluation_run_id,
                            question_id,
                            top_k
                        )
                        VALUES (%s, %s, 0)
                        """,
                        (evaluation_run_id, question_id),
                    )
                connection.rollback()

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO golden_evaluation_results (
                            evaluation_run_id,
                            question_id,
                            top_k,
                            matched_visible_count
                        )
                        VALUES (%s, %s, 5, -1)
                        """,
                        (evaluation_run_id, question_id),
                    )
                connection.rollback()
    finally:
        _cleanup_question_set(migrated_database_url, set_name)
