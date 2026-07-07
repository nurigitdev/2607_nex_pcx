from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    SearchResultFeedbackInput,
    create_search_log,
    create_search_log_results,
    create_search_result_feedback,
    get_search_log,
    get_search_log_result,
    list_search_log_results,
)

pytestmark = pytest.mark.integration


def _create_search_fixture(database_url: str) -> tuple[int, int, int]:
    checksum = f"search-repository-{uuid4()}"
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
                VALUES (%s, %s, '.md', 1, %s, %s, %s, 'slice-026')
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
                VALUES (%s, %s, 'slice-026', %s, 'personal')
                RETURNING document_id
                """,
                (file_id, f"Search repository fixture {checksum}", user_id),
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
                VALUES (%s, 0, 'Search repository chunk', %s, 'heading_512_64', 23)
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


def test_search_log_repository_persists_results_and_feedback(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id, user_id = _create_search_fixture(migrated_database_url)
    try:
        search_log = create_search_log(
            migrated_database_url,
            SearchLogInput(
                query_text="What is the repository foundation?",
                normalized_query_text="what is the repository foundation?",
                actor_user_id=user_id,
                requested_search_scope="mine",
                effective_search_scope="mine",
                permission_filter_metadata={"actor_user_id": user_id},
                document_group="slice-026",
                file_type=".md",
                chunk_policy_name="heading_512_64",
                top_k=3,
                profiles=("kure_v1_1024",),
                query_runtime_metadata={"adapter": "mock"},
                total_elapsed_ms=15,
                created_by="integration-test",
                created_by_user_id=user_id,
            ),
        )
        results = create_search_log_results(
            migrated_database_url,
            [
                SearchLogResultInput(
                    search_log_id=search_log.search_log_id,
                    profile_name="kure_v1_1024",
                    rank=1,
                    chunk_id=chunk_id,
                    distance=0.12,
                    score=0.88,
                    profile_elapsed_ms=7,
                )
            ],
        )
        feedback = create_search_result_feedback(
            migrated_database_url,
            SearchResultFeedbackInput(
                search_log_result_id=results[0].search_log_result_id,
                relevance_label="correct",
                comment="expected result",
                created_by="integration-test",
                created_by_user_id=user_id,
            ),
        )
        stored_log = get_search_log(migrated_database_url, search_log.search_log_id)
        stored_result = get_search_log_result(
            migrated_database_url,
            results[0].search_log_result_id,
        )
        stored_results = list_search_log_results(migrated_database_url, search_log.search_log_id)

        assert stored_log == search_log
        assert stored_result == results[0]
        assert search_log.profiles == ("kure_v1_1024",)
        assert search_log.permission_filter_metadata == {"actor_user_id": user_id}
        assert search_log.query_runtime_metadata == {"adapter": "mock"}
        assert stored_results == results
        assert results[0].rank == 1
        assert results[0].chunk_id == chunk_id
        assert results[0].distance == pytest.approx(0.12)
        assert feedback.relevance_label == "correct"
        assert feedback.comment == "expected result"
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_search_log_repository_returns_none_and_empty_results_for_missing_log(
    migrated_database_url: str,
) -> None:
    assert get_search_log(migrated_database_url, 999999999) is None
    assert get_search_log_result(migrated_database_url, 999999999) is None
    assert list_search_log_results(migrated_database_url, 999999999) == []
