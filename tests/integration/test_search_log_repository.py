from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.search_logs import (
    InvalidSearchLogError,
    SearchLogInput,
    SearchLogResultInput,
    SearchLogReviewMetadataInput,
    SearchResultFeedbackInput,
    create_search_log,
    create_search_log_results,
    create_search_result_feedback,
    get_search_log,
    get_search_log_detail,
    get_search_log_result,
    list_search_log_results,
    list_search_logs,
    summarize_search_feedback,
    update_search_log_review_metadata,
)

pytestmark = pytest.mark.integration


def _create_search_fixture(database_url: str) -> tuple[int, int, int, str]:
    checksum = f"search-repository-{uuid4()}"
    document_group = f"slice-026-{uuid4()}"
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
                VALUES (%s, %s, '.md', 1, %s, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    user_id,
                    document_group,
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
                VALUES (%s, %s, %s, %s, 'personal')
                RETURNING document_id
                """,
                (file_id, f"Search repository fixture {checksum}", document_group, user_id),
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

    return file_id, chunk_id, user_id, document_group


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
    file_id, chunk_id, user_id, document_group = _create_search_fixture(migrated_database_url)
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
                document_group=document_group,
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
                    score_components={"vector_score": 0.88},
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
        updated_log = update_search_log_review_metadata(
            migrated_database_url,
            SearchLogReviewMetadataInput(
                search_log_id=search_log.search_log_id,
                review_tags=("baseline", "needs-review"),
                review_memo=" Review after initial experiment ",
                reviewed_by_user_id=user_id,
            ),
        )
        stored_log = get_search_log(migrated_database_url, search_log.search_log_id)
        stored_result = get_search_log_result(
            migrated_database_url,
            results[0].search_log_result_id,
        )
        stored_results = list_search_log_results(migrated_database_url, search_log.search_log_id)
        summary = summarize_search_feedback(
            migrated_database_url,
            document_group=document_group,
        )
        history = list_search_logs(
            migrated_database_url,
            actor_user_id=user_id,
            requested_search_scope="mine",
            document_group=document_group,
        )
        detail = get_search_log_detail(migrated_database_url, search_log.search_log_id)
        profile_summary = next(
            profile for profile in summary.profiles if profile.profile_name == "kure_v1_1024"
        )

        assert search_log.review_tags == ()
        assert search_log.review_memo is None
        assert search_log.reviewed_by_user_id is None
        assert search_log.reviewed_at is None
        assert updated_log is not None
        assert stored_log == updated_log
        assert updated_log.review_tags == ("baseline", "needs-review")
        assert updated_log.review_memo == "Review after initial experiment"
        assert updated_log.reviewed_by_user_id == user_id
        assert updated_log.reviewed_at is not None
        assert stored_result == results[0]
        assert updated_log.profiles == ("kure_v1_1024",)
        assert updated_log.strategy_name == "vector_cosine"
        assert updated_log.permission_filter_metadata == {"actor_user_id": user_id}
        assert updated_log.query_runtime_metadata == {"adapter": "mock"}
        assert stored_results == results
        assert results[0].rank == 1
        assert results[0].chunk_id == chunk_id
        assert results[0].distance == pytest.approx(0.12)
        assert results[0].search_profile_name == "kure_v1_1024"
        assert results[0].retrieval_strategy == "vector_cosine"
        assert results[0].score_components == {"vector_score": 0.88}
        assert feedback.relevance_label == "correct"
        assert feedback.comment == "expected result"
        assert summary.feedback_count == 1
        assert summary.search_log_count == 1
        assert summary.result_count == 1
        assert profile_summary.feedback_count == 1
        assert profile_summary.correct_count == 1
        assert profile_summary.relevant_count == 1
        assert profile_summary.average_rank == pytest.approx(1)
        assert profile_summary.average_score == pytest.approx(0.88)
        assert len(history) == 1
        assert history[0].search_log == updated_log
        assert history[0].actor_login_id == "alice.member"
        assert history[0].result_count == 1
        assert history[0].feedback_count == 1
        assert detail is not None
        assert detail.search_log == updated_log
        assert detail.actor_login_id == "alice.member"
        assert len(detail.results) == 1
        assert detail.results[0].search_log_result == results[0]
        assert detail.results[0].document_group == document_group
        assert detail.results[0].feedback == (feedback,)
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_search_log_repository_returns_none_and_empty_results_for_missing_log(
    migrated_database_url: str,
) -> None:
    assert get_search_log(migrated_database_url, 999999999) is None
    assert get_search_log_detail(migrated_database_url, 999999999) is None
    assert get_search_log_result(migrated_database_url, 999999999) is None
    assert list_search_log_results(migrated_database_url, 999999999) == []
    assert (
        update_search_log_review_metadata(
            migrated_database_url,
            SearchLogReviewMetadataInput(
                search_log_id=999999999,
                review_tags=("missing",),
            ),
        )
        is None
    )


def test_search_log_review_metadata_validation(migrated_database_url: str) -> None:
    with pytest.raises(InvalidSearchLogError, match="review_tags must be unique"):
        update_search_log_review_metadata(
            migrated_database_url,
            SearchLogReviewMetadataInput(
                search_log_id=1,
                review_tags=("duplicate", "duplicate"),
            ),
        )

    with pytest.raises(InvalidSearchLogError, match="review_tag must not be blank"):
        update_search_log_review_metadata(
            migrated_database_url,
            SearchLogReviewMetadataInput(
                search_log_id=1,
                review_tags=(" ",),
            ),
        )
