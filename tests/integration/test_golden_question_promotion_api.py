from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.golden_questions import (
    GoldenQuestionSetInput,
    create_golden_question_set,
    get_golden_question_detail,
)
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    SearchResultFeedbackInput,
    create_search_log,
    create_search_log_results,
    create_search_result_feedback,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_promotion_fixture(database_url: str) -> dict[str, object]:
    checksum = f"golden-promotion-{uuid4()}"
    document_group = f"slice-043-{uuid4()}"
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
                VALUES (%s, %s, %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, f"Golden promotion fixture {checksum}", document_group, user_id),
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
                    'Golden promotion expected chunk',
                    %s,
                    'heading_512_64',
                    ARRAY['Promotion', 'Scope'],
                    31
                )
                RETURNING chunk_id
                """,
                (document_id, f"chunk-{checksum}"),
            )
            chunk_id = cursor.fetchone()["chunk_id"]

    question_set = create_golden_question_set(
        database_url,
        GoldenQuestionSetInput(
            set_name=f"slice-043-{uuid4()}",
            description="Slice 043 promotion fixture",
            created_by_user_id=user_id,
        ),
    )
    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text="Which policy can be promoted?",
            normalized_query_text="which policy can be promoted?",
            actor_user_id=user_id,
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"actor_user_id": user_id},
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024",),
            query_runtime_metadata={"adapter": "mock"},
            total_elapsed_ms=11,
            created_by="integration-test",
            created_by_user_id=user_id,
        ),
    )
    result = create_search_log_results(
        database_url,
        [
            SearchLogResultInput(
                search_log_id=search_log.search_log_id,
                profile_name="kure_v1_1024",
                rank=1,
                chunk_id=chunk_id,
                distance=0.1,
                score=0.9,
                profile_elapsed_ms=4,
            )
        ],
    )[0]
    create_search_result_feedback(
        database_url,
        SearchResultFeedbackInput(
            search_log_result_id=result.search_log_result_id,
            relevance_label="correct",
            comment="promote this",
            created_by="integration-test",
            created_by_user_id=user_id,
        ),
    )
    return {
        "file_id": file_id,
        "question_set_id": question_set.question_set_id,
        "question_set_name": question_set.set_name,
        "document_group": document_group,
        "search_log_id": search_log.search_log_id,
        "search_log_result_id": result.search_log_result_id,
        "chunk_id": chunk_id,
        "user_id": user_id,
    }


def _cleanup_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM golden_question_sets WHERE question_set_id = %s",
                (fixture["question_set_id"],),
            )
            cursor.execute(
                "DELETE FROM search_logs WHERE search_log_id = %s",
                (fixture["search_log_id"],),
            )
            cursor.execute("DELETE FROM files WHERE file_id = %s", (fixture["file_id"],))


def test_search_result_can_be_promoted_to_golden_question(
    migrated_database_url: str,
) -> None:
    fixture = _create_promotion_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        with TestClient(app) as client:
            candidate_response = client.get(
                "/api/evaluations/golden-question-candidates",
                params={"document_group": fixture["document_group"]},
            )
            response = client.post(
                f"/api/search/results/{fixture['search_log_result_id']}/promote-golden-question",
                json={
                    "question_set_id": fixture["question_set_id"],
                    "metadata": {"case": "slice-043"},
                    "notes": "promoted from feedback",
                },
            )
            remaining_candidate_response = client.get(
                "/api/evaluations/golden-question-candidates",
                params={"document_group": fixture["document_group"]},
            )
            promoted_candidate_response = client.get(
                "/api/evaluations/golden-question-candidates",
                params={
                    "document_group": fixture["document_group"],
                    "include_promoted": True,
                },
            )
            page_response = client.get(f"/search/logs?search_log_id={fixture['search_log_id']}")

        candidate = candidate_response.json()["candidates"][0]
        body = response.json()["promotion"]
        question = body["question"]
        target = body["expected_target"]
        detail = get_golden_question_detail(migrated_database_url, question["question_id"])

        assert candidate_response.status_code == 200
        assert candidate["search_log_result_id"] == fixture["search_log_result_id"]
        assert candidate["search_log_id"] == fixture["search_log_id"]
        assert candidate["query_text"] == "Which policy can be promoted?"
        assert candidate["document_group"] == fixture["document_group"]
        assert candidate["profile_name"] == "kure_v1_1024"
        assert candidate["feedback_count"] == 1
        assert candidate["correct_count"] == 1
        assert candidate["partial_count"] == 0
        assert candidate["feedback_labels"] == ["correct"]
        assert candidate["latest_feedback_comment"] == "promote this"
        assert candidate["already_promoted"] is False
        assert response.status_code == 201
        assert question["question_set_id"] == fixture["question_set_id"]
        assert question["question_text"] == "Which policy can be promoted?"
        assert question["actor_user_id"] == fixture["user_id"]
        assert question["requested_search_scope"] == "company"
        assert question["document_group"].startswith("slice-043-")
        assert question["metadata"]["case"] == "slice-043"
        assert question["metadata"]["promotion"]["search_log_id"] == fixture["search_log_id"]
        assert question["metadata"]["promotion"]["feedback_labels"] == ["correct"]
        assert target["question_id"] == question["question_id"]
        assert target["chunk_id"] == fixture["chunk_id"]
        assert target["expected_heading_path"] == ["Promotion", "Scope"]
        assert target["notes"] == "promoted from feedback"
        assert body["source"]["search_log_result_id"] == fixture["search_log_result_id"]
        assert detail is not None
        assert len(detail.expected_targets) == 1
        assert remaining_candidate_response.status_code == 200
        assert remaining_candidate_response.json()["candidates"] == []
        assert promoted_candidate_response.status_code == 200
        assert promoted_candidate_response.json()["candidates"][0]["already_promoted"] is True
        assert page_response.status_code == 200
        assert "search-history-promotion" in page_response.text
        assert "/promote-golden-question" in page_response.text
        assert fixture["question_set_name"] in page_response.text
    finally:
        _cleanup_fixture(migrated_database_url, fixture)


def test_search_result_promotion_returns_not_found_for_missing_result(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/results/999999999/promote-golden-question",
            json={"question_set_id": 1},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Search result not found."}
