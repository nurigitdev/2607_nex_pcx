from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Json

from app.core.config import Settings
from app.core.database import connect
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _seed_owner_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id = 'alice.member'
                """)
            user_row = cursor.fetchone()
            cursor.execute("""
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name = 'NeX Company'
                """)
            org_row = cursor.fetchone()
    return {
        "alice.member": int(user_row["user_id"]),
        "NeX Company": int(org_row["org_unit_id"]),
    }


def _create_citation_readiness_fixture(database_url: str) -> tuple[int, int]:
    ids = _seed_owner_ids(database_url)
    checksum = f"citation-readiness-{uuid4()}"
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
                    storage_path,
                    uploaded_by_user_id,
                    document_group
                )
                VALUES (%s, %s, '.md', 128, %s, %s, %s, 'slice-333')
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    ids["alice.member"],
                ),
            )
            file_id = int(cursor.fetchone()["file_id"])
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    owner_user_id,
                    owner_org_unit_id,
                    access_scope
                )
                VALUES (%s, 'Citation readiness document', 'slice-333', %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, ids["alice.member"], ids["NeX Company"]),
            )
            document_id = int(cursor.fetchone()["document_id"])
            chunk_text = "생성 답변의 citation readiness를 확인하기 위한 핵심 근거입니다."
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_type,
                    chunk_text,
                    content_markdown,
                    content_hash,
                    chunk_policy_name,
                    heading_path,
                    source_anchor,
                    page_no,
                    source_char_start,
                    source_char_end,
                    token_count,
                    char_count,
                    metadata
                )
                VALUES (
                    %s, 0, 'text', %s, %s, %s, 'heading_512_64',
                    %s, %s, 3, 0, %s, 8, %s, %s
                )
                RETURNING chunk_id
                """,
                (
                    document_id,
                    chunk_text,
                    chunk_text,
                    f"chunk-{checksum}",
                    ["Citation", "Readiness"],
                    Json({"start_line": 1, "end_line": 1}),
                    len(chunk_text),
                    len(chunk_text),
                    Json({"fixture": "slice-333"}),
                ),
            )
            chunk_id = int(cursor.fetchone()["chunk_id"])

    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text="citation readiness 테스트",
            normalized_query_text="citation readiness 테스트",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"scope": "company"},
            document_group="slice-333",
            top_k=1,
            profiles=("reranked_vector_cosine",),
            chunk_policy_name="heading_512_64",
            strategy_name="reranked_vector_cosine",
            created_by="slice-333-test",
        ),
    )
    create_search_log_results(
        database_url,
        [
            SearchLogResultInput(
                search_log_id=search_log.search_log_id,
                profile_name="reranked_vector_cosine",
                search_profile_name="reranked_vector_cosine",
                retrieval_strategy="reranked",
                rank=1,
                chunk_id=chunk_id,
                score=0.92,
                profile_elapsed_ms=12,
            )
        ],
    )
    return file_id, search_log.search_log_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM search_logs WHERE document_group = 'slice-333'")
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_citation_readiness_api_and_ui(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_citation_readiness_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            api_response = client.get(
                f"/api/search/logs/{search_log_id}/citation-readiness",
                params={
                    "max_context_chars": 4000,
                    "include_neighbors": "true",
                    "max_items": 5,
                },
            )
            missing_response = client.get("/api/search/logs/999999999/citation-readiness")
            invalid_response = client.get("/api/search/logs/0/citation-readiness")
            page_response = client.get(
                "/search/citation-readiness",
                params={"search_log_id": search_log_id, "max_context_chars": 4000},
            )
            list_page_response = client.get("/search/citation-readiness")

        body = api_response.json()
        assert api_response.status_code == 200
        assert body["search_log_id"] == search_log_id
        assert body["retrieval_confidence"]["status"] == "answerable"
        assert body["summary"]["included_candidate_count"] == 1
        assert body["summary"]["source_anchor_coverage_percent"] == "100.00"
        assert body["summary"]["warning_count"] == 1
        assert body["summary"]["status"] == "warning"
        assert body["candidates"][0]["citation_key"] == "RCP-001"
        assert body["candidates"][0]["has_source_anchor"] is True
        assert body["candidates"][0]["has_lineage_reference"] is False
        assert body["candidates"][0]["issues"][0]["code"] == "missing_artifact_block_reference"
        assert missing_response.status_code == 404
        assert invalid_response.status_code == 400
        assert page_response.status_code == 200
        assert "Citation Coverage 점검" in page_response.text
        assert "data-citation-readiness-report" in page_response.text
        assert "data-retrieval-confidence-guardrail" not in page_response.text
        assert "missing_artifact_block_reference" in page_response.text
        assert list_page_response.status_code == 200
        assert "최근 검색 로그" in list_page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)
