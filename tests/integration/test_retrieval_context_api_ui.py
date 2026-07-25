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


def _create_retrieval_context_fixture(
    database_url: str,
    *,
    link_neighbors: bool = True,
) -> tuple[int, int]:
    ids = _seed_owner_ids(database_url)
    checksum = f"retrieval-context-{uuid4()}"
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
                VALUES (%s, %s, '.md', 128, %s, %s, %s, 'slice-332')
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
                VALUES (%s, 'Retrieval package document', 'slice-332', %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, ids["alice.member"], ids["NeX Company"]),
            )
            document_id = int(cursor.fetchone()["document_id"])
            chunk_ids: list[int] = []
            chunk_texts = [
                "이전 근거 문장입니다.",
                "생성 단계에서 반드시 전달되어야 하는 핵심 검색 근거입니다.",
                "다음 근거 문장입니다.",
            ]
            for index, chunk_text in enumerate(chunk_texts):
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
                        %s, %s, 'text', %s, %s, %s, 'heading_512_64',
                        %s, %s, 2, %s, %s, 8, %s, %s
                    )
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        index,
                        chunk_text,
                        chunk_text,
                        f"chunk-{checksum}-{index}",
                        ["Retrieval", "Package"],
                        Json({"start_line": index + 1, "end_line": index + 1}),
                        index * 20,
                        index * 20 + len(chunk_text),
                        len(chunk_text),
                        Json({"fixture": "slice-332", "index": index}),
                    ),
                )
                chunk_ids.append(int(cursor.fetchone()["chunk_id"]))
            if link_neighbors:
                cursor.execute(
                    "UPDATE chunks SET next_chunk_id = %s WHERE chunk_id = %s",
                    (chunk_ids[1], chunk_ids[0]),
                )
                cursor.execute(
                    """
                    UPDATE chunks
                    SET prev_chunk_id = %s,
                        next_chunk_id = %s
                    WHERE chunk_id = %s
                    """,
                    (chunk_ids[0], chunk_ids[2], chunk_ids[1]),
                )
                cursor.execute(
                    "UPDATE chunks SET prev_chunk_id = %s WHERE chunk_id = %s",
                    (chunk_ids[1], chunk_ids[2]),
                )

    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text="생성 context package 테스트",
            normalized_query_text="생성 context package 테스트",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"scope": "company"},
            document_group="slice-332",
            top_k=2,
            profiles=("reranked_vector_cosine",),
            chunk_policy_name="heading_512_64",
            strategy_name="reranked_vector_cosine",
            created_by="slice-332-test",
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
                chunk_id=chunk_ids[1],
                score=0.92,
                profile_elapsed_ms=12,
            )
        ],
    )
    return file_id, search_log.search_log_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                DELETE FROM search_logs
                WHERE document_group = 'slice-332'
                """)
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_retrieval_context_package_api_and_ui(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_retrieval_context_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            api_response = client.get(
                f"/api/search/logs/{search_log_id}/retrieval-context",
                params={
                    "max_context_chars": 4000,
                    "include_neighbors": "false",
                    "max_items": 5,
                },
            )
            missing_response = client.get("/api/search/logs/999999999/retrieval-context")
            invalid_response = client.get("/api/search/logs/0/retrieval-context")
            page_response = client.get(
                "/search/context",
                params={"search_log_id": search_log_id, "max_context_chars": 4000},
            )
            list_page_response = client.get("/search/context")

        body = api_response.json()
        assert api_response.status_code == 200
        assert body["search_log"]["search_log_id"] == search_log_id
        assert body["retrieval_confidence"]["status"] == "answerable"
        assert body["retrieval_confidence"]["withhold_generation_context"] is False
        assert body["summary"]["included_count"] == 1
        assert body["summary"]["include_neighbors"] is False
        assert body["included_candidates"][0]["citation"]["citation_key"] == "RCP-001"
        assert body["included_candidates"][0]["citation"]["source_label"].endswith("/ p.2")
        assert len(body["included_candidates"][0]["chunks"]) == 1
        assert "핵심 검색 근거" in body["generation_context_text"]
        assert missing_response.status_code == 404
        assert invalid_response.status_code == 400
        assert page_response.status_code == 200
        assert "검색 Context 패키지" in page_response.text
        assert "data-retrieval-context-package" in page_response.text
        assert "data-retrieval-confidence-guardrail" not in page_response.text
        assert "생성 입력 Context" in page_response.text
        assert list_page_response.status_code == 200
        assert "최근 검색 로그" in list_page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_retrieval_context_package_handles_null_neighbor_links(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_retrieval_context_fixture(
        migrated_database_url,
        link_neighbors=False,
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.get(
                f"/api/search/logs/{search_log_id}/retrieval-context",
                params={"max_context_chars": 4000, "include_neighbors": "true"},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["summary"]["included_count"] == 1
        assert [chunk["position"] for chunk in body["included_candidates"][0]["chunks"]] == [
            "previous",
            "current",
            "next",
        ]
    finally:
        _cleanup_file(migrated_database_url, file_id)
