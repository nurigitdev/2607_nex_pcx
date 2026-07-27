from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Json

from app.core.bm25_keyword_index import (
    KOREAN_NGRAM_BM25_TOKENIZER_NAME,
    refresh_chunk_policy_keyword_index,
)
from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.generation_runs import GENERATION_PROVIDER_MODE_MOCK
from app.main import create_app

pytestmark = pytest.mark.integration


def _seed_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id = 'alice.member'
                """)
            user = cursor.fetchone()
            cursor.execute("""
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name = 'NeX Company'
                """)
            org = cursor.fetchone()
    return {
        "alice.member": int(user["user_id"]),
        "NeX Company": int(org["org_unit_id"]),
    }


def _create_chunk_policy(database_url: str, chunk_policy_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chunk_policies (
                    chunk_policy_name,
                    target_token_size,
                    overlap_token_size,
                    split_strategy,
                    description
                )
                VALUES (%s, 512, 64, 'direct-generation-fixture', 'Direct generation fixture')
                """,
                (chunk_policy_name,),
            )


def _create_direct_generation_chunk(
    database_url: str,
    *,
    owner_org_unit_id: int,
    document_group: str,
    chunk_policy_name: str,
) -> tuple[int, int]:
    checksum = f"direct-generation-{uuid4()}"
    chunk_text = (
        "Direct generation BM25 anchor 문서는 사용자가 prompt query만 입력해도 "
        "내부 검색과 context packaging 후 grounded answer를 생성한다."
    )
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
                    document_group
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    document_group,
                ),
            )
            file_id = int(cursor.fetchone()["file_id"])
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    owner_org_unit_id,
                    access_scope
                )
                VALUES (%s, 'direct generation fixture', %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, document_group, owner_org_unit_id),
            )
            document_id = int(cursor.fetchone()["document_id"])
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
                    %s, 0, 'text', %s, %s, %s, %s,
                    %s, %s, 1, 0, %s, 16, %s, %s
                )
                RETURNING chunk_id
                """,
                (
                    document_id,
                    chunk_text,
                    chunk_text,
                    f"chunk-{checksum}",
                    chunk_policy_name,
                    ["Direct Generation"],
                    Json({"start_line": 1, "end_line": 1}),
                    len(chunk_text),
                    len(chunk_text),
                    Json({"fixture": "slice-359"}),
                ),
            )
            chunk_id = int(cursor.fetchone()["chunk_id"])
    return file_id, chunk_id


def _restore_mock_generation_provider(database_url: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                UPDATE generation_provider_configs
                SET is_default = false
                WHERE is_default
                """)
            cursor.execute("""
                UPDATE generation_provider_configs
                SET provider_mode = 'mock',
                    provider_base_url = NULL,
                    is_default = true,
                    is_active = true
                WHERE provider_name = 'mock_qwen36_27b_nvfp4'
                """)


def _cleanup_fixture(database_url: str, file_id: int, chunk_policy_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                DELETE FROM search_logs
                WHERE document_group LIKE 'slice-359-%'
                """)
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            cursor.execute(
                "DELETE FROM chunk_keyword_statistics WHERE chunk_policy_name = %s",
                (chunk_policy_name,),
            )
            cursor.execute(
                "DELETE FROM chunk_policies WHERE chunk_policy_name = %s",
                (chunk_policy_name,),
            )


def test_direct_generation_query_api_searches_packages_and_creates_mock_run(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    document_group = f"slice-359-{uuid4()}"
    chunk_policy_name = f"direct_generation_{uuid4().hex}"
    _create_chunk_policy(migrated_database_url, chunk_policy_name)
    file_id, chunk_id = _create_direct_generation_chunk(
        migrated_database_url,
        owner_org_unit_id=ids["NeX Company"],
        document_group=document_group,
        chunk_policy_name=chunk_policy_name,
    )
    _restore_mock_generation_provider(migrated_database_url)
    refresh_chunk_policy_keyword_index(
        migrated_database_url,
        chunk_policy_name=chunk_policy_name,
        tokenizer_name=KOREAN_NGRAM_BM25_TOKENIZER_NAME,
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/generation/direct-runs",
                json={
                    "query_text": "direct generation BM25 anchor",
                    "actor_user_id": ids["alice.member"],
                    "requested_search_scope": "company",
                    "provider_mode": GENERATION_PROVIDER_MODE_MOCK,
                    "top_k": 3,
                    "profiles": ["bm25_keyword"],
                    "chunk_policy_name": chunk_policy_name,
                    "document_group": document_group,
                    "bm25_tokenizer_name": KOREAN_NGRAM_BM25_TOKENIZER_NAME,
                    "include_neighbors": False,
                    "max_context_chars": 4000,
                    "max_items": 5,
                },
            )
            invalid_response = client.post(
                "/api/generation/direct-runs",
                json={
                    "query_text": "direct generation BM25 anchor",
                    "actor_user_id": ids["alice.member"],
                    "provider_mode": "unsupported",
                },
            )

        body = response.json()
        run_row = fetch_one(
            migrated_database_url,
            """
            SELECT search_log_id, provider_mode, status, created_by, created_by_user_id
            FROM generation_runs
            WHERE generation_run_id = %s
            """,
            (body["generation_run_id"],),
        )

        assert response.status_code == 201
        assert body["mode"] == "direct_query"
        assert body["search_log_id"] == body["search"]["search_log_id"]
        assert body["generation_run_id"] == body["generation"]["run"]["generation_run_id"]
        assert body["search"]["profiles"][0]["profile_name"] == "bm25_keyword"
        assert body["search"]["profiles"][0]["results"][0]["chunk_id"] == chunk_id
        assert body["retrieval_context"]["summary"]["included_count"] == 1
        assert (
            body["retrieval_context"]["included_candidates"][0]["citation"]["citation_key"]
            == "RCP-001"
        )
        assert body["generation"]["run"]["provider_mode"] == "mock"
        assert body["generation"]["run"]["status"] == "succeeded"
        assert body["generation"]["run"]["created_by"] == "api_direct_generation"
        assert body["generation"]["run"]["created_by_user_id"] == ids["alice.member"]
        assert body["links"]["generation_run"] == f"/generation/runs/{body['generation_run_id']}"
        assert run_row["search_log_id"] == body["search_log_id"]
        assert run_row["provider_mode"] == "mock"
        assert run_row["created_by"] == "api_direct_generation"
        assert run_row["created_by_user_id"] == ids["alice.member"]
        assert invalid_response.status_code == 400
        assert "provider_mode is not supported" in invalid_response.text
    finally:
        _cleanup_fixture(migrated_database_url, file_id, chunk_policy_name)


def test_direct_generation_query_api_returns_503_without_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/generation/direct-runs",
            json={
                "query_text": "direct generation",
                "actor_user_id": 1,
                "provider_mode": GENERATION_PROVIDER_MODE_MOCK,
            },
        )

    assert response.status_code == 503
