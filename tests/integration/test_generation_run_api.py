from urllib.parse import parse_qs, urlsplit
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


def _create_generation_api_fixture(database_url: str) -> tuple[int, int]:
    ids = _seed_owner_ids(database_url)
    checksum = f"generation-api-{uuid4()}"
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
                VALUES (%s, %s, '.md', 128, %s, %s, %s, 'slice-340')
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
                VALUES (%s, 'Generation API document', 'slice-340', %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, ids["alice.member"], ids["NeX Company"]),
            )
            document_id = int(cursor.fetchone()["document_id"])
            chunk_text = "생성 API 테스트에서 답변 근거로 사용되는 핵심 검색 근거입니다."
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
                    %s, %s, 2, 0, %s, 8, %s, %s
                )
                RETURNING chunk_id
                """,
                (
                    document_id,
                    chunk_text,
                    chunk_text,
                    f"chunk-{checksum}",
                    ["Generation", "API"],
                    Json({"start_line": 1, "end_line": 1}),
                    len(chunk_text),
                    len(chunk_text),
                    Json({"fixture": "slice-340"}),
                ),
            )
            chunk_id = int(cursor.fetchone()["chunk_id"])

    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text="generation API 테스트",
            normalized_query_text="generation API 테스트",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"scope": "company"},
            document_group="slice-340",
            top_k=1,
            profiles=("reranked_vector_cosine",),
            chunk_policy_name="heading_512_64",
            strategy_name="reranked_vector_cosine",
            created_by="slice-340-test",
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
            cursor.execute("DELETE FROM search_logs WHERE document_group = 'slice-340'")
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _generation_run_count(database_url: str, search_log_id: int) -> int:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*) AS run_count
                FROM generation_runs
                WHERE search_log_id = %s
                """,
                (search_log_id,),
            )
            return int(cursor.fetchone()["run_count"])


def test_mock_generation_run_api_creates_and_reads_generation_run(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            create_response = client.post(
                f"/api/search/logs/{search_log_id}/generation-runs/mock",
                params={
                    "max_context_chars": 4000,
                    "include_neighbors": "false",
                    "max_items": 5,
                },
            )
            create_body = create_response.json()
            run_id = create_body["run"]["generation_run_id"]
            read_response = client.get(f"/api/generation/runs/{run_id}")
            invalid_response = client.get("/api/generation/runs/0")
            missing_response = client.post("/api/search/logs/999999999/generation-runs/mock")

        read_body = read_response.json()
        assert create_response.status_code == 201
        assert create_body["provider"]["provider_mode"] == "mock"
        assert create_body["run"]["search_log_id"] == search_log_id
        assert create_body["run"]["status"] == "succeeded"
        assert create_body["run"]["finish_reason"] == "mock_completed"
        assert create_body["run"]["retrieval_confidence_status"] == "answerable"
        assert create_body["run"]["citation_readiness_status"] == "warning"
        assert "[RCP-001]" in create_body["run"]["answer_text"]
        assert create_body["prompt_package"]["messages"][0]["role"] == "system"
        assert create_body["prompt_package"]["blocked"] is False
        assert create_body["citations"][0]["citation_key"] == "RCP-001"
        assert create_body["citations"][0]["was_cited"] is True
        assert read_response.status_code == 200
        assert read_body["run"]["generation_run_id"] == run_id
        assert read_body["citations"][0]["source_label"].endswith("/ p.2")
        assert invalid_response.status_code == 400
        assert missing_response.status_code == 404
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_generation_prompt_preview_api_returns_messages_without_creating_run(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        before_count = _generation_run_count(migrated_database_url, search_log_id)
        with TestClient(app) as client:
            preview_response = client.get(
                f"/api/search/logs/{search_log_id}/generation-prompt/preview",
                params={
                    "max_context_chars": "4000",
                    "include_neighbors": "false",
                    "max_items": "5",
                    "response_language": "ko",
                },
            )
            missing_response = client.get("/api/search/logs/999999999/generation-prompt/preview")

        after_count = _generation_run_count(migrated_database_url, search_log_id)
        body = preview_response.json()
        prompt_package = body["prompt_package"]
        assert preview_response.status_code == 200
        assert body["retrieval_context"]["search_log"]["search_log_id"] == search_log_id
        assert prompt_package["messages"][0]["role"] == "system"
        assert prompt_package["messages"][1]["role"] == "user"
        assert prompt_package["response_language"] == "ko"
        assert prompt_package["blocked"] is False
        assert prompt_package["block_reason"] is None
        assert prompt_package["citation_keys"] == ["RCP-001"]
        assert prompt_package["prompt_hash"]
        assert prompt_package["context_hash"]
        assert after_count == before_count
        assert missing_response.status_code == 404
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_generation_run_api_returns_503_without_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        create_response = client.post("/api/search/logs/1/generation-runs/mock")
        read_response = client.get("/api/generation/runs/1")
        preview_response = client.get("/api/search/logs/1/generation-prompt/preview")
        metrics_response = client.get("/api/admin/generation-provider-metrics/snapshot")

    assert create_response.status_code == 503
    assert read_response.status_code == 503
    assert preview_response.status_code == 503
    assert metrics_response.status_code == 503


def test_generation_provider_metric_snapshot_api_reads_mock_persisted_metrics(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            create_response = client.post(
                f"/api/search/logs/{search_log_id}/generation-runs/mock",
                params={"include_neighbors": "false"},
            )
            run_id = create_response.json()["run"]["generation_run_id"]
            snapshot_response = client.get(
                "/api/admin/generation-provider-metrics/snapshot",
                params={"limit": "10"},
            )
            invalid_limit_response = client.get(
                "/api/admin/generation-provider-metrics/snapshot",
                params={"limit": "0"},
            )

        snapshot_body = snapshot_response.json()
        matching_runs = [run for run in snapshot_body["runs"] if run["generation_run_id"] == run_id]
        assert create_response.status_code == 201
        assert snapshot_response.status_code == 200
        assert snapshot_body["summary"]["run_count"] >= 1
        assert snapshot_body["summary"]["metric_present_count"] >= 1
        assert matching_runs
        assert matching_runs[0]["provider_name"] == "mock_qwen36_27b_nvfp4"
        assert matching_runs[0]["metric_present"] is True
        assert matching_runs[0]["succeeded"] is True
        assert matching_runs[0]["finish_reason"] == "mock_completed"
        assert (
            matching_runs[0]["total_token_count"]
            == create_response.json()["run"]["total_token_count"]
        )
        assert invalid_limit_response.status_code == 400
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_generation_run_ui_loads_context_and_creates_mock_run(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            landing_response = client.get("/generation")
            context_response = client.get(
                "/generation",
                params={"search_log_id": search_log_id, "include_neighbors": "false"},
            )
            post_response = client.post(
                "/generation/runs/mock",
                data={
                    "search_log_id": str(search_log_id),
                    "max_context_chars": "4000",
                    "max_items": "5",
                },
                follow_redirects=False,
            )
            detail_response = client.get(post_response.headers["location"])

        assert landing_response.status_code == 200
        assert 'href="/generation"' in landing_response.text
        assert "생성 실행" in landing_response.text
        assert context_response.status_code == 200
        assert "생성 입력 준비" in context_response.text
        assert "Prompt Preview" in context_response.text
        assert "grounded_answer_v1" in context_response.text
        assert "Mock 생성" in context_response.text
        assert post_response.status_code == 303
        assert "generation_run_id=" in post_response.headers["location"]
        assert detail_response.status_code == 200
        assert "Mock 생성 실행이 완료되었습니다." in detail_response.text
        assert "생성 결과" in detail_response.text
        assert "상세" in detail_response.text
        assert "제공된 문서 근거에 따르면" in detail_response.text
        assert "Citation Trace" in detail_response.text
        assert "RCP-001" in detail_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_generation_run_detail_ui_shows_prompt_metadata_and_citations(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            post_response = client.post(
                "/generation/runs/mock",
                data={
                    "search_log_id": str(search_log_id),
                    "max_context_chars": "4000",
                    "max_items": "5",
                },
                follow_redirects=False,
            )
            location = post_response.headers["location"]
            query = parse_qs(urlsplit(location).query)
            run_id = int(query["generation_run_id"][0])
            detail_response = client.get(f"/generation/runs/{run_id}")
            missing_response = client.get("/generation/runs/999999999")

        assert post_response.status_code == 303
        assert detail_response.status_code == 200
        assert "생성 실행 상세" in detail_response.text
        assert "Prompt Messages" in detail_response.text
        assert "Request Metadata" in detail_response.text
        assert "Response Metadata" in detail_response.text
        assert "Guardrail Metadata" in detail_response.text
        assert "grounded_answer_v1" in detail_response.text
        assert "generation_ui_mock" in detail_response.text
        assert "RCP-001" in detail_response.text
        assert "Generation API document" in detail_response.text
        assert missing_response.status_code == 200
        assert "Generation run not found." in missing_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)
