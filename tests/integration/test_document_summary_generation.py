from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Json

from app.core.config import Settings
from app.core.database import connect
from app.main import create_app

pytestmark = pytest.mark.integration

MOCK_PROVIDER_NAME = "mock_qwen36_27b_nvfp4"


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


def _restore_mock_generation_provider(database_url: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                UPDATE generation_provider_configs
                SET is_default = false
                WHERE is_default
                """)
            cursor.execute(
                """
                UPDATE generation_provider_configs
                SET provider_mode = 'mock',
                    provider_base_url = NULL,
                    is_default = true,
                    is_active = true
                WHERE provider_name = %s
                """,
                (MOCK_PROVIDER_NAME,),
            )


def _create_document_summary_fixture(database_url: str) -> tuple[int, int, int]:
    ids = _seed_ids(database_url)
    checksum = f"document-summary-{uuid4()}"
    document_group = f"slice-379-{uuid4()}"
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
                    document_group,
                    parse_status
                )
                VALUES (%s, %s, '.md', 256, %s, %s, %s, %s, 'succeeded')
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    ids["alice.member"],
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
                    owner_user_id,
                    owner_org_unit_id,
                    access_scope
                )
                VALUES (%s, 'Document Summary API document', %s, %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, document_group, ids["alice.member"], ids["NeX Company"]),
            )
            document_id = int(cursor.fetchone()["document_id"])
            chunk_ids: list[int] = []
            for seq, chunk_text in enumerate(
                (
                    "문서 요약 테스트 첫 번째 chunk는 시장 현황과 핵심 배경을 설명합니다.",
                    "문서 요약 테스트 두 번째 chunk는 리스크와 후속 조치 계획을 설명합니다.",
                )
            ):
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
                        %s, %s, %s, 0, %s, 10, %s, %s
                    )
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        seq,
                        chunk_text,
                        chunk_text,
                        f"chunk-{checksum}-{seq}",
                        ["Document Summary", f"Section {seq + 1}"],
                        Json({"start_line": seq + 1, "end_line": seq + 1}),
                        seq + 1,
                        len(chunk_text),
                        len(chunk_text),
                        Json({"fixture": "slice-379"}),
                    ),
                )
                chunk_ids.append(int(cursor.fetchone()["chunk_id"]))
            cursor.execute(
                """
                UPDATE chunks
                SET next_chunk_id = %s
                WHERE chunk_id = %s
                """,
                (chunk_ids[1], chunk_ids[0]),
            )
            cursor.execute(
                """
                UPDATE chunks
                SET prev_chunk_id = %s
                WHERE chunk_id = %s
                """,
                (chunk_ids[0], chunk_ids[1]),
            )
    return file_id, document_id, ids["alice.member"]


def _cleanup_fixture(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM search_logs WHERE document_group LIKE 'slice-379-%'")
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_document_summary_api_creates_search_log_and_generation_run(
    migrated_database_url: str,
) -> None:
    file_id, document_id, actor_user_id = _create_document_summary_fixture(migrated_database_url)
    _restore_mock_generation_provider(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/documents/{document_id}/summary-runs",
                json={
                    "actor_user_id": actor_user_id,
                    "summary_instruction": "리스크와 후속 조치 중심",
                    "provider_mode": "mock",
                    "generation_template_key": "summary",
                    "max_chunks": 2,
                    "max_context_chars": 4000,
                    "include_neighbors": False,
                    "chunk_policy_name": "heading_512_64",
                },
            )
            invalid_response = client.post(
                f"/api/documents/{document_id}/summary-runs",
                json={
                    "actor_user_id": actor_user_id,
                    "provider_mode": "unsupported",
                },
            )
            missing_response = client.post(
                "/api/documents/999999999/summary-runs",
                json={"actor_user_id": actor_user_id, "provider_mode": "mock"},
            )

        body = response.json()
        assert response.status_code == 201
        assert body["mode"] == "document_summary"
        assert body["document"]["document_id"] == document_id
        assert body["source_chunk_count"] == 2
        assert body["retrieval_context"]["summary"]["included_count"] == 2
        assert body["retrieval_context"]["search_log"]["strategy_name"] == "document_summary"
        assert body["retrieval_context"]["search_log"]["query_runtime_metadata"]["operation"] == (
            "document_summary"
        )
        assert body["generation"]["provider"]["provider_mode"] == "mock"
        assert body["generation"]["prompt_package"]["template_key"] == "summary"
        assert body["generation"]["run"]["status"] == "succeeded"
        assert body["generation"]["run"]["created_by"] == "api_document_summary"
        assert "핵심 요약" in body["generation"]["run"]["answer_text"]
        assert "[RCP-001]" in body["generation"]["run"]["answer_text"]
        assert body["generation"]["citations"][0]["document_id"] == document_id
        assert invalid_response.status_code == 400
        assert "provider_mode" in invalid_response.json()["detail"]
        assert missing_response.status_code == 400
        assert "document was not found" in missing_response.json()["detail"]
    finally:
        _cleanup_fixture(migrated_database_url, file_id)


def test_generation_ui_document_summary_form_redirects_to_created_run(
    migrated_database_url: str,
) -> None:
    file_id, document_id, actor_user_id = _create_document_summary_fixture(migrated_database_url)
    _restore_mock_generation_provider(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            page_response = client.get("/generation")
            response = client.post(
                "/generation/document-summaries",
                data={
                    "summary_document_id": str(document_id),
                    "summary_actor_user_id": str(actor_user_id),
                    "summary_instruction": "핵심 요약",
                    "summary_provider_mode": "mock",
                    "summary_generation_template_key": "summary",
                    "summary_max_chunks": "2",
                    "summary_max_context_chars": "4000",
                    "summary_chunk_policy_name": "heading_512_64",
                },
                follow_redirects=False,
            )
            missing_response = client.post(
                "/generation/document-summaries",
                data={
                    "summary_document_id": "999999999",
                    "summary_actor_user_id": str(actor_user_id),
                    "summary_provider_mode": "mock",
                },
                follow_redirects=False,
            )

        location = response.headers["location"]
        query = parse_qs(urlsplit(location).query)
        assert page_response.status_code == 200
        assert "data-document-summary-generation-form" in page_response.text
        assert "문서 요약 생성" in page_response.text
        assert "임원 보고 요약" in page_response.text
        assert "리스크/후속 조치 요약" in page_response.text
        assert "보고서 초안 (report)" in page_response.text
        assert "Document Summary API document" in page_response.text
        assert "document-summary-progress" in page_response.text
        summary_select_start = page_response.text.index('id="summary_generation_template_key"')
        summary_select_end = page_response.text.index("</select>", summary_select_start)
        summary_select_html = page_response.text[summary_select_start:summary_select_end]
        assert "임원 보고 요약" in summary_select_html
        assert "리스크/후속 조치 요약" in summary_select_html
        assert "보고서 초안 (report)" not in summary_select_html
        assert response.status_code == 303
        assert query["generation_status"] == ["document_summary_created"]
        assert int(query["search_log_id"][0]) > 0
        assert int(query["generation_run_id"][0]) > 0
        assert missing_response.status_code == 303
        assert "generation_error=document+was+not+found" in missing_response.headers["location"]
    finally:
        _cleanup_fixture(migrated_database_url, file_id)


def test_document_summary_history_api_and_ui_filter_summary_runs(
    migrated_database_url: str,
) -> None:
    file_id, document_id, actor_user_id = _create_document_summary_fixture(migrated_database_url)
    _restore_mock_generation_provider(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            first_response = client.post(
                f"/api/documents/{document_id}/summary-runs",
                json={
                    "actor_user_id": actor_user_id,
                    "provider_mode": "mock",
                    "generation_template_key": "summary_risk_action",
                    "summary_instruction": "리스크 중심",
                    "max_chunks": 2,
                },
            )
            first_body = first_response.json()
            second_response = client.post(
                f"/api/documents/{document_id}/summary-runs",
                json={
                    "actor_user_id": actor_user_id,
                    "provider_mode": "mock",
                    "generation_template_key": "summary_executive",
                    "summary_instruction": "임원 보고 중심",
                    "max_chunks": 1,
                },
            )
            api_response = client.get(
                "/api/generation/document-summaries",
                params={
                    "limit": 10,
                    "run_status": "succeeded",
                    "generation_template_key": "summary_risk_action",
                },
            )
            invalid_response = client.get(
                "/api/generation/document-summaries",
                params={"run_status": "unsupported"},
            )
            markdown_export_response = client.get(
                "/api/generation/document-summaries/"
                f"{first_body['generation_run_id']}/export/markdown"
            )
            docx_export_response = client.get(
                "/api/generation/document-summaries/"
                f"{first_body['generation_run_id']}/export/docx"
            )
            missing_export_response = client.get(
                "/api/generation/document-summaries/999999999/export/markdown"
            )
            page_response = client.get(
                "/generation/document-summaries",
                params={"generation_template_key": "summary_risk_action"},
            )

        second_body = second_response.json()
        history_body = api_response.json()
        runs = history_body["runs"]
        assert first_response.status_code == 201
        assert second_response.status_code == 201
        assert first_body["generation"]["prompt_package"]["template_key"] == "summary_risk_action"
        assert second_body["generation"]["prompt_package"]["template_key"] == "summary_executive"
        assert api_response.status_code == 200
        assert history_body["filters"]["generation_template_key"] == "summary_risk_action"
        assert history_body["summary"]["run_count"] == 1
        assert history_body["summary"]["succeeded_count"] == 1
        assert runs[0]["document_id"] == document_id
        assert runs[0]["document_label"] == "Document Summary API document"
        assert runs[0]["template_key"] == "summary_risk_action"
        assert runs[0]["summary_instruction"] == "리스크 중심"
        assert runs[0]["source_chunk_count"] == 2
        assert runs[0]["answer_quality_status"] == "warning"
        assert runs[0]["citation_coverage_percent"] == 50.0
        assert runs[0]["expected_citation_count"] == 2
        assert runs[0]["cited_citation_count"] == 1
        assert runs[0]["missing_citation_count"] == 1
        assert runs[0]["links"]["generation_run"] == (
            f"/generation/runs/{first_body['generation_run_id']}"
        )
        assert invalid_response.status_code == 400
        assert "run_status" in invalid_response.json()["detail"]
        assert markdown_export_response.status_code == 200
        assert markdown_export_response.headers["content-disposition"] == (
            f'attachment; filename="document-summary-run-{first_body["generation_run_id"]}.md"'
        )
        assert markdown_export_response.headers["content-type"].startswith("text/markdown")
        assert "# Generation Run" in markdown_export_response.text
        assert "Document Summary API document" in markdown_export_response.text
        assert docx_export_response.status_code == 200
        assert docx_export_response.headers["content-disposition"] == (
            f'attachment; filename="document-summary-run-{first_body["generation_run_id"]}.docx"'
        )
        assert docx_export_response.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert docx_export_response.headers["x-nex-pcx-export-readiness"]
        assert docx_export_response.content.startswith(b"PK")
        assert missing_export_response.status_code == 404
        assert page_response.status_code == 200
        assert "문서 요약 이력" in page_response.text
        assert "Document Summary API document" in page_response.text
        assert "리스크/후속 조치 요약" in page_response.text
        assert "주의" in page_response.text
        assert "50.00%" in page_response.text
        assert "summary_risk_action" in page_response.text
        assert (
            "/api/generation/document-summaries/"
            f"{first_body['generation_run_id']}/export/markdown"
        ) in page_response.text
        assert (
            "/api/generation/document-summaries/" f"{first_body['generation_run_id']}/export/docx"
        ) in page_response.text
        assert "data-document-summary-history-table" in page_response.text
    finally:
        _cleanup_fixture(migrated_database_url, file_id)


def test_document_summary_history_api_returns_503_without_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/generation/document-summaries")
        export_response = client.get("/api/generation/document-summaries/1/export/markdown")
        page_response = client.get("/generation/document-summaries")

    assert response.status_code == 503
    assert export_response.status_code == 503
    assert page_response.status_code == 200
    assert "NEX_PCX_DATABASE_URL is not configured." in page_response.text
