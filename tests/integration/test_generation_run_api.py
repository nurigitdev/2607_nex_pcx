import re
from io import BytesIO
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from docx import Document
from fastapi.testclient import TestClient
from psycopg.types.json import Json

from app.core.config import Settings
from app.core.database import connect
from app.core.generation_executor import execute_remote_generation_run
from app.core.generation_provider_metrics import parse_openai_chat_completion_metrics
from app.core.generation_providers import (
    GenerationChatCompletionRequest,
    GenerationChatCompletionResponse,
)
from app.core.generation_runs import (
    DGX_VLLM_GENERATION_MODEL_ID,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GenerationRunInput,
    create_generation_run,
    seed_dgx_vllm_generation_provider_config,
)
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)
from app.main import create_app

pytestmark = pytest.mark.integration

MOCK_PROVIDER_NAME = "mock_qwen36_27b_nvfp4"


class _FakeRemoteGenerationProvider:
    def __init__(self, response: GenerationChatCompletionResponse) -> None:
        self.response = response
        self.requests: list[GenerationChatCompletionRequest] = []

    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        self.requests.append(request)
        return self.response


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


def _restore_generation_provider_defaults(database_url: str, provider_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM generation_provider_configs WHERE provider_name = %s",
                (provider_name,),
            )
            cursor.execute(
                """
                UPDATE generation_provider_configs
                SET is_default = false
                WHERE is_default
                  AND provider_name <> %s
                """,
                (MOCK_PROVIDER_NAME,),
            )
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


def _create_failed_answer_quality_run(database_url: str, search_log_id: int) -> int:
    run = create_generation_run(
        database_url,
        GenerationRunInput(
            search_log_id=search_log_id,
            retrieval_package_key="pytest-history-failed",
            provider_name=MOCK_PROVIDER_NAME,
            provider_mode="mock",
            model_id="nvidia/Qwen3.6-27B-NVFP4",
            retrieval_confidence_status="answerable",
            citation_readiness_status="warning",
            query_text="generation history failed quality",
            status="succeeded",
            guardrail_status="allowed",
            answer_text="citation 없는 생성 답변입니다.",
            finish_reason="mock_completed",
            input_token_count=4,
            output_token_count=4,
            total_token_count=8,
            elapsed_ms=11,
            response_metadata={
                "answer_quality": {
                    "contract_version": "generation_answer_quality_v1",
                    "status": "failed",
                    "expected_citation_keys": ["RCP-001"],
                    "cited_citation_keys": [],
                    "recognized_citation_keys": [],
                    "missing_citation_keys": ["RCP-001"],
                    "unrecognized_citation_keys": [],
                    "citation_coverage_percent": 0.0,
                    "reason_codes": ["missing_required_citation"],
                }
            },
            guardrail_metadata={
                "answer_quality_status": "failed",
                "answer_quality_reason_codes": ["missing_required_citation"],
            },
            created_by="generation_history_fixture",
        ),
    )
    return run.generation_run_id


def _remote_response(provider_name: str) -> GenerationChatCompletionResponse:
    answer_text = "remote vLLM 생성 API 응답입니다. [RCP-001]"
    metrics = parse_openai_chat_completion_metrics(
        {
            "id": "chatcmpl-api-remote",
            "object": "chat.completion",
            "created": 1785000000,
            "model": DGX_VLLM_GENERATION_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": answer_text},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 9,
                "total_tokens": 109,
            },
        },
        provider_name=provider_name,
        provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
        requested_model_id=DGX_VLLM_GENERATION_MODEL_ID,
        http_status_code=200,
        elapsed_ms=71,
        provider_elapsed_ms=68,
    )
    return GenerationChatCompletionResponse(
        answer_text=answer_text,
        finish_reason="stop",
        provider_model_id=DGX_VLLM_GENERATION_MODEL_ID,
        response_id="chatcmpl-api-remote",
        input_token_count=100,
        output_token_count=9,
        total_token_count=109,
        elapsed_ms=71,
        provider_metrics=metrics,
        response_metadata={"provider_name": provider_name},
        raw_response={},
    )


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
            completeness_response = client.get(
                f"/api/generation/runs/{run_id}/template-completeness"
            )
            export_response = client.get(
                f"/api/generation/runs/{run_id}/export/markdown",
            )
            docx_export_response = client.get(
                f"/api/generation/runs/{run_id}/export/docx",
            )
            invalid_response = client.get("/api/generation/runs/0")
            invalid_completeness_response = client.get(
                "/api/generation/runs/0/template-completeness"
            )
            missing_response = client.post("/api/search/logs/999999999/generation-runs/mock")
            missing_completeness_response = client.get(
                "/api/generation/runs/999999999/template-completeness"
            )
            missing_export_response = client.get(
                "/api/generation/runs/999999999/export/markdown",
            )
            missing_docx_export_response = client.get(
                "/api/generation/runs/999999999/export/docx",
            )

        read_body = read_response.json()
        completeness_body = completeness_response.json()
        docx_document = Document(BytesIO(docx_export_response.content))
        docx_text = "\n".join(
            paragraph.text for paragraph in docx_document.paragraphs if paragraph.text
        )
        assert create_response.status_code == 201
        assert create_body["provider"]["provider_mode"] == "mock"
        assert create_body["run"]["search_log_id"] == search_log_id
        assert create_body["run"]["status"] == "succeeded"
        assert create_body["run"]["finish_reason"] == "mock_completed"
        assert (
            create_body["run"]["generation_template_id"]
            == create_body["prompt_package"]["generation_template_id"]
        )
        assert create_body["run"]["retrieval_confidence_status"] == "answerable"
        assert create_body["run"]["citation_readiness_status"] == "warning"
        assert create_body["run"]["response_metadata"]["answer_quality"]["status"] == "passed"
        assert (
            create_body["run"]["response_metadata"]["answer_quality"]["citation_coverage_percent"]
            == 100.0
        )
        assert (
            create_body["run"]["template_completeness"]["contract_version"]
            == "generation_template_completeness_v1"
        )
        assert create_body["run"]["guardrail_metadata"]["answer_quality_status"] == "passed"
        assert "[RCP-001]" in create_body["run"]["answer_text"]
        assert create_body["prompt_package"]["messages"][0]["role"] == "system"
        assert create_body["prompt_package"]["prompt_version"] == "grounded_answer_v1_prompt_v1"
        assert (
            create_body["prompt_package"]["prompt_contract_version"] == "template_aware_prompt_v1"
        )
        assert create_body["prompt_package"]["template_key"] == "grounded_answer"
        assert create_body["prompt_package"]["generation_template"]["template_key"] == (
            "grounded_answer"
        )
        assert create_body["prompt_package"]["blocked"] is False
        assert create_body["citations"][0]["citation_key"] == "RCP-001"
        assert create_body["citations"][0]["was_cited"] is True
        assert read_response.status_code == 200
        assert read_body["run"]["generation_run_id"] == run_id
        assert read_body["run"]["template_completeness"]["template_key"] == "grounded_answer"
        assert read_body["citations"][0]["source_label"].endswith("/ p.2")
        assert completeness_response.status_code == 200
        assert completeness_body["generation_run_id"] == run_id
        assert (
            completeness_body["template_completeness"]["contract_version"]
            == "generation_template_completeness_v1"
        )
        assert completeness_body["template_completeness"]["template_key"] == "grounded_answer"
        assert export_response.status_code == 200
        assert export_response.headers["content-type"].startswith("text/markdown")
        assert export_response.headers["content-disposition"] == (
            f'attachment; filename="generation-run-{run_id}.md"'
        )
        assert f"# Generation Run #{run_id}" in export_response.text
        assert "## Metadata" in export_response.text
        assert "## Answer" in export_response.text
        assert "## Citations" in export_response.text
        assert "- [RCP-001]" in export_response.text
        assert "## Raw Runtime Metadata" in export_response.text
        assert "grounded_answer" in export_response.text
        assert docx_export_response.status_code == 200
        assert docx_export_response.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert docx_export_response.headers["content-disposition"] == (
            f'attachment; filename="generation-run-{run_id}.docx"'
        )
        assert f"Generation Run #{run_id}" in docx_text
        assert "Metadata" in docx_text
        assert "Answer" in docx_text
        assert "Citations" in docx_text
        assert "RCP-001" in docx_text
        assert invalid_response.status_code == 400
        assert invalid_completeness_response.status_code == 400
        assert missing_response.status_code == 404
        assert missing_completeness_response.status_code == 404
        assert missing_export_response.status_code == 404
        assert missing_docx_export_response.status_code == 404
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_remote_generation_run_api_creates_remote_generation_run(
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    provider_name = f"pytest_remote_generation_api_{search_log_id}"
    seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        is_default=True,
    )
    fake_provider = _FakeRemoteGenerationProvider(_remote_response(provider_name))

    def fake_execute_remote_generation_run(
        database_url: str,
        package: object,
        *,
        generation_template_key: str | None = None,
        api_key: str | None = None,
        created_by: str | None = None,
        created_by_user_id: int | None = None,
    ):
        assert generation_template_key is None
        assert api_key == "pytest-remote-secret"
        assert created_by == "api_remote_generation"
        return execute_remote_generation_run(
            database_url,
            package,  # type: ignore[arg-type]
            generation_template_key=generation_template_key,
            provider_client=fake_provider,
            api_key=api_key,
            created_by=created_by,
            created_by_user_id=created_by_user_id,
        )

    monkeypatch.setattr(
        "app.main.execute_remote_generation_run",
        fake_execute_remote_generation_run,
    )
    app = create_app(
        Settings(
            database_url=migrated_database_url,
            remote_generation_provider_api_key="pytest-remote-secret",
        )
    )

    try:
        with TestClient(app) as client:
            create_response = client.post(
                f"/api/search/logs/{search_log_id}/generation-runs/remote",
                params={
                    "max_context_chars": 4000,
                    "include_neighbors": "false",
                    "max_items": 5,
                },
            )
            create_body = create_response.json()

        assert create_response.status_code == 201
        assert len(fake_provider.requests) == 1
        assert create_body["provider"]["provider_name"] == provider_name
        assert create_body["provider"]["provider_mode"] == "remote_openai_compatible"
        assert create_body["run"]["status"] == "succeeded"
        assert create_body["run"]["created_by"] == "api_remote_generation"
        assert create_body["run"]["provider_name"] == provider_name
        assert create_body["run"]["total_token_count"] == 109
        assert create_body["run"]["response_metadata"]["provider_model_id"] == (
            DGX_VLLM_GENERATION_MODEL_ID
        )
        assert create_body["citations"][0]["citation_key"] == "RCP-001"
        assert "pytest-remote-secret" not in create_response.text
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _cleanup_file(migrated_database_url, file_id)


def test_remote_generation_run_api_rejects_missing_api_key_env(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    provider_name = f"pytest_remote_generation_missing_key_{search_log_id}"
    seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        api_key_env="NEX_PCX_PYTEST_MISSING_GENERATION_API_KEY",
        is_default=True,
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/search/logs/{search_log_id}/generation-runs/remote",
                params={"include_neighbors": "false"},
            )

        assert response.status_code == 400
        assert "environment variable is not set" in response.json()["detail"]
        assert "NEX_PCX_PYTEST_MISSING_GENERATION_API_KEY" in response.json()["detail"]
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _cleanup_file(migrated_database_url, file_id)


def test_remote_generation_run_api_rejects_mock_default_provider(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/search/logs/{search_log_id}/generation-runs/remote",
                params={"include_neighbors": "false"},
            )

        assert response.status_code == 400
        assert "remote_openai_compatible" in response.json()["detail"]
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
                    "generation_template_key": "report",
                },
            )
            missing_response = client.get("/api/search/logs/999999999/generation-prompt/preview")
            invalid_template_response = client.get(
                f"/api/search/logs/{search_log_id}/generation-prompt/preview",
                params={"generation_template_key": "missing_template"},
            )

        after_count = _generation_run_count(migrated_database_url, search_log_id)
        body = preview_response.json()
        prompt_package = body["prompt_package"]
        assert preview_response.status_code == 200
        assert body["retrieval_context"]["search_log"]["search_log_id"] == search_log_id
        assert prompt_package["messages"][0]["role"] == "system"
        assert prompt_package["messages"][1]["role"] == "user"
        assert prompt_package["response_language"] == "ko"
        assert prompt_package["prompt_version"] == "report_v1_prompt_v1"
        assert prompt_package["prompt_contract_version"] == "template_aware_prompt_v1"
        assert prompt_package["template_key"] == "report"
        assert prompt_package["template_version"] == "v1"
        assert prompt_package["output_format"] == "markdown"
        assert prompt_package["generation_template"]["template_key"] == "report"
        assert "template_name: 보고서 초안" in prompt_package["messages"][0]["content"]
        assert prompt_package["blocked"] is False
        assert prompt_package["block_reason"] is None
        assert prompt_package["citation_keys"] == ["RCP-001"]
        assert prompt_package["prompt_hash"]
        assert prompt_package["context_hash"]
        assert after_count == before_count
        assert missing_response.status_code == 404
        assert invalid_template_response.status_code == 400
        assert "active generation template" in invalid_template_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_generation_template_list_api_returns_active_templates(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get("/api/generation/templates")
        no_database_response = TestClient(create_app(Settings(database_url=None))).get(
            "/api/generation/templates"
        )

    body = response.json()
    template_keys = {template["template_key"] for template in body["templates"]}
    report_template = next(
        template for template in body["templates"] if template["template_key"] == "report"
    )

    assert response.status_code == 200
    assert body["default_template_key"] == "grounded_answer"
    assert {"grounded_answer", "report", "proposal", "summary", "meeting_minutes"} <= template_keys
    assert report_template["document_type"] == "report"
    assert report_template["output_format"] == "markdown"
    assert report_template["section_schema"]
    assert no_database_response.status_code == 503


def test_generation_run_markdown_export_returns_503_without_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/generation/runs/1/export/markdown")
        docx_response = client.get("/api/generation/runs/1/export/docx")

    assert response.status_code == 503
    assert docx_response.status_code == 503


def test_generation_run_api_returns_503_without_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        create_response = client.post("/api/search/logs/1/generation-runs/mock")
        remote_create_response = client.post("/api/search/logs/1/generation-runs/remote")
        list_response = client.get("/api/generation/runs")
        read_response = client.get("/api/generation/runs/1")
        completeness_response = client.get("/api/generation/runs/1/template-completeness")
        preview_response = client.get("/api/search/logs/1/generation-prompt/preview")
        metrics_response = client.get("/api/admin/generation-provider-metrics/snapshot")

    assert create_response.status_code == 503
    assert remote_create_response.status_code == 503
    assert list_response.status_code == 503
    assert read_response.status_code == 503
    assert completeness_response.status_code == 503
    assert preview_response.status_code == 503
    assert metrics_response.status_code == 503


def test_generation_run_history_api_filters_answer_quality(
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
            passed_run_id = create_response.json()["run"]["generation_run_id"]
            failed_run_id = _create_failed_answer_quality_run(
                migrated_database_url,
                search_log_id,
            )
            passed_response = client.get(
                "/api/generation/runs",
                params={
                    "answer_quality_status": "passed",
                    "provider_mode": "mock",
                    "run_status": "succeeded",
                    "limit": "25",
                },
            )
            failed_response = client.get(
                "/api/generation/runs",
                params={
                    "answer_quality_status": "failed",
                    "provider_mode": "mock",
                    "run_status": "succeeded",
                    "limit": "25",
                },
            )
            invalid_response = client.get(
                "/api/generation/runs",
                params={"answer_quality_status": "unsupported"},
            )

        passed_body = passed_response.json()
        failed_body = failed_response.json()
        passed_ids = {run["generation_run_id"] for run in passed_body["runs"]}
        failed_ids = {run["generation_run_id"] for run in failed_body["runs"]}
        assert create_response.status_code == 201
        assert passed_response.status_code == 200
        assert passed_body["filters"]["answer_quality_status"] == "passed"
        assert passed_body["summary"]["passed_count"] >= 1
        assert passed_run_id in passed_ids
        assert failed_run_id not in passed_ids
        assert failed_response.status_code == 200
        assert failed_body["filters"]["answer_quality_status"] == "failed"
        assert failed_body["summary"]["failed_count"] >= 1
        assert failed_run_id in failed_ids
        failed_run = next(
            run for run in failed_body["runs"] if run["generation_run_id"] == failed_run_id
        )
        assert failed_run["answer_quality_reason_codes"] == ["missing_required_citation"]
        assert failed_run["citation_coverage_percent"] == 0.0
        assert invalid_response.status_code == 400
        assert "answer_quality_status is not supported" in invalid_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_generation_run_history_ui_shows_quality_filter_and_detail_links(
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
            page_response = client.get(
                "/generation/runs",
                params={
                    "answer_quality_status": "passed",
                    "provider_mode": "mock",
                    "run_status": "succeeded",
                    "limit": "10",
                },
            )

        assert create_response.status_code == 201
        assert page_response.status_code == 200
        assert "생성 이력" in page_response.text
        assert "data-generation-run-history-filters" in page_response.text
        assert "data-generation-run-history-summary" in page_response.text
        assert "data-generation-run-history-table" in page_response.text
        assert "답변 품질" in page_response.text
        assert "100.00%" in page_response.text
        assert f'href="/generation/runs/{run_id}"' in page_response.text
        api_link_match = re.search(
            r'href="(?P<href>/api/generation/runs[^"]*)"',
            page_response.text,
        )
        assert api_link_match is not None
        api_link = urlsplit(api_link_match.group("href"))
        api_query = parse_qs(api_link.query)
        assert api_link.path == "/api/generation/runs"
        assert api_query["limit"] == ["10"]
        assert api_query["answer_quality_status"] == ["passed"]
        assert api_query["provider_mode"] == ["mock"]
        assert api_query["run_status"] == ["succeeded"]
    finally:
        _cleanup_file(migrated_database_url, file_id)


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
        assert any(
            summary["provider_mode"] == "mock" for summary in snapshot_body["mode_summaries"]
        )
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


def test_generation_provider_metric_snapshot_api_groups_provider_modes(
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    provider_name = f"pytest_remote_generation_metrics_{search_log_id}"
    app = create_app(
        Settings(
            database_url=migrated_database_url,
            remote_generation_provider_api_key="pytest-metrics-secret",
        )
    )

    try:
        with TestClient(app) as client:
            mock_response = client.post(
                f"/api/search/logs/{search_log_id}/generation-runs/mock",
                params={"include_neighbors": "false"},
            )

        seed_dgx_vllm_generation_provider_config(
            migrated_database_url,
            provider_name=provider_name,
            is_default=True,
        )
        fake_provider = _FakeRemoteGenerationProvider(_remote_response(provider_name))

        def fake_execute_remote_generation_run(
            database_url: str,
            package: object,
            *,
            generation_template_key: str | None = None,
            api_key: str | None = None,
            created_by: str | None = None,
            created_by_user_id: int | None = None,
        ):
            assert generation_template_key is None
            assert api_key == "pytest-metrics-secret"
            return execute_remote_generation_run(
                database_url,
                package,  # type: ignore[arg-type]
                generation_template_key=generation_template_key,
                provider_client=fake_provider,
                api_key=api_key,
                created_by=created_by,
                created_by_user_id=created_by_user_id,
            )

        monkeypatch.setattr(
            "app.main.execute_remote_generation_run",
            fake_execute_remote_generation_run,
        )

        with TestClient(app) as client:
            remote_response = client.post(
                f"/api/search/logs/{search_log_id}/generation-runs/remote",
                params={"include_neighbors": "false"},
            )
            snapshot_response = client.get(
                "/api/admin/generation-provider-metrics/snapshot",
                params={"limit": "2"},
            )

        snapshot_body = snapshot_response.json()
        summaries = {
            summary["provider_mode"]: summary for summary in snapshot_body["mode_summaries"]
        }
        assert mock_response.status_code == 201
        assert remote_response.status_code == 201
        assert snapshot_response.status_code == 200
        assert snapshot_body["summary"]["run_count"] == 2
        assert summaries["mock"]["run_count"] == 1
        assert summaries["mock"]["succeeded_count"] == 1
        assert summaries["remote_openai_compatible"]["run_count"] == 1
        assert summaries["remote_openai_compatible"]["metric_present_count"] == 1
        assert summaries["remote_openai_compatible"]["total_token_count"] == 109
        assert summaries["remote_openai_compatible"]["average_provider_elapsed_ms"] == 68
        assert "pytest-metrics-secret" not in snapshot_response.text
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _cleanup_file(migrated_database_url, file_id)


def test_generation_provider_metric_snapshot_ui_shows_recent_metrics(
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
            page_response = client.get(
                "/admin/generation-provider-metrics",
                params={"limit": "10"},
            )
            invalid_limit_response = client.get(
                "/admin/generation-provider-metrics",
                params={"limit": "0"},
            )

        assert create_response.status_code == 201
        assert page_response.status_code == 200
        assert "생성 Provider Metrics" in page_response.text
        assert "data-generation-provider-metrics-summary" in page_response.text
        assert "data-generation-provider-metrics-mode-breakdown" in page_response.text
        assert "Provider Mode Breakdown" in page_response.text
        assert "data-generation-provider-metrics-runs" in page_response.text
        assert "mock_qwen36_27b_nvfp4" in page_response.text
        assert "mock_completed" in page_response.text
        assert f'href="/generation/runs/{run_id}"' in page_response.text
        assert "/api/admin/generation-provider-metrics/snapshot?limit=10" in page_response.text
        assert "Raw Snapshot JSON" in page_response.text
        assert invalid_limit_response.status_code == 200
        assert "limit must be between 1 and 500" in invalid_limit_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_generation_provider_metric_snapshot_ui_reports_missing_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        page_response = client.get("/admin/generation-provider-metrics")

    assert page_response.status_code == 200
    assert "생성 Provider Metrics" in page_response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in page_response.text


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
        assert "grounded_answer_v1_prompt_v1" in context_response.text
        assert "template_aware_prompt_v1" in context_response.text
        assert "Mock 생성" in context_response.text
        assert "Remote vLLM Runtime" in context_response.text
        assert "Remote vLLM 생성" in context_response.text
        assert "Remote 실행 불가" in context_response.text
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


def test_generation_run_ui_creates_remote_run_with_runtime_feedback(
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    provider_name = f"pytest_remote_generation_ui_{search_log_id}"
    seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        is_default=True,
    )
    fake_provider = _FakeRemoteGenerationProvider(_remote_response(provider_name))

    def fake_execute_remote_generation_run(
        database_url: str,
        package: object,
        *,
        generation_template_key: str | None = None,
        api_key: str | None = None,
        created_by: str | None = None,
        created_by_user_id: int | None = None,
    ):
        assert generation_template_key is None
        assert api_key == "pytest-ui-secret"
        assert created_by == "generation_ui_remote"
        return execute_remote_generation_run(
            database_url,
            package,  # type: ignore[arg-type]
            generation_template_key=generation_template_key,
            provider_client=fake_provider,
            api_key=api_key,
            created_by=created_by,
            created_by_user_id=created_by_user_id,
        )

    monkeypatch.setattr(
        "app.main.execute_remote_generation_run",
        fake_execute_remote_generation_run,
    )
    app = create_app(
        Settings(
            database_url=migrated_database_url,
            remote_generation_provider_api_key="pytest-ui-secret",
        )
    )

    try:
        with TestClient(app) as client:
            context_response = client.get(
                "/generation",
                params={"search_log_id": search_log_id, "include_neighbors": "false"},
            )
            post_response = client.post(
                "/generation/runs/remote",
                data={
                    "search_log_id": str(search_log_id),
                    "max_context_chars": "4000",
                    "max_items": "5",
                },
                follow_redirects=False,
            )
            detail_response = client.get(post_response.headers["location"])

        assert context_response.status_code == 200
        assert "Remote vLLM Runtime" in context_response.text
        assert provider_name in context_response.text
        assert "Remote 준비됨" in context_response.text
        assert "env 설정됨" in context_response.text
        assert "Remote vLLM 생성" in context_response.text
        assert post_response.status_code == 303
        assert "generation_status=remote_created" in post_response.headers["location"]
        assert detail_response.status_code == 200
        assert "Remote vLLM 생성 실행이 완료되었습니다." in detail_response.text
        assert "remote vLLM 생성 API 응답입니다." in detail_response.text
        assert provider_name in detail_response.text
        assert "pytest-ui-secret" not in detail_response.text
        assert len(fake_provider.requests) == 1
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _cleanup_file(migrated_database_url, file_id)


def test_generation_run_ui_disables_remote_action_when_api_key_env_is_missing(
    migrated_database_url: str,
) -> None:
    file_id, search_log_id = _create_generation_api_fixture(migrated_database_url)
    provider_name = f"pytest_remote_generation_ui_missing_key_{search_log_id}"
    seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        api_key_env="NEX_PCX_PYTEST_UI_MISSING_GENERATION_API_KEY",
        is_default=True,
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            context_response = client.get(
                "/generation",
                params={"search_log_id": search_log_id, "include_neighbors": "false"},
            )
            post_response = client.post(
                "/generation/runs/remote",
                data={
                    "search_log_id": str(search_log_id),
                    "max_context_chars": "4000",
                    "max_items": "5",
                },
                follow_redirects=False,
            )
            error_page_response = client.get(post_response.headers["location"])

        assert context_response.status_code == 200
        assert "Remote 실행 불가" in context_response.text
        assert "env 누락" in context_response.text
        assert "Remote 실행을 위해서는" in context_response.text
        assert post_response.status_code == 303
        assert "generation_error=" in post_response.headers["location"]
        assert error_page_response.status_code == 200
        assert "environment variable is not set" in error_page_response.text
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
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
            result_response = client.get(location)
            detail_response = client.get(f"/generation/runs/{run_id}")
            missing_response = client.get("/generation/runs/999999999")

        assert post_response.status_code == 303
        assert result_response.status_code == 200
        assert "data-generation-answer-quality" in result_response.text
        assert "data-generation-template-completeness" in result_response.text
        assert "답변 품질" in result_response.text
        assert "Template 완성도" in result_response.text
        assert "100.00%" in result_response.text
        assert detail_response.status_code == 200
        assert "생성 실행 상세" in detail_response.text
        assert "data-generation-answer-quality-detail" in detail_response.text
        assert "data-generation-template-completeness-detail" in detail_response.text
        assert "기대 Citation Key" in detail_response.text
        assert "필수 Section Coverage" in detail_response.text
        assert "Prompt Messages" in detail_response.text
        assert "Request Metadata" in detail_response.text
        assert "Response Metadata" in detail_response.text
        assert "Guardrail Metadata" in detail_response.text
        assert "Markdown 내보내기" in detail_response.text
        assert "DOCX 내보내기" in detail_response.text
        assert f"/api/generation/runs/{run_id}/export/markdown" in detail_response.text
        assert f"/api/generation/runs/{run_id}/export/docx" in detail_response.text
        assert "grounded_answer_v1_prompt_v1" in detail_response.text
        assert "template_aware_prompt_v1" in detail_response.text
        assert "generation_ui_mock" in detail_response.text
        assert "RCP-001" in detail_response.text
        assert "Generation API document" in detail_response.text
        assert missing_response.status_code == 200
        assert "Generation run not found." in missing_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)
