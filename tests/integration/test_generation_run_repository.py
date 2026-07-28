from datetime import UTC, datetime
from uuid import uuid4

import pytest
from psycopg.types.json import Json

from app.core.database import connect
from app.core.generation_executor import (
    MOCK_FINISH_REASON_COMPLETED,
    MOCK_FINISH_REASON_NO_ANSWER,
    MOCK_NO_ANSWER_TEXT,
    execute_mock_generation_run,
    execute_remote_generation_run,
)
from app.core.generation_provider_metrics import parse_openai_chat_completion_metrics
from app.core.generation_providers import (
    GenerationChatCompletionRequest,
    GenerationChatCompletionResponse,
    GenerationProviderRequestError,
)
from app.core.generation_runs import (
    DGX_VLLM_GENERATION_BASE_URL,
    DGX_VLLM_GENERATION_MODEL_ID,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GENERATION_STATUS_FAILED,
    GENERATION_STATUS_NO_ANSWER,
    GENERATION_STATUS_SUCCEEDED,
    GenerationRunCitationInput,
    GenerationRunInput,
    InvalidGenerationRunError,
    create_generation_run,
    create_generation_run_citation,
    get_default_generation_provider_config,
    get_generation_run,
    list_generation_run_citations,
    seed_dgx_vllm_generation_provider_config,
)
from app.core.retrieval_confidence import (
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
    RETRIEVAL_CONFIDENCE_LOW,
    RetrievalConfidenceAssessment,
)
from app.core.retrieval_context import (
    RetrievalContextCandidate,
    RetrievalContextChunkEntry,
    RetrievalContextCitation,
    RetrievalContextPackage,
    RetrievalContextResultReference,
    RetrievalContextSummary,
)
from app.core.search_logs import (
    SearchLogDetailRecord,
    SearchLogInput,
    SearchLogRecord,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 7, 26, 10, 0, tzinfo=UTC)
MOCK_PROVIDER_NAME = "mock_qwen36_27b_nvfp4"


class _FakeRemoteGenerationProvider:
    def __init__(
        self,
        *,
        response: GenerationChatCompletionResponse | None = None,
        error: GenerationProviderRequestError | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.requests: list[GenerationChatCompletionRequest] = []

    def complete(
        self,
        request: GenerationChatCompletionRequest,
    ) -> GenerationChatCompletionResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _delete_search_log(database_url: str, search_log_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM search_logs WHERE search_log_id = %s", (search_log_id,))
        conn.commit()


def _delete_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
        conn.commit()


def _restore_generation_provider_defaults(database_url: str, provider_name: str) -> None:
    with connect(database_url) as conn:
        conn.execute(
            "DELETE FROM generation_provider_configs WHERE provider_name = %s",
            (provider_name,),
        )
        conn.execute(
            """
            UPDATE generation_provider_configs
            SET is_default = false
            WHERE is_default
              AND provider_name <> %s
            """,
            (MOCK_PROVIDER_NAME,),
        )
        conn.execute(
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
        conn.commit()


def _search_log(
    database_url: str, query_text: str = "사내 보안 규정은 무엇인가?"
) -> SearchLogRecord:
    return create_search_log(
        database_url,
        SearchLogInput(
            query_text=query_text,
            normalized_query_text=query_text,
            top_k=3,
            profiles=("reranked_vector_cosine",),
            requested_search_scope="company",
            effective_search_scope="company",
            document_group="policy",
            file_type="md",
            strategy_name="reranked_vector_cosine",
            similarity_metric="cosine",
            query_runtime_metadata={"test": "generation_run_repository"},
            created_by="pytest",
        ),
    )


def _create_citation_source_fixture(
    database_url: str,
    search_log: SearchLogRecord,
) -> dict[str, int | str]:
    checksum = f"generation-run-citation-{uuid4()}"
    chunk_text = "사내 보안 규정은 계정 공유를 금지하고 정기적인 비밀번호 변경을 요구한다."
    chunk_policy_name = "heading_512_64"
    with connect(database_url) as conn:
        file_row = conn.execute(
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
            VALUES (%s, %s, '.md', 128, %s, %s, 'policy')
            RETURNING file_id
            """,
            (
                f"{checksum}.md",
                f"{checksum}.stored.md",
                checksum,
                f"/tmp/{checksum}.md",
            ),
        ).fetchone()
        assert file_row is not None
        file_id = int(file_row["file_id"])

        document_row = conn.execute(
            """
            INSERT INTO documents (
                file_id,
                document_title,
                document_group,
                access_scope
            )
            VALUES (%s, '사내 보안 규정', 'policy', 'company')
            RETURNING document_id
            """,
            (file_id,),
        ).fetchone()
        assert document_row is not None
        document_id = int(document_row["document_id"])

        chunk_row = conn.execute(
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
                %s, 1, 'text', %s, %s, %s, %s,
                %s, %s, 1, 0, %s, 20, %s, %s
            )
            RETURNING chunk_id
            """,
            (
                document_id,
                chunk_text,
                chunk_text,
                f"chunk-{checksum}",
                chunk_policy_name,
                ["보안", "계정 관리"],
                Json({"start_line": 3, "end_line": 8}),
                len(chunk_text),
                len(chunk_text),
                Json({"fixture": "generation-run-citation"}),
            ),
        ).fetchone()
        assert chunk_row is not None
        chunk_id = int(chunk_row["chunk_id"])
        conn.commit()

    result = create_search_log_results(
        database_url,
        [
            SearchLogResultInput(
                search_log_id=search_log.search_log_id,
                profile_name="reranked_vector_cosine",
                search_profile_name="reranked_vector_cosine",
                retrieval_strategy="reranked",
                rank=1,
                chunk_id=chunk_id,
                distance=0.1,
                score=0.9,
                score_components={"source_score": 0.8, "raw_cross_encoder_score": 2.5},
                profile_elapsed_ms=45,
            )
        ],
    )[0]
    return {
        "file_id": file_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "search_log_result_id": result.search_log_result_id,
        "chunk_policy_name": chunk_policy_name,
    }


def _search_detail(search_log: SearchLogRecord) -> SearchLogDetailRecord:
    return SearchLogDetailRecord(
        search_log=search_log,
        actor_login_id="pytest",
        actor_display_name="Pytest User",
        results=(),
    )


def _result_reference(
    *,
    search_log_result_id: int = 501,
    chunk_id: int = 1001,
) -> RetrievalContextResultReference:
    return RetrievalContextResultReference(
        search_log_result_id=search_log_result_id,
        profile_name="reranked_vector_cosine",
        search_profile_name="reranked_vector_cosine",
        retrieval_strategy="reranked",
        rank=1,
        chunk_id=chunk_id,
        distance=0.1,
        score=0.9,
        score_components={"source_score": 0.8, "raw_cross_encoder_score": 2.5},
        profile_elapsed_ms=45,
        created_at=NOW,
    )


def _candidate(
    *,
    citation_key: str | None = "RCP-001",
    search_log_result_id: int = 501,
    chunk_id: int = 1001,
    document_id: int = 10,
    file_id: int = 20,
    chunk_policy_name: str = "heading_1000_100",
) -> RetrievalContextCandidate:
    citation = RetrievalContextCitation(
        citation_key=citation_key,
        chunk_id=chunk_id,
        document_id=document_id,
        file_id=file_id,
        document_title="사내 보안 규정",
        original_file_name="security_policy.md",
        file_ext="md",
        document_group="policy",
        chunk_policy_name=chunk_policy_name,
        chunk_seq=1,
        heading_path=("보안", "계정 관리"),
        page_no=1,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        artifact_id=30,
        block_id=40,
        source_anchor={"start_line": 3, "end_line": 8},
        source_label="security_policy.md / p.1",
    )
    chunk = RetrievalContextChunkEntry(
        position="current",
        chunk_id=chunk_id,
        chunk_seq=1,
        chunk_text="사내 보안 규정은 계정 공유를 금지하고 정기적인 비밀번호 변경을 요구한다.",
        chunk_preview="사내 보안 규정은 계정 공유를 금지한다.",
        char_count=43,
        token_count=20,
        source_anchor={"start_line": 3, "end_line": 8},
    )
    context_text = (
        "[RCP-001] 사내 보안 규정\n"
        "Source: security_policy.md / p.1\n\n"
        "current:\n"
        f"{chunk.chunk_text}"
    )
    return RetrievalContextCandidate(
        included=True,
        exclusion_reason=None,
        citation=citation,
        primary_result=_result_reference(
            search_log_result_id=search_log_result_id,
            chunk_id=chunk_id,
        ),
        supporting_results=(),
        chunks=(chunk,),
        context_text=context_text,
        context_char_count=len(context_text),
        original_context_char_count=len(context_text),
    )


def _summary(included_count: int = 1, excluded_count: int = 0) -> RetrievalContextSummary:
    return RetrievalContextSummary(
        candidate_result_count=included_count + excluded_count,
        unique_candidate_count=included_count + excluded_count,
        included_count=included_count,
        excluded_count=excluded_count,
        duplicate_supporting_result_count=0,
        max_context_chars=12000,
        used_context_chars=200 if included_count else 0,
        remaining_context_chars=11800,
        truncated_count=0,
        source_context_missing_count=0,
        include_neighbors=True,
        max_items=20,
    )


def _confidence(status: str = RETRIEVAL_CONFIDENCE_ANSWERABLE) -> RetrievalConfidenceAssessment:
    answerable = status == RETRIEVAL_CONFIDENCE_ANSWERABLE
    return RetrievalConfidenceAssessment(
        status=status,
        answerable=answerable,
        withhold_generation_context=not answerable,
        profile_count=1,
        result_count=1 if answerable else 0,
        answerable_profile_count=1 if answerable else 0,
        low_confidence_profile_count=0 if answerable else 1,
        no_context_profile_count=0,
        failed_profile_count=0,
        reason_codes=() if answerable else ("weak_source_vector_score",),
        profiles=(),
    )


def _remote_response(
    provider_name: str,
    model_id: str,
    *,
    answer_text: str = "사내 보안 규정은 계정 공유를 금지합니다. [RCP-001]",
    finish_reason: str = "stop",
    input_token_count: int = 120,
    output_token_count: int = 12,
) -> GenerationChatCompletionResponse:
    total_token_count = input_token_count + output_token_count
    metrics = parse_openai_chat_completion_metrics(
        {
            "id": "chatcmpl-pytest-remote",
            "object": "chat.completion",
            "created": 1785000000,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": answer_text},
                }
            ],
            "usage": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": total_token_count,
            },
        },
        provider_name=provider_name,
        provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
        requested_model_id=model_id,
        http_status_code=200,
        elapsed_ms=88,
        provider_elapsed_ms=80,
    )
    return GenerationChatCompletionResponse(
        answer_text=answer_text,
        finish_reason=finish_reason,
        provider_model_id=model_id,
        response_id="chatcmpl-pytest-remote",
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        total_token_count=total_token_count,
        elapsed_ms=88,
        provider_metrics=metrics,
        response_metadata={"provider_name": provider_name, "metrics": {"succeeded": True}},
        raw_response={},
    )


def _remote_error(provider_name: str, model_id: str) -> GenerationProviderRequestError:
    metrics = parse_openai_chat_completion_metrics(
        {
            "error": {
                "message": "model is warming up",
                "type": "server_error",
                "code": "overloaded",
            }
        },
        provider_name=provider_name,
        provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
        requested_model_id=model_id,
        http_status_code=503,
        elapsed_ms=44,
        provider_elapsed_ms=40,
    )
    return GenerationProviderRequestError(
        "Remote generation provider returned HTTP 503: model is warming up",
        metrics=metrics,
        payload={"error": {"message": "model is warming up", "code": "overloaded"}},
    )


def _package(
    search_log: SearchLogRecord,
    *,
    confidence_status: str = RETRIEVAL_CONFIDENCE_ANSWERABLE,
    candidate: RetrievalContextCandidate | None = None,
) -> RetrievalContextPackage:
    candidate = candidate or _candidate()
    included = confidence_status == RETRIEVAL_CONFIDENCE_ANSWERABLE
    included_candidate = candidate if included else None
    excluded_candidate = (
        RetrievalContextCandidate(
            included=False,
            exclusion_reason="low_confidence",
            citation=candidate.citation,
            primary_result=candidate.primary_result,
            supporting_results=(),
            chunks=(),
            context_text="",
            context_char_count=0,
            original_context_char_count=0,
        )
        if not included
        else None
    )
    candidates = (included_candidate,) if included_candidate else (excluded_candidate,)
    return RetrievalContextPackage(
        package_key=f"pkg-generation-{search_log.search_log_id}",
        search_log=_search_detail(search_log),
        summary=_summary(included_count=1 if included else 0, excluded_count=0 if included else 1),
        candidates=candidates,
        generation_context_text=candidate.context_text if included else "",
        confidence_assessment=_confidence(confidence_status),
    )


def test_generation_run_repository_persists_run_and_citations(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    provider = get_default_generation_provider_config(migrated_database_url)
    assert provider is not None

    try:
        run = create_generation_run(
            migrated_database_url,
            GenerationRunInput(
                search_log_id=search_log.search_log_id,
                retrieval_package_key="pkg-repository-001",
                provider_config_id=provider.provider_config_id,
                provider_name=provider.provider_name,
                provider_mode=provider.provider_mode,
                model_id=provider.model_id,
                retrieval_confidence_status="answerable",
                citation_readiness_status="ready",
                query_text=search_log.query_text,
                status=GENERATION_STATUS_SUCCEEDED,
                answer_text="근거 기반 답변입니다. [RCP-001]",
                request_metadata={"prompt_hash": "abc"},
                response_metadata={"mock": True},
                guardrail_metadata={"status": "allowed"},
            ),
        )
        citation = create_generation_run_citation(
            migrated_database_url,
            GenerationRunCitationInput(
                generation_run_id=run.generation_run_id,
                citation_key="RCP-001",
                citation_index=1,
                source_label="security_policy.md / p.1",
                source_anchor={"start_line": 3},
                citation_payload={"chunk_id": 1001},
                was_cited=True,
            ),
        )

        stored = get_generation_run(migrated_database_url, run.generation_run_id)
        citations = list_generation_run_citations(migrated_database_url, run.generation_run_id)

        assert stored == run
        assert citation in citations
        assert citations[0].citation_payload["chunk_id"] == 1001
        assert citations[0].was_cited is True
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)


def test_generation_run_repository_returns_none_for_missing_run(
    migrated_database_url: str,
) -> None:
    assert get_generation_run(migrated_database_url, 999999999) is None


@pytest.mark.parametrize(
    ("field_name", "run_input"),
    (
        (
            "status",
            GenerationRunInput(
                search_log_id=1,
                retrieval_package_key="pkg",
                provider_name="mock",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                retrieval_confidence_status="answerable",
                citation_readiness_status="ready",
                query_text="query",
                status="unknown",
            ),
        ),
        (
            "guardrail_status",
            GenerationRunInput(
                search_log_id=1,
                retrieval_package_key="pkg",
                provider_name="mock",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                retrieval_confidence_status="answerable",
                citation_readiness_status="ready",
                query_text="query",
                guardrail_status="unknown",
            ),
        ),
        (
            "provider_mode",
            GenerationRunInput(
                search_log_id=1,
                retrieval_package_key="pkg",
                provider_name="mock",
                provider_mode="unsupported",
                model_id="model",
                retrieval_confidence_status="answerable",
                citation_readiness_status="ready",
                query_text="query",
            ),
        ),
        (
            "retrieval_confidence_status",
            GenerationRunInput(
                search_log_id=1,
                retrieval_package_key="pkg",
                provider_name="mock",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                retrieval_confidence_status="unknown",
                citation_readiness_status="ready",
                query_text="query",
            ),
        ),
        (
            "citation_readiness_status",
            GenerationRunInput(
                search_log_id=1,
                retrieval_package_key="pkg",
                provider_name="mock",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                retrieval_confidence_status="answerable",
                citation_readiness_status="unknown",
                query_text="query",
            ),
        ),
        (
            "elapsed_ms",
            GenerationRunInput(
                search_log_id=1,
                retrieval_package_key="pkg",
                provider_name="mock",
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
                model_id="model",
                retrieval_confidence_status="answerable",
                citation_readiness_status="ready",
                query_text="query",
                elapsed_ms=-1,
            ),
        ),
    ),
)
def test_generation_run_repository_rejects_invalid_input_before_insert(
    migrated_database_url: str,
    field_name: str,
    run_input: GenerationRunInput,
) -> None:
    with pytest.raises(InvalidGenerationRunError, match=field_name):
        create_generation_run(migrated_database_url, run_input)


def test_generation_run_citation_repository_rejects_invalid_optional_fk(
    migrated_database_url: str,
) -> None:
    with pytest.raises(InvalidGenerationRunError, match="chunk_id"):
        create_generation_run_citation(
            migrated_database_url,
            GenerationRunCitationInput(
                generation_run_id=1,
                citation_key="RCP-001",
                citation_index=1,
                chunk_id=0,
            ),
        )


def test_mock_generation_executor_persists_answer_and_citations(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    fixture = _create_citation_source_fixture(migrated_database_url, search_log)
    candidate = _candidate(
        search_log_result_id=fixture["search_log_result_id"],
        chunk_id=fixture["chunk_id"],
        document_id=fixture["document_id"],
        file_id=fixture["file_id"],
        chunk_policy_name=str(fixture["chunk_policy_name"]),
    )

    try:
        report = execute_mock_generation_run(
            migrated_database_url,
            _package(search_log, candidate=candidate),
            created_by="pytest",
        )

        assert report.run.status == GENERATION_STATUS_SUCCEEDED
        assert report.run.finish_reason == MOCK_FINISH_REASON_COMPLETED
        assert report.run.guardrail_status == "allowed"
        assert "[RCP-001]" in (report.run.answer_text or "")
        assert report.run.generation_template_id == report.prompt_package.generation_template_id
        assert report.prompt_package.template_key == "grounded_answer"
        assert report.run.prompt_hash == report.prompt_package.prompt_hash
        assert report.run.request_metadata["messages"][0]["role"] == "system"
        assert report.run.request_metadata["template_key"] == "grounded_answer"
        assert report.run.request_metadata["generation_template"]["template_key"] == (
            "grounded_answer"
        )
        assert report.run.response_metadata["deterministic"] is True
        provider_metrics = report.run.response_metadata["provider_metrics"]
        assert provider_metrics["provider_name"] == "mock_qwen36_27b_nvfp4"
        assert provider_metrics["provider_mode"] == "mock"
        assert provider_metrics["finish_reason"] == MOCK_FINISH_REASON_COMPLETED
        assert provider_metrics["total_token_count"] == report.run.total_token_count
        assert provider_metrics["succeeded"] is True
        answer_quality = report.run.response_metadata["answer_quality"]
        assert answer_quality["status"] == "passed"
        assert answer_quality["recognized_citation_keys"] == ["RCP-001"]
        assert report.run.guardrail_metadata["answer_quality_status"] == "passed"
        assert len(report.citations) == 1
        assert report.citations[0].was_cited is True
        assert report.citations[0].search_log_result_id == fixture["search_log_result_id"]
        assert report.citations[0].chunk_id == fixture["chunk_id"]
        assert report.citations[0].document_id == fixture["document_id"]
        assert report.citations[0].file_id == fixture["file_id"]
        assert report.citations[0].citation_payload["original_file_name"] == "security_policy.md"
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)
        _delete_file(migrated_database_url, fixture["file_id"])


def test_mock_generation_executor_shapes_report_template_output(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    fixture = _create_citation_source_fixture(migrated_database_url, search_log)
    candidate = _candidate(
        search_log_result_id=fixture["search_log_result_id"],
        chunk_id=fixture["chunk_id"],
        document_id=fixture["document_id"],
        file_id=fixture["file_id"],
        chunk_policy_name=str(fixture["chunk_policy_name"]),
    )

    try:
        report = execute_mock_generation_run(
            migrated_database_url,
            _package(search_log, candidate=candidate),
            generation_template_key=" report ",
            created_by="pytest-report-template",
        )

        assert report.run.status == GENERATION_STATUS_SUCCEEDED
        assert report.prompt_package.template_key == "report"
        assert report.run.request_metadata["template_key"] == "report"
        assert report.run.answer_text is not None
        assert report.run.answer_text.startswith("# 보고서 초안")
        assert "## 제목" in report.run.answer_text
        assert "## 요약" in report.run.answer_text
        assert "## 배경" in report.run.answer_text
        assert "## 주요 내용" in report.run.answer_text
        assert "## 근거" in report.run.answer_text
        assert "## 리스크" in report.run.answer_text
        assert "## 후속 조치" in report.run.answer_text
        assert "[RCP-001]" in report.run.answer_text
        assert report.run.response_metadata["template"] == {
            "template_key": "report",
            "template_name": "보고서 초안",
            "template_version": "v1",
            "document_type": "report",
            "output_format": "markdown",
            "template_section_keys": [
                "title",
                "overview",
                "background",
                "findings",
                "evidence",
                "risks",
                "next_steps",
            ],
            "required_template_section_keys": [
                "title",
                "overview",
                "background",
                "findings",
                "evidence",
                "risks",
                "next_steps",
            ],
        }
        assert report.run.response_metadata["answer_quality"]["status"] == "passed"
        assert len(report.citations) == 1
        assert report.citations[0].was_cited is True
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)
        _delete_file(migrated_database_url, fixture["file_id"])


def test_mock_generation_executor_records_no_answer_for_low_confidence(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)

    try:
        report = execute_mock_generation_run(
            migrated_database_url,
            _package(search_log, confidence_status=RETRIEVAL_CONFIDENCE_LOW),
        )

        assert report.run.status == GENERATION_STATUS_NO_ANSWER
        assert report.run.finish_reason == MOCK_FINISH_REASON_NO_ANSWER
        assert report.run.answer_text == MOCK_NO_ANSWER_TEXT
        assert report.run.guardrail_status == "no_answer"
        assert report.run.retrieval_confidence_status == RETRIEVAL_CONFIDENCE_LOW
        assert report.run.guardrail_metadata["prompt_blocked"] is True
        assert report.run.response_metadata["answer_quality"]["status"] == "passed"
        assert report.run.guardrail_metadata["answer_quality_status"] == "passed"
        assert report.citations == ()
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)


def test_mock_generation_executor_rejects_missing_template_key(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)

    try:
        with pytest.raises(InvalidGenerationRunError, match="generation template"):
            execute_mock_generation_run(
                migrated_database_url,
                _package(search_log),
                generation_template_key="missing_template",
            )
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)


def test_mock_generation_executor_fails_when_default_provider_is_not_active(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    provider = get_default_generation_provider_config(migrated_database_url)
    assert provider is not None

    try:
        with connect(migrated_database_url) as conn:
            conn.execute(
                """
                UPDATE generation_provider_configs
                SET is_active = false
                WHERE provider_config_id = %s
                """,
                (provider.provider_config_id,),
            )
            conn.commit()

        assert get_default_generation_provider_config(migrated_database_url) is None
        with pytest.raises(InvalidGenerationRunError, match="active mock generation provider"):
            execute_mock_generation_run(migrated_database_url, _package(search_log))
    finally:
        with connect(migrated_database_url) as conn:
            conn.execute(
                """
                UPDATE generation_provider_configs
                SET is_active = true
                WHERE provider_config_id = %s
                """,
                (provider.provider_config_id,),
            )
            conn.commit()
        _delete_search_log(migrated_database_url, search_log.search_log_id)


def test_mock_generation_executor_fails_when_default_provider_is_remote(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    provider = get_default_generation_provider_config(migrated_database_url)
    assert provider is not None

    try:
        with connect(migrated_database_url) as conn:
            conn.execute(
                """
                UPDATE generation_provider_configs
                SET provider_mode = 'remote_openai_compatible',
                    provider_base_url = 'http://127.0.0.1:8001'
                WHERE provider_config_id = %s
                """,
                (provider.provider_config_id,),
            )
            conn.commit()

        with pytest.raises(InvalidGenerationRunError, match="active mock generation provider"):
            execute_mock_generation_run(migrated_database_url, _package(search_log))
    finally:
        with connect(migrated_database_url) as conn:
            conn.execute(
                """
                UPDATE generation_provider_configs
                SET provider_mode = 'mock',
                    provider_base_url = NULL
                WHERE provider_config_id = %s
                """,
                (provider.provider_config_id,),
            )
            conn.commit()
        _delete_search_log(migrated_database_url, search_log.search_log_id)


def test_remote_generation_executor_persists_success_and_citation_trace(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    fixture = _create_citation_source_fixture(migrated_database_url, search_log)
    candidate = _candidate(
        search_log_result_id=fixture["search_log_result_id"],
        chunk_id=fixture["chunk_id"],
        document_id=fixture["document_id"],
        file_id=fixture["file_id"],
        chunk_policy_name=str(fixture["chunk_policy_name"]),
    )
    provider_name = f"pytest_remote_generation_{search_log.search_log_id}"
    provider = seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        is_default=False,
    )
    default_provider = get_default_generation_provider_config(migrated_database_url)
    assert default_provider is not None
    assert default_provider.provider_name == MOCK_PROVIDER_NAME
    fake_provider = _FakeRemoteGenerationProvider(
        response=_remote_response(provider.provider_name, provider.model_id)
    )

    try:
        report = execute_remote_generation_run(
            migrated_database_url,
            _package(search_log, candidate=candidate),
            provider_client=fake_provider,
            api_key="pytest-secret",
            created_by="pytest-remote",
        )
        stored = get_generation_run(migrated_database_url, report.run.generation_run_id)
        citations = list_generation_run_citations(
            migrated_database_url,
            report.run.generation_run_id,
        )

        assert len(fake_provider.requests) == 1
        assert fake_provider.requests[0].model_id == DGX_VLLM_GENERATION_MODEL_ID
        assert fake_provider.requests[0].max_tokens == 4096
        assert fake_provider.requests[0].extra_body == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert report.provider.provider_name == provider_name
        assert report.run.status == GENERATION_STATUS_SUCCEEDED
        assert report.run.provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
        assert report.run.answer_text == "사내 보안 규정은 계정 공유를 금지합니다. [RCP-001]"
        assert report.run.finish_reason == "stop"
        assert report.run.total_token_count == 132
        assert report.run.response_metadata["provider_model_id"] == provider.model_id
        assert report.run.response_metadata["response_id"] == "chatcmpl-pytest-remote"
        assert report.run.response_metadata["provider_metrics"]["succeeded"] is True
        assert report.run.response_metadata["template"]["template_key"] == "grounded_answer"
        assert report.run.response_metadata["template"]["required_template_section_keys"] == [
            "answer",
            "evidence",
        ]
        assert report.run.response_metadata["answer_quality"]["status"] == "passed"
        assert report.run.response_metadata["answer_quality"]["recognized_citation_keys"] == [
            "RCP-001"
        ]
        assert report.run.guardrail_metadata["answer_quality_status"] == "passed"
        assert report.run.request_metadata["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert report.run.request_metadata["configured_max_tokens"] == 4096
        assert report.run.request_metadata["max_tokens"] == 4096
        assert report.run.request_metadata["max_token_policy"]["resolution_reason"] == "configured"
        assert report.run.response_metadata["truncation"]["truncated"] is False
        assert stored == report.run
        assert len(citations) == 1
        assert citations[0].was_cited is True
        assert citations[0].citation_key == "RCP-001"
        assert citations[0].search_log_result_id == fixture["search_log_result_id"]
        assert citations[0].chunk_id == fixture["chunk_id"]
        assert citations[0].document_id == fixture["document_id"]
        assert citations[0].file_id == fixture["file_id"]
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _delete_search_log(migrated_database_url, search_log.search_log_id)
        _delete_file(migrated_database_url, fixture["file_id"])


def test_remote_generation_executor_raises_dgx_long_form_budget_and_records_truncation(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    fixture = _create_citation_source_fixture(migrated_database_url, search_log)
    candidate = _candidate(
        search_log_result_id=fixture["search_log_result_id"],
        chunk_id=fixture["chunk_id"],
        document_id=fixture["document_id"],
        file_id=fixture["file_id"],
        chunk_policy_name=str(fixture["chunk_policy_name"]),
    )
    provider_name = f"pytest_remote_generation_report_budget_{search_log.search_log_id}"
    provider = seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        max_tokens=1024,
        is_default=True,
    )
    report_answer = "\n\n".join(
        (
            "# 보고서 초안",
            "## 1. 제목\n사내 보안 규정 보고서",
            "## 2. 요약\n계정 공유 금지 규정이 확인되었습니다. [RCP-001]",
            "## 3. 배경\n보안 정책 문서를 기준으로 확인했습니다. [RCP-001]",
            "## 4. 주요 내용\n- 계정 공유를 금지합니다. [RCP-001]",
            "## 5. 근거\n- [RCP-001] security_policy.md / p.1",
            "## 6. 리스크\n- 추가 세부 규정은 별도 확인이 필요합니다. [RCP-001]",
            "## 7. 후속 조치\n- 담당자 검토를 진행합니다. [RCP-001]",
        )
    )
    fake_provider = _FakeRemoteGenerationProvider(
        response=_remote_response(
            provider.provider_name,
            provider.model_id,
            answer_text=report_answer,
            finish_reason="length",
            input_token_count=6124,
            output_token_count=8192,
        )
    )

    try:
        report = execute_remote_generation_run(
            migrated_database_url,
            _package(search_log, candidate=candidate),
            generation_template_key="report",
            provider_client=fake_provider,
            api_key="pytest-secret",
            created_by="pytest-remote-report-budget",
        )

        assert len(fake_provider.requests) == 1
        assert fake_provider.requests[0].max_tokens == 8192
        assert report.run.status == GENERATION_STATUS_SUCCEEDED
        assert report.run.finish_reason == "length"
        assert report.run.output_token_count == 8192
        assert report.run.request_metadata["configured_max_tokens"] == 1024
        assert report.run.request_metadata["max_tokens"] == 8192
        assert (
            report.run.request_metadata["max_token_policy"]["resolution_reason"]
            == "dgx_long_form_document_type"
        )
        assert report.run.response_metadata["template"]["required_template_section_keys"] == [
            "title",
            "overview",
            "background",
            "findings",
            "evidence",
            "risks",
            "next_steps",
        ]
        assert report.run.response_metadata["truncation"] == {
            "contract_version": "generation_truncation_v1",
            "status": "truncated",
            "truncated": True,
            "reason_code": "finish_reason_length",
            "finish_reason": "length",
            "requested_max_tokens": 8192,
            "output_token_count": 8192,
        }
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _delete_search_log(migrated_database_url, search_log.search_log_id)
        _delete_file(migrated_database_url, fixture["file_id"])


def test_remote_generation_executor_persists_provider_failure(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    provider_name = f"pytest_remote_generation_failure_{search_log.search_log_id}"
    provider = seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        provider_base_url=DGX_VLLM_GENERATION_BASE_URL,
        is_default=True,
    )
    fake_provider = _FakeRemoteGenerationProvider(
        error=_remote_error(provider.provider_name, provider.model_id)
    )

    try:
        report = execute_remote_generation_run(
            migrated_database_url,
            _package(search_log),
            provider_client=fake_provider,
        )

        assert len(fake_provider.requests) == 1
        assert report.run.status == GENERATION_STATUS_FAILED
        assert report.run.error_message is not None
        assert "HTTP 503" in report.run.error_message
        assert report.run.elapsed_ms == 44
        assert report.run.response_metadata["provider_error"] is True
        assert report.run.response_metadata["provider_metrics"]["error_code"] == "overloaded"
        assert report.run.response_metadata["provider_metrics"]["succeeded"] is False
        assert report.run.response_metadata["answer_quality"]["status"] == "not_evaluated"
        assert report.run.response_metadata["answer_quality"]["reason_codes"] == ["provider_error"]
        assert report.run.guardrail_metadata["answer_quality_status"] == "not_evaluated"
        assert report.citations == ()
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _delete_search_log(migrated_database_url, search_log.search_log_id)


def test_remote_generation_executor_skips_provider_call_for_low_confidence(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    provider_name = f"pytest_remote_generation_guardrail_{search_log.search_log_id}"
    seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        is_default=True,
    )
    fake_provider = _FakeRemoteGenerationProvider(
        response=_remote_response(provider_name, DGX_VLLM_GENERATION_MODEL_ID)
    )

    try:
        report = execute_remote_generation_run(
            migrated_database_url,
            _package(search_log, confidence_status=RETRIEVAL_CONFIDENCE_LOW),
            provider_client=fake_provider,
        )

        assert fake_provider.requests == []
        assert report.run.status == GENERATION_STATUS_NO_ANSWER
        assert report.run.answer_text == MOCK_NO_ANSWER_TEXT
        assert report.run.response_metadata["skipped_provider_call"] is True
        assert report.run.response_metadata["answer_quality"]["status"] == "passed"
        assert report.run.guardrail_metadata["answer_quality_status"] == "passed"
        assert report.citations == ()
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _delete_search_log(migrated_database_url, search_log.search_log_id)


def test_remote_generation_executor_records_failed_answer_quality_for_missing_citation(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)
    fixture = _create_citation_source_fixture(migrated_database_url, search_log)
    candidate = _candidate(
        search_log_result_id=fixture["search_log_result_id"],
        chunk_id=fixture["chunk_id"],
        document_id=fixture["document_id"],
        file_id=fixture["file_id"],
        chunk_policy_name=str(fixture["chunk_policy_name"]),
    )
    provider_name = f"pytest_remote_generation_quality_{search_log.search_log_id}"
    provider = seed_dgx_vllm_generation_provider_config(
        migrated_database_url,
        provider_name=provider_name,
        is_default=True,
    )
    fake_provider = _FakeRemoteGenerationProvider(
        response=_remote_response(
            provider.provider_name,
            provider.model_id,
            answer_text="사내 보안 규정은 계정 공유를 금지합니다.",
        )
    )

    try:
        report = execute_remote_generation_run(
            migrated_database_url,
            _package(search_log, candidate=candidate),
            provider_client=fake_provider,
        )

        answer_quality = report.run.response_metadata["answer_quality"]
        assert report.run.status == GENERATION_STATUS_SUCCEEDED
        assert answer_quality["status"] == "failed"
        assert answer_quality["reason_codes"] == ["missing_required_citation"]
        assert answer_quality["missing_citation_keys"] == ["RCP-001"]
        assert report.run.guardrail_metadata["answer_quality_status"] == "failed"
        assert report.citations[0].was_cited is False
        assert report.citations[0].search_log_result_id == fixture["search_log_result_id"]
        assert report.citations[0].chunk_id == fixture["chunk_id"]
        assert report.citations[0].document_id == fixture["document_id"]
        assert report.citations[0].file_id == fixture["file_id"]
    finally:
        _restore_generation_provider_defaults(migrated_database_url, provider_name)
        _delete_search_log(migrated_database_url, search_log.search_log_id)
        _delete_file(migrated_database_url, fixture["file_id"])


def test_remote_generation_executor_fails_when_default_provider_is_mock(
    migrated_database_url: str,
) -> None:
    search_log = _search_log(migrated_database_url)

    try:
        with pytest.raises(InvalidGenerationRunError, match="remote_openai_compatible"):
            execute_remote_generation_run(migrated_database_url, _package(search_log))
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)
