from datetime import UTC, datetime

import pytest

from app.core.database import connect
from app.core.generation_executor import (
    MOCK_FINISH_REASON_COMPLETED,
    MOCK_FINISH_REASON_NO_ANSWER,
    MOCK_NO_ANSWER_TEXT,
    execute_mock_generation_run,
)
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
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
    create_search_log,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 7, 26, 10, 0, tzinfo=UTC)


def _delete_search_log(database_url: str, search_log_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM search_logs WHERE search_log_id = %s", (search_log_id,))
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


def _search_detail(search_log: SearchLogRecord) -> SearchLogDetailRecord:
    return SearchLogDetailRecord(
        search_log=search_log,
        actor_login_id="pytest",
        actor_display_name="Pytest User",
        results=(),
    )


def _result_reference() -> RetrievalContextResultReference:
    return RetrievalContextResultReference(
        search_log_result_id=501,
        profile_name="reranked_vector_cosine",
        search_profile_name="reranked_vector_cosine",
        retrieval_strategy="reranked",
        rank=1,
        chunk_id=1001,
        distance=0.1,
        score=0.9,
        score_components={"source_score": 0.8, "raw_cross_encoder_score": 2.5},
        profile_elapsed_ms=45,
        created_at=NOW,
    )


def _candidate(*, citation_key: str | None = "RCP-001") -> RetrievalContextCandidate:
    citation = RetrievalContextCitation(
        citation_key=citation_key,
        chunk_id=1001,
        document_id=10,
        file_id=20,
        document_title="사내 보안 규정",
        original_file_name="security_policy.md",
        file_ext="md",
        document_group="policy",
        chunk_policy_name="heading_1000_100",
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
        chunk_id=1001,
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
        primary_result=_result_reference(),
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


def _package(
    search_log: SearchLogRecord,
    *,
    confidence_status: str = RETRIEVAL_CONFIDENCE_ANSWERABLE,
) -> RetrievalContextPackage:
    candidate = _candidate()
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

    try:
        report = execute_mock_generation_run(
            migrated_database_url,
            _package(search_log),
            created_by="pytest",
        )

        assert report.run.status == GENERATION_STATUS_SUCCEEDED
        assert report.run.finish_reason == MOCK_FINISH_REASON_COMPLETED
        assert report.run.guardrail_status == "allowed"
        assert "[RCP-001]" in (report.run.answer_text or "")
        assert report.run.prompt_hash == report.prompt_package.prompt_hash
        assert report.run.request_metadata["messages"][0]["role"] == "system"
        assert report.run.response_metadata["deterministic"] is True
        provider_metrics = report.run.response_metadata["provider_metrics"]
        assert provider_metrics["provider_name"] == "mock_qwen36_27b_nvfp4"
        assert provider_metrics["provider_mode"] == "mock"
        assert provider_metrics["finish_reason"] == MOCK_FINISH_REASON_COMPLETED
        assert provider_metrics["total_token_count"] == report.run.total_token_count
        assert provider_metrics["succeeded"] is True
        assert len(report.citations) == 1
        assert report.citations[0].was_cited is True
        assert report.citations[0].citation_payload["original_file_name"] == "security_policy.md"
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)


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
        assert report.citations == ()
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
        with pytest.raises(InvalidGenerationRunError, match="default generation provider"):
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

        with pytest.raises(InvalidGenerationRunError, match="not mock"):
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
