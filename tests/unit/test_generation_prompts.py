from datetime import UTC, datetime

import pytest

from app.core.generation_prompts import (
    DEFAULT_GENERATION_PROMPT_VERSION,
    GENERATION_BLOCK_LOW_CONFIDENCE,
    InvalidGenerationPromptError,
    build_generation_prompt_package,
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
from app.core.search_logs import SearchLogDetailRecord, SearchLogRecord

NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


def _search_log(query_text: str = "사내 보안 규정은 무엇인가?") -> SearchLogDetailRecord:
    return SearchLogDetailRecord(
        search_log=SearchLogRecord(
            search_log_id=77,
            query_text=query_text,
            normalized_query_text=query_text,
            actor_user_id=1,
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"scope": "company"},
            document_group="policy",
            file_type="md",
            chunk_policy_name="heading_1000_100",
            strategy_name="reranked_vector_cosine",
            top_k=3,
            similarity_metric="cosine",
            profiles=("reranked_vector_cosine",),
            query_runtime_metadata={},
            total_elapsed_ms=123,
            created_by="pytest",
            created_by_user_id=1,
            review_tags=(),
            review_memo=None,
            reviewed_by_user_id=None,
            reviewed_at=None,
            created_at=NOW,
        ),
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


def _candidate() -> RetrievalContextCandidate:
    citation = RetrievalContextCitation(
        citation_key="RCP-001",
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
    primary = _result_reference()
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
        primary_result=primary,
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
    *,
    confidence_status: str = RETRIEVAL_CONFIDENCE_ANSWERABLE,
    query_text: str = "사내 보안 규정은 무엇인가?",
) -> RetrievalContextPackage:
    candidate = _candidate()
    included = confidence_status == RETRIEVAL_CONFIDENCE_ANSWERABLE
    return RetrievalContextPackage(
        package_key="pkg-test-001",
        search_log=_search_log(query_text),
        summary=_summary(included_count=1 if included else 0, excluded_count=0 if included else 1),
        candidates=(candidate,),
        generation_context_text=candidate.context_text if included else "",
        confidence_assessment=_confidence(confidence_status),
    )


def test_build_generation_prompt_package_returns_openai_messages() -> None:
    package = build_generation_prompt_package(_package(), response_language="ko")

    assert package.prompt_version == DEFAULT_GENERATION_PROMPT_VERSION
    assert package.response_language == "ko"
    assert package.blocked is False
    assert package.citation_keys == ("RCP-001",)
    assert package.search_log_id == 77
    assert package.retrieval_package_key == "pkg-test-001"
    assert package.context_hash
    assert package.prompt_hash
    assert package.openai_messages[0]["role"] == "system"
    assert package.openai_messages[1]["role"] == "user"
    assert "Use citation keys such as [RCP-001]" in package.messages[0].content
    assert "사내 보안 규정은 계정 공유를 금지" in package.messages[1].content
    assert "citations: RCP-001" in package.messages[1].content


def test_build_generation_prompt_package_blocks_low_confidence_context() -> None:
    prompt_package = build_generation_prompt_package(
        _package(confidence_status=RETRIEVAL_CONFIDENCE_LOW)
    )

    assert prompt_package.blocked is True
    assert prompt_package.block_reason == GENERATION_BLOCK_LOW_CONFIDENCE
    assert prompt_package.context_text == ""
    assert "Generation is blocked before LLM execution." in prompt_package.messages[1].content
    assert "unsupported context" in prompt_package.messages[1].content


def test_build_generation_prompt_package_rejects_empty_prompt_version() -> None:
    with pytest.raises(InvalidGenerationPromptError, match="prompt_version"):
        build_generation_prompt_package(_package(), prompt_version=" ")
