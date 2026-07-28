from types import SimpleNamespace

import pytest

from app.core.bm25_search import BM25_SEARCH_PROFILE_NAME
from app.core.chat_search import (
    CHAT_SEARCH_SUMMARY_EXECUTION_MODE,
    CHAT_SEARCH_SUMMARY_PROMPT_VERSION,
    ChatSearchSummaryInput,
    InvalidChatSearchError,
    execute_chat_search_summary,
    validate_chat_search_summary_input,
)
from app.core.permissions import PermissionSearchFilter
from app.core.retrieval_confidence import (
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
    RetrievalConfidenceAssessment,
)
from app.core.search_compare import (
    SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
    SearchCompareInput,
    SearchCompareProfileResult,
    SearchCompareResult,
    SearchCompareResultItem,
)


def _permission_filter() -> PermissionSearchFilter:
    return PermissionSearchFilter(
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        where_sql="TRUE",
        params=(),
        metadata={"display_name": "Alice"},
    )


def _confidence(result_count: int) -> RetrievalConfidenceAssessment:
    return RetrievalConfidenceAssessment(
        status=RETRIEVAL_CONFIDENCE_ANSWERABLE,
        answerable=True,
        withhold_generation_context=False,
        profile_count=1,
        result_count=result_count,
        answerable_profile_count=1,
        low_confidence_profile_count=0,
        no_context_profile_count=0,
        failed_profile_count=0,
        reason_codes=(),
        profiles=(),
    )


def _search_result(*, result_count: int = 1) -> SearchCompareResult:
    results = ()
    if result_count:
        vector_result = SimpleNamespace(
            rank=1,
            chunk_id=101,
            document_title="회의 준비 가이드",
            chunk_preview="회의 목적, 참석자, 결정사항을 먼저 정리합니다.",
            score=0.42,
        )
        results = (
            SearchCompareResultItem(
                search_log_result_id=501,
                vector_result=vector_result,
            ),
        )
    return SearchCompareResult(
        search_log_id=301,
        query_text="회의 준비 방법을 검색해서 요약해줘",
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        permission_filter=_permission_filter(),
        top_k=5,
        profiles=(
            SearchCompareProfileResult(
                profile_name=BM25_SEARCH_PROFILE_NAME,
                elapsed_ms=3,
                results=results,
                status=SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
            ),
        ),
        total_elapsed_ms=5,
        confidence_assessment=_confidence(len(results)),
    )


def test_execute_chat_search_summary_uses_bm25_baseline_and_formats_top_result() -> None:
    captured: dict[str, object] = {}

    def runner(database_url: str, search_input: SearchCompareInput, **kwargs):
        captured["database_url"] = database_url
        captured["search_input"] = search_input
        captured["fallback_runtime_config"] = kwargs.get("fallback_runtime_config")
        return _search_result()

    result = execute_chat_search_summary(
        "postgresql://db",
        ChatSearchSummaryInput(
            content="회의 준비 방법을 검색해서 요약해줘",
            actor_user_id=1,
            profiles=None,
            runtime_metadata={"chat_session_id": 9},
        ),
        search_compare_runner=runner,
    )

    search_input = captured["search_input"]
    assert isinstance(search_input, SearchCompareInput)
    assert search_input.profiles == (BM25_SEARCH_PROFILE_NAME,)
    assert result.execution_mode == CHAT_SEARCH_SUMMARY_EXECUTION_MODE
    assert result.prompt_version == CHAT_SEARCH_SUMMARY_PROMPT_VERSION
    assert result.result_count == 1
    assert "Search Log: #301" in result.answer_text
    assert "회의 준비 가이드" in result.answer_text
    assert result.request_metadata["runtime_metadata"] == {"chat_session_id": 9}
    assert result.response_metadata["search_log_id"] == 301


def test_execute_chat_search_summary_handles_empty_results() -> None:
    result = execute_chat_search_summary(
        "postgresql://db",
        ChatSearchSummaryInput(
            content="없는 내용을 검색해서 요약해줘",
            actor_user_id=1,
        ),
        search_compare_runner=lambda *args, **kwargs: _search_result(result_count=0),
    )

    assert result.result_count == 0
    assert "관련 chunk를 찾지 못했습니다" in result.answer_text
    assert result.profile_status_counts == {"succeeded": 1, "failed": 0}


@pytest.mark.parametrize(
    ("summary_input", "message"),
    (
        (ChatSearchSummaryInput(content="", actor_user_id=1), "content"),
        (ChatSearchSummaryInput(content="검색", actor_user_id=None), "actor_user_id"),
        (ChatSearchSummaryInput(content="검색", actor_user_id=0), "actor_user_id"),
        (ChatSearchSummaryInput(content="검색", actor_user_id=1, top_k=0), "top_k"),
        (
            ChatSearchSummaryInput(
                content="검색",
                actor_user_id=1,
                requested_search_scope=" ",
            ),
            "requested_search_scope",
        ),
        (
            ChatSearchSummaryInput(content="검색", actor_user_id=1, profiles=()),
            "profiles",
        ),
        (
            ChatSearchSummaryInput(
                content="검색",
                actor_user_id=1,
                profiles=("bm25_keyword", "bm25_keyword"),
            ),
            "profiles",
        ),
        (
            ChatSearchSummaryInput(content="검색", actor_user_id=1, runtime_metadata=[]),
            "runtime_metadata",
        ),
    ),
)
def test_validate_chat_search_summary_input_rejects_invalid_values(
    summary_input: ChatSearchSummaryInput,
    message: str,
) -> None:
    with pytest.raises(InvalidChatSearchError, match=message):
        validate_chat_search_summary_input(summary_input)
