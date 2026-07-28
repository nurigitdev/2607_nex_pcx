"""Search-backed chat orchestration helpers."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.bm25_keyword_index import DEFAULT_BM25_TOKENIZER_NAME
from app.core.bm25_search import BM25_SEARCH_PROFILE_NAME
from app.core.embedding_providers import EmbeddingProviderRuntimeConfig
from app.core.rerankers import RerankerRuntimeConfig
from app.core.search_compare import (
    SearchCompareInput,
    SearchCompareProfileResult,
    SearchCompareResult,
    run_search_compare,
)

CHAT_SEARCH_SUMMARY_PROMPT_VERSION = "chat_search_summary_v1"
CHAT_SEARCH_SUMMARY_EXECUTION_MODE = "search_compare_summary"
DEFAULT_CHAT_SEARCH_TOP_K = 5
DEFAULT_CHAT_SEARCH_PROFILES = (BM25_SEARCH_PROFILE_NAME,)

SearchCompareRunner = Callable[..., SearchCompareResult]


@dataclass(frozen=True)
class ChatSearchSummaryInput:
    content: str
    actor_user_id: int | None
    requested_search_scope: str = "company"
    top_k: int = DEFAULT_CHAT_SEARCH_TOP_K
    profiles: tuple[str, ...] | None = DEFAULT_CHAT_SEARCH_PROFILES
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    bm25_tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME
    hybrid_vector_profile_name: str | None = None
    reranked_vector_profile_name: str | None = None
    allow_mock_fallback: bool = True
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatSearchSummaryResult:
    answer_text: str
    search_result: SearchCompareResult
    prompt_version: str
    execution_mode: str
    result_count: int
    profile_status_counts: dict[str, int]
    retrieval_confidence_status: str | None
    request_metadata: dict[str, Any]
    response_metadata: dict[str, Any]


class InvalidChatSearchError(ValueError):
    """Raised when a chat search orchestration request is invalid."""


def _validate_nonblank(value: str | None, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise InvalidChatSearchError(f"{field_name} must not be blank")
    return normalized


def _validate_positive_int(value: int | None, field_name: str) -> int:
    if value is None or value <= 0:
        raise InvalidChatSearchError(f"{field_name} must be greater than 0")
    return value


def _validate_profiles(profiles: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if profiles is None:
        return None
    normalized = tuple(_validate_nonblank(profile, "profile") for profile in profiles)
    if not normalized:
        raise InvalidChatSearchError("profiles must not be empty")
    if len(set(normalized)) != len(normalized):
        raise InvalidChatSearchError("profiles must be unique")
    return normalized


def validate_chat_search_summary_input(
    summary_input: ChatSearchSummaryInput,
) -> ChatSearchSummaryInput:
    if not isinstance(summary_input.runtime_metadata, Mapping):
        raise InvalidChatSearchError("runtime_metadata must be a JSON object")
    return ChatSearchSummaryInput(
        content=_validate_nonblank(summary_input.content, "content"),
        actor_user_id=_validate_positive_int(summary_input.actor_user_id, "actor_user_id"),
        requested_search_scope=_validate_nonblank(
            summary_input.requested_search_scope,
            "requested_search_scope",
        ),
        top_k=_validate_positive_int(summary_input.top_k, "top_k"),
        profiles=_validate_profiles(summary_input.profiles) or DEFAULT_CHAT_SEARCH_PROFILES,
        chunk_policy_name=(
            _validate_nonblank(summary_input.chunk_policy_name, "chunk_policy_name")
            if summary_input.chunk_policy_name is not None
            else None
        ),
        document_group=(
            _validate_nonblank(summary_input.document_group, "document_group")
            if summary_input.document_group is not None
            else None
        ),
        file_type=(
            _validate_nonblank(summary_input.file_type, "file_type")
            if summary_input.file_type is not None
            else None
        ),
        bm25_tokenizer_name=_validate_nonblank(
            summary_input.bm25_tokenizer_name,
            "bm25_tokenizer_name",
        ),
        hybrid_vector_profile_name=(
            _validate_nonblank(
                summary_input.hybrid_vector_profile_name,
                "hybrid_vector_profile_name",
            )
            if summary_input.hybrid_vector_profile_name is not None
            else None
        ),
        reranked_vector_profile_name=(
            _validate_nonblank(
                summary_input.reranked_vector_profile_name,
                "reranked_vector_profile_name",
            )
            if summary_input.reranked_vector_profile_name is not None
            else None
        ),
        allow_mock_fallback=bool(summary_input.allow_mock_fallback),
        runtime_metadata=dict(summary_input.runtime_metadata),
    )


def _profile_status_counts(profiles: tuple[SearchCompareProfileResult, ...]) -> dict[str, int]:
    counts: dict[str, int] = {"succeeded": 0, "failed": 0}
    for profile in profiles:
        counts[profile.status] = counts.get(profile.status, 0) + 1
    return counts


def _result_count(profiles: tuple[SearchCompareProfileResult, ...]) -> int:
    return sum(len(profile.results) for profile in profiles)


def _first_result_text(result: SearchCompareResult) -> str:
    for profile in result.profiles:
        if not profile.results:
            continue
        top_result = profile.results[0].vector_result
        title = getattr(top_result, "document_title", None) or "문서 제목 없음"
        preview = getattr(top_result, "chunk_preview", None) or ""
        chunk_id = getattr(top_result, "chunk_id", None)
        return f"- 최상위 결과: {title} / chunk #{chunk_id}\n- 미리보기: {preview}"
    return "- 최상위 결과: 없음"


def _retrieval_confidence_status(result: SearchCompareResult) -> str | None:
    if result.confidence_assessment is None:
        return None
    return result.confidence_assessment.status


def _answer_text(result: SearchCompareResult) -> str:
    result_count = _result_count(result.profiles)
    confidence = _retrieval_confidence_status(result) or "not_available"
    if result_count == 0:
        return (
            "검색 결과 요약입니다.\n\n"
            f"- Search Log: #{result.search_log_id}\n"
            f"- 검색 범위: {result.effective_search_scope}\n"
            f"- 검색 confidence: {confidence}\n"
            "- 관련 chunk를 찾지 못했습니다. 검색어, 권한 범위, chunk policy를 다시 확인하세요."
        )
    profile_counts = ", ".join(
        f"{profile.profile_name} {len(profile.results)}건" for profile in result.profiles
    )
    return (
        "검색 결과 요약입니다.\n\n"
        f"- Search Log: #{result.search_log_id}\n"
        f"- 검색 범위: {result.effective_search_scope}\n"
        f"- 검색 confidence: {confidence}\n"
        f"- 프로필 결과: {profile_counts}\n"
        f"{_first_result_text(result)}"
    )


def _request_metadata(
    summary_input: ChatSearchSummaryInput,
    search_input: SearchCompareInput,
) -> dict[str, Any]:
    return {
        "prompt_version": CHAT_SEARCH_SUMMARY_PROMPT_VERSION,
        "query_text": search_input.query_text,
        "actor_user_id": search_input.actor_user_id,
        "requested_search_scope": search_input.requested_search_scope,
        "top_k": search_input.top_k,
        "profiles": list(search_input.profiles or ()),
        "chunk_policy_name": search_input.chunk_policy_name,
        "document_group": search_input.document_group,
        "file_type": search_input.file_type,
        "bm25_tokenizer_name": search_input.bm25_tokenizer_name,
        "runtime_metadata": dict(summary_input.runtime_metadata),
    }


def execute_chat_search_summary(
    database_url: str,
    summary_input: ChatSearchSummaryInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    fallback_reranker_runtime_config: RerankerRuntimeConfig | None = None,
    search_compare_runner: SearchCompareRunner = run_search_compare,
) -> ChatSearchSummaryResult:
    database_url = _validate_nonblank(database_url, "database_url")
    validated = validate_chat_search_summary_input(summary_input)
    search_input = SearchCompareInput(
        query_text=validated.content,
        actor_user_id=validated.actor_user_id or 0,
        requested_search_scope=validated.requested_search_scope,
        top_k=validated.top_k,
        profiles=validated.profiles,
        chunk_policy_name=validated.chunk_policy_name,
        document_group=validated.document_group,
        file_type=validated.file_type,
        bm25_tokenizer_name=validated.bm25_tokenizer_name,
        hybrid_vector_profile_name=validated.hybrid_vector_profile_name,
        reranked_vector_profile_name=validated.reranked_vector_profile_name,
        allow_mock_fallback=validated.allow_mock_fallback,
    )
    search_result = search_compare_runner(
        database_url,
        search_input,
        fallback_runtime_config=fallback_runtime_config,
        fallback_reranker_runtime_config=fallback_reranker_runtime_config,
    )
    result_count = _result_count(search_result.profiles)
    return ChatSearchSummaryResult(
        answer_text=_answer_text(search_result),
        search_result=search_result,
        prompt_version=CHAT_SEARCH_SUMMARY_PROMPT_VERSION,
        execution_mode=CHAT_SEARCH_SUMMARY_EXECUTION_MODE,
        result_count=result_count,
        profile_status_counts=_profile_status_counts(search_result.profiles),
        retrieval_confidence_status=_retrieval_confidence_status(search_result),
        request_metadata=_request_metadata(validated, search_input),
        response_metadata={
            "search_log_id": search_result.search_log_id,
            "result_count": result_count,
            "profile_status_counts": _profile_status_counts(search_result.profiles),
            "retrieval_confidence_status": _retrieval_confidence_status(search_result),
            "total_elapsed_ms": search_result.total_elapsed_ms,
        },
    )
