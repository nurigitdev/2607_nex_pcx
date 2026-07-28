"""Grounded answer orchestration helpers for conversational UX."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.bm25_keyword_index import DEFAULT_BM25_TOKENIZER_NAME
from app.core.bm25_search import BM25_SEARCH_PROFILE_NAME
from app.core.direct_generation import (
    DirectGenerationInput,
    DirectGenerationResult,
    run_direct_generation_query,
)
from app.core.embedding_providers import EmbeddingProviderRuntimeConfig
from app.core.generation_providers import GenerationProvider
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
)
from app.core.rerankers import RerankerRuntimeConfig
from app.core.retrieval_context import DEFAULT_CONTEXT_CHAR_BUDGET, DEFAULT_CONTEXT_MAX_ITEMS

CHAT_GROUNDED_ANSWER_EXECUTION_MODE = "direct_grounded_generation"
CHAT_GROUNDED_ANSWER_PROMPT_VERSION = "chat_grounded_answer_v1"
DEFAULT_CHAT_GROUNDED_TOP_K = 5
DEFAULT_CHAT_GROUNDED_PROFILES = (BM25_SEARCH_PROFILE_NAME,)
CHAT_GROUNDED_PROVIDER_MODES = {
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
}

DirectGenerationRunner = Callable[..., DirectGenerationResult]


@dataclass(frozen=True)
class ChatGroundedAnswerInput:
    content: str
    actor_user_id: int | None
    requested_search_scope: str = "company"
    provider_mode: str = GENERATION_PROVIDER_MODE_MOCK
    generation_template_key: str | None = None
    top_k: int = DEFAULT_CHAT_GROUNDED_TOP_K
    profiles: tuple[str, ...] | None = DEFAULT_CHAT_GROUNDED_PROFILES
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    bm25_tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME
    hybrid_vector_profile_name: str | None = None
    reranked_vector_profile_name: str | None = None
    allow_mock_fallback: bool = True
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET
    include_neighbors: bool = True
    max_items: int = DEFAULT_CONTEXT_MAX_ITEMS


@dataclass(frozen=True)
class ChatGroundedAnswerResult:
    answer_text: str
    direct_generation_result: DirectGenerationResult
    prompt_version: str
    execution_mode: str
    search_log_id: int
    generation_run_id: int
    retrieval_context_included_count: int
    generation_status: str
    guardrail_status: str
    request_metadata: dict[str, Any]
    response_metadata: dict[str, Any]


class InvalidChatGroundedGenerationError(ValueError):
    """Raised when a chat grounded generation request is invalid."""


def _validate_nonblank(value: str | None, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise InvalidChatGroundedGenerationError(f"{field_name} must not be blank")
    return normalized


def _validate_positive_int(value: int | None, field_name: str) -> int:
    if value is None or value <= 0:
        raise InvalidChatGroundedGenerationError(f"{field_name} must be greater than 0")
    return value


def _validate_provider_mode(provider_mode: str) -> str:
    normalized = provider_mode.strip().lower()
    if normalized not in CHAT_GROUNDED_PROVIDER_MODES:
        raise InvalidChatGroundedGenerationError("provider_mode is not supported")
    return normalized


def _validate_profiles(profiles: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if profiles is None:
        return DEFAULT_CHAT_GROUNDED_PROFILES
    normalized = tuple(_validate_nonblank(profile, "profile") for profile in profiles)
    if not normalized:
        raise InvalidChatGroundedGenerationError("profiles must not be empty")
    if len(set(normalized)) != len(normalized):
        raise InvalidChatGroundedGenerationError("profiles must be unique")
    return normalized


def validate_chat_grounded_answer_input(
    grounded_input: ChatGroundedAnswerInput,
) -> ChatGroundedAnswerInput:
    return ChatGroundedAnswerInput(
        content=_validate_nonblank(grounded_input.content, "content"),
        actor_user_id=_validate_positive_int(grounded_input.actor_user_id, "actor_user_id"),
        requested_search_scope=_validate_nonblank(
            grounded_input.requested_search_scope,
            "requested_search_scope",
        ),
        provider_mode=_validate_provider_mode(grounded_input.provider_mode),
        generation_template_key=(
            _validate_nonblank(grounded_input.generation_template_key, "generation_template_key")
            if grounded_input.generation_template_key is not None
            else None
        ),
        top_k=_validate_positive_int(grounded_input.top_k, "top_k"),
        profiles=_validate_profiles(grounded_input.profiles),
        chunk_policy_name=(
            _validate_nonblank(grounded_input.chunk_policy_name, "chunk_policy_name")
            if grounded_input.chunk_policy_name is not None
            else None
        ),
        document_group=(
            _validate_nonblank(grounded_input.document_group, "document_group")
            if grounded_input.document_group is not None
            else None
        ),
        file_type=(
            _validate_nonblank(grounded_input.file_type, "file_type")
            if grounded_input.file_type is not None
            else None
        ),
        bm25_tokenizer_name=_validate_nonblank(
            grounded_input.bm25_tokenizer_name,
            "bm25_tokenizer_name",
        ),
        hybrid_vector_profile_name=(
            _validate_nonblank(
                grounded_input.hybrid_vector_profile_name,
                "hybrid_vector_profile_name",
            )
            if grounded_input.hybrid_vector_profile_name is not None
            else None
        ),
        reranked_vector_profile_name=(
            _validate_nonblank(
                grounded_input.reranked_vector_profile_name,
                "reranked_vector_profile_name",
            )
            if grounded_input.reranked_vector_profile_name is not None
            else None
        ),
        allow_mock_fallback=bool(grounded_input.allow_mock_fallback),
        max_context_chars=_validate_positive_int(
            grounded_input.max_context_chars,
            "max_context_chars",
        ),
        include_neighbors=bool(grounded_input.include_neighbors),
        max_items=_validate_positive_int(grounded_input.max_items, "max_items"),
    )


def _request_metadata(grounded_input: ChatGroundedAnswerInput) -> dict[str, Any]:
    return {
        "prompt_version": CHAT_GROUNDED_ANSWER_PROMPT_VERSION,
        "query_text": grounded_input.content,
        "actor_user_id": grounded_input.actor_user_id,
        "requested_search_scope": grounded_input.requested_search_scope,
        "provider_mode": grounded_input.provider_mode,
        "generation_template_key": grounded_input.generation_template_key,
        "top_k": grounded_input.top_k,
        "profiles": list(grounded_input.profiles or ()),
        "chunk_policy_name": grounded_input.chunk_policy_name,
        "document_group": grounded_input.document_group,
        "file_type": grounded_input.file_type,
        "bm25_tokenizer_name": grounded_input.bm25_tokenizer_name,
        "max_context_chars": grounded_input.max_context_chars,
        "include_neighbors": grounded_input.include_neighbors,
        "max_items": grounded_input.max_items,
    }


def _answer_text(result: DirectGenerationResult) -> str:
    answer_text = result.generation_report.run.answer_text
    if answer_text and answer_text.strip():
        return answer_text
    return "근거 기반 답변 생성을 완료했지만 표시할 답변 본문이 없습니다."


def execute_chat_grounded_answer(
    database_url: str,
    grounded_input: ChatGroundedAnswerInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    fallback_reranker_runtime_config: RerankerRuntimeConfig | None = None,
    api_key: str | None = None,
    generation_provider_client: GenerationProvider | None = None,
    direct_generation_runner: DirectGenerationRunner = run_direct_generation_query,
) -> ChatGroundedAnswerResult:
    database_url = _validate_nonblank(database_url, "database_url")
    validated = validate_chat_grounded_answer_input(grounded_input)
    direct_result = direct_generation_runner(
        database_url,
        DirectGenerationInput(
            query_text=validated.content,
            actor_user_id=validated.actor_user_id or 0,
            requested_search_scope=validated.requested_search_scope,
            provider_mode=validated.provider_mode,
            generation_template_key=validated.generation_template_key,
            top_k=validated.top_k,
            profiles=validated.profiles,
            chunk_policy_name=validated.chunk_policy_name,
            document_group=validated.document_group,
            file_type=validated.file_type,
            bm25_tokenizer_name=validated.bm25_tokenizer_name,
            hybrid_vector_profile_name=validated.hybrid_vector_profile_name,
            reranked_vector_profile_name=validated.reranked_vector_profile_name,
            allow_mock_fallback=validated.allow_mock_fallback,
            max_context_chars=validated.max_context_chars,
            include_neighbors=validated.include_neighbors,
            max_items=validated.max_items,
        ),
        fallback_runtime_config=fallback_runtime_config,
        fallback_reranker_runtime_config=fallback_reranker_runtime_config,
        api_key=api_key,
        generation_provider_client=generation_provider_client,
    )
    search_log_id = direct_result.search_result.search_log_id
    generation_run = direct_result.generation_report.run
    included_count = direct_result.retrieval_package.summary.included_count
    return ChatGroundedAnswerResult(
        answer_text=_answer_text(direct_result),
        direct_generation_result=direct_result,
        prompt_version=CHAT_GROUNDED_ANSWER_PROMPT_VERSION,
        execution_mode=CHAT_GROUNDED_ANSWER_EXECUTION_MODE,
        search_log_id=search_log_id,
        generation_run_id=generation_run.generation_run_id,
        retrieval_context_included_count=included_count,
        generation_status=generation_run.status,
        guardrail_status=generation_run.guardrail_status,
        request_metadata=_request_metadata(validated),
        response_metadata={
            "search_log_id": search_log_id,
            "generation_run_id": generation_run.generation_run_id,
            "generation_status": generation_run.status,
            "guardrail_status": generation_run.guardrail_status,
            "retrieval_context_included_count": included_count,
            "retrieval_package_key": direct_result.retrieval_package.package_key,
        },
    )
