"""Template-aware document generation orchestration for chat UX."""

from dataclasses import dataclass
from typing import Any

from app.core.bm25_keyword_index import DEFAULT_BM25_TOKENIZER_NAME
from app.core.chat_grounded_generation import (
    DEFAULT_CHAT_GROUNDED_PROFILES,
    DEFAULT_CHAT_GROUNDED_TOP_K,
    ChatGroundedAnswerInput,
    ChatGroundedAnswerResult,
    DirectGenerationRunner,
    execute_chat_grounded_answer,
)
from app.core.embedding_providers import EmbeddingProviderRuntimeConfig
from app.core.generation_providers import GenerationProvider
from app.core.generation_runs import GENERATION_PROVIDER_MODE_MOCK
from app.core.rerankers import RerankerRuntimeConfig
from app.core.retrieval_context import DEFAULT_CONTEXT_CHAR_BUDGET, DEFAULT_CONTEXT_MAX_ITEMS

CHAT_DOCUMENT_GENERATION_EXECUTION_MODE = "template_direct_generation"
CHAT_DOCUMENT_GENERATION_PROMPT_VERSION = "chat_document_generation_v1"
CHAT_DOCUMENT_GENERATION_DEFAULT_TEMPLATE_KEY = "report"
CHAT_DOCUMENT_GENERATION_TEMPLATE_KEYS = {
    "report",
    "proposal",
    "summary",
    "meeting_minutes",
}


@dataclass(frozen=True)
class ChatDocumentGenerationInput:
    content: str
    actor_user_id: int | None
    requested_search_scope: str = "company"
    provider_mode: str = GENERATION_PROVIDER_MODE_MOCK
    requested_template_key: str | None = None
    top_k: int = DEFAULT_CHAT_GROUNDED_TOP_K
    profiles: tuple[str, ...] | None = DEFAULT_CHAT_GROUNDED_PROFILES
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    bm25_tokenizer_name: str | None = None
    hybrid_vector_profile_name: str | None = None
    reranked_vector_profile_name: str | None = None
    allow_mock_fallback: bool = True
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET
    include_neighbors: bool = True
    max_items: int = DEFAULT_CONTEXT_MAX_ITEMS


@dataclass(frozen=True)
class ChatDocumentGenerationResult:
    answer_text: str
    grounded_result: ChatGroundedAnswerResult
    template_key: str
    prompt_version: str
    execution_mode: str
    search_log_id: int
    generation_run_id: int
    request_metadata: dict[str, Any]
    response_metadata: dict[str, Any]


class InvalidChatDocumentGenerationError(ValueError):
    """Raised when a chat document generation request is invalid."""


def _normalize_template_key(template_key: str) -> str:
    normalized = template_key.strip().lower()
    if not normalized:
        raise InvalidChatDocumentGenerationError("generation_template_key must not be blank")
    if normalized not in CHAT_DOCUMENT_GENERATION_TEMPLATE_KEYS:
        raise InvalidChatDocumentGenerationError("generation_template_key is not supported")
    return normalized


def select_chat_document_generation_template_key(
    content: str,
    *,
    requested_template_key: str | None = None,
) -> str:
    if requested_template_key is not None:
        return _normalize_template_key(requested_template_key)
    normalized = content.strip().lower()
    if not normalized:
        raise InvalidChatDocumentGenerationError("content must not be blank")
    if "제안서" in normalized or "proposal" in normalized:
        return "proposal"
    if "회의록" in normalized or "meeting minutes" in normalized or "minutes" in normalized:
        return "meeting_minutes"
    if "요약서" in normalized or "summary document" in normalized:
        return "summary"
    return CHAT_DOCUMENT_GENERATION_DEFAULT_TEMPLATE_KEY


def _request_metadata(
    document_input: ChatDocumentGenerationInput,
    *,
    template_key: str,
) -> dict[str, Any]:
    return {
        "prompt_version": CHAT_DOCUMENT_GENERATION_PROMPT_VERSION,
        "query_text": document_input.content,
        "actor_user_id": document_input.actor_user_id,
        "requested_search_scope": document_input.requested_search_scope,
        "provider_mode": document_input.provider_mode,
        "requested_template_key": document_input.requested_template_key,
        "resolved_template_key": template_key,
        "top_k": document_input.top_k,
        "profiles": list(document_input.profiles or ()),
        "chunk_policy_name": document_input.chunk_policy_name,
        "document_group": document_input.document_group,
        "file_type": document_input.file_type,
    }


def execute_chat_document_generation(
    database_url: str,
    document_input: ChatDocumentGenerationInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    fallback_reranker_runtime_config: RerankerRuntimeConfig | None = None,
    api_key: str | None = None,
    generation_provider_client: GenerationProvider | None = None,
    direct_generation_runner: DirectGenerationRunner | None = None,
) -> ChatDocumentGenerationResult:
    template_key = select_chat_document_generation_template_key(
        document_input.content,
        requested_template_key=document_input.requested_template_key,
    )
    grounded_kwargs: dict[str, Any] = {}
    if direct_generation_runner is not None:
        grounded_kwargs["direct_generation_runner"] = direct_generation_runner
    grounded_result = execute_chat_grounded_answer(
        database_url,
        ChatGroundedAnswerInput(
            content=document_input.content,
            actor_user_id=document_input.actor_user_id,
            requested_search_scope=document_input.requested_search_scope,
            provider_mode=document_input.provider_mode,
            generation_template_key=template_key,
            top_k=document_input.top_k,
            profiles=document_input.profiles,
            chunk_policy_name=document_input.chunk_policy_name,
            document_group=document_input.document_group,
            file_type=document_input.file_type,
            bm25_tokenizer_name=(
                document_input.bm25_tokenizer_name
                if document_input.bm25_tokenizer_name is not None
                else DEFAULT_BM25_TOKENIZER_NAME
            ),
            hybrid_vector_profile_name=document_input.hybrid_vector_profile_name,
            reranked_vector_profile_name=document_input.reranked_vector_profile_name,
            allow_mock_fallback=document_input.allow_mock_fallback,
            max_context_chars=document_input.max_context_chars,
            include_neighbors=document_input.include_neighbors,
            max_items=document_input.max_items,
        ),
        fallback_runtime_config=fallback_runtime_config,
        fallback_reranker_runtime_config=fallback_reranker_runtime_config,
        api_key=api_key,
        generation_provider_client=generation_provider_client,
        **grounded_kwargs,
    )
    return ChatDocumentGenerationResult(
        answer_text=grounded_result.answer_text,
        grounded_result=grounded_result,
        template_key=template_key,
        prompt_version=CHAT_DOCUMENT_GENERATION_PROMPT_VERSION,
        execution_mode=CHAT_DOCUMENT_GENERATION_EXECUTION_MODE,
        search_log_id=grounded_result.search_log_id,
        generation_run_id=grounded_result.generation_run_id,
        request_metadata=_request_metadata(document_input, template_key=template_key),
        response_metadata={
            "search_log_id": grounded_result.search_log_id,
            "generation_run_id": grounded_result.generation_run_id,
            "template_key": template_key,
            "grounded_execution_mode": grounded_result.execution_mode,
            "retrieval_context_included_count": (grounded_result.retrieval_context_included_count),
        },
    )
