"""Direct query orchestration from retrieval search to generation run."""

from dataclasses import dataclass

from app.core.bm25_keyword_index import DEFAULT_BM25_TOKENIZER_NAME
from app.core.embedding_providers import EmbeddingProviderRuntimeConfig
from app.core.generation_executor import (
    GenerationExecutionReport,
    execute_mock_generation_run,
    execute_remote_generation_run,
)
from app.core.generation_providers import GenerationProvider
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
)
from app.core.query_embeddings import QueryEmbeddingProviderBuilder
from app.core.rerankers import RerankerProviderBuilder, RerankerRuntimeConfig
from app.core.retrieval_context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    DEFAULT_CONTEXT_MAX_ITEMS,
    RetrievalContextInput,
    RetrievalContextPackage,
    build_retrieval_context_package,
    validate_retrieval_context_input,
)
from app.core.search_compare import SearchCompareInput, SearchCompareResult, run_search_compare

DIRECT_GENERATION_PROVIDER_MODES = {
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
}


@dataclass(frozen=True)
class DirectGenerationInput:
    query_text: str
    actor_user_id: int
    requested_search_scope: str = "company"
    provider_mode: str = GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    generation_template_key: str | None = None
    top_k: int = 5
    profiles: tuple[str, ...] | None = None
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
class DirectGenerationResult:
    search_result: SearchCompareResult
    retrieval_package: RetrievalContextPackage
    generation_report: GenerationExecutionReport


class InvalidDirectGenerationError(ValueError):
    """Raised when a direct generation query cannot be orchestrated."""


def _validate_provider_mode(provider_mode: str) -> str:
    normalized = provider_mode.strip().lower()
    if normalized not in DIRECT_GENERATION_PROVIDER_MODES:
        raise InvalidDirectGenerationError("provider_mode is not supported")
    return normalized


def _validate_direct_generation_input(
    direct_input: DirectGenerationInput,
) -> DirectGenerationInput:
    if not direct_input.query_text.strip():
        raise InvalidDirectGenerationError("query_text must not be blank")
    if direct_input.actor_user_id <= 0:
        raise InvalidDirectGenerationError("actor_user_id must be greater than 0")
    if direct_input.top_k <= 0:
        raise InvalidDirectGenerationError("top_k must be greater than 0")
    profiles = direct_input.profiles
    if profiles is not None:
        profiles = tuple(profile.strip() for profile in profiles)
        if not profiles or any(not profile for profile in profiles):
            raise InvalidDirectGenerationError("profiles must not contain blank values")
        if len(set(profiles)) != len(profiles):
            raise InvalidDirectGenerationError("profiles must be unique")
    context_input = validate_retrieval_context_input(
        RetrievalContextInput(
            search_log_id=1,
            max_context_chars=direct_input.max_context_chars,
            include_neighbors=direct_input.include_neighbors,
            max_items=direct_input.max_items,
        )
    )
    return DirectGenerationInput(
        query_text=direct_input.query_text.strip(),
        actor_user_id=direct_input.actor_user_id,
        requested_search_scope=direct_input.requested_search_scope.strip(),
        provider_mode=_validate_provider_mode(direct_input.provider_mode),
        generation_template_key=(
            direct_input.generation_template_key.strip() or None
            if direct_input.generation_template_key
            else None
        ),
        top_k=direct_input.top_k,
        profiles=profiles,
        chunk_policy_name=(
            direct_input.chunk_policy_name.strip() if direct_input.chunk_policy_name else None
        ),
        document_group=direct_input.document_group.strip() if direct_input.document_group else None,
        file_type=direct_input.file_type.strip() if direct_input.file_type else None,
        bm25_tokenizer_name=direct_input.bm25_tokenizer_name,
        hybrid_vector_profile_name=(
            direct_input.hybrid_vector_profile_name.strip()
            if direct_input.hybrid_vector_profile_name
            else None
        ),
        reranked_vector_profile_name=(
            direct_input.reranked_vector_profile_name.strip()
            if direct_input.reranked_vector_profile_name
            else None
        ),
        allow_mock_fallback=direct_input.allow_mock_fallback,
        max_context_chars=context_input.max_context_chars,
        include_neighbors=context_input.include_neighbors,
        max_items=context_input.max_items,
    )


def _generation_report_for_provider_mode(
    database_url: str,
    *,
    direct_input: DirectGenerationInput,
    retrieval_package: RetrievalContextPackage,
    api_key: str | None,
    generation_provider_client: GenerationProvider | None,
) -> GenerationExecutionReport:
    if direct_input.provider_mode == GENERATION_PROVIDER_MODE_MOCK:
        return execute_mock_generation_run(
            database_url,
            retrieval_package,
            generation_template_key=direct_input.generation_template_key,
            created_by="api_direct_generation",
            created_by_user_id=direct_input.actor_user_id,
        )
    return execute_remote_generation_run(
        database_url,
        retrieval_package,
        generation_template_key=direct_input.generation_template_key,
        provider_client=generation_provider_client,
        api_key=api_key,
        created_by="api_direct_generation",
        created_by_user_id=direct_input.actor_user_id,
    )


def run_direct_generation_query(
    database_url: str,
    direct_input: DirectGenerationInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    fallback_reranker_runtime_config: RerankerRuntimeConfig | None = None,
    api_key: str | None = None,
    generation_provider_client: GenerationProvider | None = None,
    query_embedding_provider_builder: QueryEmbeddingProviderBuilder | None = None,
    reranker_provider_builder: RerankerProviderBuilder | None = None,
) -> DirectGenerationResult:
    """Run search, package retrieval context, and persist a generation run."""

    validated = _validate_direct_generation_input(direct_input)
    search_kwargs = {}
    if query_embedding_provider_builder is not None:
        search_kwargs["query_embedding_provider_builder"] = query_embedding_provider_builder
    if reranker_provider_builder is not None:
        search_kwargs["reranker_provider_builder"] = reranker_provider_builder

    search_result = run_search_compare(
        database_url,
        SearchCompareInput(
            query_text=validated.query_text,
            actor_user_id=validated.actor_user_id,
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
        ),
        fallback_runtime_config=fallback_runtime_config,
        fallback_reranker_runtime_config=fallback_reranker_runtime_config,
        **search_kwargs,
    )
    retrieval_package = build_retrieval_context_package(
        database_url,
        RetrievalContextInput(
            search_log_id=search_result.search_log_id,
            max_context_chars=validated.max_context_chars,
            include_neighbors=validated.include_neighbors,
            max_items=validated.max_items,
        ),
    )
    if retrieval_package is None:
        raise InvalidDirectGenerationError("retrieval context package was not created")
    generation_report = _generation_report_for_provider_mode(
        database_url,
        direct_input=validated,
        retrieval_package=retrieval_package,
        api_key=api_key,
        generation_provider_client=generation_provider_client,
    )
    return DirectGenerationResult(
        search_result=search_result,
        retrieval_package=retrieval_package,
        generation_report=generation_report,
    )
