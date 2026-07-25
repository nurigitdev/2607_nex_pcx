"""Helpers for applying reranker providers to search result candidates."""

from dataclasses import dataclass, field
from typing import Any

from app.core.bm25_search import BM25SearchResult
from app.core.hybrid_search import HybridSearchResult
from app.core.rerankers import (
    RERANK_RETRIEVAL_STRATEGY,
    RerankCandidate,
    RerankerProvider,
    RerankRequest,
    rerank_candidates,
)
from app.core.vector_search import MAX_TOP_K, VectorSearchResult

RERANKED_SEARCH_PROFILE_NAME = "reranked_vector_cosine"
SearchRerankSourceResult = VectorSearchResult | BM25SearchResult | HybridSearchResult


@dataclass(frozen=True)
class RerankedSearchResult:
    profile_name: str
    distance: float | None
    embedding_elapsed_ms: int | None
    search_profile_name: str
    retrieval_strategy: str
    rank: int
    chunk_id: int
    document_id: int
    file_id: int
    score: float
    chunk_text: str
    chunk_preview: str
    content_hash: str
    chunk_policy_name: str
    heading_path: tuple[str, ...]
    page_no: int | None
    slide_no: int | None
    sheet_name: str | None
    cell_range: str | None
    document_title: str | None
    document_group: str
    original_file_name: str
    file_ext: str | None
    score_components: dict[str, Any] = field(default_factory=dict)


class InvalidRerankedSearchError(ValueError):
    """Raised when search results cannot be converted into rerank candidates."""


def _source_search_profile_name(result: SearchRerankSourceResult) -> str:
    return str(getattr(result, "search_profile_name", result.profile_name))


def _source_retrieval_strategy(result: SearchRerankSourceResult) -> str:
    return str(getattr(result, "retrieval_strategy", "vector_cosine"))


def _candidate_key(result: SearchRerankSourceResult) -> str:
    return f"{_source_search_profile_name(result)}:{result.chunk_id}"


def build_rerank_candidates(
    results: tuple[SearchRerankSourceResult, ...] | list[SearchRerankSourceResult],
) -> tuple[RerankCandidate, ...]:
    candidates: list[RerankCandidate] = []
    seen_keys: set[str] = set()
    for result in results:
        candidate_key = _candidate_key(result)
        if candidate_key in seen_keys:
            raise InvalidRerankedSearchError("search result candidates must be unique")
        seen_keys.add(candidate_key)
        candidates.append(
            RerankCandidate(
                candidate_key=candidate_key,
                rank=result.rank,
                text=result.chunk_text,
                source_profile_name=_source_search_profile_name(result),
                source_retrieval_strategy=_source_retrieval_strategy(result),
                source_score=result.score,
                chunk_id=result.chunk_id,
                metadata={
                    "document_id": result.document_id,
                    "file_id": result.file_id,
                    "content_hash": result.content_hash,
                    "chunk_policy_name": result.chunk_policy_name,
                },
            )
        )
    return tuple(candidates)


def _copy_result_as_reranked(
    source: SearchRerankSourceResult,
    *,
    rank: int,
    score: float,
    score_components: dict[str, Any],
) -> RerankedSearchResult:
    return RerankedSearchResult(
        profile_name=RERANKED_SEARCH_PROFILE_NAME,
        distance=getattr(source, "distance", None),
        embedding_elapsed_ms=getattr(source, "embedding_elapsed_ms", None),
        search_profile_name=RERANKED_SEARCH_PROFILE_NAME,
        retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
        rank=rank,
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        file_id=source.file_id,
        score=score,
        chunk_text=source.chunk_text,
        chunk_preview=source.chunk_preview,
        content_hash=source.content_hash,
        chunk_policy_name=source.chunk_policy_name,
        heading_path=source.heading_path,
        page_no=source.page_no,
        slide_no=source.slide_no,
        sheet_name=source.sheet_name,
        cell_range=source.cell_range,
        document_title=source.document_title,
        document_group=source.document_group,
        original_file_name=source.original_file_name,
        file_ext=source.file_ext,
        score_components=score_components,
    )


def rerank_search_results(
    *,
    query_text: str,
    results: tuple[SearchRerankSourceResult, ...] | list[SearchRerankSourceResult],
    top_k: int,
    provider: RerankerProvider | None = None,
) -> tuple[RerankedSearchResult, ...]:
    if top_k <= 0:
        raise InvalidRerankedSearchError("top_k must be greater than 0")
    if top_k > MAX_TOP_K:
        raise InvalidRerankedSearchError(f"top_k must be less than or equal to {MAX_TOP_K}")
    if not results:
        return ()

    source_by_key = {_candidate_key(result): result for result in results}
    if len(source_by_key) != len(results):
        raise InvalidRerankedSearchError("search result candidates must be unique")
    rerank_result = rerank_candidates(
        RerankRequest(
            query_text=query_text,
            candidates=build_rerank_candidates(results),
            top_k=top_k,
        ),
        provider=provider,
    )
    reranked_results: list[RerankedSearchResult] = []
    for item in rerank_result.results:
        source = source_by_key[item.candidate.candidate_key]
        reranked_results.append(
            _copy_result_as_reranked(
                source,
                rank=item.rank,
                score=item.score,
                score_components={
                    **item.score_components,
                    "source_profile_name": _source_search_profile_name(source),
                    "source_retrieval_strategy": _source_retrieval_strategy(source),
                    "source_rank": source.rank,
                    "source_score": source.score,
                    "source_score_components": getattr(source, "score_components", {}),
                    "reranker_profile_name": rerank_result.reranker_profile_name,
                    "reranker_model_id": rerank_result.reranker_model_id,
                    "reranker_provider_type": rerank_result.provider_type,
                    "candidate_count": rerank_result.candidate_count,
                },
            )
        )
    return tuple(reranked_results)
