"""Hybrid retrieval helpers for combining keyword and vector candidates."""

from dataclasses import dataclass, field
from typing import Any

from app.core.bm25_search import BM25SearchResult
from app.core.vector_search import MAX_TOP_K, VectorSearchResult

HYBRID_SEARCH_PROFILE_NAME = "hybrid_keyword_vector"
HYBRID_RETRIEVAL_STRATEGY = "hybrid_keyword_vector"
DEFAULT_RRF_K = 60
DEFAULT_HYBRID_CANDIDATE_MULTIPLIER = 4
SearchFusionCandidate = VectorSearchResult | BM25SearchResult


@dataclass(frozen=True)
class HybridSearchInput:
    query_text: str
    vector_profile_name: str
    top_k: int = 5
    chunk_policy_name: str | None = None
    bm25_tokenizer_name: str | None = None
    rrf_k: int = DEFAULT_RRF_K
    candidate_multiplier: int = DEFAULT_HYBRID_CANDIDATE_MULTIPLIER

    @property
    def candidate_top_k(self) -> int:
        return min(MAX_TOP_K, self.top_k * self.candidate_multiplier)


@dataclass(frozen=True)
class HybridSearchResult:
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


class InvalidHybridSearchError(ValueError):
    """Raised when hybrid search input cannot be fused safely."""


def validate_hybrid_search_input(search_input: HybridSearchInput) -> HybridSearchInput:
    query_text = search_input.query_text.strip()
    vector_profile_name = search_input.vector_profile_name.strip()
    if not query_text:
        raise InvalidHybridSearchError("query_text is required")
    if not vector_profile_name:
        raise InvalidHybridSearchError("vector_profile_name is required")
    if search_input.top_k <= 0:
        raise InvalidHybridSearchError("top_k must be greater than 0")
    if search_input.top_k > MAX_TOP_K:
        raise InvalidHybridSearchError(f"top_k must be less than or equal to {MAX_TOP_K}")
    if search_input.rrf_k <= 0:
        raise InvalidHybridSearchError("rrf_k must be greater than 0")
    if search_input.candidate_multiplier <= 0:
        raise InvalidHybridSearchError("candidate_multiplier must be greater than 0")
    return HybridSearchInput(
        query_text=query_text,
        vector_profile_name=vector_profile_name,
        top_k=search_input.top_k,
        chunk_policy_name=(
            search_input.chunk_policy_name.strip()
            if search_input.chunk_policy_name is not None
            else None
        ),
        bm25_tokenizer_name=(
            search_input.bm25_tokenizer_name.strip()
            if search_input.bm25_tokenizer_name is not None
            else None
        ),
        rrf_k=search_input.rrf_k,
        candidate_multiplier=search_input.candidate_multiplier,
    )


def _rrf_contribution(rank: int, rrf_k: int) -> float:
    return 1.0 / (rrf_k + rank)


def _copy_candidate_as_hybrid_result(
    candidate: SearchFusionCandidate,
    *,
    rank: int,
    score: float,
    score_components: dict[str, Any],
) -> HybridSearchResult:
    return HybridSearchResult(
        profile_name=HYBRID_SEARCH_PROFILE_NAME,
        distance=getattr(candidate, "distance", None),
        embedding_elapsed_ms=getattr(candidate, "embedding_elapsed_ms", None),
        search_profile_name=HYBRID_SEARCH_PROFILE_NAME,
        retrieval_strategy=HYBRID_RETRIEVAL_STRATEGY,
        rank=rank,
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        file_id=candidate.file_id,
        score=score,
        chunk_text=candidate.chunk_text,
        chunk_preview=candidate.chunk_preview,
        content_hash=candidate.content_hash,
        chunk_policy_name=candidate.chunk_policy_name,
        heading_path=candidate.heading_path,
        page_no=candidate.page_no,
        slide_no=candidate.slide_no,
        sheet_name=candidate.sheet_name,
        cell_range=candidate.cell_range,
        document_title=candidate.document_title,
        document_group=candidate.document_group,
        original_file_name=candidate.original_file_name,
        file_ext=candidate.file_ext,
        score_components=score_components,
    )


def reciprocal_rank_fuse_results(
    *,
    vector_results: tuple[VectorSearchResult, ...] | list[VectorSearchResult],
    keyword_results: tuple[BM25SearchResult, ...] | list[BM25SearchResult],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[HybridSearchResult, ...]:
    if top_k <= 0:
        raise InvalidHybridSearchError("top_k must be greater than 0")
    if top_k > MAX_TOP_K:
        raise InvalidHybridSearchError(f"top_k must be less than or equal to {MAX_TOP_K}")
    if rrf_k <= 0:
        raise InvalidHybridSearchError("rrf_k must be greater than 0")

    candidates: dict[int, SearchFusionCandidate] = {}
    scores: dict[int, float] = {}
    components: dict[int, dict[str, Any]] = {}

    def add_candidate(candidate: SearchFusionCandidate, source: str) -> None:
        chunk_id = candidate.chunk_id
        contribution = _rrf_contribution(candidate.rank, rrf_k)
        candidates.setdefault(chunk_id, candidate)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
        source_rank_key = f"{source}_rank"
        source_score_key = f"{source}_score"
        source_rrf_key = f"{source}_rrf"
        components.setdefault(
            chunk_id,
            {
                "fusion": "rrf",
                "rrf_k": rrf_k,
                "sources": [],
                "source_count": 0,
            },
        )
        if source not in components[chunk_id]["sources"]:
            components[chunk_id]["sources"].append(source)
            components[chunk_id]["source_count"] += 1
        components[chunk_id][source_rank_key] = candidate.rank
        components[chunk_id][source_score_key] = candidate.score
        components[chunk_id][source_rrf_key] = contribution
        if hasattr(candidate, "score_components"):
            components[chunk_id][f"{source}_score_components"] = candidate.score_components

    for result in vector_results:
        add_candidate(result, "vector")
    for result in keyword_results:
        add_candidate(result, "keyword")

    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            components[chunk_id].get("vector_rank", MAX_TOP_K + 1),
            components[chunk_id].get("keyword_rank", MAX_TOP_K + 1),
            chunk_id,
        ),
    )[:top_k]

    return tuple(
        _copy_candidate_as_hybrid_result(
            candidates[chunk_id],
            rank=index,
            score=scores[chunk_id],
            score_components={
                **components[chunk_id],
                "hybrid_score": scores[chunk_id],
            },
        )
        for index, chunk_id in enumerate(ranked_chunk_ids, start=1)
    )
