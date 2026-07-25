import pytest

from app.core.bm25_search import BM25SearchResult
from app.core.hybrid_search import (
    DEFAULT_HYBRID_CANDIDATE_MULTIPLIER,
    DEFAULT_RRF_K,
    HYBRID_RETRIEVAL_STRATEGY,
    HYBRID_SEARCH_PROFILE_NAME,
    HybridSearchInput,
    InvalidHybridSearchError,
    reciprocal_rank_fuse_results,
    validate_hybrid_search_input,
)
from app.core.vector_search import VectorSearchResult


def _vector_result(chunk_id: int, rank: int, score: float = 0.9) -> VectorSearchResult:
    return VectorSearchResult(
        profile_name="qwen3_4b_2560",
        rank=rank,
        chunk_id=chunk_id,
        document_id=chunk_id + 100,
        file_id=chunk_id + 200,
        distance=1.0 - score,
        score=score,
        chunk_text=f"vector chunk {chunk_id}",
        chunk_preview=f"vector chunk {chunk_id}",
        content_hash=f"hash-{chunk_id}",
        chunk_policy_name="heading_1000_100",
        heading_path=("Vector",),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title="Vector Doc",
        document_group="default",
        original_file_name="vector.md",
        file_ext=".md",
        embedding_elapsed_ms=12,
    )


def _keyword_result(chunk_id: int, rank: int, score: float = 1.7) -> BM25SearchResult:
    return BM25SearchResult(
        profile_name="bm25_keyword",
        distance=None,
        embedding_elapsed_ms=None,
        search_profile_name="bm25_keyword",
        retrieval_strategy="bm25_keyword",
        rank=rank,
        chunk_id=chunk_id,
        document_id=chunk_id + 100,
        file_id=chunk_id + 200,
        score=score,
        chunk_text=f"keyword chunk {chunk_id}",
        chunk_preview=f"keyword chunk {chunk_id}",
        content_hash=f"hash-{chunk_id}",
        chunk_policy_name="heading_1000_100",
        heading_path=("Keyword",),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title="Keyword Doc",
        document_group="default",
        original_file_name="keyword.md",
        file_ext=".md",
        matched_term_count=2,
        document_length=10.0,
        score_components={"query_terms": ["policy"]},
    )


def test_validate_hybrid_search_input_normalizes_values_and_candidate_limit() -> None:
    validated = validate_hybrid_search_input(
        HybridSearchInput(
            query_text=" 정책 검색 ",
            vector_profile_name=" qwen3_4b_2560 ",
            top_k=3,
            chunk_policy_name=" heading_1000_100 ",
            bm25_tokenizer_name=" unicode_word_v1 ",
        )
    )

    assert validated.query_text == "정책 검색"
    assert validated.vector_profile_name == "qwen3_4b_2560"
    assert validated.chunk_policy_name == "heading_1000_100"
    assert validated.bm25_tokenizer_name == "unicode_word_v1"
    assert validated.rrf_k == DEFAULT_RRF_K
    assert validated.candidate_multiplier == DEFAULT_HYBRID_CANDIDATE_MULTIPLIER
    assert validated.candidate_top_k == 12


@pytest.mark.parametrize(
    ("search_input", "message"),
    [
        (
            HybridSearchInput(query_text=" ", vector_profile_name="qwen3_4b_2560"),
            "query_text",
        ),
        (
            HybridSearchInput(query_text="query", vector_profile_name=" "),
            "vector_profile_name",
        ),
        (
            HybridSearchInput(query_text="query", vector_profile_name="qwen3_4b_2560", top_k=0),
            "top_k",
        ),
        (
            HybridSearchInput(
                query_text="query",
                vector_profile_name="qwen3_4b_2560",
                top_k=101,
            ),
            "top_k",
        ),
        (
            HybridSearchInput(
                query_text="query",
                vector_profile_name="qwen3_4b_2560",
                rrf_k=0,
            ),
            "rrf_k",
        ),
        (
            HybridSearchInput(
                query_text="query",
                vector_profile_name="qwen3_4b_2560",
                candidate_multiplier=0,
            ),
            "candidate_multiplier",
        ),
    ],
)
def test_validate_hybrid_search_input_rejects_invalid_values(
    search_input: HybridSearchInput,
    message: str,
) -> None:
    with pytest.raises(InvalidHybridSearchError, match=message):
        validate_hybrid_search_input(search_input)


def test_reciprocal_rank_fuse_results_combines_duplicate_chunks_with_components() -> None:
    results = reciprocal_rank_fuse_results(
        vector_results=(
            _vector_result(10, 1, score=0.95),
            _vector_result(20, 2, score=0.88),
        ),
        keyword_results=(
            _keyword_result(20, 1, score=3.5),
            _keyword_result(30, 2, score=2.0),
        ),
        top_k=3,
        rrf_k=60,
    )

    assert [result.chunk_id for result in results] == [20, 10, 30]
    assert [result.rank for result in results] == [1, 2, 3]
    top = results[0]
    assert top.profile_name == HYBRID_SEARCH_PROFILE_NAME
    assert top.search_profile_name == HYBRID_SEARCH_PROFILE_NAME
    assert top.retrieval_strategy == HYBRID_RETRIEVAL_STRATEGY
    assert top.score_components["fusion"] == "rrf"
    assert top.score_components["rrf_k"] == 60
    assert top.score_components["source_count"] == 2
    assert top.score_components["sources"] == ["vector", "keyword"]
    assert top.score_components["vector_rank"] == 2
    assert top.score_components["keyword_rank"] == 1
    assert top.score_components["keyword_score_components"] == {"query_terms": ["policy"]}


def test_reciprocal_rank_fuse_results_uses_stable_tie_breaks_and_limits() -> None:
    results = reciprocal_rank_fuse_results(
        vector_results=(
            _vector_result(30, 1),
            _vector_result(10, 1),
            _vector_result(20, 2),
        ),
        keyword_results=(),
        top_k=2,
        rrf_k=60,
    )

    assert [result.chunk_id for result in results] == [10, 30]


@pytest.mark.parametrize(
    ("top_k", "rrf_k", "message"),
    [
        (0, 60, "top_k"),
        (101, 60, "top_k"),
        (5, 0, "rrf_k"),
    ],
)
def test_reciprocal_rank_fuse_results_rejects_invalid_parameters(
    top_k: int,
    rrf_k: int,
    message: str,
) -> None:
    with pytest.raises(InvalidHybridSearchError, match=message):
        reciprocal_rank_fuse_results(
            vector_results=(),
            keyword_results=(),
            top_k=top_k,
            rrf_k=rrf_k,
        )
