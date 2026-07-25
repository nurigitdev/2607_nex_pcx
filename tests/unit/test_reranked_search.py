import pytest

from app.core.bm25_search import BM25SearchResult
from app.core.reranked_search import (
    RERANKED_SEARCH_PROFILE_NAME,
    InvalidRerankedSearchError,
    build_rerank_candidates,
    rerank_search_results,
)
from app.core.rerankers import RERANK_RETRIEVAL_STRATEGY
from app.core.vector_search import VectorSearchResult


def _vector_result(chunk_id: int, rank: int, text: str, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        profile_name="qwen3_4b_2560",
        rank=rank,
        chunk_id=chunk_id,
        document_id=chunk_id + 100,
        file_id=chunk_id + 200,
        distance=1.0 - score,
        score=score,
        chunk_text=text,
        chunk_preview=text,
        content_hash=f"hash-{chunk_id}",
        chunk_policy_name="heading_1000_100",
        heading_path=("Policy",),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title="Policy",
        document_group="default",
        original_file_name="policy.md",
        file_ext=".md",
        embedding_elapsed_ms=4,
    )


def _bm25_result(chunk_id: int, rank: int, text: str, score: float) -> BM25SearchResult:
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
        chunk_text=text,
        chunk_preview=text,
        content_hash=f"hash-{chunk_id}",
        chunk_policy_name="heading_1000_100",
        heading_path=("Policy",),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title="Policy",
        document_group="default",
        original_file_name="policy.md",
        file_ext=".md",
        matched_term_count=2,
        document_length=8.0,
        score_components={"query_terms": ["policy"]},
    )


def test_build_rerank_candidates_preserves_source_identity_and_metadata() -> None:
    candidates = build_rerank_candidates(
        (
            _vector_result(10, 1, "policy vector", 0.8),
            _bm25_result(20, 2, "policy keyword", 2.1),
        )
    )

    assert [candidate.candidate_key for candidate in candidates] == [
        "qwen3_4b_2560:10",
        "bm25_keyword:20",
    ]
    assert candidates[0].source_profile_name == "qwen3_4b_2560"
    assert candidates[0].source_retrieval_strategy == "vector_cosine"
    assert candidates[1].source_profile_name == "bm25_keyword"
    assert candidates[1].source_retrieval_strategy == "bm25_keyword"
    assert candidates[1].metadata["chunk_policy_name"] == "heading_1000_100"


def test_build_rerank_candidates_rejects_duplicate_source_chunk_keys() -> None:
    with pytest.raises(InvalidRerankedSearchError, match="unique"):
        build_rerank_candidates(
            (
                _vector_result(10, 1, "policy vector", 0.8),
                _vector_result(10, 2, "policy vector duplicate", 0.7),
            )
        )


def test_rerank_search_results_returns_reranked_result_shape_and_components() -> None:
    results = rerank_search_results(
        query_text="policy guide",
        results=(
            _vector_result(10, 1, "unrelated", 0.9),
            _vector_result(20, 2, "policy guide text", 0.7),
        ),
        top_k=2,
    )

    assert [result.chunk_id for result in results] == [20, 10]
    assert [result.rank for result in results] == [1, 2]
    top = results[0]
    assert top.profile_name == RERANKED_SEARCH_PROFILE_NAME
    assert top.search_profile_name == RERANKED_SEARCH_PROFILE_NAME
    assert top.retrieval_strategy == RERANK_RETRIEVAL_STRATEGY
    assert top.score_components["source_profile_name"] == "qwen3_4b_2560"
    assert top.score_components["source_retrieval_strategy"] == "vector_cosine"
    assert top.score_components["source_rank"] == 2
    assert top.score_components["reranker_model_id"] == "Qwen/Qwen3-Reranker-4B"
    assert top.score_components["candidate_count"] == 2


def test_rerank_search_results_returns_empty_for_no_candidates() -> None:
    assert rerank_search_results(query_text="policy", results=(), top_k=5) == ()


@pytest.mark.parametrize("top_k", [0, 101])
def test_rerank_search_results_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(InvalidRerankedSearchError, match="top_k"):
        rerank_search_results(
            query_text="policy",
            results=(_vector_result(10, 1, "policy", 0.9),),
            top_k=top_k,
        )


def test_rerank_search_results_rejects_duplicate_candidates_before_provider_call() -> None:
    with pytest.raises(InvalidRerankedSearchError, match="unique"):
        rerank_search_results(
            query_text="policy",
            results=(
                _bm25_result(10, 1, "policy", 1.0),
                _bm25_result(10, 2, "policy duplicate", 0.9),
            ),
            top_k=2,
        )
