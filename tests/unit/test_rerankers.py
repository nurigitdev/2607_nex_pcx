import pytest

from app.core.rerankers import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    DEFAULT_RERANKER_PROVIDER_TYPE,
    MAX_RERANK_CANDIDATES,
    RERANK_RETRIEVAL_STRATEGY,
    InvalidRerankerError,
    RerankCandidate,
    RerankRequest,
    RerankResult,
    rerank_candidates,
    validate_rerank_candidate,
    validate_rerank_request,
)


def _candidate(
    key: str,
    *,
    rank: int,
    text: str,
    source_score: float | None = None,
    chunk_id: int | None = None,
) -> RerankCandidate:
    return RerankCandidate(
        candidate_key=key,
        rank=rank,
        text=text,
        source_profile_name="hybrid_keyword_vector",
        source_retrieval_strategy="hybrid_keyword_vector",
        source_score=source_score,
        chunk_id=chunk_id,
        metadata={"source": "unit"},
    )


def test_validate_rerank_request_normalizes_candidates_and_limits_top_k() -> None:
    request = validate_rerank_request(
        RerankRequest(
            query_text=" 정책 검색 ",
            candidates=(
                _candidate(" c1 ", rank=1, text=" 정책 문서 ", source_score=1, chunk_id=10),
                _candidate("c2", rank=2, text="다른 문서"),
            ),
            top_k=5,
        )
    )

    assert request.query_text == "정책 검색"
    assert request.top_k == 2
    assert request.reranker_profile_name == DEFAULT_RERANKER_PROFILE_NAME
    assert request.reranker_model_id == DEFAULT_RERANKER_MODEL_ID
    assert request.candidates[0].candidate_key == "c1"
    assert request.candidates[0].text == "정책 문서"
    assert request.candidates[0].source_score == 1.0
    assert request.candidates[0].metadata == {"source": "unit"}


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (_candidate(" ", rank=1, text="text"), "candidate_key"),
        (_candidate("c1", rank=0, text="text"), "rank"),
        (_candidate("c1", rank=1, text=" "), "candidate text"),
        (
            RerankCandidate(
                candidate_key="c1",
                rank=1,
                text="text",
                source_profile_name=" ",
                source_retrieval_strategy="hybrid",
            ),
            "source_profile_name",
        ),
        (
            RerankCandidate(
                candidate_key="c1",
                rank=1,
                text="text",
                source_profile_name="hybrid",
                source_retrieval_strategy=" ",
            ),
            "source_retrieval_strategy",
        ),
        (
            RerankCandidate(
                candidate_key="c1",
                rank=1,
                text="text",
                source_profile_name="hybrid",
                source_retrieval_strategy="hybrid",
                source_score="bad",  # type: ignore[arg-type]
            ),
            "source_score",
        ),
    ],
)
def test_validate_rerank_candidate_rejects_invalid_values(
    candidate: RerankCandidate,
    message: str,
) -> None:
    with pytest.raises(InvalidRerankerError, match=message):
        validate_rerank_candidate(candidate)


@pytest.mark.parametrize(
    ("rerank_request", "message"),
    [
        (
            RerankRequest(query_text=" ", candidates=(_candidate("c1", rank=1, text="text"),)),
            "query_text",
        ),
        (RerankRequest(query_text="query", candidates=()), "candidates"),
        (
            RerankRequest(
                query_text="query",
                candidates=tuple(
                    _candidate(f"c{index}", rank=index + 1, text="text")
                    for index in range(MAX_RERANK_CANDIDATES + 1)
                ),
            ),
            "candidates",
        ),
        (
            RerankRequest(
                query_text="query", candidates=(_candidate("c1", rank=1, text="text"),), top_k=0
            ),
            "top_k",
        ),
        (
            RerankRequest(
                query_text="query",
                candidates=(
                    _candidate("c1", rank=1, text="text"),
                    _candidate("c1", rank=2, text="text"),
                ),
            ),
            "candidate_key",
        ),
        (
            RerankRequest(
                query_text="query",
                candidates=(_candidate("c1", rank=1, text="text"),),
                reranker_profile_name=" ",
            ),
            "reranker_profile_name",
        ),
        (
            RerankRequest(
                query_text="query",
                candidates=(_candidate("c1", rank=1, text="text"),),
                reranker_model_id=" ",
            ),
            "reranker_model_id",
        ),
    ],
)
def test_validate_rerank_request_rejects_invalid_values(
    rerank_request: RerankRequest,
    message: str,
) -> None:
    with pytest.raises(InvalidRerankerError, match=message):
        validate_rerank_request(rerank_request)


def test_mock_lexical_overlap_reranker_orders_by_overlap_then_source_rank() -> None:
    result = rerank_candidates(
        RerankRequest(
            query_text="hybrid 정책 검색",
            candidates=(
                _candidate("c1", rank=1, text="hybrid unrelated", source_score=0.9, chunk_id=10),
                _candidate(
                    "c2", rank=2, text="hybrid 정책 검색 문서", source_score=0.1, chunk_id=20
                ),
                _candidate("c3", rank=3, text="정책 검색", chunk_id=30),
            ),
            top_k=2,
        )
    )

    assert result.provider_type == DEFAULT_RERANKER_PROVIDER_TYPE
    assert result.retrieval_strategy == RERANK_RETRIEVAL_STRATEGY
    assert result.candidate_count == 3
    assert result.returned_count == 2
    assert [item.candidate.candidate_key for item in result.results] == ["c2", "c3"]
    assert [item.rank for item in result.results] == [1, 2]
    assert result.results[0].score_components["matched_terms"] == ["hybrid", "검색", "정책"]
    assert result.results[0].score_components["rerank_rank"] == 1
    assert result.runtime_metadata["query_term_count"] == 3


def test_mock_lexical_overlap_reranker_uses_source_score_as_small_hint() -> None:
    result = rerank_candidates(
        RerankRequest(
            query_text="query",
            candidates=(
                _candidate("c1", rank=2, text="query", source_score=0.0),
                _candidate("c2", rank=1, text="query", source_score=1.0),
            ),
            top_k=2,
        )
    )

    assert [item.candidate.candidate_key for item in result.results] == ["c2", "c1"]
    assert result.results[0].score_components["source_score_hint"] == 1.0


def test_rerank_candidates_rejects_invalid_provider_result() -> None:
    class BadProvider:
        provider_type = "bad"

        def rerank(self, request: RerankRequest) -> RerankResult:
            return RerankResult(
                query_text=request.query_text,
                reranker_profile_name=request.reranker_profile_name,
                reranker_model_id=request.reranker_model_id,
                provider_type=self.provider_type,
                retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
                candidate_count=1,
                returned_count=2,
                top_k=1,
                results=(),
            )

    with pytest.raises(InvalidRerankerError, match="returned_count"):
        rerank_candidates(
            RerankRequest(
                query_text="query",
                candidates=(_candidate("c1", rank=1, text="query"),),
            ),
            provider=BadProvider(),
        )
