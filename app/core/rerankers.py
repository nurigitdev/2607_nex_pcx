"""Reranker runtime contracts and deterministic baseline provider."""

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.vector_search import MAX_TOP_K

DEFAULT_RERANKER_PROFILE_NAME = "qwen3_reranker_4b"
DEFAULT_RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-4B"
DEFAULT_RERANKER_PROVIDER_TYPE = "mock_lexical_overlap"
RERANK_RETRIEVAL_STRATEGY = "reranked"
MAX_RERANK_CANDIDATES = MAX_TOP_K
_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)


@dataclass(frozen=True)
class RerankCandidate:
    candidate_key: str
    rank: int
    text: str
    source_profile_name: str
    source_retrieval_strategy: str
    source_score: float | None = None
    chunk_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankRequest:
    query_text: str
    candidates: tuple[RerankCandidate, ...]
    top_k: int = 5
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME
    reranker_model_id: str = DEFAULT_RERANKER_MODEL_ID


@dataclass(frozen=True)
class RerankResultItem:
    candidate: RerankCandidate
    rank: int
    score: float
    score_components: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    query_text: str
    reranker_profile_name: str
    reranker_model_id: str
    provider_type: str
    retrieval_strategy: str
    candidate_count: int
    returned_count: int
    top_k: int
    results: tuple[RerankResultItem, ...]
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


class RerankerProvider(Protocol):
    provider_type: str

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank candidates for the supplied query."""


class InvalidRerankerError(ValueError):
    """Raised when reranking input or runtime output is invalid."""


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidRerankerError(f"{field_name} is required")
    return normalized


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_PATTERN.findall(text))


def validate_rerank_candidate(candidate: RerankCandidate) -> RerankCandidate:
    if candidate.rank <= 0:
        raise InvalidRerankerError("candidate rank must be greater than 0")
    if candidate.source_score is not None and not isinstance(candidate.source_score, int | float):
        raise InvalidRerankerError("source_score must be numeric")
    return RerankCandidate(
        candidate_key=_validate_nonblank(candidate.candidate_key, "candidate_key"),
        rank=candidate.rank,
        text=_validate_nonblank(candidate.text, "candidate text"),
        source_profile_name=_validate_nonblank(
            candidate.source_profile_name,
            "source_profile_name",
        ),
        source_retrieval_strategy=_validate_nonblank(
            candidate.source_retrieval_strategy,
            "source_retrieval_strategy",
        ),
        source_score=(
            float(candidate.source_score) if candidate.source_score is not None else None
        ),
        chunk_id=candidate.chunk_id,
        metadata=dict(candidate.metadata),
    )


def validate_rerank_request(request: RerankRequest) -> RerankRequest:
    if not request.candidates:
        raise InvalidRerankerError("candidates are required")
    if len(request.candidates) > MAX_RERANK_CANDIDATES:
        raise InvalidRerankerError(
            f"candidates must be less than or equal to {MAX_RERANK_CANDIDATES}"
        )
    if request.top_k <= 0:
        raise InvalidRerankerError("top_k must be greater than 0")
    if request.top_k > MAX_TOP_K:
        raise InvalidRerankerError(f"top_k must be less than or equal to {MAX_TOP_K}")

    candidates = tuple(validate_rerank_candidate(candidate) for candidate in request.candidates)
    candidate_keys = [candidate.candidate_key for candidate in candidates]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise InvalidRerankerError("candidate_key values must be unique")
    return RerankRequest(
        query_text=_validate_nonblank(request.query_text, "query_text"),
        candidates=candidates,
        top_k=min(request.top_k, len(candidates)),
        reranker_profile_name=_validate_nonblank(
            request.reranker_profile_name,
            "reranker_profile_name",
        ),
        reranker_model_id=_validate_nonblank(request.reranker_model_id, "reranker_model_id"),
    )


class MockLexicalOverlapReranker:
    provider_type = DEFAULT_RERANKER_PROVIDER_TYPE

    def rerank(self, request: RerankRequest) -> RerankResult:
        validated = validate_rerank_request(request)
        query_terms = set(_tokenize(validated.query_text))
        scored_items = tuple(
            self._score_candidate(candidate, query_terms=query_terms)
            for candidate in validated.candidates
        )
        ranked_items = sorted(
            scored_items,
            key=lambda item: (
                -item.score,
                item.candidate.rank,
                item.candidate.chunk_id or 0,
                item.candidate.candidate_key,
            ),
        )[: validated.top_k]
        results = tuple(
            RerankResultItem(
                candidate=item.candidate,
                rank=index,
                score=item.score,
                score_components={**item.score_components, "rerank_rank": index},
            )
            for index, item in enumerate(ranked_items, start=1)
        )
        return RerankResult(
            query_text=validated.query_text,
            reranker_profile_name=validated.reranker_profile_name,
            reranker_model_id=validated.reranker_model_id,
            provider_type=self.provider_type,
            retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
            candidate_count=len(validated.candidates),
            returned_count=len(results),
            top_k=validated.top_k,
            results=results,
            runtime_metadata={
                "provider_type": self.provider_type,
                "query_term_count": len(query_terms),
                "candidate_count": len(validated.candidates),
                "top_k": validated.top_k,
            },
        )

    def _score_candidate(
        self,
        candidate: RerankCandidate,
        *,
        query_terms: set[str],
    ) -> RerankResultItem:
        candidate_terms = set(_tokenize(candidate.text))
        matched_terms = sorted(query_terms & candidate_terms)
        denominator = max(1, len(query_terms))
        lexical_score = len(matched_terms) / denominator
        source_score_hint = (
            max(0.0, min(1.0, float(candidate.source_score)))
            if candidate.source_score is not None
            else 0.0
        )
        score = lexical_score + (source_score_hint * 0.001)
        return RerankResultItem(
            candidate=candidate,
            rank=candidate.rank,
            score=score,
            score_components={
                "reranker": self.provider_type,
                "matched_terms": matched_terms,
                "matched_term_count": len(matched_terms),
                "query_term_count": len(query_terms),
                "lexical_score": lexical_score,
                "source_score_hint": source_score_hint,
                "source_rank": candidate.rank,
                "source_profile_name": candidate.source_profile_name,
                "source_retrieval_strategy": candidate.source_retrieval_strategy,
            },
        )


def rerank_candidates(
    request: RerankRequest,
    *,
    provider: RerankerProvider | None = None,
) -> RerankResult:
    reranker = provider or MockLexicalOverlapReranker()
    result = reranker.rerank(request)
    if result.returned_count != len(result.results):
        raise InvalidRerankerError("returned_count must match result length")
    if result.returned_count > result.top_k:
        raise InvalidRerankerError("returned_count must be less than or equal to top_k")
    return result
