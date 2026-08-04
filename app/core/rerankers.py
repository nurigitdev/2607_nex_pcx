"""Reranker runtime contracts and deterministic baseline provider."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urljoin

from app.core.vector_search import MAX_TOP_K

DEFAULT_RERANKER_PROFILE_NAME = "qwen3_reranker_0_6b"
DEFAULT_RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANKER_PROVIDER_TYPE = "mock_lexical_overlap"
RERANK_RETRIEVAL_STRATEGY = "reranked"
MOCK_RERANKER_PROVIDER_MODE = "mock"
REMOTE_RERANKER_PROVIDER_MODE = "remote"
RERANKER_PROVIDER_MODES = (MOCK_RERANKER_PROVIDER_MODE, REMOTE_RERANKER_PROVIDER_MODE)
REMOTE_RERANKER_HEALTH_PATH = "/healthz"
REMOTE_RERANKER_RERANK_PATH = "/v1/rerank"
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


@dataclass(frozen=True)
class RerankerHealth:
    ready: bool
    provider_type: str
    provider_model_id: str
    reranker_profile_name: str
    device: str | None = None
    runtime_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankerRuntimeConfig:
    mode: str = MOCK_RERANKER_PROVIDER_MODE
    remote_base_url: str | None = None
    remote_timeout_seconds: float = 60.0
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME
    reranker_model_id: str = DEFAULT_RERANKER_MODEL_ID
    remote_headers: Mapping[str, str] = field(default_factory=dict)


class RerankerProvider(Protocol):
    provider_type: str

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank candidates for the supplied query."""


class InvalidRerankerError(ValueError):
    """Raised when reranking input or runtime output is invalid."""


RerankerProviderBuilder = Callable[[RerankerRuntimeConfig], RerankerProvider]


def reranker_runtime_config_from_settings(settings: object) -> RerankerRuntimeConfig:
    return normalize_reranker_runtime_config(
        RerankerRuntimeConfig(
            mode=str(getattr(settings, "reranker_provider_mode", MOCK_RERANKER_PROVIDER_MODE)),
            remote_base_url=getattr(settings, "remote_reranker_provider_url", None),
            remote_timeout_seconds=float(
                getattr(settings, "remote_reranker_provider_timeout_seconds", 60.0)
            ),
            reranker_profile_name=str(
                getattr(settings, "reranker_profile_name", DEFAULT_RERANKER_PROFILE_NAME)
            ),
            reranker_model_id=str(
                getattr(settings, "reranker_model_id", DEFAULT_RERANKER_MODEL_ID)
            ),
        )
    )


def normalize_reranker_runtime_config(config: RerankerRuntimeConfig) -> RerankerRuntimeConfig:
    mode = config.mode.strip().lower()
    if mode not in RERANKER_PROVIDER_MODES:
        raise InvalidRerankerError(f"Unsupported reranker provider mode: {config.mode}")
    if config.remote_timeout_seconds <= 0:
        raise InvalidRerankerError(
            "remote_reranker_provider_timeout_seconds must be greater than 0"
        )

    remote_base_url = config.remote_base_url.strip().rstrip("/") if config.remote_base_url else None
    if mode == REMOTE_RERANKER_PROVIDER_MODE and not remote_base_url:
        raise InvalidRerankerError(
            "remote_reranker_provider_url is required for remote provider mode"
        )

    return RerankerRuntimeConfig(
        mode=mode,
        remote_base_url=remote_base_url,
        remote_timeout_seconds=config.remote_timeout_seconds,
        reranker_profile_name=_validate_nonblank(
            config.reranker_profile_name,
            "reranker_profile_name",
        ),
        reranker_model_id=_validate_nonblank(config.reranker_model_id, "reranker_model_id"),
        remote_headers=_normalize_remote_headers(config.remote_headers),
    )


def build_reranker_provider_from_runtime_config(
    config: RerankerRuntimeConfig,
    *,
    http_client: object | None = None,
) -> RerankerProvider:
    normalized = normalize_reranker_runtime_config(config)
    if normalized.mode == MOCK_RERANKER_PROVIDER_MODE:
        return MockLexicalOverlapReranker()
    if normalized.remote_base_url is None:
        raise InvalidRerankerError("remote_reranker_provider_url is required")
    return RemoteRerankerProviderClient(
        normalized.remote_base_url,
        timeout_seconds=normalized.remote_timeout_seconds,
        headers=normalized.remote_headers,
        http_client=http_client,
    )


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


class RemoteRerankerProviderClient:
    provider_type = REMOTE_RERANKER_PROVIDER_MODE

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 60.0,
        headers: Mapping[str, str] | None = None,
        http_client: object | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise InvalidRerankerError("base_url is required")
        if timeout_seconds <= 0:
            raise InvalidRerankerError("timeout_seconds must be greater than 0")
        self.base_url = normalized_base_url
        self.timeout_seconds = timeout_seconds
        self.headers = _normalize_remote_headers(headers or {})
        self._owns_client = http_client is None
        self._client = http_client or _create_httpx_client(timeout_seconds=timeout_seconds)

    def health(self) -> RerankerHealth:
        payload = self._request_json("GET", REMOTE_RERANKER_HEALTH_PATH)
        try:
            ready = payload["ready"]
            if not isinstance(ready, bool):
                raise ValueError("ready must be a boolean")
            return RerankerHealth(
                ready=ready,
                provider_type=_validate_nonblank(str(payload["provider_type"]), "provider_type"),
                provider_model_id=_validate_nonblank(
                    str(payload["provider_model_id"]),
                    "provider_model_id",
                ),
                reranker_profile_name=_validate_nonblank(
                    str(payload["reranker_profile_name"]),
                    "reranker_profile_name",
                ),
                device=str(payload["device"]) if payload.get("device") is not None else None,
                runtime_metadata=dict(payload.get("runtime_metadata") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidRerankerError("Invalid reranker health response") from exc

    def rerank(self, request: RerankRequest) -> RerankResult:
        validated = validate_rerank_request(request)
        response_payload = self._request_json(
            "POST",
            REMOTE_RERANKER_RERANK_PATH,
            json=_rerank_request_payload(validated),
        )
        return _remote_rerank_result_from_payload(response_payload, validated)

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def _request_json(self, method: str, path: str, **kwargs) -> dict[str, object]:
        request_headers = dict(self.headers)
        if kwargs.get("headers"):
            request_headers.update(dict(kwargs.pop("headers")))
        try:
            response = self._client.request(  # type: ignore[attr-defined]
                method,
                urljoin(f"{self.base_url}/", path.lstrip("/")),
                timeout=self.timeout_seconds,
                headers=request_headers or None,
                **kwargs,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise InvalidRerankerError(f"Remote reranker request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidRerankerError("Remote reranker response must be a JSON object")
        return payload


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


def _rerank_request_payload(request: RerankRequest) -> dict[str, object]:
    return {
        "query_text": request.query_text,
        "top_k": request.top_k,
        "reranker_profile_name": request.reranker_profile_name,
        "reranker_model_id": request.reranker_model_id,
        "candidates": [
            {
                "candidate_key": candidate.candidate_key,
                "rank": candidate.rank,
                "text": candidate.text,
                "source_profile_name": candidate.source_profile_name,
                "source_retrieval_strategy": candidate.source_retrieval_strategy,
                "source_score": candidate.source_score,
                "chunk_id": candidate.chunk_id,
                "metadata": dict(candidate.metadata),
            }
            for candidate in request.candidates
        ],
    }


def _remote_rerank_result_from_payload(
    payload: Mapping[str, object],
    request: RerankRequest,
) -> RerankResult:
    candidate_by_key = {candidate.candidate_key: candidate for candidate in request.candidates}
    try:
        result_items = tuple(
            _remote_rerank_result_item_from_payload(item, candidate_by_key)
            for item in payload["results"]  # type: ignore[index]
        )
        result = RerankResult(
            query_text=str(payload.get("query_text") or request.query_text),
            reranker_profile_name=str(
                payload.get("reranker_profile_name") or request.reranker_profile_name
            ),
            reranker_model_id=str(payload.get("reranker_model_id") or request.reranker_model_id),
            provider_type=str(payload.get("provider_type") or REMOTE_RERANKER_PROVIDER_MODE),
            retrieval_strategy=str(payload.get("retrieval_strategy") or RERANK_RETRIEVAL_STRATEGY),
            candidate_count=int(payload.get("candidate_count") or len(request.candidates)),
            returned_count=int(payload.get("returned_count") or len(result_items)),
            top_k=int(payload.get("top_k") or request.top_k),
            results=result_items,
            runtime_metadata=dict(payload.get("runtime_metadata") or {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidRerankerError("Invalid remote reranker response") from exc

    if result.reranker_profile_name != request.reranker_profile_name:
        raise InvalidRerankerError("remote reranker_profile_name mismatch")
    if result.reranker_model_id != request.reranker_model_id:
        raise InvalidRerankerError("remote reranker_model_id mismatch")
    if result.candidate_count != len(request.candidates):
        raise InvalidRerankerError("remote candidate_count mismatch")
    if result.top_k != request.top_k:
        raise InvalidRerankerError("remote top_k mismatch")
    if result.retrieval_strategy != RERANK_RETRIEVAL_STRATEGY:
        raise InvalidRerankerError("remote retrieval_strategy mismatch")
    return result


def _remote_rerank_result_item_from_payload(
    payload: Mapping[str, object],
    candidate_by_key: Mapping[str, RerankCandidate],
) -> RerankResultItem:
    candidate_key = _validate_nonblank(str(payload["candidate_key"]), "candidate_key")
    try:
        candidate = candidate_by_key[candidate_key]
    except KeyError as exc:
        raise InvalidRerankerError(f"Unknown rerank candidate_key: {candidate_key}") from exc
    return RerankResultItem(
        candidate=candidate,
        rank=int(payload["rank"]),
        score=float(payload["score"]),
        score_components=dict(payload.get("score_components") or {}),
    )


def _create_httpx_client(*, timeout_seconds: float):
    try:
        import httpx
    except ImportError as exc:
        raise InvalidRerankerError("httpx is required for remote reranker providers.") from exc
    return httpx.Client(timeout=timeout_seconds)


def _normalize_remote_headers(headers: Mapping[str, str]) -> dict[str, str]:
    normalized_headers: dict[str, str] = {}
    for key, value in headers.items():
        header_name = _validate_nonblank(str(key), "header")
        if any(char in header_name for char in ("\r", "\n", ":")):
            raise InvalidRerankerError("header contains invalid characters")
        header_value = str(value).strip()
        if any(char in header_value for char in ("\r", "\n")):
            raise InvalidRerankerError("header value contains invalid characters")
        normalized_headers[header_name] = header_value
    return normalized_headers
