"""Search strategy registry for experiment runs."""

import math
from dataclasses import dataclass, field
from typing import Any

from app.core.vector_search import MAX_TOP_K

SEARCH_STRATEGY_MODES = {"vector", "keyword", "hybrid", "rerank"}
SEARCH_STRATEGY_STAGES = {"active", "planned"}
SEARCH_STRATEGY_SIMILARITY_METRICS = {"cosine", "l2", "inner_product", "bm25"}


@dataclass(frozen=True)
class SearchStrategyDefinition:
    strategy_name: str
    display_name: str
    description: str
    mode: str
    stage: str
    similarity_metric: str
    default_top_k: int = 5
    max_top_k: int = MAX_TOP_K
    supports_score_threshold: bool = False
    supports_permission_filter: bool = True
    supports_metadata_filters: bool = True
    supports_multi_profile: bool = True
    runtime_parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.stage == "active"


@dataclass(frozen=True)
class SearchStrategySelection:
    strategy: SearchStrategyDefinition
    top_k: int
    score_threshold: float | None
    runtime_parameters: dict[str, Any]


class InvalidSearchStrategyError(ValueError):
    """Raised when a search strategy selection is invalid."""


SEARCH_STRATEGY_REGISTRY: dict[str, SearchStrategyDefinition] = {
    "vector_cosine": SearchStrategyDefinition(
        strategy_name="vector_cosine",
        display_name="Vector Cosine",
        description="Baseline pgvector cosine similarity search.",
        mode="vector",
        stage="active",
        similarity_metric="cosine",
    ),
    "vector_cosine_threshold": SearchStrategyDefinition(
        strategy_name="vector_cosine_threshold",
        display_name="Vector Cosine + Score Threshold",
        description="Cosine vector search with a minimum score threshold.",
        mode="vector",
        stage="active",
        similarity_metric="cosine",
        supports_score_threshold=True,
        runtime_parameters={"default_score_threshold": 0.7},
    ),
    "bm25_keyword": SearchStrategyDefinition(
        strategy_name="bm25_keyword",
        display_name="BM25 Keyword",
        description="Planned BM25 keyword retrieval baseline over chunk text.",
        mode="keyword",
        stage="planned",
        similarity_metric="bm25",
        supports_score_threshold=False,
        supports_multi_profile=False,
        runtime_parameters={
            "planned": True,
            "index_source": "chunks.chunk_text",
            "tokenizer": "unicode_word_v1",
            "scoring": "okapi_bm25",
            "k1": 1.2,
            "b": 0.75,
        },
    ),
    "hybrid_keyword_vector": SearchStrategyDefinition(
        strategy_name="hybrid_keyword_vector",
        display_name="Hybrid Keyword + Vector",
        description="Planned BM25 keyword and vector blended retrieval strategy.",
        mode="hybrid",
        stage="planned",
        similarity_metric="cosine",
        supports_score_threshold=True,
        runtime_parameters={
            "planned": True,
            "keyword_strategy": "bm25_keyword",
            "vector_strategy": "vector_cosine",
            "fusion": "rrf",
            "rrf_k": 60,
        },
    ),
    "reranked_vector_cosine": SearchStrategyDefinition(
        strategy_name="reranked_vector_cosine",
        display_name="Reranked Vector Cosine",
        description="Planned vector retrieval followed by reranking.",
        mode="rerank",
        stage="planned",
        similarity_metric="cosine",
        supports_score_threshold=True,
        runtime_parameters={"planned": True},
    ),
}


def _validate_nonblank(value: str | None, field_name: str) -> str:
    if value is None:
        raise InvalidSearchStrategyError(f"{field_name} is required")
    normalized = value.strip()
    if not normalized:
        raise InvalidSearchStrategyError(f"{field_name} must not be blank")
    return normalized


def _validate_strategy_definition(definition: SearchStrategyDefinition) -> None:
    _validate_nonblank(definition.strategy_name, "strategy_name")
    _validate_nonblank(definition.display_name, "display_name")
    _validate_nonblank(definition.description, "description")
    if definition.mode not in SEARCH_STRATEGY_MODES:
        raise InvalidSearchStrategyError(f"Unsupported strategy mode: {definition.mode}")
    if definition.stage not in SEARCH_STRATEGY_STAGES:
        raise InvalidSearchStrategyError(f"Unsupported strategy stage: {definition.stage}")
    if definition.similarity_metric not in SEARCH_STRATEGY_SIMILARITY_METRICS:
        raise InvalidSearchStrategyError(
            f"Unsupported similarity_metric: {definition.similarity_metric}"
        )
    if definition.default_top_k <= 0:
        raise InvalidSearchStrategyError("default_top_k must be greater than 0")
    if definition.max_top_k <= 0:
        raise InvalidSearchStrategyError("max_top_k must be greater than 0")
    if definition.default_top_k > definition.max_top_k:
        raise InvalidSearchStrategyError("default_top_k must be less than or equal to max_top_k")
    if not isinstance(definition.runtime_parameters, dict):
        raise InvalidSearchStrategyError("runtime_parameters must be a JSON object")


def list_search_strategies(
    *,
    active_only: bool = True,
) -> tuple[SearchStrategyDefinition, ...]:
    definitions = tuple(
        sorted(
            SEARCH_STRATEGY_REGISTRY.values(),
            key=lambda strategy: strategy.strategy_name,
        )
    )
    if active_only:
        return tuple(strategy for strategy in definitions if strategy.is_active)
    return definitions


def get_search_strategy(
    strategy_name: str,
    *,
    active_only: bool = True,
) -> SearchStrategyDefinition | None:
    normalized = _validate_nonblank(strategy_name, "strategy_name")
    strategy = SEARCH_STRATEGY_REGISTRY.get(normalized)
    if strategy is None:
        return None
    if active_only and not strategy.is_active:
        return None
    return strategy


def _validate_top_k(strategy: SearchStrategyDefinition, top_k: int | None) -> int:
    selected_top_k = strategy.default_top_k if top_k is None else top_k
    if selected_top_k <= 0:
        raise InvalidSearchStrategyError("top_k must be greater than 0")
    if selected_top_k > strategy.max_top_k:
        raise InvalidSearchStrategyError(
            f"top_k must be less than or equal to {strategy.max_top_k}"
        )
    return selected_top_k


def _validate_score_threshold(
    strategy: SearchStrategyDefinition,
    score_threshold: float | None,
) -> float | None:
    if score_threshold is None:
        return None
    if not strategy.supports_score_threshold:
        raise InvalidSearchStrategyError(
            f"{strategy.strategy_name} does not support score_threshold"
        )
    parsed = float(score_threshold)
    if not math.isfinite(parsed):
        raise InvalidSearchStrategyError("score_threshold must be finite")
    return parsed


def validate_search_strategy_selection(
    strategy_name: str,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    runtime_parameters: dict[str, Any] | None = None,
    active_only: bool = True,
) -> SearchStrategySelection:
    strategy = get_search_strategy(strategy_name, active_only=active_only)
    if strategy is None:
        raise InvalidSearchStrategyError(f"Unsupported search strategy: {strategy_name}")
    _validate_strategy_definition(strategy)
    if runtime_parameters is not None and not isinstance(runtime_parameters, dict):
        raise InvalidSearchStrategyError("runtime_parameters must be a JSON object")
    return SearchStrategySelection(
        strategy=strategy,
        top_k=_validate_top_k(strategy, top_k),
        score_threshold=_validate_score_threshold(strategy, score_threshold),
        runtime_parameters={
            **strategy.runtime_parameters,
            **(runtime_parameters or {}),
        },
    )
