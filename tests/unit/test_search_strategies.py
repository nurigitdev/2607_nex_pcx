import math

import pytest

from app.core.search_experiments import (
    InvalidSearchExperimentError,
    SearchExperimentRunInput,
    validate_search_experiment_run_input,
)
from app.core.search_strategies import (
    InvalidSearchStrategyError,
    SearchStrategyDefinition,
    _validate_strategy_definition,
    get_search_strategy,
    list_search_strategies,
    validate_search_strategy_selection,
)


def test_search_strategy_registry_lists_active_and_planned_strategies() -> None:
    active_names = {strategy.strategy_name for strategy in list_search_strategies()}
    all_names = {strategy.strategy_name for strategy in list_search_strategies(active_only=False)}

    assert "vector_cosine" in active_names
    assert "vector_cosine_threshold" in active_names
    assert "bm25_keyword" not in active_names
    assert "bm25_keyword" in all_names
    assert "hybrid_keyword_vector" in active_names
    assert "hybrid_keyword_vector" in all_names
    assert "reranked_vector_cosine" in active_names
    assert "reranked_vector_cosine" in all_names
    assert get_search_strategy("bm25_keyword") is None
    assert get_search_strategy("bm25_keyword", active_only=False) is not None
    assert get_search_strategy("hybrid_keyword_vector") is not None
    assert get_search_strategy("reranked_vector_cosine") is not None


def test_hybrid_keyword_vector_strategy_contract_is_active_rrf_foundation() -> None:
    strategy = get_search_strategy("hybrid_keyword_vector")

    assert strategy is not None
    assert strategy.mode == "hybrid"
    assert strategy.stage == "active"
    assert strategy.similarity_metric == "cosine"
    assert strategy.runtime_parameters == {
        "keyword_strategy": "bm25_keyword",
        "vector_strategy": "vector_cosine",
        "fusion": "rrf",
        "rrf_k": 60,
    }


def test_bm25_keyword_strategy_contract_is_planned_baseline() -> None:
    strategy = get_search_strategy("bm25_keyword", active_only=False)

    assert strategy is not None
    assert strategy.mode == "keyword"
    assert strategy.stage == "planned"
    assert strategy.similarity_metric == "bm25"
    assert strategy.supports_multi_profile is False
    assert strategy.runtime_parameters == {
        "planned": True,
        "index_source": "chunks.chunk_text",
        "tokenizer": "unicode_word_v1",
        "scoring": "okapi_bm25",
        "k1": 1.2,
        "b": 0.75,
    }


def test_reranked_vector_cosine_strategy_contract_is_active_rerank_foundation() -> None:
    strategy = get_search_strategy("reranked_vector_cosine")

    assert strategy is not None
    assert strategy.mode == "rerank"
    assert strategy.stage == "active"
    assert strategy.similarity_metric == "cosine"
    assert strategy.runtime_parameters == {
        "source_strategy": "vector_cosine",
        "retrieval_strategy": "reranked",
        "reranker_profile_name": "qwen3_reranker_4b",
        "reranker_model_id": "Qwen/Qwen3-Reranker-4B",
        "candidate_multiplier": 4,
    }


def test_search_strategy_selection_merges_defaults_and_runtime_parameters() -> None:
    selection = validate_search_strategy_selection(
        "vector_cosine_threshold",
        top_k=12,
        score_threshold=0.82,
        runtime_parameters={"tie_break": "chunk_id"},
    )

    assert selection.strategy.strategy_name == "vector_cosine_threshold"
    assert selection.top_k == 12
    assert selection.score_threshold == 0.82
    assert selection.runtime_parameters == {
        "default_score_threshold": 0.7,
        "tie_break": "chunk_id",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy_name": "unknown"}, "Unsupported search strategy"),
        ({"strategy_name": "bm25_keyword"}, "Unsupported search strategy"),
        ({"strategy_name": "vector_cosine", "top_k": 0}, "top_k"),
        ({"strategy_name": "vector_cosine", "top_k": 101}, "top_k"),
        (
            {"strategy_name": "vector_cosine", "score_threshold": 0.5},
            "does not support score_threshold",
        ),
        (
            {"strategy_name": "vector_cosine_threshold", "score_threshold": math.inf},
            "score_threshold",
        ),
        (
            {"strategy_name": "vector_cosine", "runtime_parameters": []},
            "runtime_parameters",
        ),
    ],
)
def test_search_strategy_selection_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(InvalidSearchStrategyError, match=message):
        validate_search_strategy_selection(**kwargs)


def test_search_strategy_definition_validation_rejects_bad_definitions() -> None:
    with pytest.raises(InvalidSearchStrategyError, match="Unsupported strategy mode"):
        _validate_strategy_definition(
            SearchStrategyDefinition(
                strategy_name="bad",
                display_name="Bad",
                description="Bad strategy",
                mode="graph",
                stage="active",
                similarity_metric="cosine",
            )
        )
    with pytest.raises(InvalidSearchStrategyError, match="Unsupported similarity_metric"):
        _validate_strategy_definition(
            SearchStrategyDefinition(
                strategy_name="bad_metric",
                display_name="Bad Metric",
                description="Bad strategy",
                mode="keyword",
                stage="active",
                similarity_metric="tfidf",
            )
        )
    with pytest.raises(InvalidSearchStrategyError, match="default_top_k"):
        _validate_strategy_definition(
            SearchStrategyDefinition(
                strategy_name="bad_top_k",
                display_name="Bad Top K",
                description="Bad strategy",
                mode="vector",
                stage="active",
                similarity_metric="cosine",
                default_top_k=10,
                max_top_k=5,
            )
        )


def test_search_experiment_validation_rejects_unregistered_strategy() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="Unsupported search strategy"):
        validate_search_experiment_run_input(
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                strategy_name="ad_hoc_strategy",
            )
        )


def test_search_experiment_validation_requires_strategy_similarity_metric() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="requires cosine"):
        validate_search_experiment_run_input(
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                strategy_name="vector_cosine",
                similarity_metric="l2",
            )
        )
