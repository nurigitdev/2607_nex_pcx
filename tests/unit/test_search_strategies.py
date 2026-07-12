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
    assert "hybrid_keyword_vector" not in active_names
    assert "hybrid_keyword_vector" in all_names
    assert get_search_strategy("hybrid_keyword_vector") is None
    assert get_search_strategy("hybrid_keyword_vector", active_only=False) is not None


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
        ({"strategy_name": "hybrid_keyword_vector"}, "Unsupported search strategy"),
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
