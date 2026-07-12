import math

import pytest

from app.core.search_experiments import (
    InvalidSearchExperimentError,
    SearchExperimentProfileRunInput,
    SearchExperimentRunInput,
    _validate_limit,
    validate_search_experiment_profile_run_input,
    validate_search_experiment_run_input,
)


def test_search_experiment_run_validation_deduplicates_profiles() -> None:
    validated = validate_search_experiment_run_input(
        SearchExperimentRunInput(
            run_name="  Strategy Trial  ",
            query_text="  inverter manual  ",
            profile_names=("bge_m3_1024", "bge_m3_1024", "kure_v1_1024"),
            requested_search_scope="company",
            strategy_name="vector_cosine_threshold",
            score_threshold=0.25,
            runtime_metadata={"slice": 165},
        )
    )

    assert validated.run_name == "Strategy Trial"
    assert validated.query_text == "inverter manual"
    assert validated.profile_names == ("bge_m3_1024", "kure_v1_1024")
    assert validated.requested_search_scope == "company"
    assert validated.score_threshold == 0.25


@pytest.mark.parametrize(
    ("run_input", "message"),
    [
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text=" ",
                profile_names=("bge_m3_1024",),
            ),
            "query_text must not be blank",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=(),
            ),
            "profile_names must not be empty",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                requested_search_scope="everyone",
            ),
            "Unsupported requested_search_scope",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                similarity_metric="jaccard",
            ),
            "Unsupported similarity_metric",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                status="done",
            ),
            "Unsupported experiment status",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                top_k=0,
            ),
            "top_k must be greater than 0",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                actor_user_id=0,
            ),
            "actor_user_id must be greater than 0",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                runtime_metadata=[],
            ),
            "runtime_metadata must be a JSON object",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                strategy_name="vector_cosine_threshold",
                score_threshold=math.inf,
            ),
            "score_threshold must be finite",
        ),
    ],
)
def test_search_experiment_run_validation_rejects_invalid_inputs(
    run_input: SearchExperimentRunInput,
    message: str,
) -> None:
    with pytest.raises(InvalidSearchExperimentError, match=message):
        validate_search_experiment_run_input(run_input)


def test_search_experiment_profile_validation_rejects_invalid_status() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="Unsupported profile status"):
        validate_search_experiment_profile_run_input(
            SearchExperimentProfileRunInput(
                experiment_run_id=1,
                profile_name="bge_m3_1024",
                status="done",
            )
        )


def test_search_experiment_profile_validation_rejects_negative_metrics() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="result_count"):
        validate_search_experiment_profile_run_input(
            SearchExperimentProfileRunInput(
                experiment_run_id=1,
                profile_name="bge_m3_1024",
                result_count=-1,
            )
        )
    with pytest.raises(InvalidSearchExperimentError, match="elapsed_ms"):
        validate_search_experiment_profile_run_input(
            SearchExperimentProfileRunInput(
                experiment_run_id=1,
                profile_name="bge_m3_1024",
                elapsed_ms=-1,
            )
        )


def test_search_experiment_limit_validation_rejects_out_of_range_values() -> None:
    assert _validate_limit(10) == 10
    with pytest.raises(InvalidSearchExperimentError, match="greater than 0"):
        _validate_limit(0)
    with pytest.raises(InvalidSearchExperimentError, match="less than or equal to 5"):
        _validate_limit(6, max_limit=5)
