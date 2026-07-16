"""Search experiment execution orchestration."""

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from app.core.embedding_jobs import list_active_embedding_profiles
from app.core.embedding_providers import (
    EmbeddingProviderRuntimeConfig,
    build_embedding_provider_from_runtime_config,
)
from app.core.query_embeddings import QueryEmbeddingProviderBuilder
from app.core.search_compare import (
    SEARCH_COMPARE_PROFILE_STATUS_FAILED,
    SearchCompareInput,
    SearchCompareResult,
    run_search_compare,
)
from app.core.search_experiments import (
    SearchExperimentProfileRunInput,
    SearchExperimentRunInput,
    SearchExperimentRunRecord,
    create_search_experiment_run,
    update_search_experiment_run_status,
    upsert_search_experiment_profile_run,
)
from app.core.search_strategies import (
    InvalidSearchStrategyError,
    SearchStrategySelection,
    validate_search_strategy_selection,
)
from app.core.vector_search import VectorSearchResult


@dataclass(frozen=True)
class SearchExperimentExecutionInput:
    run_name: str
    query_text: str
    actor_user_id: int
    requested_search_scope: str = "company"
    profiles: tuple[str, ...] | None = None
    strategy_name: str = "vector_cosine"
    top_k: int = 5
    score_threshold: float | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    created_by: str | None = None
    created_by_user_id: int | None = None
    allow_mock_fallback: bool = True


@dataclass(frozen=True)
class SearchExperimentProfileExecutionSummary:
    profile_name: str
    raw_result_count: int
    retained_result_count: int
    excluded_by_threshold_count: int
    top_score: float | None
    average_score: float | None
    elapsed_ms: int


@dataclass(frozen=True)
class SearchExperimentExecutionReport:
    run: SearchExperimentRunRecord
    search_result: SearchCompareResult
    strategy_selection: SearchStrategySelection
    profile_summaries: tuple[SearchExperimentProfileExecutionSummary, ...]


class InvalidSearchExperimentExecutionError(ValueError):
    """Raised when a search experiment cannot be executed."""


def _default_profiles(database_url: str) -> tuple[str, ...]:
    profiles = tuple(
        profile.profile_name for profile in list_active_embedding_profiles(database_url)
    )
    if not profiles:
        raise InvalidSearchExperimentExecutionError("No active embedding profiles are configured")
    return profiles


def _normalize_profiles(profiles: tuple[str, ...] | None, database_url: str) -> tuple[str, ...]:
    selected_profiles = profiles or _default_profiles(database_url)
    normalized: list[str] = []
    seen: set[str] = set()
    for profile_name in selected_profiles:
        profile = profile_name.strip()
        if not profile:
            raise InvalidSearchExperimentExecutionError("profile_name must not be blank")
        if profile not in seen:
            normalized.append(profile)
            seen.add(profile)
    return tuple(normalized)


def _filter_results_by_threshold(
    results: tuple[VectorSearchResult, ...],
    score_threshold: float | None,
) -> tuple[VectorSearchResult, ...]:
    if score_threshold is None:
        return results
    return tuple(result for result in results if result.score >= score_threshold)


def _profile_summary(
    *,
    profile_name: str,
    raw_results: tuple[VectorSearchResult, ...],
    retained_results: tuple[VectorSearchResult, ...],
    elapsed_ms: int,
) -> SearchExperimentProfileExecutionSummary:
    scores = [result.score for result in retained_results]
    return SearchExperimentProfileExecutionSummary(
        profile_name=profile_name,
        raw_result_count=len(raw_results),
        retained_result_count=len(retained_results),
        excluded_by_threshold_count=len(raw_results) - len(retained_results),
        top_score=max(scores) if scores else None,
        average_score=fmean(scores) if scores else None,
        elapsed_ms=elapsed_ms,
    )


def execute_search_experiment(
    database_url: str,
    execution_input: SearchExperimentExecutionInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    query_embedding_provider_builder: QueryEmbeddingProviderBuilder = (
        build_embedding_provider_from_runtime_config
    ),
) -> SearchExperimentExecutionReport:
    try:
        strategy_selection = validate_search_strategy_selection(
            execution_input.strategy_name,
            top_k=execution_input.top_k,
            score_threshold=execution_input.score_threshold,
        )
    except InvalidSearchStrategyError as exc:
        raise InvalidSearchExperimentExecutionError(str(exc)) from exc

    if strategy_selection.strategy.mode != "vector":
        raise InvalidSearchExperimentExecutionError(
            f"{strategy_selection.strategy.strategy_name} is not executable yet"
        )

    profiles = _normalize_profiles(execution_input.profiles, database_url)
    run = create_search_experiment_run(
        database_url,
        SearchExperimentRunInput(
            run_name=execution_input.run_name,
            query_text=execution_input.query_text,
            profile_names=profiles,
            actor_user_id=execution_input.actor_user_id,
            requested_search_scope=execution_input.requested_search_scope,
            strategy_name=strategy_selection.strategy.strategy_name,
            similarity_metric=strategy_selection.strategy.similarity_metric,
            top_k=strategy_selection.top_k,
            score_threshold=strategy_selection.score_threshold,
            chunk_policy_name=execution_input.chunk_policy_name,
            document_group=execution_input.document_group,
            file_type=execution_input.file_type,
            status="running",
            runtime_metadata={
                **execution_input.runtime_metadata,
                "strategy_runtime_parameters": strategy_selection.runtime_parameters,
                "allow_mock_fallback": execution_input.allow_mock_fallback,
                "real_provider_required": not execution_input.allow_mock_fallback,
            },
            created_by=execution_input.created_by,
            created_by_user_id=execution_input.created_by_user_id,
        ),
    )

    try:
        search_result = run_search_compare(
            database_url,
            SearchCompareInput(
                query_text=execution_input.query_text,
                actor_user_id=execution_input.actor_user_id,
                requested_search_scope=execution_input.requested_search_scope,
                top_k=strategy_selection.top_k,
                profiles=profiles,
                chunk_policy_name=execution_input.chunk_policy_name,
                document_group=execution_input.document_group,
                file_type=execution_input.file_type,
                allow_mock_fallback=execution_input.allow_mock_fallback,
            ),
            fallback_runtime_config=fallback_runtime_config,
            query_embedding_provider_builder=query_embedding_provider_builder,
        )
    except Exception as exc:
        update_search_experiment_run_status(
            database_url,
            run.experiment_run_id,
            status="failed",
            error_message=str(exc),
        )
        raise

    summaries: list[SearchExperimentProfileExecutionSummary] = []
    for profile_result in search_result.profiles:
        raw_results = tuple(item.vector_result for item in profile_result.results)
        retained_results = _filter_results_by_threshold(
            raw_results,
            strategy_selection.score_threshold,
        )
        summary = _profile_summary(
            profile_name=profile_result.profile_name,
            raw_results=raw_results,
            retained_results=retained_results,
            elapsed_ms=profile_result.elapsed_ms,
        )
        upsert_search_experiment_profile_run(
            database_url,
            SearchExperimentProfileRunInput(
                experiment_run_id=run.experiment_run_id,
                profile_name=profile_result.profile_name,
                search_log_id=search_result.search_log_id,
                status=(
                    "failed"
                    if profile_result.status == SEARCH_COMPARE_PROFILE_STATUS_FAILED
                    else "succeeded"
                ),
                result_count=summary.retained_result_count,
                top_score=summary.top_score,
                average_score=summary.average_score,
                elapsed_ms=summary.elapsed_ms,
                runtime_metadata={
                    "profile_status": profile_result.status,
                    "profile_error_code": profile_result.error_code,
                    "query_runtime_metadata": profile_result.query_runtime_metadata,
                    "raw_result_count": summary.raw_result_count,
                    "score_threshold": strategy_selection.score_threshold,
                    "excluded_by_threshold_count": summary.excluded_by_threshold_count,
                },
                error_message=profile_result.error_message,
            ),
        )
        summaries.append(summary)

    completed_run = update_search_experiment_run_status(
        database_url,
        run.experiment_run_id,
        status="succeeded",
        total_elapsed_ms=search_result.total_elapsed_ms,
        runtime_metadata={"search_log_id": search_result.search_log_id},
    )
    if completed_run is None:
        raise RuntimeError("Search experiment run disappeared during execution")

    return SearchExperimentExecutionReport(
        run=completed_run,
        search_result=search_result,
        strategy_selection=strategy_selection,
        profile_summaries=tuple(summaries),
    )
