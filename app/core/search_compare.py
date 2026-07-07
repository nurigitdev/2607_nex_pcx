"""Search compare service for profile-by-profile vector retrieval."""

from dataclasses import dataclass
from time import perf_counter

from app.core.embedding_jobs import list_active_embedding_profiles
from app.core.permissions import PermissionSearchFilter, resolve_permission_search_filter
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)
from app.core.vector_search import VectorSearchInput, VectorSearchResult, search_similar_chunks


@dataclass(frozen=True)
class SearchCompareInput:
    query_text: str
    actor_user_id: int
    requested_search_scope: str
    top_k: int = 5
    profiles: tuple[str, ...] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


@dataclass(frozen=True)
class SearchCompareResultItem:
    search_log_result_id: int
    vector_result: VectorSearchResult


@dataclass(frozen=True)
class SearchCompareProfileResult:
    profile_name: str
    elapsed_ms: int
    results: tuple[SearchCompareResultItem, ...]


@dataclass(frozen=True)
class SearchCompareResult:
    search_log_id: int
    query_text: str
    actor_user_id: int
    requested_search_scope: str
    effective_search_scope: str
    permission_filter: PermissionSearchFilter
    top_k: int
    profiles: tuple[SearchCompareProfileResult, ...]
    total_elapsed_ms: int


class InvalidSearchCompareError(ValueError):
    """Raised when search compare input is invalid before reaching repositories."""


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidSearchCompareError(f"{field_name} must not be blank")
    return normalized


def _validate_search_compare_input(search_input: SearchCompareInput) -> SearchCompareInput:
    query_text = _validate_nonblank(search_input.query_text, "query_text")
    if search_input.actor_user_id <= 0:
        raise InvalidSearchCompareError("actor_user_id must be greater than 0")
    if search_input.top_k <= 0:
        raise InvalidSearchCompareError("top_k must be greater than 0")
    profiles = None
    if search_input.profiles is not None:
        profiles = tuple(
            _validate_nonblank(profile, "profile_name") for profile in search_input.profiles
        )
        if not profiles:
            raise InvalidSearchCompareError("profiles must not be empty")
        if len(set(profiles)) != len(profiles):
            raise InvalidSearchCompareError("profiles must be unique")
    return SearchCompareInput(
        query_text=query_text or search_input.query_text,
        actor_user_id=search_input.actor_user_id,
        requested_search_scope=_validate_nonblank(
            search_input.requested_search_scope,
            "requested_search_scope",
        )
        or search_input.requested_search_scope,
        top_k=search_input.top_k,
        profiles=profiles,
        chunk_policy_name=_validate_nonblank(search_input.chunk_policy_name, "chunk_policy_name"),
        document_group=_validate_nonblank(search_input.document_group, "document_group"),
        file_type=_validate_nonblank(search_input.file_type, "file_type"),
    )


def _default_profiles(database_url: str) -> tuple[str, ...]:
    profiles = tuple(
        profile.profile_name for profile in list_active_embedding_profiles(database_url)
    )
    if not profiles:
        raise InvalidSearchCompareError("No active embedding profiles are configured")
    return profiles


def run_search_compare(
    database_url: str,
    search_input: SearchCompareInput,
) -> SearchCompareResult:
    validated = _validate_search_compare_input(search_input)
    profiles = validated.profiles or _default_profiles(database_url)
    permission_filter = resolve_permission_search_filter(
        database_url,
        actor_user_id=validated.actor_user_id,
        requested_search_scope=validated.requested_search_scope,
    )

    started_at = perf_counter()
    raw_profile_results: list[tuple[str, int, tuple[VectorSearchResult, ...]]] = []
    for profile_name in profiles:
        profile_started_at = perf_counter()
        results = search_similar_chunks(
            database_url,
            VectorSearchInput(
                query_text=validated.query_text,
                profile_name=profile_name,
                top_k=validated.top_k,
                chunk_policy_name=validated.chunk_policy_name,
                document_group=validated.document_group,
                file_type=validated.file_type,
                permission_filter=permission_filter,
            ),
        )
        raw_profile_results.append(
            (
                profile_name,
                max(0, int((perf_counter() - profile_started_at) * 1000)),
                tuple(results),
            )
        )

    total_elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text=validated.query_text,
            normalized_query_text=validated.query_text.strip().lower(),
            actor_user_id=validated.actor_user_id,
            requested_search_scope=permission_filter.requested_search_scope,
            effective_search_scope=permission_filter.effective_search_scope,
            permission_filter_metadata=permission_filter.metadata,
            document_group=validated.document_group,
            file_type=validated.file_type,
            chunk_policy_name=validated.chunk_policy_name,
            top_k=validated.top_k,
            similarity_metric="cosine",
            profiles=profiles,
            query_runtime_metadata={
                "adapter": "mock",
                "search_mode": "compare_mvp",
            },
            total_elapsed_ms=total_elapsed_ms,
            created_by_user_id=validated.actor_user_id,
        ),
    )
    result_inputs = [
        SearchLogResultInput(
            search_log_id=search_log.search_log_id,
            profile_name=profile_name,
            rank=result.rank,
            chunk_id=result.chunk_id,
            distance=result.distance,
            score=result.score,
            profile_elapsed_ms=elapsed_ms,
        )
        for profile_name, elapsed_ms, results in raw_profile_results
        for result in results
    ]
    stored_results = []
    if result_inputs:
        stored_results = create_search_log_results(database_url, result_inputs)

    result_id_iter = iter(stored_results)
    profile_results = [
        SearchCompareProfileResult(
            profile_name=profile_name,
            elapsed_ms=elapsed_ms,
            results=tuple(
                SearchCompareResultItem(
                    search_log_result_id=next(result_id_iter).search_log_result_id,
                    vector_result=result,
                )
                for result in results
            ),
        )
        for profile_name, elapsed_ms, results in raw_profile_results
    ]

    return SearchCompareResult(
        search_log_id=search_log.search_log_id,
        query_text=validated.query_text,
        actor_user_id=validated.actor_user_id,
        requested_search_scope=permission_filter.requested_search_scope,
        effective_search_scope=permission_filter.effective_search_scope,
        permission_filter=permission_filter,
        top_k=validated.top_k,
        profiles=tuple(profile_results),
        total_elapsed_ms=total_elapsed_ms,
    )
