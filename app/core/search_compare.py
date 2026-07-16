"""Search compare service for profile-by-profile vector retrieval."""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from app.core.database import connect
from app.core.embedding_jobs import list_active_embedding_profiles
from app.core.embedding_providers import (
    EmbeddingProviderRuntimeConfig,
    build_embedding_provider_from_runtime_config,
)
from app.core.permissions import PermissionSearchFilter, resolve_permission_search_filter
from app.core.query_embeddings import (
    QueryEmbeddingProviderBuilder,
    QueryEmbeddingResult,
    embed_query_for_profile,
    query_embedding_runtime_metadata,
)
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)
from app.core.vector_search import VectorSearchInput, VectorSearchResult, search_similar_chunks

MAX_PERMISSION_MATRIX_ENTRIES = 12
ACCESS_SCOPE_ORDER = ("personal", "team", "org_tree", "company")


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


@dataclass(frozen=True)
class SearchPermissionMatrixEntryInput:
    actor_user_id: int
    requested_search_scope: str


@dataclass(frozen=True)
class SearchPermissionMatrixInput:
    query_text: str
    entries: tuple[SearchPermissionMatrixEntryInput, ...]
    top_k: int = 5
    profiles: tuple[str, ...] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


@dataclass(frozen=True)
class SearchPermissionMatrixEntryResult:
    search_log_id: int
    actor_user_id: int
    requested_search_scope: str
    effective_search_scope: str
    permission_filter: PermissionSearchFilter
    result_count: int
    unique_chunk_count: int
    top_result: VectorSearchResult | None
    profiles: tuple[SearchCompareProfileResult, ...]
    total_elapsed_ms: int


@dataclass(frozen=True)
class SearchPermissionMatrixResult:
    query_text: str
    top_k: int
    entries: tuple[SearchPermissionMatrixEntryResult, ...]
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


def _validate_permission_matrix_input(
    matrix_input: SearchPermissionMatrixInput,
) -> SearchPermissionMatrixInput:
    if not matrix_input.entries:
        raise InvalidSearchCompareError("entries must not be empty")
    if len(matrix_input.entries) > MAX_PERMISSION_MATRIX_ENTRIES:
        raise InvalidSearchCompareError(f"entries must be {MAX_PERMISSION_MATRIX_ENTRIES} or fewer")

    normalized_entries: list[SearchPermissionMatrixEntryInput] = []
    seen_entries: set[tuple[int, str]] = set()
    for entry in matrix_input.entries:
        if entry.actor_user_id <= 0:
            raise InvalidSearchCompareError("actor_user_id must be greater than 0")
        requested_scope = (
            _validate_nonblank(entry.requested_search_scope, "requested_search_scope")
            or entry.requested_search_scope
        )
        normalized_key = (entry.actor_user_id, requested_scope)
        if normalized_key in seen_entries:
            raise InvalidSearchCompareError("entries must be unique by actor and scope")
        seen_entries.add(normalized_key)
        normalized_entries.append(
            SearchPermissionMatrixEntryInput(
                actor_user_id=entry.actor_user_id,
                requested_search_scope=requested_scope,
            )
        )

    first_entry = normalized_entries[0]
    common = _validate_search_compare_input(
        SearchCompareInput(
            query_text=matrix_input.query_text,
            actor_user_id=first_entry.actor_user_id,
            requested_search_scope=first_entry.requested_search_scope,
            top_k=matrix_input.top_k,
            profiles=matrix_input.profiles,
            chunk_policy_name=matrix_input.chunk_policy_name,
            document_group=matrix_input.document_group,
            file_type=matrix_input.file_type,
        )
    )
    return SearchPermissionMatrixInput(
        query_text=common.query_text,
        entries=tuple(normalized_entries),
        top_k=common.top_k,
        profiles=common.profiles,
        chunk_policy_name=common.chunk_policy_name,
        document_group=common.document_group,
        file_type=common.file_type,
    )


def _default_profiles(database_url: str) -> tuple[str, ...]:
    profiles = tuple(
        profile.profile_name for profile in list_active_embedding_profiles(database_url)
    )
    if not profiles:
        raise InvalidSearchCompareError("No active embedding profiles are configured")
    return profiles


def _document_visibility_filters(
    search_input: SearchCompareInput,
) -> tuple[str, list[object]]:
    where_clauses = ["d.document_status = 'active'"]
    params: list[object] = []

    if search_input.document_group is not None:
        where_clauses.append("d.document_group = %s")
        params.append(search_input.document_group)
    if search_input.file_type is not None:
        where_clauses.append("f.file_ext = %s")
        params.append(search_input.file_type)
    if search_input.chunk_policy_name is not None:
        where_clauses.append("""
            EXISTS (
                SELECT 1
                FROM chunks visibility_chunk
                WHERE visibility_chunk.document_id = d.document_id
                  AND visibility_chunk.chunk_policy_name = %s
            )
            """)
        params.append(search_input.chunk_policy_name)

    return " AND ".join(where_clauses), params


def _permission_explainability_summary(
    database_url: str,
    search_input: SearchCompareInput,
    permission_filter: PermissionSearchFilter,
) -> dict[str, Any]:
    base_where_sql, base_params = _document_visibility_filters(search_input)
    visibility_params = [*base_params, *permission_filter.params]
    access_scope_counts = dict.fromkeys(ACCESS_SCOPE_ORDER, 0)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT count(*) AS count
                FROM documents d
                JOIN files f ON f.file_id = d.file_id
                WHERE {base_where_sql}
                """,
                tuple(base_params),
            )
            candidate_document_count = int(cursor.fetchone()["count"])

            cursor.execute(
                f"""
                SELECT d.access_scope, count(*) AS count
                FROM documents d
                JOIN files f ON f.file_id = d.file_id
                WHERE {base_where_sql}
                  AND {permission_filter.where_sql}
                GROUP BY d.access_scope
                """,
                tuple(visibility_params),
            )
            for row in cursor.fetchall():
                access_scope = str(row["access_scope"])
                if access_scope in access_scope_counts:
                    access_scope_counts[access_scope] = int(row["count"])

    visible_document_count = sum(access_scope_counts.values())
    return {
        "actor_user_id": permission_filter.metadata["actor_user_id"],
        "actor_login_id": permission_filter.metadata["login_id"],
        "actor_display_name": permission_filter.metadata.get("display_name"),
        "role_name": permission_filter.metadata["role_name"],
        "primary_org_unit_id": permission_filter.metadata["primary_org_unit_id"],
        "primary_org_unit_name": permission_filter.metadata["primary_org_unit_name"],
        "requested_search_scope": permission_filter.requested_search_scope,
        "effective_search_scope": permission_filter.effective_search_scope,
        "scope_was_downgraded": (
            permission_filter.requested_search_scope != permission_filter.effective_search_scope
        ),
        "ancestor_org_unit_count": len(permission_filter.metadata.get("ancestor_org_unit_ids", ())),
        "managed_org_unit_count": len(permission_filter.metadata.get("managed_org_unit_ids", ())),
        "includes_company_documents": permission_filter.metadata["includes_company_documents"],
        "filter_clause_count": permission_filter.metadata["filter_clause_count"],
        "candidate_document_count": candidate_document_count,
        "visible_document_count": visible_document_count,
        "excluded_document_count": max(
            0,
            candidate_document_count - visible_document_count,
        ),
        "visible_access_scope_counts": access_scope_counts,
        "included_access_scopes": [
            scope for scope, count in access_scope_counts.items() if count > 0
        ],
    }


def _with_permission_explainability(
    database_url: str,
    search_input: SearchCompareInput,
    permission_filter: PermissionSearchFilter,
) -> PermissionSearchFilter:
    summary = _permission_explainability_summary(
        database_url,
        search_input,
        permission_filter,
    )
    metadata = {
        **permission_filter.metadata,
        "included_access_scopes": summary["included_access_scopes"],
        "permission_explainability": summary,
    }
    return replace(permission_filter, metadata=metadata)


def _search_compare_query_runtime_metadata(
    query_embedding_results: dict[str, QueryEmbeddingResult],
) -> dict[str, object]:
    profiles = {
        profile_name: query_embedding_runtime_metadata(result)
        for profile_name, result in query_embedding_results.items()
    }
    provider_types = sorted(
        {
            result.provider_type
            for result in query_embedding_results.values()
        }
    )
    runtime_sources = sorted(
        {
            result.runtime_source
            for result in query_embedding_results.values()
        }
    )
    return {
        "adapter": "query_embedding_bridge",
        "search_mode": "compare_mvp",
        "query_embedding_bridge": True,
        "query_embedding_profile_count": len(query_embedding_results),
        "query_embedding_provider_types": provider_types,
        "query_embedding_runtime_sources": runtime_sources,
        "profile_query_embeddings": profiles,
    }


def run_search_compare(
    database_url: str,
    search_input: SearchCompareInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    query_embedding_provider_builder: QueryEmbeddingProviderBuilder = (
        build_embedding_provider_from_runtime_config
    ),
) -> SearchCompareResult:
    validated = _validate_search_compare_input(search_input)
    profiles = validated.profiles or _default_profiles(database_url)
    fallback_config = fallback_runtime_config or EmbeddingProviderRuntimeConfig(mode="mock")
    permission_filter = resolve_permission_search_filter(
        database_url,
        actor_user_id=validated.actor_user_id,
        requested_search_scope=validated.requested_search_scope,
    )
    permission_filter = _with_permission_explainability(
        database_url,
        validated,
        permission_filter,
    )

    started_at = perf_counter()
    raw_profile_results: list[tuple[str, int, tuple[VectorSearchResult, ...]]] = []
    query_embedding_results: dict[str, QueryEmbeddingResult] = {}
    for profile_name in profiles:
        profile_started_at = perf_counter()
        query_embedding = embed_query_for_profile(
            database_url,
            query_text=validated.query_text,
            profile_name=profile_name,
            fallback_runtime_config=fallback_config,
            provider_builder=query_embedding_provider_builder,
            trace_id=f"search-compare:{validated.actor_user_id}:{profile_name}",
        )
        query_embedding_results[profile_name] = query_embedding
        results = search_similar_chunks(
            database_url,
            VectorSearchInput(
                query_text=validated.query_text,
                profile_name=profile_name,
                top_k=validated.top_k,
                query_embedding=query_embedding.embedding,
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
            query_runtime_metadata=_search_compare_query_runtime_metadata(
                query_embedding_results,
            ),
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


def _first_profile_top_result(
    profiles: tuple[SearchCompareProfileResult, ...],
) -> VectorSearchResult | None:
    for profile in profiles:
        if profile.results:
            return profile.results[0].vector_result
    return None


def run_permission_search_matrix(
    database_url: str,
    matrix_input: SearchPermissionMatrixInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    query_embedding_provider_builder: QueryEmbeddingProviderBuilder = (
        build_embedding_provider_from_runtime_config
    ),
) -> SearchPermissionMatrixResult:
    validated = _validate_permission_matrix_input(matrix_input)
    profiles = validated.profiles or _default_profiles(database_url)

    started_at = perf_counter()
    entry_results: list[SearchPermissionMatrixEntryResult] = []
    for entry in validated.entries:
        compare_result = run_search_compare(
            database_url,
            SearchCompareInput(
                query_text=validated.query_text,
                actor_user_id=entry.actor_user_id,
                requested_search_scope=entry.requested_search_scope,
                top_k=validated.top_k,
                profiles=profiles,
                chunk_policy_name=validated.chunk_policy_name,
                document_group=validated.document_group,
                file_type=validated.file_type,
            ),
            fallback_runtime_config=fallback_runtime_config,
            query_embedding_provider_builder=query_embedding_provider_builder,
        )
        chunk_ids = {
            result.vector_result.chunk_id
            for profile in compare_result.profiles
            for result in profile.results
        }
        result_count = sum(len(profile.results) for profile in compare_result.profiles)
        entry_results.append(
            SearchPermissionMatrixEntryResult(
                search_log_id=compare_result.search_log_id,
                actor_user_id=compare_result.actor_user_id,
                requested_search_scope=compare_result.requested_search_scope,
                effective_search_scope=compare_result.effective_search_scope,
                permission_filter=compare_result.permission_filter,
                result_count=result_count,
                unique_chunk_count=len(chunk_ids),
                top_result=_first_profile_top_result(compare_result.profiles),
                profiles=compare_result.profiles,
                total_elapsed_ms=compare_result.total_elapsed_ms,
            )
        )

    return SearchPermissionMatrixResult(
        query_text=validated.query_text,
        top_k=validated.top_k,
        entries=tuple(entry_results),
        total_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
    )
