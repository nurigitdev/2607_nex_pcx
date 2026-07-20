"""Search compare service for profile-by-profile vector retrieval."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from time import perf_counter
from typing import Any

from app.core.bm25_search import (
    BM25_RETRIEVAL_STRATEGY,
    BM25_SEARCH_PROFILE_NAME,
    BM25SearchInput,
    BM25SearchResult,
    InvalidBM25SearchError,
    search_bm25_chunks,
)
from app.core.chunks import DEFAULT_CHUNK_POLICY_NAME
from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    EmbeddingJobRecord,
    create_embedding_job_in_connection,
    list_active_embedding_profiles,
    retry_embedding_job_in_connection,
)
from app.core.embedding_providers import (
    EmbeddingProviderRuntimeConfig,
    build_embedding_provider_from_runtime_config,
)
from app.core.embedding_vectors import EMBEDDING_VECTOR_TABLES
from app.core.permissions import PermissionSearchFilter, resolve_permission_search_filter
from app.core.query_embeddings import (
    InvalidQueryEmbeddingError,
    QueryEmbeddingProviderBuilder,
    embed_query_for_profile,
    query_embedding_runtime_metadata,
)
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)
from app.core.vector_search import (
    InvalidVectorSearchError,
    VectorSearchInput,
    VectorSearchResult,
    search_similar_chunks,
)

MAX_PERMISSION_MATRIX_ENTRIES = 12
ACCESS_SCOPE_ORDER = ("personal", "team", "org_tree", "company")
SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED = "succeeded"
SEARCH_COMPARE_PROFILE_STATUS_FAILED = "failed"
SEARCH_COMPARE_PROFILE_ERROR_QUERY_EMBEDDING_FAILED = "query_embedding_failed"
SEARCH_COMPARE_PROFILE_ERROR_VECTOR_SEARCH_FAILED = "vector_search_failed"
SEARCH_COMPARE_PROFILE_ERROR_KEYWORD_SEARCH_FAILED = "keyword_search_failed"

SearchResultLike = VectorSearchResult | BM25SearchResult


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
    allow_mock_fallback: bool = True


@dataclass(frozen=True)
class SearchCompareReadinessInput:
    actor_user_id: int
    requested_search_scope: str
    profiles: tuple[str, ...] | None = None
    chunk_policy_names: tuple[str, ...] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


@dataclass(frozen=True)
class SearchCompareCoverageReconcileInput:
    actor_user_id: int
    requested_search_scope: str
    profile_name: str
    chunk_policy_name: str
    document_group: str | None = None
    file_type: str | None = None
    max_jobs: int = 500


@dataclass(frozen=True)
class SearchCompareCoverageReconcileResult:
    actor_user_id: int
    requested_search_scope: str
    effective_search_scope: str
    profile_name: str
    chunk_policy_name: str
    document_group: str | None
    file_type: str | None
    chunk_count: int
    existing_job_count: int
    missing_job_count: int
    created_job_count: int
    failed_job_count: int
    retryable_failed_job_count: int
    retried_job_count: int
    created_jobs: tuple[EmbeddingJobRecord, ...]
    retried_jobs: tuple[EmbeddingJobRecord, ...]


@dataclass(frozen=True)
class SearchCompareResultItem:
    search_log_result_id: int
    vector_result: SearchResultLike


@dataclass(frozen=True)
class SearchCompareProfileResult:
    profile_name: str
    elapsed_ms: int
    results: tuple[SearchCompareResultItem, ...]
    status: str = SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED
    error_code: str | None = None
    error_message: str | None = None
    query_runtime_metadata: dict[str, object] = field(default_factory=dict)


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
class SearchCompareReadinessProfile:
    profile_name: str
    chunk_policy_name: str | None
    chunk_count: int
    job_count: int
    pending_count: int
    running_count: int
    failed_count: int
    succeeded_job_count: int
    skipped_count: int
    embedded_chunk_count: int
    coverage_percent: Decimal
    status: str
    latest_job_updated_at: datetime | None
    latest_embedding_at: datetime | None
    average_embedding_elapsed_ms: Decimal | None

    @property
    def missing_embedding_count(self) -> int:
        return max(0, self.chunk_count - self.embedded_chunk_count)

    @property
    def ready(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True)
class SearchCompareReadinessResult:
    actor_user_id: int
    requested_search_scope: str
    effective_search_scope: str
    document_group: str | None
    file_type: str | None
    chunk_policy_names: tuple[str, ...]
    profiles: tuple[SearchCompareReadinessProfile, ...]

    @property
    def profile_count(self) -> int:
        return len({profile.profile_name for profile in self.profiles})

    @property
    def policy_count(self) -> int:
        return len({profile.chunk_policy_name or "all" for profile in self.profiles})

    @property
    def expected_embedding_count(self) -> int:
        return sum(profile.chunk_count for profile in self.profiles)

    @property
    def embedded_chunk_count(self) -> int:
        return sum(profile.embedded_chunk_count for profile in self.profiles)

    @property
    def attention_count(self) -> int:
        return sum(1 for profile in self.profiles if profile.status in {"failed", "partial"})

    @property
    def ready(self) -> bool:
        return bool(self.profiles) and all(profile.ready for profile in self.profiles)

    @property
    def coverage_percent(self) -> Decimal:
        if self.expected_embedding_count == 0:
            return Decimal("0.00")
        return (
            Decimal(self.embedded_chunk_count)
            / Decimal(self.expected_embedding_count)
            * Decimal("100")
        ).quantize(Decimal("0.01"))


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
    allow_mock_fallback: bool = True


@dataclass(frozen=True)
class SearchPermissionMatrixEntryResult:
    search_log_id: int
    actor_user_id: int
    requested_search_scope: str
    effective_search_scope: str
    permission_filter: PermissionSearchFilter
    result_count: int
    unique_chunk_count: int
    top_result: SearchResultLike | None
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


@dataclass(frozen=True)
class _RawSearchCompareProfileResult:
    profile_name: str
    elapsed_ms: int
    results: tuple[SearchResultLike, ...]
    status: str
    error_code: str | None = None
    error_message: str | None = None
    query_runtime_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED


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
        allow_mock_fallback=search_input.allow_mock_fallback,
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
            allow_mock_fallback=matrix_input.allow_mock_fallback,
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
        allow_mock_fallback=common.allow_mock_fallback,
    )


def _default_profiles(database_url: str) -> tuple[str, ...]:
    profiles = tuple(
        profile.profile_name for profile in list_active_embedding_profiles(database_url)
    )
    if not profiles:
        raise InvalidSearchCompareError("No active embedding profiles are configured")
    return profiles


def _validate_readiness_input(
    readiness_input: SearchCompareReadinessInput,
) -> SearchCompareReadinessInput:
    if readiness_input.actor_user_id <= 0:
        raise InvalidSearchCompareError("actor_user_id must be greater than 0")
    requested_search_scope = (
        _validate_nonblank(readiness_input.requested_search_scope, "requested_search_scope")
        or readiness_input.requested_search_scope
    )
    profiles = None
    if readiness_input.profiles is not None:
        profiles = tuple(
            _validate_nonblank(profile, "profile_name") for profile in readiness_input.profiles
        )
        if not profiles:
            raise InvalidSearchCompareError("profiles must not be empty")
        if len(set(profiles)) != len(profiles):
            raise InvalidSearchCompareError("profiles must be unique")
    chunk_policy_names = None
    if readiness_input.chunk_policy_names is not None:
        chunk_policy_names = tuple(
            _validate_nonblank(policy_name, "chunk_policy_name")
            for policy_name in readiness_input.chunk_policy_names
        )
        if not chunk_policy_names:
            raise InvalidSearchCompareError("chunk_policy_names must not be empty")
        if len(set(chunk_policy_names)) != len(chunk_policy_names):
            raise InvalidSearchCompareError("chunk_policy_names must be unique")
    chunk_policy_name = _validate_nonblank(
        readiness_input.chunk_policy_name,
        "chunk_policy_name",
    )
    if chunk_policy_names is not None and chunk_policy_name is not None:
        raise InvalidSearchCompareError(
            "chunk_policy_name and chunk_policy_names cannot be used together"
        )
    return SearchCompareReadinessInput(
        actor_user_id=readiness_input.actor_user_id,
        requested_search_scope=requested_search_scope,
        profiles=profiles,
        chunk_policy_names=chunk_policy_names,
        chunk_policy_name=chunk_policy_name,
        document_group=_validate_nonblank(readiness_input.document_group, "document_group"),
        file_type=_validate_nonblank(readiness_input.file_type, "file_type"),
    )


def _readiness_status(
    *,
    chunk_count: int,
    pending_count: int,
    running_count: int,
    failed_count: int,
    succeeded_job_count: int,
    embedded_chunk_count: int,
) -> str:
    if chunk_count == 0:
        return "not_chunked"
    if embedded_chunk_count >= chunk_count:
        return "ready"
    if running_count > 0:
        return "running"
    if failed_count > 0 and pending_count == 0:
        return "failed"
    if pending_count > 0:
        return "pending"
    if embedded_chunk_count > 0 or succeeded_job_count > 0:
        return "partial"
    return "missing"


def _coverage_percent(*, chunk_count: int, embedded_chunk_count: int) -> Decimal:
    if chunk_count == 0:
        return Decimal("0.00")
    return (Decimal(embedded_chunk_count) / Decimal(chunk_count) * Decimal("100")).quantize(
        Decimal("0.01")
    )


def _validate_coverage_reconcile_input(
    reconcile_input: SearchCompareCoverageReconcileInput,
) -> SearchCompareCoverageReconcileInput:
    if reconcile_input.max_jobs <= 0:
        raise InvalidSearchCompareError("max_jobs must be greater than 0")
    if reconcile_input.max_jobs > 500:
        raise InvalidSearchCompareError("max_jobs must be less than or equal to 500")
    profile_name = _validate_nonblank(reconcile_input.profile_name, "profile_name")
    chunk_policy_name = _validate_nonblank(
        reconcile_input.chunk_policy_name,
        "chunk_policy_name",
    )
    readiness_input = _validate_readiness_input(
        SearchCompareReadinessInput(
            actor_user_id=reconcile_input.actor_user_id,
            requested_search_scope=reconcile_input.requested_search_scope,
            profiles=(profile_name or reconcile_input.profile_name,),
            chunk_policy_name=chunk_policy_name,
            document_group=reconcile_input.document_group,
            file_type=reconcile_input.file_type,
        ),
    )
    return SearchCompareCoverageReconcileInput(
        actor_user_id=readiness_input.actor_user_id,
        requested_search_scope=readiness_input.requested_search_scope,
        profile_name=(
            readiness_input.profiles[0] if readiness_input.profiles else profile_name or ""
        ),
        chunk_policy_name=readiness_input.chunk_policy_name or chunk_policy_name or "",
        document_group=readiness_input.document_group,
        file_type=readiness_input.file_type,
        max_jobs=reconcile_input.max_jobs,
    )


def _fetch_readiness_profiles(
    database_url: str,
    profiles: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if profiles is None:
        return _default_profiles(database_url)
    unsupported_profiles = sorted(
        profile for profile in profiles if profile not in EMBEDDING_VECTOR_TABLES
    )
    if unsupported_profiles:
        raise InvalidSearchCompareError(
            f"Unsupported embedding profiles: {', '.join(unsupported_profiles)}"
        )
    placeholders = ", ".join(["%s"] * len(profiles))
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT profile_name
                FROM embedding_profiles
                WHERE is_active
                  AND profile_name IN ({placeholders})
                """,
                profiles,
            )
            active_profiles = {str(row["profile_name"]) for row in cursor.fetchall()}
    inactive_profiles = sorted(set(profiles) - active_profiles)
    if inactive_profiles:
        raise InvalidSearchCompareError(
            f"Inactive embedding profiles: {', '.join(inactive_profiles)}"
        )
    return profiles


def _fetch_readiness_chunk_policies(
    database_url: str,
    readiness_input: SearchCompareReadinessInput,
) -> tuple[str | None, ...]:
    if readiness_input.chunk_policy_names is None and readiness_input.chunk_policy_name is None:
        return (None,)
    requested = readiness_input.chunk_policy_names or (readiness_input.chunk_policy_name,)
    requested = tuple(policy for policy in requested if policy is not None)
    placeholders = ", ".join(["%s"] * len(requested))
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT chunk_policy_name
                FROM chunk_policies
                WHERE chunk_policy_name IN ({placeholders})
                """,
                requested,
            )
            known_policy_names = {str(row["chunk_policy_name"]) for row in cursor.fetchall()}
    unknown_policy_names = sorted(set(requested) - known_policy_names)
    if unknown_policy_names:
        raise InvalidSearchCompareError(
            f"Unknown chunk_policy_names: {', '.join(unknown_policy_names)}"
        )
    return requested


def _readiness_filters(
    readiness_input: SearchCompareReadinessInput,
    permission_filter: PermissionSearchFilter,
    chunk_policy_name: str | None,
) -> tuple[str, list[object]]:
    clauses = ["d.document_status = 'active'"]
    params: list[object] = []
    if readiness_input.document_group is not None:
        clauses.append("d.document_group = %s")
        params.append(readiness_input.document_group)
    if readiness_input.file_type is not None:
        clauses.append("f.file_ext = %s")
        params.append(readiness_input.file_type)
    if chunk_policy_name is not None:
        clauses.append("c.chunk_policy_name = %s")
        params.append(chunk_policy_name)
    clauses.append(permission_filter.where_sql)
    params.extend(permission_filter.params)
    return " AND ".join(clauses), params


def _fetch_readiness_profile(
    database_url: str,
    *,
    readiness_input: SearchCompareReadinessInput,
    permission_filter: PermissionSearchFilter,
    profile_name: str,
    chunk_policy_name: str | None,
) -> SearchCompareReadinessProfile:
    vector_table = EMBEDDING_VECTOR_TABLES[profile_name]
    where_sql, filter_params = _readiness_filters(
        readiness_input,
        permission_filter,
        chunk_policy_name,
    )
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    count(c.chunk_id)::int AS chunk_count,
                    count(ej.job_id)::int AS job_count,
                    count(ej.job_id) FILTER (WHERE ej.status = 'pending')::int AS pending_count,
                    count(ej.job_id) FILTER (WHERE ej.status = 'running')::int AS running_count,
                    count(ej.job_id) FILTER (WHERE ej.status = 'failed')::int AS failed_count,
                    count(ej.job_id) FILTER (WHERE ej.status = 'succeeded')::int
                        AS succeeded_job_count,
                    count(ej.job_id) FILTER (WHERE ej.status = 'skipped')::int AS skipped_count,
                    count(v.chunk_id)::int AS embedded_chunk_count,
                    max(ej.updated_at) AS latest_job_updated_at,
                    max(v.created_at) AS latest_embedding_at,
                    avg(v.elapsed_ms)::numeric(12,2) AS average_embedding_elapsed_ms
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                LEFT JOIN embedding_jobs ej
                  ON ej.chunk_id = c.chunk_id
                 AND ej.profile_name = %s
                LEFT JOIN {vector_table.table_name} v ON v.chunk_id = c.chunk_id
                WHERE {where_sql}
                """,
                (profile_name, *filter_params),
            )
            row = cursor.fetchone()

    if row is None:
        raise InvalidSearchCompareError("Search readiness query returned no row")
    chunk_count = int(row["chunk_count"])
    pending_count = int(row["pending_count"])
    running_count = int(row["running_count"])
    failed_count = int(row["failed_count"])
    succeeded_job_count = int(row["succeeded_job_count"])
    embedded_chunk_count = int(row["embedded_chunk_count"])
    return SearchCompareReadinessProfile(
        profile_name=profile_name,
        chunk_policy_name=chunk_policy_name,
        chunk_count=chunk_count,
        job_count=int(row["job_count"]),
        pending_count=pending_count,
        running_count=running_count,
        failed_count=failed_count,
        succeeded_job_count=succeeded_job_count,
        skipped_count=int(row["skipped_count"]),
        embedded_chunk_count=embedded_chunk_count,
        coverage_percent=_coverage_percent(
            chunk_count=chunk_count,
            embedded_chunk_count=embedded_chunk_count,
        ),
        status=_readiness_status(
            chunk_count=chunk_count,
            pending_count=pending_count,
            running_count=running_count,
            failed_count=failed_count,
            succeeded_job_count=succeeded_job_count,
            embedded_chunk_count=embedded_chunk_count,
        ),
        latest_job_updated_at=row["latest_job_updated_at"],
        latest_embedding_at=row["latest_embedding_at"],
        average_embedding_elapsed_ms=row["average_embedding_elapsed_ms"],
    )


def get_search_compare_readiness(
    database_url: str,
    readiness_input: SearchCompareReadinessInput,
) -> SearchCompareReadinessResult:
    validated = _validate_readiness_input(readiness_input)
    profiles = _fetch_readiness_profiles(database_url, validated.profiles)
    chunk_policy_names = _fetch_readiness_chunk_policies(database_url, validated)
    permission_filter = resolve_permission_search_filter(
        database_url,
        actor_user_id=validated.actor_user_id,
        requested_search_scope=validated.requested_search_scope,
    )
    readiness_profiles = tuple(
        _fetch_readiness_profile(
            database_url,
            readiness_input=validated,
            permission_filter=permission_filter,
            profile_name=profile_name,
            chunk_policy_name=chunk_policy_name,
        )
        for chunk_policy_name in chunk_policy_names
        for profile_name in profiles
    )
    return SearchCompareReadinessResult(
        actor_user_id=validated.actor_user_id,
        requested_search_scope=permission_filter.requested_search_scope,
        effective_search_scope=permission_filter.effective_search_scope,
        document_group=validated.document_group,
        file_type=validated.file_type,
        chunk_policy_names=tuple(policy or "all" for policy in chunk_policy_names),
        profiles=readiness_profiles,
    )


def reconcile_search_compare_policy_coverage(
    database_url: str,
    reconcile_input: SearchCompareCoverageReconcileInput,
) -> SearchCompareCoverageReconcileResult:
    validated = _validate_coverage_reconcile_input(reconcile_input)
    profile_name = _fetch_readiness_profiles(database_url, (validated.profile_name,))[0]
    chunk_policy_name = _fetch_readiness_chunk_policies(
        database_url,
        SearchCompareReadinessInput(
            actor_user_id=validated.actor_user_id,
            requested_search_scope=validated.requested_search_scope,
            chunk_policy_name=validated.chunk_policy_name,
        ),
    )[0]
    if chunk_policy_name is None:
        raise InvalidSearchCompareError("chunk_policy_name is required")
    permission_filter = resolve_permission_search_filter(
        database_url,
        actor_user_id=validated.actor_user_id,
        requested_search_scope=validated.requested_search_scope,
    )
    readiness_input = SearchCompareReadinessInput(
        actor_user_id=validated.actor_user_id,
        requested_search_scope=validated.requested_search_scope,
        document_group=validated.document_group,
        file_type=validated.file_type,
    )
    where_sql, filter_params = _readiness_filters(
        readiness_input,
        permission_filter,
        chunk_policy_name,
    )

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    count(c.chunk_id)::int AS chunk_count,
                    count(ej.job_id)::int AS existing_job_count,
                    count(c.chunk_id) FILTER (WHERE ej.job_id IS NULL)::int
                        AS missing_job_count,
                    count(ej.job_id) FILTER (WHERE ej.status = 'failed')::int
                        AS failed_job_count,
                    count(ej.job_id) FILTER (
                        WHERE ej.status = 'failed' AND ej.attempts < ej.max_attempts
                    )::int AS retryable_failed_job_count
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                LEFT JOIN embedding_jobs ej
                  ON ej.chunk_id = c.chunk_id
                 AND ej.profile_name = %s
                WHERE {where_sql}
                """,
                (profile_name, *filter_params),
            )
            summary = dict(cursor.fetchone())
            cursor.execute(
                f"""
                SELECT c.chunk_id
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                LEFT JOIN embedding_jobs ej
                  ON ej.chunk_id = c.chunk_id
                 AND ej.profile_name = %s
                WHERE {where_sql}
                  AND ej.job_id IS NULL
                ORDER BY d.document_id ASC, c.chunk_seq ASC, c.chunk_id ASC
                LIMIT %s
                """,
                (profile_name, *filter_params, validated.max_jobs),
            )
            missing_chunk_ids = [int(row["chunk_id"]) for row in cursor.fetchall()]
            cursor.execute(
                f"""
                SELECT ej.job_id
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                JOIN embedding_jobs ej
                  ON ej.chunk_id = c.chunk_id
                 AND ej.profile_name = %s
                WHERE {where_sql}
                  AND ej.status = 'failed'
                  AND ej.attempts < ej.max_attempts
                ORDER BY d.document_id ASC, c.chunk_seq ASC, c.chunk_id ASC, ej.job_id ASC
                LIMIT %s
                """,
                (profile_name, *filter_params, validated.max_jobs),
            )
            retryable_job_ids = [int(row["job_id"]) for row in cursor.fetchall()]

        created_jobs = tuple(
            result.job
            for chunk_id in missing_chunk_ids
            if (
                result := create_embedding_job_in_connection(
                    connection,
                    EmbeddingJobInput(
                        chunk_id=chunk_id,
                        profile_name=profile_name,
                        runtime_metadata={
                            "reconcile_source": "search_compare_readiness",
                            "chunk_policy_name": chunk_policy_name,
                            "actor_user_id": validated.actor_user_id,
                            "requested_search_scope": permission_filter.requested_search_scope,
                            "effective_search_scope": permission_filter.effective_search_scope,
                            "document_group": validated.document_group,
                            "file_type": validated.file_type,
                        },
                    ),
                )
            ).created
        )
        retried_jobs = tuple(
            job
            for job_id in retryable_job_ids
            if (job := retry_embedding_job_in_connection(connection, job_id)) is not None
        )

    return SearchCompareCoverageReconcileResult(
        actor_user_id=validated.actor_user_id,
        requested_search_scope=permission_filter.requested_search_scope,
        effective_search_scope=permission_filter.effective_search_scope,
        profile_name=profile_name,
        chunk_policy_name=chunk_policy_name,
        document_group=validated.document_group,
        file_type=validated.file_type,
        chunk_count=int(summary["chunk_count"] or 0),
        existing_job_count=int(summary["existing_job_count"] or 0),
        missing_job_count=int(summary["missing_job_count"] or 0),
        created_job_count=len(created_jobs),
        failed_job_count=int(summary["failed_job_count"] or 0),
        retryable_failed_job_count=int(summary["retryable_failed_job_count"] or 0),
        retried_job_count=len(retried_jobs),
        created_jobs=created_jobs,
        retried_jobs=retried_jobs,
    )


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


def _profile_status_counts(
    profile_results: (
        tuple[_RawSearchCompareProfileResult, ...] | tuple[SearchCompareProfileResult, ...]
    ),
) -> dict[str, int]:
    counts = {
        SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED: 0,
        SEARCH_COMPARE_PROFILE_STATUS_FAILED: 0,
    }
    for profile_result in profile_results:
        counts[profile_result.status] = counts.get(profile_result.status, 0) + 1
    return counts


def _profile_failure_metadata(
    profile_name: str,
    *,
    error_code: str,
    error_message: str,
    elapsed_ms: int,
) -> dict[str, object]:
    return {
        "profile_name": profile_name,
        "status": SEARCH_COMPARE_PROFILE_STATUS_FAILED,
        "error_code": error_code,
        "error_message": error_message,
        "elapsed_ms": elapsed_ms,
    }


def _search_compare_query_runtime_metadata(
    profile_results: tuple[_RawSearchCompareProfileResult, ...],
    *,
    allow_mock_fallback: bool,
) -> dict[str, object]:
    profiles = {
        profile_result.profile_name: profile_result.query_runtime_metadata
        for profile_result in profile_results
        if profile_result.succeeded
    }
    query_embedding_profiles = {
        profile_name: metadata
        for profile_name, metadata in profiles.items()
        if metadata.get("retrieval_strategy") != BM25_RETRIEVAL_STRATEGY
    }
    keyword_search_profiles = {
        profile_name: metadata
        for profile_name, metadata in profiles.items()
        if metadata.get("retrieval_strategy") == BM25_RETRIEVAL_STRATEGY
    }
    has_query_embedding_profile = any(
        not _is_bm25_search_profile(profile_result.profile_name)
        for profile_result in profile_results
    )
    profile_failures = {
        profile_result.profile_name: _profile_failure_metadata(
            profile_result.profile_name,
            error_code=profile_result.error_code or "unknown_error",
            error_message=profile_result.error_message or "Unknown profile failure",
            elapsed_ms=profile_result.elapsed_ms,
        )
        for profile_result in profile_results
        if not profile_result.succeeded
    }
    provider_types = sorted(
        {
            str(profile_metadata["provider_type"])
            for profile_metadata in query_embedding_profiles.values()
            if profile_metadata.get("provider_type") is not None
        }
    )
    runtime_sources = sorted(
        {
            str(profile_metadata["runtime_source"])
            for profile_metadata in query_embedding_profiles.values()
            if profile_metadata.get("runtime_source") is not None
        }
    )
    status_counts = _profile_status_counts(profile_results)
    return {
        "adapter": (
            "query_embedding_bridge" if not keyword_search_profiles else "search_compare_runtime"
        ),
        "search_mode": "compare_mvp",
        "query_embedding_bridge": has_query_embedding_profile,
        "allow_mock_fallback": allow_mock_fallback,
        "real_provider_required": not allow_mock_fallback,
        "selected_profile_count": len(profile_results),
        "query_embedding_profile_count": len(query_embedding_profiles),
        "query_embedding_success_count": len(query_embedding_profiles),
        "keyword_search_profile_count": len(keyword_search_profiles),
        "profile_status_counts": status_counts,
        "profile_failure_count": status_counts.get(SEARCH_COMPARE_PROFILE_STATUS_FAILED, 0),
        "query_embedding_provider_types": provider_types,
        "query_embedding_runtime_sources": runtime_sources,
        "profile_query_embeddings": query_embedding_profiles,
        "profile_keyword_searches": keyword_search_profiles,
        "profile_failures": profile_failures,
    }


def _profile_error_code(exc: Exception) -> str:
    if isinstance(exc, InvalidQueryEmbeddingError):
        return SEARCH_COMPARE_PROFILE_ERROR_QUERY_EMBEDDING_FAILED
    if isinstance(exc, InvalidBM25SearchError):
        return SEARCH_COMPARE_PROFILE_ERROR_KEYWORD_SEARCH_FAILED
    return SEARCH_COMPARE_PROFILE_ERROR_VECTOR_SEARCH_FAILED


def _is_bm25_search_profile(profile_name: str) -> bool:
    return profile_name == BM25_SEARCH_PROFILE_NAME


def _bm25_query_runtime_metadata(
    *,
    tokenizer_name: str,
    k1: float,
    b: float,
) -> dict[str, object]:
    return {
        "provider_type": "keyword",
        "provider_model_id": BM25_SEARCH_PROFILE_NAME,
        "runtime_source": "local_keyword_index",
        "query_embedding_bridge": False,
        "retrieval_strategy": BM25_RETRIEVAL_STRATEGY,
        "search_profile_name": BM25_SEARCH_PROFILE_NAME,
        "tokenizer_name": tokenizer_name,
        "k1": k1,
        "b": b,
    }


def _search_compare_strategy_name(profiles: tuple[str, ...]) -> str:
    bm25_profile_count = sum(
        1 for profile_name in profiles if _is_bm25_search_profile(profile_name)
    )
    if bm25_profile_count == 0:
        return "vector_cosine"
    if bm25_profile_count == len(profiles):
        return BM25_RETRIEVAL_STRATEGY
    return "mixed_vector_bm25"


def _search_compare_similarity_metric(profiles: tuple[str, ...]) -> str:
    if profiles and all(_is_bm25_search_profile(profile_name) for profile_name in profiles):
        return "bm25"
    return "cosine"


def _failed_profile_result(
    profile_name: str,
    *,
    started_at: float,
    exc: Exception,
) -> _RawSearchCompareProfileResult:
    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    error_code = _profile_error_code(exc)
    return _RawSearchCompareProfileResult(
        profile_name=profile_name,
        elapsed_ms=elapsed_ms,
        results=(),
        status=SEARCH_COMPARE_PROFILE_STATUS_FAILED,
        error_code=error_code,
        error_message=str(exc),
        query_runtime_metadata=_profile_failure_metadata(
            profile_name,
            error_code=error_code,
            error_message=str(exc),
            elapsed_ms=elapsed_ms,
        ),
    )


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
    raw_profile_results: list[_RawSearchCompareProfileResult] = []
    for profile_name in profiles:
        profile_started_at = perf_counter()
        try:
            if _is_bm25_search_profile(profile_name):
                bm25_input = BM25SearchInput(
                    query_text=validated.query_text,
                    top_k=validated.top_k,
                    chunk_policy_name=validated.chunk_policy_name or DEFAULT_CHUNK_POLICY_NAME,
                    document_group=validated.document_group,
                    file_type=validated.file_type,
                    permission_filter=permission_filter,
                )
                results = search_bm25_chunks(database_url, bm25_input)
                query_runtime_metadata = _bm25_query_runtime_metadata(
                    tokenizer_name=bm25_input.tokenizer_name,
                    k1=bm25_input.k1,
                    b=bm25_input.b,
                )
            else:
                query_embedding = embed_query_for_profile(
                    database_url,
                    query_text=validated.query_text,
                    profile_name=profile_name,
                    fallback_runtime_config=fallback_config,
                    provider_builder=query_embedding_provider_builder,
                    trace_id=f"search-compare:{validated.actor_user_id}:{profile_name}",
                    allow_mock_fallback=validated.allow_mock_fallback,
                )
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
                query_runtime_metadata = query_embedding_runtime_metadata(query_embedding)
            raw_profile_results.append(
                _RawSearchCompareProfileResult(
                    profile_name=profile_name,
                    elapsed_ms=max(0, int((perf_counter() - profile_started_at) * 1000)),
                    results=tuple(results),
                    status=SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
                    query_runtime_metadata=query_runtime_metadata,
                )
            )
        except (
            InvalidQueryEmbeddingError,
            InvalidVectorSearchError,
            InvalidBM25SearchError,
        ) as exc:
            raw_profile_results.append(
                _failed_profile_result(
                    profile_name,
                    started_at=profile_started_at,
                    exc=exc,
                )
            )

    raw_profile_results_tuple = tuple(raw_profile_results)
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
            profiles=profiles,
            query_runtime_metadata=_search_compare_query_runtime_metadata(
                raw_profile_results_tuple,
                allow_mock_fallback=validated.allow_mock_fallback,
            ),
            strategy_name=_search_compare_strategy_name(profiles),
            total_elapsed_ms=total_elapsed_ms,
            similarity_metric=_search_compare_similarity_metric(profiles),
            created_by_user_id=validated.actor_user_id,
        ),
    )
    result_inputs = [
        SearchLogResultInput(
            search_log_id=search_log.search_log_id,
            profile_name=profile_result.profile_name,
            rank=result.rank,
            chunk_id=result.chunk_id,
            distance=result.distance,
            score=result.score,
            search_profile_name=getattr(
                result,
                "search_profile_name",
                profile_result.profile_name,
            ),
            retrieval_strategy=getattr(result, "retrieval_strategy", "vector_cosine"),
            score_components=getattr(result, "score_components", {}),
            profile_elapsed_ms=profile_result.elapsed_ms,
        )
        for profile_result in raw_profile_results_tuple
        for result in profile_result.results
    ]
    stored_results = []
    if result_inputs:
        stored_results = create_search_log_results(database_url, result_inputs)

    result_id_iter = iter(stored_results)
    profile_results = [
        SearchCompareProfileResult(
            profile_name=profile_result.profile_name,
            elapsed_ms=profile_result.elapsed_ms,
            results=tuple(
                SearchCompareResultItem(
                    search_log_result_id=next(result_id_iter).search_log_result_id,
                    vector_result=result,
                )
                for result in profile_result.results
            ),
            status=profile_result.status,
            error_code=profile_result.error_code,
            error_message=profile_result.error_message,
            query_runtime_metadata=profile_result.query_runtime_metadata,
        )
        for profile_result in raw_profile_results_tuple
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
                allow_mock_fallback=validated.allow_mock_fallback,
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
