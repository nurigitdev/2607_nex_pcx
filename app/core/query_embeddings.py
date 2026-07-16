"""Query embedding bridge for provider-backed vector search."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from app.core.embedding_model_distribution import (
    InvalidEmbeddingModelDistributionError,
    get_embedding_model_distribution_for_profile,
)
from app.core.embedding_provider_route_auth import (
    InvalidEmbeddingProviderRouteAuthError,
    resolve_embedding_provider_route_request_headers,
)
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteRecord,
    list_embedding_provider_routes,
)
from app.core.embedding_providers import (
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProvider,
    EmbeddingProviderRequest,
    EmbeddingProviderRuntimeConfig,
    InvalidEmbeddingProviderError,
    build_embedding_provider_from_runtime_config,
    normalize_embedding_provider_runtime_config,
)
from app.core.embedding_vectors import (
    InvalidEmbeddingVectorError,
    get_embedding_vector_table,
)

QUERY_EMBEDDING_INPUT_TYPE = "query"
QUERY_EMBEDDING_RUNTIME_SOURCE_FALLBACK = "fallback_runtime_config"
QUERY_EMBEDDING_RUNTIME_SOURCE_ROUTE = "route"

QueryEmbeddingProviderBuilder = Callable[[EmbeddingProviderRuntimeConfig], EmbeddingProvider]
QueryEmbeddingRouteCandidatesSelector = Callable[
    [str, str],
    Sequence[EmbeddingProviderRouteRecord],
]


@dataclass(frozen=True)
class QueryEmbeddingResult:
    profile_name: str
    embedding: tuple[float, ...]
    dimension: int
    provider_type: str
    provider_model_id: str
    provider_elapsed_ms: int
    total_elapsed_ms: int
    runtime_source: str
    runtime_metadata: dict[str, object]


class InvalidQueryEmbeddingError(ValueError):
    """Raised when query embedding cannot be generated for vector search."""


def embed_query_for_profile(
    database_url: str,
    *,
    query_text: str,
    profile_name: str,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    provider_builder: QueryEmbeddingProviderBuilder = build_embedding_provider_from_runtime_config,
    route_candidates_selector: QueryEmbeddingRouteCandidatesSelector | None = None,
    trace_id: str | None = None,
) -> QueryEmbeddingResult:
    """Generate one query embedding using an active provider route or fallback config."""

    normalized_query = _validate_nonblank(query_text, "query_text")
    normalized_profile = _validate_nonblank(profile_name, "profile_name")
    fallback_config = fallback_runtime_config or EmbeddingProviderRuntimeConfig(mode="mock")
    try:
        table = get_embedding_vector_table(normalized_profile)
        distribution = get_embedding_model_distribution_for_profile(normalized_profile)
    except (InvalidEmbeddingVectorError, InvalidEmbeddingModelDistributionError) as exc:
        raise InvalidQueryEmbeddingError(str(exc)) from exc

    routes = list(
        (route_candidates_selector or _select_query_embedding_route_candidates)(
            database_url,
            normalized_profile,
        )
    )
    candidates: list[EmbeddingProviderRouteRecord | None] = [*routes] if routes else [None]
    failed_attempts: list[dict[str, object]] = []

    for attempt_index, route in enumerate(candidates, start=1):
        provider: EmbeddingProvider | None = None
        try:
            runtime_config = _runtime_config_for_query(
                route,
                fallback_runtime_config=fallback_config,
            )
            provider = provider_builder(runtime_config)
            started_at = perf_counter()
            response = provider.embed(
                EmbeddingProviderRequest(
                    profile_name=normalized_profile,
                    model_key=distribution.model_key,
                    input_type=QUERY_EMBEDDING_INPUT_TYPE,
                    texts=(normalized_query,),
                    output_dimension=table.dimension,
                    trace_id=trace_id,
                    runtime_metadata={
                        "query_embedding_bridge": True,
                        "query_embedding_attempt": attempt_index,
                    },
                )
            )
            total_elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
            embedding = response.embeddings[0]
            runtime_metadata = _runtime_metadata_for_query_provider(
                route,
                runtime_config=runtime_config,
                response_metadata=response.runtime_metadata,
                failed_attempts=failed_attempts,
                attempt_index=attempt_index,
                candidate_count=len(candidates),
                table_name=table.table_name,
                storage_type=table.storage_type,
                trace_id=trace_id,
            )
            return QueryEmbeddingResult(
                profile_name=normalized_profile,
                embedding=embedding,
                dimension=response.dimension,
                provider_type=response.provider_type,
                provider_model_id=response.provider_model_id,
                provider_elapsed_ms=response.elapsed_ms,
                total_elapsed_ms=total_elapsed_ms,
                runtime_source=str(runtime_metadata["provider_runtime_source"]),
                runtime_metadata=runtime_metadata,
            )
        except InvalidEmbeddingProviderError as exc:
            failed_attempts.append(_failed_attempt_metadata(route, str(exc)))
        finally:
            if provider is not None and hasattr(provider, "close"):
                provider.close()  # type: ignore[attr-defined]

    raise InvalidQueryEmbeddingError(_failed_query_embedding_message(failed_attempts))


def query_embedding_runtime_metadata(
    result: QueryEmbeddingResult,
) -> dict[str, object]:
    """Return a JSON-safe metadata summary without embedding values."""

    return {
        "profile_name": result.profile_name,
        "dimension": result.dimension,
        "provider_type": result.provider_type,
        "provider_model_id": result.provider_model_id,
        "provider_elapsed_ms": result.provider_elapsed_ms,
        "total_elapsed_ms": result.total_elapsed_ms,
        "runtime_source": result.runtime_source,
        "runtime_metadata": dict(result.runtime_metadata),
    }


def _select_query_embedding_route_candidates(
    database_url: str,
    profile_name: str,
) -> Sequence[EmbeddingProviderRouteRecord]:
    return list_embedding_provider_routes(
        database_url,
        profile_name=profile_name,
        active_only=True,
    )


def _runtime_config_for_query(
    route: EmbeddingProviderRouteRecord | None,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig,
) -> EmbeddingProviderRuntimeConfig:
    if route is None:
        return normalize_embedding_provider_runtime_config(fallback_runtime_config)
    try:
        remote_headers = (
            resolve_embedding_provider_route_request_headers(route.runtime_metadata)
            if route.provider_mode == REMOTE_EMBEDDING_PROVIDER_TYPE
            else {}
        )
    except InvalidEmbeddingProviderRouteAuthError as exc:
        raise InvalidEmbeddingProviderError(str(exc)) from exc
    return normalize_embedding_provider_runtime_config(
        EmbeddingProviderRuntimeConfig(
            mode=route.provider_mode,
            remote_base_url=route.provider_base_url,
            remote_timeout_seconds=route.timeout_seconds,
            remote_headers=remote_headers,
        )
    )


def _runtime_metadata_for_query_provider(
    route: EmbeddingProviderRouteRecord | None,
    *,
    runtime_config: EmbeddingProviderRuntimeConfig,
    response_metadata: Mapping[str, object],
    failed_attempts: Sequence[Mapping[str, object]],
    attempt_index: int,
    candidate_count: int,
    table_name: str,
    storage_type: str,
    trace_id: str | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "query_embedding_bridge": True,
        "query_embedding_input_type": QUERY_EMBEDDING_INPUT_TYPE,
        "provider_runtime_source": (
            QUERY_EMBEDDING_RUNTIME_SOURCE_ROUTE
            if route is not None
            else QUERY_EMBEDDING_RUNTIME_SOURCE_FALLBACK
        ),
        "provider_runtime_mode": runtime_config.mode,
        "provider_runtime_base_url": runtime_config.remote_base_url,
        "provider_runtime_timeout_seconds": runtime_config.remote_timeout_seconds,
        "embedding_table_name": table_name,
        "embedding_storage_type": storage_type,
        "provider_response_metadata": dict(response_metadata),
    }
    if trace_id is not None:
        metadata["trace_id"] = trace_id
    if route is not None:
        metadata.update(
            {
                "provider_route_id": route.route_id,
                "provider_route_name": route.provider_name,
                "provider_route_priority": route.priority,
            }
        )
        if route.runtime_metadata:
            metadata["provider_route_metadata"] = dict(route.runtime_metadata)
    if candidate_count > 1:
        metadata["provider_route_failover_attempt"] = attempt_index
        metadata["provider_route_failover_candidate_count"] = candidate_count
    if failed_attempts:
        metadata["provider_route_failed_attempts"] = [dict(attempt) for attempt in failed_attempts]
    return metadata


def _failed_attempt_metadata(
    route: EmbeddingProviderRouteRecord | None,
    error_message: str,
) -> dict[str, object]:
    if route is None:
        return {
            "provider_runtime_source": QUERY_EMBEDDING_RUNTIME_SOURCE_FALLBACK,
            "error_message": error_message,
        }
    return {
        "route_id": route.route_id,
        "provider_name": route.provider_name,
        "provider_mode": route.provider_mode,
        "priority": route.priority,
        "error_message": error_message,
    }


def _failed_query_embedding_message(
    failed_attempts: Sequence[Mapping[str, object]],
) -> str:
    if not failed_attempts:
        return "Query embedding failed"
    details = "; ".join(
        f"{attempt.get('provider_name') or attempt.get('provider_runtime_source')}: "
        f"{attempt.get('error_message')}"
        for attempt in failed_attempts
    )
    return f"Query embedding failed for all provider candidates: {details}"


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidQueryEmbeddingError(f"{field_name} is required")
    return normalized
