"""Health aggregation for profile-specific embedding provider routes."""

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
    MOCK_EMBEDDING_PROVIDER_TYPE,
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProviderHealth,
    InvalidEmbeddingProviderError,
    RemoteEmbeddingProviderClient,
)
from app.core.embedding_vectors import InvalidEmbeddingVectorError, get_embedding_vector_table

EmbeddingProviderRouteHTTPClientFactory = Callable[
    [EmbeddingProviderRouteRecord],
    object | None,
]


@dataclass(frozen=True)
class EmbeddingProviderRouteHealthResult:
    route: EmbeddingProviderRouteRecord
    checked: bool
    ready: bool
    status: str
    elapsed_ms: int | None
    provider_type: str | None
    provider_model_id: str | None
    model_key: str | None
    profile_names: tuple[str, ...]
    dimension: int | None
    device: str | None
    runtime_metadata: dict[str, object]
    validation_errors: tuple[str, ...] = ()
    error_message: str | None = None


@dataclass(frozen=True)
class EmbeddingProviderRouteHealthSummary:
    routes: tuple[EmbeddingProviderRouteHealthResult, ...]

    @property
    def route_count(self) -> int:
        return len(self.routes)

    @property
    def ready_count(self) -> int:
        return sum(1 for route in self.routes if route.ready)

    @property
    def checked_count(self) -> int:
        return sum(1 for route in self.routes if route.checked)


def get_embedding_provider_route_health_summary(
    database_url: str,
    *,
    profile_name: str | None = None,
    active_only: bool = True,
    http_client_factory: EmbeddingProviderRouteHTTPClientFactory | None = None,
) -> EmbeddingProviderRouteHealthSummary:
    routes = list_embedding_provider_routes(
        database_url,
        profile_name=profile_name,
        active_only=active_only,
    )
    return summarize_embedding_provider_route_health(
        routes,
        http_client_factory=http_client_factory,
    )


def summarize_embedding_provider_route_health(
    routes: Sequence[EmbeddingProviderRouteRecord],
    *,
    http_client_factory: EmbeddingProviderRouteHTTPClientFactory | None = None,
) -> EmbeddingProviderRouteHealthSummary:
    return EmbeddingProviderRouteHealthSummary(
        routes=tuple(
            check_embedding_provider_route_health(
                route,
                http_client=http_client_factory(route) if http_client_factory else None,
            )
            for route in routes
        )
    )


def check_embedding_provider_route_health(
    route: EmbeddingProviderRouteRecord,
    *,
    http_client: object | None = None,
) -> EmbeddingProviderRouteHealthResult:
    if not route.health_check_enabled:
        return _unchecked_route_health(route, status="skipped")
    if route.provider_mode == MOCK_EMBEDDING_PROVIDER_TYPE:
        return _mock_route_health(route)
    if route.provider_mode == REMOTE_EMBEDDING_PROVIDER_TYPE:
        return _remote_route_health(route, http_client=http_client)
    return _unchecked_route_health(
        route,
        status="unsupported",
        error_message=f"Unsupported provider_mode: {route.provider_mode}",
    )


def _mock_route_health(route: EmbeddingProviderRouteRecord) -> EmbeddingProviderRouteHealthResult:
    return EmbeddingProviderRouteHealthResult(
        route=route,
        checked=True,
        ready=True,
        status="ready",
        elapsed_ms=0,
        provider_type=MOCK_EMBEDDING_PROVIDER_TYPE,
        provider_model_id="mock-provider",
        model_key=None,
        profile_names=(route.profile_name,),
        dimension=None,
        device=None,
        runtime_metadata={"provider": MOCK_EMBEDDING_PROVIDER_TYPE},
    )


def _remote_route_health(
    route: EmbeddingProviderRouteRecord,
    *,
    http_client: object | None,
) -> EmbeddingProviderRouteHealthResult:
    started_at = perf_counter()
    client = None
    try:
        client = RemoteEmbeddingProviderClient(
            route.provider_base_url or "",
            timeout_seconds=route.timeout_seconds,
            headers=resolve_embedding_provider_route_request_headers(route.runtime_metadata),
            http_client=http_client,
        )
        health = client.health()
    except (InvalidEmbeddingProviderError, InvalidEmbeddingProviderRouteAuthError) as exc:
        return EmbeddingProviderRouteHealthResult(
            route=route,
            checked=True,
            ready=False,
            status="unreachable",
            elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            provider_type=REMOTE_EMBEDDING_PROVIDER_TYPE,
            provider_model_id=None,
            model_key=None,
            profile_names=(),
            dimension=None,
            device=None,
            runtime_metadata={},
            error_message=str(exc),
        )
    finally:
        if client is not None:
            client.close()

    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    validation_errors = _validate_route_health_match(route, health)
    ready = health.ready and not validation_errors
    if validation_errors:
        route_status = "mismatch"
    elif health.ready:
        route_status = "ready"
    else:
        route_status = "not_ready"

    return EmbeddingProviderRouteHealthResult(
        route=route,
        checked=True,
        ready=ready,
        status=route_status,
        elapsed_ms=elapsed_ms,
        provider_type=health.provider_type,
        provider_model_id=health.provider_model_id,
        model_key=health.model_key,
        profile_names=health.profile_names,
        dimension=health.dimension,
        device=health.device,
        runtime_metadata=dict(health.runtime_metadata),
        validation_errors=validation_errors,
    )


def _validate_route_health_match(
    route: EmbeddingProviderRouteRecord,
    health: EmbeddingProviderHealth,
) -> tuple[str, ...]:
    validation_errors = []
    if health.provider_type != REMOTE_EMBEDDING_PROVIDER_TYPE:
        validation_errors.append(f"provider_type mismatch: {health.provider_type}")
    if health.profile_names and route.profile_name not in health.profile_names:
        validation_errors.append(f"profile_name missing from provider health: {route.profile_name}")

    try:
        expected_distribution = get_embedding_model_distribution_for_profile(route.profile_name)
        if health.model_key != expected_distribution.model_key:
            validation_errors.append(
                f"model_key mismatch: expected {expected_distribution.model_key}, "
                f"got {health.model_key}"
            )
    except InvalidEmbeddingModelDistributionError as exc:
        validation_errors.append(str(exc))

    try:
        expected_table = get_embedding_vector_table(route.profile_name)
        provider_dimension = provider_health_dimension_for_profile(
            health.dimension,
            profile_name=route.profile_name,
            runtime_metadata=health.runtime_metadata,
        )
        if provider_dimension is not None and provider_dimension != expected_table.dimension:
            validation_errors.append(
                f"dimension mismatch: expected {expected_table.dimension}, got {provider_dimension}"
            )
    except InvalidEmbeddingVectorError as exc:
        validation_errors.append(str(exc))

    return tuple(validation_errors)


def provider_health_dimension_for_profile(
    dimension: int | None,
    *,
    profile_name: str,
    runtime_metadata: Mapping[str, object],
) -> int | None:
    profile_dimensions = runtime_metadata.get("profile_dimensions")
    if isinstance(profile_dimensions, Mapping):
        profile_dimension = profile_dimensions.get(profile_name)
        if profile_dimension is not None:
            try:
                return int(profile_dimension)
            except (TypeError, ValueError):
                return dimension
    return dimension


def _unchecked_route_health(
    route: EmbeddingProviderRouteRecord,
    *,
    status: str,
    error_message: str | None = None,
) -> EmbeddingProviderRouteHealthResult:
    return EmbeddingProviderRouteHealthResult(
        route=route,
        checked=False,
        ready=False,
        status=status,
        elapsed_ms=None,
        provider_type=None,
        provider_model_id=None,
        model_key=None,
        profile_names=(),
        dimension=None,
        device=None,
        runtime_metadata={},
        error_message=error_message,
    )
