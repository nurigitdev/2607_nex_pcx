"""Provider route and local model availability matrix."""

from dataclasses import dataclass
from pathlib import Path

from app.core.embedding_model_distribution import (
    EmbeddingModelReadiness,
    audit_embedding_model_readiness,
)
from app.core.embedding_provider_route_readiness import (
    EmbeddingProviderRouteReadinessItem,
    EmbeddingProviderRouteReadinessSummary,
    get_embedding_provider_route_readiness_summary,
)


@dataclass(frozen=True)
class ProviderModelAvailabilityRow:
    profile_name: str
    model_key: str
    repo_id: str
    local_dir: Path
    model_ready: bool
    model_exists: bool
    model_status: str
    route_count: int
    active_route_count: int
    ready_route_count: int
    blocked_route_count: int
    status: str
    route_status_counts: dict[str, int]
    provider_names: tuple[str, ...]


@dataclass(frozen=True)
class ProviderModelAvailabilityMatrix:
    rows: tuple[ProviderModelAvailabilityRow, ...]

    @property
    def profile_count(self) -> int:
        return len(self.rows)

    @property
    def ready_count(self) -> int:
        return sum(1 for row in self.rows if row.status == "ready")

    @property
    def blocked_count(self) -> int:
        return self.profile_count - self.ready_count

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts


def get_provider_model_availability_matrix(
    database_url: str,
    *,
    models_dir: Path,
) -> ProviderModelAvailabilityMatrix:
    return build_provider_model_availability_matrix(
        audit_embedding_model_readiness(models_dir),
        get_embedding_provider_route_readiness_summary(database_url, active_only=False),
    )


def build_provider_model_availability_matrix(
    model_readiness: tuple[EmbeddingModelReadiness, ...],
    route_readiness: EmbeddingProviderRouteReadinessSummary,
) -> ProviderModelAvailabilityMatrix:
    route_items_by_profile: dict[str, list[EmbeddingProviderRouteReadinessItem]] = {}
    for item in route_readiness.routes:
        route_items_by_profile.setdefault(item.route.profile_name, []).append(item)

    rows: list[ProviderModelAvailabilityRow] = []
    for item in model_readiness:
        for profile_name in item.distribution.profile_names:
            profile_routes = tuple(route_items_by_profile.get(profile_name, ()))
            active_routes = tuple(route for route in profile_routes if route.route.is_active)
            ready_routes = tuple(route for route in profile_routes if route.ready)
            route_status_counts = _route_status_counts(profile_routes)
            rows.append(
                ProviderModelAvailabilityRow(
                    profile_name=profile_name,
                    model_key=item.distribution.model_key,
                    repo_id=item.distribution.repo_id,
                    local_dir=item.local_dir,
                    model_ready=item.ready,
                    model_exists=item.exists,
                    model_status=_model_status(item),
                    route_count=len(profile_routes),
                    active_route_count=len(active_routes),
                    ready_route_count=len(ready_routes),
                    blocked_route_count=len(profile_routes) - len(ready_routes),
                    status=_availability_status(item, profile_routes),
                    route_status_counts=route_status_counts,
                    provider_names=tuple(
                        sorted({route.route.provider_name for route in profile_routes})
                    ),
                )
            )
    return ProviderModelAvailabilityMatrix(rows=tuple(rows))


def _availability_status(
    model_readiness: EmbeddingModelReadiness,
    routes: tuple[EmbeddingProviderRouteReadinessItem, ...],
) -> str:
    if not model_readiness.ready:
        return _model_status(model_readiness)
    if any(route.ready for route in routes):
        return "ready"
    if any(route.route.is_active for route in routes):
        return "route_not_ready"
    if routes:
        return "route_inactive"
    return "missing_route"


def _model_status(model_readiness: EmbeddingModelReadiness) -> str:
    if model_readiness.ready:
        return "ready"
    if model_readiness.exists:
        return "model_incomplete"
    return "model_missing"


def _route_status_counts(
    routes: tuple[EmbeddingProviderRouteReadinessItem, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        counts[route.status] = counts.get(route.status, 0) + 1
    return counts
