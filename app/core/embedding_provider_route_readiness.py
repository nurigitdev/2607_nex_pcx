"""Readiness aggregation for embedding provider routes."""

from dataclasses import dataclass

from app.core.embedding_provider_route_contract_snapshots import (
    EmbeddingProviderRouteContractSnapshotRecord,
    list_latest_embedding_provider_route_contract_snapshots,
)
from app.core.embedding_provider_route_health_snapshots import (
    EmbeddingProviderRouteHealthSnapshotRecord,
    list_latest_embedding_provider_route_health_snapshots,
)
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteRecord,
    list_embedding_provider_routes,
)


@dataclass(frozen=True)
class EmbeddingProviderRouteReadinessItem:
    route: EmbeddingProviderRouteRecord
    ready: bool
    status: str
    reasons: tuple[str, ...]
    latest_health_snapshot: EmbeddingProviderRouteHealthSnapshotRecord | None
    latest_contract_snapshot: EmbeddingProviderRouteContractSnapshotRecord | None


@dataclass(frozen=True)
class EmbeddingProviderRouteReadinessSummary:
    routes: tuple[EmbeddingProviderRouteReadinessItem, ...]

    @property
    def route_count(self) -> int:
        return len(self.routes)

    @property
    def active_count(self) -> int:
        return sum(1 for item in self.routes if item.route.is_active)

    @property
    def ready_count(self) -> int:
        return sum(1 for item in self.routes if item.ready)

    @property
    def blocked_count(self) -> int:
        return self.route_count - self.ready_count

    @property
    def needs_preflight_count(self) -> int:
        return sum(1 for item in self.routes if item.status == "needs_contract")

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.routes:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts


def get_embedding_provider_route_readiness_summary(
    database_url: str,
    *,
    profile_name: str | None = None,
    active_only: bool = False,
) -> EmbeddingProviderRouteReadinessSummary:
    routes = list_embedding_provider_routes(
        database_url,
        profile_name=profile_name,
        active_only=active_only,
    )
    route_ids = tuple(route.route_id for route in routes)
    latest_health_snapshots = list_latest_embedding_provider_route_health_snapshots(
        database_url,
        route_ids,
    )
    latest_contract_snapshots = list_latest_embedding_provider_route_contract_snapshots(
        database_url,
        route_ids,
    )

    return EmbeddingProviderRouteReadinessSummary(
        routes=tuple(
            _route_readiness_item(
                route,
                latest_health_snapshots.get(route.route_id),
                latest_contract_snapshots.get(route.route_id),
            )
            for route in routes
        )
    )


def _route_readiness_item(
    route: EmbeddingProviderRouteRecord,
    health: EmbeddingProviderRouteHealthSnapshotRecord | None,
    contract: EmbeddingProviderRouteContractSnapshotRecord | None,
) -> EmbeddingProviderRouteReadinessItem:
    status, ready, reasons = _classify_route_readiness(route, health, contract)
    return EmbeddingProviderRouteReadinessItem(
        route=route,
        ready=ready,
        status=status,
        reasons=reasons,
        latest_health_snapshot=health,
        latest_contract_snapshot=contract,
    )


def _classify_route_readiness(
    route: EmbeddingProviderRouteRecord,
    health: EmbeddingProviderRouteHealthSnapshotRecord | None,
    contract: EmbeddingProviderRouteContractSnapshotRecord | None,
) -> tuple[str, bool, tuple[str, ...]]:
    if not route.is_active:
        return "inactive", False, ("route_inactive",)
    if contract is None:
        return "needs_contract", False, ("contract_snapshot_missing",)
    if not contract.passed:
        return "contract_failed", False, (f"contract:{contract.status}",)
    if health is not None and not health.ready and health.checked_at >= contract.checked_at:
        return "health_not_ready", False, (f"health:{health.status}",)
    return "ready", True, ()
