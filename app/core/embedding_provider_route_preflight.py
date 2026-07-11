"""Shared provider route preflight runner."""

from app.core.embedding_provider_contract_sample_sets import (
    get_default_embedding_provider_contract_sample_set,
)
from app.core.embedding_provider_route_contract_snapshots import (
    record_embedding_provider_route_contract_snapshot,
)
from app.core.embedding_provider_route_contracts import (
    check_embedding_provider_route_contract,
)
from app.core.embedding_provider_route_health_snapshots import (
    record_embedding_provider_route_health_snapshot,
)
from app.core.embedding_provider_routes import list_embedding_provider_routes


def run_embedding_provider_route_preflight(
    database_url: str,
    *,
    profile_name: str | None = None,
    active_only: bool = True,
) -> dict[str, object]:
    routes = list_embedding_provider_routes(
        database_url,
        profile_name=profile_name,
        active_only=active_only,
    )
    sample_set = get_default_embedding_provider_contract_sample_set(database_url)
    results = []
    for route in routes:
        contract = check_embedding_provider_route_contract(
            route,
            sample_texts=sample_set.sample_texts,
            input_type=sample_set.input_type,
            sample_set_name=sample_set.sample_set_name,
        )
        health_snapshot = None
        if contract.health is not None:
            health_snapshot = record_embedding_provider_route_health_snapshot(
                database_url,
                contract.health,
            )
        contract_snapshot = record_embedding_provider_route_contract_snapshot(
            database_url,
            contract,
        )
        results.append(
            {
                "route_id": route.route_id,
                "profile_name": route.profile_name,
                "provider_name": route.provider_name,
                "provider_mode": route.provider_mode,
                "health_status": contract.health.status if contract.health else None,
                "health_snapshot_id": (
                    health_snapshot.snapshot_id if health_snapshot is not None else None
                ),
                "contract_passed": contract.passed,
                "contract_status": contract.status,
                "contract_snapshot_id": contract_snapshot.snapshot_id,
                "provider_model_id": contract.provider_model_id,
                "dimension": contract.dimension,
                "elapsed_ms": contract.elapsed_ms,
                "validation_errors": list(contract.validation_errors),
                "error_message": contract.error_message,
            }
        )

    passed_count = sum(1 for result in results if result["contract_passed"])
    return {
        "route_count": len(routes),
        "passed_count": passed_count,
        "failed_count": len(routes) - passed_count,
        "profile_name": profile_name,
        "active_only": active_only,
        "sample_set": {
            "sample_set_name": sample_set.sample_set_name,
            "input_type": sample_set.input_type,
            "sample_text_count": len(sample_set.sample_texts),
        },
        "results": results,
    }
