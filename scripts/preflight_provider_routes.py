"""Run provider route preflight checks from the command line."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_provider_route_contract_snapshots import (  # noqa: E402
    record_embedding_provider_route_contract_snapshot,
)
from app.core.embedding_provider_route_contracts import (  # noqa: E402
    check_embedding_provider_route_contract,
)
from app.core.embedding_provider_route_health_snapshots import (  # noqa: E402
    record_embedding_provider_route_health_snapshot,
)
from app.core.embedding_provider_routes import list_embedding_provider_routes  # noqa: E402


def run_preflight(
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
    results = []
    for route in routes:
        contract = check_embedding_provider_route_contract(route)
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
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run health and embedding contract preflight checks for provider routes.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--profile-name", default=None)
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive provider routes. Defaults to active routes only.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even when one or more route contracts fail.",
    )
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    payload = run_preflight(
        database_url,
        profile_name=args.profile_name,
        active_only=not args.include_inactive,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if args.allow_failures or payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
