"""Register embedding provider routes from local launch presets."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_provider_presets import (  # noqa: E402
    EmbeddingProviderPreset,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from app.core.embedding_provider_routes import (  # noqa: E402
    EmbeddingProviderRouteInput,
    EmbeddingProviderRouteRecord,
    upsert_embedding_provider_route,
)


@dataclass(frozen=True)
class EmbeddingProviderRoutePlan:
    preset_name: str
    profile_name: str
    provider_name: str
    provider_mode: str
    provider_base_url: str
    provider_port: int | None
    timeout_seconds: float
    priority: int
    is_active: bool
    health_check_enabled: bool
    runtime_metadata: dict[str, object]


def select_presets(provider: str) -> tuple[EmbeddingProviderPreset, ...]:
    if provider == "all":
        return list_embedding_provider_presets()
    return (get_embedding_provider_preset(provider),)


def build_route_plans(
    presets: tuple[EmbeddingProviderPreset, ...],
    *,
    host: str | None = None,
    port: int | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
    priority: int = 100,
    is_active: bool = True,
    health_check_enabled: bool = True,
    provider_name: str | None = None,
) -> tuple[EmbeddingProviderRoutePlan, ...]:
    if base_url is not None and len(presets) != 1:
        raise ValueError("--base-url can only be used with a single provider preset")
    if port is not None and len(presets) != 1:
        raise ValueError("--port can only be used with a single provider preset")
    if base_url is not None and port is not None:
        raise ValueError("--base-url and --port cannot be used together")
    if provider_name is not None and len(presets) != 1:
        raise ValueError("--provider-name can only be used with a single provider preset")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    if priority < 0:
        raise ValueError("priority must be greater than or equal to 0")

    plans = []
    for preset in presets:
        selected_host = host or preset.default_host
        selected_port = port or preset.default_port
        selected_base_url = base_url or f"http://{selected_host}:{selected_port}"
        provider_port = urlparse(selected_base_url).port or selected_port
        selected_provider_name = provider_name or preset.provider_name
        for profile_name in preset.profile_names:
            plans.append(
                EmbeddingProviderRoutePlan(
                    preset_name=preset.preset_name,
                    profile_name=profile_name,
                    provider_name=selected_provider_name,
                    provider_mode="remote",
                    provider_base_url=selected_base_url.rstrip("/"),
                    provider_port=provider_port,
                    timeout_seconds=timeout_seconds,
                    priority=priority,
                    is_active=is_active,
                    health_check_enabled=health_check_enabled,
                    runtime_metadata={
                        "preset_name": preset.preset_name,
                        "backend": preset.backend,
                        "model_key": preset.model_key,
                        "provider_model_id": preset.provider_model_id,
                        "default_port": preset.default_port,
                        "script": "register_embedding_provider_routes.py",
                    },
                )
            )
    return tuple(plans)


def register_route_plans(
    database_url: str,
    plans: tuple[EmbeddingProviderRoutePlan, ...],
) -> tuple[EmbeddingProviderRouteRecord, ...]:
    records = []
    for plan in plans:
        records.append(
            upsert_embedding_provider_route(
                database_url,
                EmbeddingProviderRouteInput(
                    profile_name=plan.profile_name,
                    provider_name=plan.provider_name,
                    provider_mode=plan.provider_mode,
                    provider_base_url=plan.provider_base_url,
                    timeout_seconds=plan.timeout_seconds,
                    priority=plan.priority,
                    is_active=plan.is_active,
                    health_check_enabled=plan.health_check_enabled,
                    runtime_metadata=plan.runtime_metadata,
                ),
            )
        )
    return tuple(records)


def _plan_payload(plan: EmbeddingProviderRoutePlan) -> dict[str, object]:
    return asdict(plan)


def _record_payload(record: EmbeddingProviderRouteRecord) -> dict[str, object]:
    return {
        "route_id": record.route_id,
        "profile_name": record.profile_name,
        "provider_name": record.provider_name,
        "provider_mode": record.provider_mode,
        "provider_base_url": record.provider_base_url,
        "timeout_seconds": record.timeout_seconds,
        "priority": record.priority,
        "is_active": record.is_active,
        "health_check_enabled": record.health_check_enabled,
        "runtime_metadata": record.runtime_metadata,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


def _print_human_plan(plans: tuple[EmbeddingProviderRoutePlan, ...], *, dry_run: bool) -> None:
    action = "Dry-run route registration plan" if dry_run else "Registered provider routes"
    print(action)
    for plan in plans:
        print(
            "- "
            f"{plan.profile_name}: {plan.provider_name} "
            f"{plan.provider_base_url} port={plan.provider_port or '-'} "
            f"priority={plan.priority} active={plan.is_active}"
        )


def main() -> int:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description="Register remote embedding provider routes from NeX_PCX provider presets.",
    )
    parser.add_argument("--provider", choices=[*preset_names, "all"], required=True)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--provider-name", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--inactive", action="store_true")
    parser.add_argument("--disable-health-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        plans = build_route_plans(
            select_presets(args.provider),
            host=args.host,
            port=args.port,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            priority=args.priority,
            is_active=not args.inactive,
            health_check_enabled=not args.disable_health_check,
            provider_name=args.provider_name,
        )
    except ValueError as exc:
        parser.error(str(exc))

    database_url = args.database_url or get_settings().database_url
    if not args.dry_run and not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required unless --dry-run is used")

    records: tuple[EmbeddingProviderRouteRecord, ...] = ()
    if not args.dry_run:
        records = register_route_plans(database_url or "", plans)

    if args.json:
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "route_count": len(plans),
                    "routes": [_plan_payload(plan) for plan in plans],
                    "registered_routes": [_record_payload(record) for record in records],
                },
                ensure_ascii=False,
            )
        )
    else:
        _print_human_plan(plans, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
