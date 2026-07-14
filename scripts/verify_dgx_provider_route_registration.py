"""Verify DGX provider route registration in a NeX_PCX database."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_provider_presets import (  # noqa: E402
    InvalidEmbeddingProviderPresetError,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from app.core.embedding_provider_routes import (  # noqa: E402
    EmbeddingProviderRouteRecord,
    list_embedding_provider_routes,
)
from scripts.plan_remote_provider_foreground_smoke import DEFAULT_GPU_HOST  # noqa: E402
from scripts.register_embedding_provider_routes import (  # noqa: E402
    EmbeddingProviderRoutePlan,
    build_route_plans,
    register_route_plans,
    select_presets,
)

DEFAULT_PROVIDER_ORDER = ("kure", "bge", "qwen")
DEFAULT_PROVIDER_TIMEOUT_SECONDS = {
    "kure": 120.0,
    "bge": 120.0,
    "qwen": 300.0,
}


@dataclass(frozen=True)
class DgxProviderRouteVerificationResult:
    profile_name: str
    provider_name: str
    expected_route: EmbeddingProviderRoutePlan
    actual_route: EmbeddingProviderRouteRecord | None
    mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.actual_route is not None and not self.mismatches


@dataclass(frozen=True)
class DgxProviderRouteVerificationReport:
    expected_route_count: int
    verified_count: int
    missing_count: int
    mismatched_count: int
    applied: bool
    results: tuple[DgxProviderRouteVerificationResult, ...]

    @property
    def passed(self) -> bool:
        return self.expected_route_count > 0 and all(result.passed for result in self.results)


def build_dgx_route_plans(
    provider_names: tuple[str, ...] | None = None,
    *,
    host: str = DEFAULT_GPU_HOST,
    timeout_seconds: float | None = None,
    qwen_timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS["qwen"],
    priority: int = 100,
    is_active: bool = True,
    health_check_enabled: bool = True,
) -> tuple[EmbeddingProviderRoutePlan, ...]:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    if qwen_timeout_seconds <= 0:
        raise ValueError("qwen_timeout_seconds must be greater than 0")
    if priority < 0:
        raise ValueError("priority must be greater than or equal to 0")

    plans: list[EmbeddingProviderRoutePlan] = []
    for provider_name in _normalize_provider_names(provider_names):
        provider_timeout = timeout_seconds or (
            qwen_timeout_seconds
            if provider_name == "qwen"
            else DEFAULT_PROVIDER_TIMEOUT_SECONDS[provider_name]
        )
        plans.extend(
            build_route_plans(
                select_presets(provider_name),
                host=host,
                timeout_seconds=provider_timeout,
                priority=priority,
                is_active=is_active,
                health_check_enabled=health_check_enabled,
            )
        )
    return tuple(plans)


def verify_dgx_route_registration(
    database_url: str,
    expected_routes: tuple[EmbeddingProviderRoutePlan, ...],
    *,
    apply: bool = False,
) -> DgxProviderRouteVerificationReport:
    if apply:
        register_route_plans(database_url, expected_routes)

    actual_routes = list_embedding_provider_routes(database_url, active_only=False)
    actual_by_key = {(route.profile_name, route.provider_name): route for route in actual_routes}
    results = tuple(
        _verify_route(expected, actual_by_key.get((expected.profile_name, expected.provider_name)))
        for expected in expected_routes
    )
    missing_count = sum(1 for result in results if result.actual_route is None)
    mismatched_count = sum(
        1 for result in results if result.actual_route is not None and result.mismatches
    )
    verified_count = sum(1 for result in results if result.passed)
    return DgxProviderRouteVerificationReport(
        expected_route_count=len(expected_routes),
        verified_count=verified_count,
        missing_count=missing_count,
        mismatched_count=mismatched_count,
        applied=apply,
        results=results,
    )


def _verify_route(
    expected: EmbeddingProviderRoutePlan,
    actual: EmbeddingProviderRouteRecord | None,
) -> DgxProviderRouteVerificationResult:
    if actual is None:
        return DgxProviderRouteVerificationResult(
            profile_name=expected.profile_name,
            provider_name=expected.provider_name,
            expected_route=expected,
            actual_route=None,
            mismatches=("route is missing",),
        )

    mismatches: list[str] = []
    expected_values = {
        "provider_mode": expected.provider_mode,
        "provider_base_url": expected.provider_base_url,
        "timeout_seconds": expected.timeout_seconds,
        "priority": expected.priority,
        "is_active": expected.is_active,
        "health_check_enabled": expected.health_check_enabled,
    }
    actual_values = {
        "provider_mode": actual.provider_mode,
        "provider_base_url": actual.provider_base_url,
        "timeout_seconds": actual.timeout_seconds,
        "priority": actual.priority,
        "is_active": actual.is_active,
        "health_check_enabled": actual.health_check_enabled,
    }
    for field_name, expected_value in expected_values.items():
        actual_value = actual_values[field_name]
        if actual_value != expected_value:
            mismatches.append(f"{field_name}: expected {expected_value!r}, got {actual_value!r}")

    for metadata_key, expected_value in expected.runtime_metadata.items():
        actual_value = actual.runtime_metadata.get(metadata_key)
        if actual_value != expected_value:
            mismatches.append(
                "runtime_metadata."
                f"{metadata_key}: expected {expected_value!r}, got {actual_value!r}"
            )

    return DgxProviderRouteVerificationResult(
        profile_name=expected.profile_name,
        provider_name=expected.provider_name,
        expected_route=expected,
        actual_route=actual,
        mismatches=tuple(mismatches),
    )


def _normalize_provider_names(provider_names: tuple[str, ...] | None) -> tuple[str, ...]:
    selected_names = provider_names or DEFAULT_PROVIDER_ORDER
    normalized_names = tuple(name.strip().lower() for name in selected_names if name.strip())
    if not normalized_names:
        raise InvalidEmbeddingProviderPresetError("At least one provider is required")
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in normalized_names:
        get_embedding_provider_preset(name)
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    return tuple(unique_names)


def _route_plan_payload(plan: EmbeddingProviderRoutePlan) -> dict[str, object]:
    return asdict(plan)


def _route_record_payload(record: EmbeddingProviderRouteRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
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
        "created_at": _isoformat(record.created_at),
        "updated_at": _isoformat(record.updated_at),
    }


def _result_payload(result: DgxProviderRouteVerificationResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "profile_name": result.profile_name,
        "provider_name": result.provider_name,
        "expected_route": _route_plan_payload(result.expected_route),
        "actual_route": _route_record_payload(result.actual_route),
        "mismatches": list(result.mismatches),
    }


def _report_payload(report: DgxProviderRouteVerificationReport) -> dict[str, object]:
    return {
        "passed": report.passed,
        "expected_route_count": report.expected_route_count,
        "verified_count": report.verified_count,
        "missing_count": report.missing_count,
        "mismatched_count": report.mismatched_count,
        "applied": report.applied,
        "results": [_result_payload(result) for result in report.results],
    }


def _isoformat(value: datetime) -> str:
    return value.isoformat()


def _print_human_dry_run(plans: tuple[EmbeddingProviderRoutePlan, ...]) -> None:
    print("DGX provider route registration dry-run")
    for plan in plans:
        print(
            "- "
            f"{plan.profile_name}: {plan.provider_name} "
            f"{plan.provider_base_url} timeout={plan.timeout_seconds:g}s "
            f"active={plan.is_active}"
        )


def _print_human_report(report: DgxProviderRouteVerificationReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"DGX provider route registration verification: {status}")
    print(f"- applied: {report.applied}")
    print(f"- expected_route_count: {report.expected_route_count}")
    print(f"- verified_count: {report.verified_count}")
    print(f"- missing_count: {report.missing_count}")
    print(f"- mismatched_count: {report.mismatched_count}")
    for result in report.results:
        print("- " f"{result.profile_name}/{result.provider_name}: " f"passed={result.passed}")
        for mismatch in result.mismatches:
            print(f"  - {mismatch}")


def _build_arg_parser() -> argparse.ArgumentParser:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description="Verify DGX remote embedding provider routes in a NeX_PCX database.",
    )
    parser.add_argument("--provider", choices=preset_names, action="append", default=[])
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--qwen-timeout-seconds",
        type=float,
        default=DEFAULT_PROVIDER_TIMEOUT_SECONDS["qwen"],
    )
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--inactive", action="store_true")
    parser.add_argument("--disable-health-check", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upsert the expected DGX routes before verifying them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        plans = build_dgx_route_plans(
            provider_names=tuple(args.provider) or None,
            host=args.host,
            timeout_seconds=args.timeout_seconds,
            qwen_timeout_seconds=args.qwen_timeout_seconds,
            priority=args.priority,
            is_active=not args.inactive,
            health_check_enabled=not args.disable_health_check,
        )
    except (InvalidEmbeddingProviderPresetError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "routes": [_route_plan_payload(plan) for plan in plans]}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            _print_human_dry_run(plans)
        return 0

    database_url = args.database_url or get_settings().database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required unless --dry-run is used")

    report = verify_dgx_route_registration(database_url, plans, apply=args.apply)
    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
