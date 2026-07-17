"""Validate the operations startup checklist from the command line."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.operations_startup_validation import (  # noqa: E402
    STARTUP_STATUS_BLOCKED,
    STARTUP_STATUS_WARNING,
    OperationsStartupValidationOptions,
    build_operations_startup_validation_report,
    operations_startup_validation_report_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate NeX_PCX startup readiness from the operations runbook.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--app-url",
        default=None,
        help="Base URL for the running NeX_PCX app, for example http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=float,
        default=5.0,
        help="Timeout for the optional application /healthz check.",
    )
    parser.add_argument(
        "--run-provider-preflight",
        action="store_true",
        help="Run provider route contract preflight and persist snapshots.",
    )
    parser.add_argument("--profile-name", default=None)
    parser.add_argument(
        "--include-inactive-routes",
        action="store_true",
        help="Include inactive provider routes when running provider preflight.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when the report has warnings.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON report.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.database_url:
        settings = replace(settings, database_url=args.database_url)

    report = build_operations_startup_validation_report(
        settings,
        options=OperationsStartupValidationOptions(
            app_base_url=args.app_url,
            run_provider_preflight=args.run_provider_preflight,
            profile_name=args.profile_name,
            include_inactive_routes=args.include_inactive_routes,
            health_timeout_seconds=args.health_timeout_seconds,
        ),
    )
    payload = operations_startup_validation_report_payload(report)
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )
    if report.status == STARTUP_STATUS_BLOCKED:
        return 1
    if args.strict and report.status == STARTUP_STATUS_WARNING:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
