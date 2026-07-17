"""Validate final NeX_PCX production environment readiness."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.production_environment_validation import (  # noqa: E402
    PRODUCTION_STATUS_BLOCKED,
    PRODUCTION_STATUS_WARNING,
    ProductionValidationOptions,
    build_production_environment_validation_report,
    payload_to_json,
    production_environment_validation_payload,
    render_production_environment_validation_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run final NeX_PCX production environment validation.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--app-url", default=None)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--expected-database-name", default=None)
    parser.add_argument("--health-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--run-provider-preflight", action="store_true")
    parser.add_argument("--profile-name", default=None)
    parser.add_argument("--include-inactive-routes", action="store_true")
    parser.add_argument(
        "--allow-non-production",
        action="store_true",
        help="Warn instead of blocking when NEX_PCX_ENV is not production.",
    )
    parser.add_argument(
        "--allow-non-remote-provider",
        action="store_true",
        help="Warn instead of blocking when provider mode is not remote.",
    )
    parser.add_argument(
        "--allow-route-readiness-disabled",
        action="store_true",
        help="Warn instead of blocking when route readiness is disabled.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.database_url:
        settings = replace(settings, database_url=args.database_url)

    report = build_production_environment_validation_report(
        settings,
        project_root=Path(args.project_root),
        options=ProductionValidationOptions(
            app_url=args.app_url,
            expected_database_name=args.expected_database_name,
            require_production_env=not args.allow_non_production,
            require_remote_provider=not args.allow_non_remote_provider,
            require_route_readiness=not args.allow_route_readiness_disabled,
            run_provider_preflight=args.run_provider_preflight,
            profile_name=args.profile_name,
            include_inactive_routes=args.include_inactive_routes,
            health_timeout_seconds=args.health_timeout_seconds,
        ),
    )
    payload = production_environment_validation_payload(report)
    json_text = payload_to_json(payload, pretty=args.pretty)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(
            Path(args.markdown_output),
            render_production_environment_validation_markdown(payload),
        )

    if report.status == PRODUCTION_STATUS_BLOCKED:
        return 1
    if args.strict and report.status == PRODUCTION_STATUS_WARNING:
        return 1
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
