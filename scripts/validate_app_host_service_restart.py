"""Validate app-host managed service restart readiness."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.app_host_service_restart_validation import (  # noqa: E402
    DEFAULT_EMBEDDING_WORKER_SERVICE_NAME,
    DEFAULT_EXPECTED_APP_NAME,
    DEFAULT_PIPELINE_WORKER_SERVICE_NAME,
    DEFAULT_WEB_SERVICE_NAME,
    AppHostServiceRestartValidationOptions,
    app_host_service_restart_validation_report_payload,
    build_app_host_service_restart_validation_report,
    payload_to_json,
    render_app_host_service_restart_validation_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate NeX-PCX app-host systemd service restart readiness.",
    )
    parser.add_argument("--scope", choices=("user", "system"), default="user")
    parser.add_argument("--systemctl-path", default="systemctl")
    parser.add_argument("--web-service-name", default=DEFAULT_WEB_SERVICE_NAME)
    parser.add_argument(
        "--pipeline-service-name",
        default=DEFAULT_PIPELINE_WORKER_SERVICE_NAME,
    )
    parser.add_argument(
        "--embedding-service-name",
        default=DEFAULT_EMBEDDING_WORKER_SERVICE_NAME,
    )
    parser.add_argument(
        "--service-name",
        action="append",
        default=None,
        help="Additional/override service name to inspect. Repeatable.",
    )
    parser.add_argument("--app-url", default=None)
    parser.add_argument("--expected-app-name", default=DEFAULT_EXPECTED_APP_NAME)
    parser.add_argument("--health-timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--restart-web",
        action="store_true",
        help="Restart the web service before checking app health and identity.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    service_names = tuple(
        args.service_name
        or [
            args.web_service_name,
            args.pipeline_service_name,
            args.embedding_service_name,
        ]
    )
    options = AppHostServiceRestartValidationOptions(
        scope=args.scope,
        service_names=service_names,
        web_service_name=args.web_service_name,
        app_base_url=args.app_url,
        expected_app_name=args.expected_app_name,
        restart_web=args.restart_web,
        health_timeout_seconds=args.health_timeout_seconds,
        systemctl_path=args.systemctl_path,
    )
    report = build_app_host_service_restart_validation_report(options)
    payload = app_host_service_restart_validation_report_payload(report)
    json_text = payload_to_json(payload, pretty=args.pretty)
    markdown_text = render_app_host_service_restart_validation_markdown(payload)

    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)
    if args.markdown_output:
        _write_text(Path(args.markdown_output), markdown_text)
    return 0 if report.status == "ready" else 1


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
