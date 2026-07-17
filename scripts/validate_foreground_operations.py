"""Validate foreground app-host operation readiness."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.foreground_operations_validation import (  # noqa: E402
    DEFAULT_EXPECTED_APP_NAME,
    ForegroundOperationsValidationOptions,
    build_foreground_operations_validation_report,
    foreground_operations_validation_report_payload,
    payload_to_json,
    render_foreground_operations_validation_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate NeX-PCX foreground operation mode.",
    )
    parser.add_argument("--app-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-app-name", default=DEFAULT_EXPECTED_APP_NAME)
    parser.add_argument("--health-timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--acknowledge-no-auto-restart",
        action="store_true",
        help="Accept that foreground operation has no automatic restart guarantee.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    options = ForegroundOperationsValidationOptions(
        app_base_url=args.app_url,
        expected_app_name=args.expected_app_name,
        acknowledge_no_auto_restart=args.acknowledge_no_auto_restart,
        health_timeout_seconds=args.health_timeout_seconds,
    )
    report = build_foreground_operations_validation_report(options)
    payload = foreground_operations_validation_report_payload(report)
    json_text = payload_to_json(payload, pretty=args.pretty)
    markdown_text = render_foreground_operations_validation_markdown(payload)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)
    if args.markdown_output:
        _write_text(Path(args.markdown_output), markdown_text)
    return 0 if report.status in {"ready", "warning"} else 1


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
