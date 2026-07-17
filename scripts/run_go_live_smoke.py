"""Run HTTP go-live smoke checks against a NeX_PCX app."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.go_live_smoke import (  # noqa: E402
    SMOKE_STATUS_BLOCKED,
    SMOKE_STATUS_WARNING,
    go_live_smoke_report_payload,
    render_go_live_smoke_markdown,
    run_go_live_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NeX_PCX end-to-end go-live HTTP smoke checks.",
    )
    parser.add_argument("--app-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when the smoke report has warnings.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    report = run_go_live_smoke(
        args.app_url,
        timeout_seconds=args.timeout_seconds,
    )
    payload = go_live_smoke_report_payload(report)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(Path(args.markdown_output), render_go_live_smoke_markdown(payload))

    if report.status == SMOKE_STATUS_BLOCKED:
        return 1
    if args.strict and report.status == SMOKE_STATUS_WARNING:
        return 1
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
