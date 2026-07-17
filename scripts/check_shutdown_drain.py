"""Check whether NeX_PCX queues are drained before shutdown."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.shutdown_drain_check import (  # noqa: E402
    DRAIN_STATUS_BLOCKED,
    DRAIN_STATUS_WARNING,
    build_shutdown_drain_report,
    render_shutdown_drain_markdown,
    shutdown_drain_report_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check NeX_PCX queue drain status before planned shutdown.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path. Prints JSON to stdout when omitted.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional Markdown output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when the drain report has warnings.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.database_url:
        settings = replace(settings, database_url=args.database_url)

    report = build_shutdown_drain_report(settings)
    payload = shutdown_drain_report_payload(report)
    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
    )
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(Path(args.markdown_output), render_shutdown_drain_markdown(payload))

    if report.status == DRAIN_STATUS_BLOCKED:
        return 1
    if args.strict and report.status == DRAIN_STATUS_WARNING:
        return 1
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
