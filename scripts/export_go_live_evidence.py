"""Export go-live evidence snapshots as JSON and Markdown."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.go_live_evidence import (  # noqa: E402
    GO_LIVE_EVIDENCE_STATUS_BLOCKED,
    GO_LIVE_EVIDENCE_STATUS_WARNING,
    GoLiveEvidenceSnapshotOptions,
    build_go_live_evidence_snapshot,
    render_go_live_evidence_markdown,
)
from app.core.operations_startup_validation import OperationsStartupValidationOptions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export NeX_PCX go-live evidence snapshot.",
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
        help="Return a non-zero exit code when the snapshot has warnings.",
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

    snapshot = build_go_live_evidence_snapshot(
        settings,
        options=GoLiveEvidenceSnapshotOptions(
            startup_validation=OperationsStartupValidationOptions(
                app_base_url=args.app_url,
                run_provider_preflight=args.run_provider_preflight,
                profile_name=args.profile_name,
                include_inactive_routes=args.include_inactive_routes,
                health_timeout_seconds=args.health_timeout_seconds,
            )
        ),
        project_root=PROJECT_ROOT,
    )
    json_text = json.dumps(
        snapshot,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
    )
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(Path(args.markdown_output), render_go_live_evidence_markdown(snapshot))

    if snapshot["status"] == GO_LIVE_EVIDENCE_STATUS_BLOCKED:
        return 1
    if args.strict and snapshot["status"] == GO_LIVE_EVIDENCE_STATUS_WARNING:
        return 1
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
