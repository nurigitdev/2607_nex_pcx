"""Verify operational retention settings and cleanup dry-run previews."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.operational_retention_verification import (  # noqa: E402
    RETENTION_STATUS_BLOCKED,
    RETENTION_STATUS_WARNING,
    build_operational_retention_verification_report,
    operational_retention_verification_report_payload,
    render_operational_retention_verification_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify NeX_PCX operational retention and cleanup settings.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-retention-days", type=int, default=90)
    parser.add_argument("--artifact-retention-days", type=int, default=30)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when the verification report has warnings.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if args.database_url:
        settings = replace(settings, database_url=args.database_url)

    report = build_operational_retention_verification_report(
        settings.database_url or database_url,
        project_root=Path(args.project_root),
        max_retention_days=args.max_retention_days,
        artifact_retention_days=args.artifact_retention_days,
    )
    payload = operational_retention_verification_report_payload(report)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(
            Path(args.markdown_output),
            render_operational_retention_verification_markdown(payload),
        )

    if report.status == RETENTION_STATUS_BLOCKED:
        return 1
    if args.strict and report.status == RETENTION_STATUS_WARNING:
        return 1
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
