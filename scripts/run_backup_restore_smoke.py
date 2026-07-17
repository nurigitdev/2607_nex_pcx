"""Run NeX_PCX backup and restore smoke checks."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.backup_restore_smoke import (  # noqa: E402
    BACKUP_SMOKE_STATUS_BLOCKED,
    BACKUP_SMOKE_STATUS_WARNING,
    backup_restore_smoke_report_payload,
    build_backup_restore_smoke_report,
    render_backup_restore_smoke_markdown,
)
from app.core.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a NeX_PCX backup + restore smoke report and command manifest.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--restore-database-url", default=None)
    parser.add_argument("--backup-dir", default="artifacts/backups/latest")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when the smoke report has warnings.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.database_url:
        settings = replace(settings, database_url=args.database_url)

    report = build_backup_restore_smoke_report(
        settings,
        backup_dir=Path(args.backup_dir),
        restore_database_url=args.restore_database_url,
        project_root=Path(args.project_root),
    )
    payload = backup_restore_smoke_report_payload(report)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(Path(args.markdown_output), render_backup_restore_smoke_markdown(payload))

    if report.status == BACKUP_SMOKE_STATUS_BLOCKED:
        return 1
    if args.strict and report.status == BACKUP_SMOKE_STATUS_WARNING:
        return 1
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
