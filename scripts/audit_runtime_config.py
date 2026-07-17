"""Audit NeX_PCX runtime configuration from the command line."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.runtime_config_audit import (  # noqa: E402
    CONFIG_AUDIT_STATUS_BLOCKED,
    CONFIG_AUDIT_STATUS_WARNING,
    build_runtime_config_audit_report,
    render_runtime_config_audit_markdown,
    runtime_config_audit_report_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit NeX_PCX runtime configuration for operations.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--test-database-url", default=None)
    parser.add_argument("--environment", default=None)
    parser.add_argument("--upload-storage-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--embedding-provider-mode", default=None)
    parser.add_argument("--remote-provider-url", default=None)
    parser.add_argument(
        "--require-route-readiness",
        dest="require_route_readiness",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-require-route-readiness",
        dest="require_route_readiness",
        action="store_false",
    )
    parser.add_argument("--route-readiness-failure-mode", default=None)
    parser.add_argument("--route-readiness-defer-seconds", type=int, default=None)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when the audit has warnings.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    settings = _settings_from_args(args)
    report = build_runtime_config_audit_report(
        settings,
        project_root=Path(args.project_root),
    )
    payload = runtime_config_audit_report_payload(report)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(Path(args.markdown_output), render_runtime_config_audit_markdown(payload))

    if report.status == CONFIG_AUDIT_STATUS_BLOCKED:
        return 1
    if args.strict and report.status == CONFIG_AUDIT_STATUS_WARNING:
        return 1
    return 0


def _settings_from_args(args: argparse.Namespace):
    settings = get_settings()
    updates = {}
    if args.database_url is not None:
        updates["database_url"] = args.database_url
    if args.test_database_url is not None:
        updates["test_database_url"] = args.test_database_url
    if args.environment is not None:
        updates["environment"] = args.environment
    if args.upload_storage_dir is not None:
        updates["upload_storage_dir"] = Path(args.upload_storage_dir)
    if args.models_dir is not None:
        updates["embedding_models_dir"] = Path(args.models_dir)
    if args.embedding_provider_mode is not None:
        updates["embedding_provider_mode"] = args.embedding_provider_mode
    if args.remote_provider_url is not None:
        updates["remote_embedding_provider_url"] = args.remote_provider_url
    if args.require_route_readiness is not None:
        updates["embedding_require_route_readiness"] = args.require_route_readiness
    if args.route_readiness_failure_mode is not None:
        updates["embedding_route_readiness_failure_mode"] = args.route_readiness_failure_mode
    if args.route_readiness_defer_seconds is not None:
        updates["embedding_route_readiness_defer_seconds"] = (
            args.route_readiness_defer_seconds
        )
    return replace(settings, **updates) if updates else settings


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
