"""Go-live evidence snapshot export."""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app.core.config import Settings
from app.core.go_live_readiness import (
    GO_LIVE_STATUS_BLOCKED,
    GO_LIVE_STATUS_READY,
    build_go_live_readiness_report,
    go_live_readiness_report_payload,
)
from app.core.operations_startup_validation import (
    STARTUP_STATUS_BLOCKED,
    STARTUP_STATUS_READY,
    OperationsStartupValidationOptions,
    build_operations_startup_validation_report,
    operations_startup_validation_report_payload,
)

GO_LIVE_EVIDENCE_VERSION = 1
GO_LIVE_EVIDENCE_STATUS_READY = "ready"
GO_LIVE_EVIDENCE_STATUS_WARNING = "warning"
GO_LIVE_EVIDENCE_STATUS_BLOCKED = "blocked"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GoLiveEvidenceSnapshotOptions:
    startup_validation: OperationsStartupValidationOptions = field(
        default_factory=OperationsStartupValidationOptions
    )


def build_go_live_evidence_snapshot(
    settings: Settings,
    *,
    options: GoLiveEvidenceSnapshotOptions | None = None,
    generated_at: datetime | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    selected_options = options or GoLiveEvidenceSnapshotOptions()
    exported_at = generated_at or datetime.now(UTC)
    startup_report = build_operations_startup_validation_report(
        settings,
        options=selected_options.startup_validation,
        checked_at=exported_at,
    )
    go_live_report = build_go_live_readiness_report(settings, checked_at=exported_at)
    startup_payload = operations_startup_validation_report_payload(startup_report)
    go_live_payload = go_live_readiness_report_payload(go_live_report)
    status = _combined_status(
        startup_status=str(startup_payload["status"]),
        go_live_status=str(go_live_payload["status"]),
    )
    return {
        "version": GO_LIVE_EVIDENCE_VERSION,
        "status": status,
        "exported_at": exported_at.isoformat(),
        "exported_at_label": exported_at.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": _runtime_payload(settings),
        "provenance": _provenance_payload(project_root or PROJECT_ROOT),
        "summary": {
            "startup_validation_status": startup_payload["status"],
            "go_live_readiness_status": go_live_payload["status"],
            "startup_validation_failed_count": startup_payload["failed_count"],
            "go_live_readiness_failed_count": go_live_payload["failed_count"],
            "startup_validation_warning_count": startup_payload["warning_count"],
            "go_live_readiness_warning_count": go_live_payload["warning_count"],
        },
        "startup_validation": startup_payload,
        "go_live_readiness": go_live_payload,
    }


def render_go_live_evidence_markdown(snapshot: dict[str, object]) -> str:
    runtime = _dict(snapshot.get("runtime"))
    provenance = _dict(snapshot.get("provenance"))
    startup = _dict(snapshot.get("startup_validation"))
    go_live = _dict(snapshot.get("go_live_readiness"))
    lines = [
        "# NeX_PCX Go-Live Evidence Snapshot",
        "",
        f"- Exported At: {_text(snapshot.get('exported_at_label'))}",
        f"- Overall Status: {_text(snapshot.get('status'))}",
        f"- App: {_text(runtime.get('app_name'))} v{_text(runtime.get('app_version'))}",
        f"- Environment: {_text(runtime.get('environment'))}",
        f"- Git Commit: {_text(provenance.get('git_commit'))}",
        f"- Git Branch: {_text(provenance.get('git_branch'))}",
        f"- Git Dirty: {_text(provenance.get('git_dirty'))}",
        f"- Database URL: {_text(runtime.get('database_url_masked'))}",
        "",
        "## Summary",
        "",
        "| Signal | Status | Failed | Warning |",
        "| --- | --- | ---: | ---: |",
        (
            "| Startup Validation | "
            f"{_md_cell(startup.get('status'))} | "
            f"{_md_cell(startup.get('failed_count'))} | "
            f"{_md_cell(startup.get('warning_count'))} |"
        ),
        (
            "| Go-Live Readiness | "
            f"{_md_cell(go_live.get('status'))} | "
            f"{_md_cell(go_live.get('failed_count'))} | "
            f"{_md_cell(go_live.get('warning_count'))} |"
        ),
        "",
        "## Startup Validation Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in startup.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} |"
        )

    lines.extend(
        [
            "",
            "## Go-Live Readiness Checks",
            "",
            "| Section | Check | Status | Detail |",
            "| --- | --- | --- | --- |",
        ]
    )
    for section in go_live.get("sections", []):
        section_payload = _dict(section)
        section_code = section_payload.get("code")
        for check in section_payload.get("checks", []):
            check_payload = _dict(check)
            lines.append(
                "| "
                f"{_md_cell(section_code)} | "
                f"{_md_cell(check_payload.get('code'))} | "
                f"{_md_cell(check_payload.get('status'))} | "
                f"{_md_cell(check_payload.get('detail'))} |"
            )

    lines.extend(
        [
            "",
            "## Runtime Configuration",
            "",
            "```json",
            json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Provenance",
            "",
            "```json",
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _combined_status(*, startup_status: str, go_live_status: str) -> str:
    if startup_status == STARTUP_STATUS_BLOCKED or go_live_status == GO_LIVE_STATUS_BLOCKED:
        return GO_LIVE_EVIDENCE_STATUS_BLOCKED
    if startup_status == STARTUP_STATUS_READY and go_live_status == GO_LIVE_STATUS_READY:
        return GO_LIVE_EVIDENCE_STATUS_READY
    return GO_LIVE_EVIDENCE_STATUS_WARNING


def _runtime_payload(settings: Settings) -> dict[str, object]:
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "database_url_configured": bool(settings.database_url),
        "database_url_masked": _mask_database_url(settings.database_url),
        "upload_storage_dir": str(settings.upload_storage_dir),
        "embedding_models_dir": str(settings.embedding_models_dir),
        "embedding_provider_mode": settings.embedding_provider_mode,
        "remote_embedding_provider_url": settings.remote_embedding_provider_url,
        "embedding_require_route_readiness": settings.embedding_require_route_readiness,
        "embedding_route_readiness_failure_mode": settings.embedding_route_readiness_failure_mode,
        "embedding_route_readiness_defer_seconds": settings.embedding_route_readiness_defer_seconds,
    }


def _mask_database_url(database_url: str | None) -> str | None:
    if not database_url:
        return None
    try:
        parsed = urlsplit(database_url)
        if not parsed.scheme or not parsed.netloc:
            return "***"
        username = parsed.username or ""
        userinfo = f"{username}:***@" if username else "***@"
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme, f"{userinfo}{host}{port}", parsed.path, "", ""))
    except ValueError:
        return "***"


def _provenance_payload(project_root: Path) -> dict[str, object]:
    git_status = _git_output(project_root, "status", "--porcelain")
    return {
        "project_root": str(project_root),
        "git_commit": _git_output(project_root, "rev-parse", "--short", "HEAD"),
        "git_branch": _git_output(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status.strip()),
    }


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
