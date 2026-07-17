"""Operational retention and cleanup verification for go-live operations."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.admin_logging import LogSettings, load_log_settings
from app.core.database import connect
from app.core.embedding_provider_route_retention import (
    ProviderRouteRetentionSettings,
    cleanup_expired_provider_route_records,
    load_provider_route_retention_settings,
)
from app.core.embedding_worker_batch_run_retention import (
    EmbeddingBatchRunRetentionSettings,
    cleanup_expired_embedding_batch_run_records,
    load_embedding_batch_run_retention_settings,
)
from app.core.search_logs import (
    SearchLogRetentionSettings,
    cleanup_expired_search_logs,
    load_search_log_retention_settings,
)

RETENTION_VERIFICATION_VERSION = 1

RETENTION_CHECK_PASSED = "passed"
RETENTION_CHECK_WARNING = "warning"
RETENTION_CHECK_FAILED = "failed"
RETENTION_CHECK_SKIPPED = "skipped"

RETENTION_STATUS_READY = "ready"
RETENTION_STATUS_WARNING = "warning"
RETENTION_STATUS_BLOCKED = "blocked"

DEFAULT_MAX_OPERATIONAL_RETENTION_DAYS = 90
DEFAULT_ARTIFACT_RETENTION_DAYS = 30


@dataclass(frozen=True)
class OperationalRetentionCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class OperationalRetentionVerificationReport:
    status: str
    checked_at: datetime
    project_root: str
    checks: tuple[OperationalRetentionCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(RETENTION_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(RETENTION_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(RETENTION_CHECK_FAILED)

    @property
    def skipped_count(self) -> int:
        return self._count_status(RETENTION_CHECK_SKIPPED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


def build_operational_retention_verification_report(
    database_url: str | None,
    *,
    project_root: Path | None = None,
    max_retention_days: int = DEFAULT_MAX_OPERATIONAL_RETENTION_DAYS,
    artifact_retention_days: int = DEFAULT_ARTIFACT_RETENTION_DAYS,
    checked_at: datetime | None = None,
) -> OperationalRetentionVerificationReport:
    root = project_root or Path.cwd()
    checks: list[OperationalRetentionCheck] = [_database_url_check(database_url)]
    database_ready = False

    if database_url:
        connectivity_check = _database_connectivity_check(database_url)
        database_ready = connectivity_check.status == RETENTION_CHECK_PASSED
        checks.append(connectivity_check)
    else:
        checks.append(
            OperationalRetentionCheck(
                code="database_connectivity",
                status=RETENTION_CHECK_SKIPPED,
                detail="Database connectivity is skipped until NEX_PCX_DATABASE_URL is configured.",
            )
        )

    checks.extend(
        _database_backed_checks(
            database_url,
            database_ready=database_ready,
            max_retention_days=max_retention_days,
        )
    )
    checks.append(
        _artifacts_directory_check(
            root / "artifacts",
            retention_days=artifact_retention_days,
            checked_at=checked_at,
        )
    )

    return OperationalRetentionVerificationReport(
        status=_overall_status(tuple(checks)),
        checked_at=checked_at or datetime.now(UTC),
        project_root=str(root),
        checks=tuple(checks),
    )


def operational_retention_verification_report_payload(
    report: OperationalRetentionVerificationReport,
) -> dict[str, object]:
    return {
        "version": RETENTION_VERIFICATION_VERSION,
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": report.project_root,
        "check_count": report.check_count,
        "passed_count": report.passed_count,
        "warning_count": report.warning_count,
        "failed_count": report.failed_count,
        "skipped_count": report.skipped_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "metadata": dict(check.metadata or {}),
            }
            for check in report.checks
        ],
    }


def render_operational_retention_verification_markdown(
    payload: dict[str, object],
) -> str:
    lines = [
        "# NeX_PCX Operational Retention Verification Report",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Overall Status: {_text(payload.get('status'))}",
        f"- Project Root: `{_text(payload.get('project_root'))}`",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
        f"- Skipped: {_text(payload.get('skipped_count'))}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} |"
        )

    lines.extend(["", "## Cleanup Previews", ""])
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        metadata = _dict(check_payload.get("metadata"))
        if not metadata:
            continue
        lines.extend(
            [
                f"### {_text(check_payload.get('code'))}",
                "",
                "```json",
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def load_admin_log_retention_settings(database_url: str) -> LogSettings:
    with connect(database_url) as connection:
        return load_log_settings(connection)


def _database_url_check(database_url: str | None) -> OperationalRetentionCheck:
    if database_url:
        return OperationalRetentionCheck(
            code="database_url",
            status=RETENTION_CHECK_PASSED,
            detail="NEX_PCX_DATABASE_URL is configured.",
            metadata={"configured": True},
        )
    return OperationalRetentionCheck(
        code="database_url",
        status=RETENTION_CHECK_FAILED,
        detail="NEX_PCX_DATABASE_URL is required for retention verification.",
        metadata={"configured": False},
    )


def _database_connectivity_check(database_url: str) -> OperationalRetentionCheck:
    try:
        with connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 AS ok")
                cursor.fetchone()
    except Exception as exc:
        return OperationalRetentionCheck(
            code="database_connectivity",
            status=RETENTION_CHECK_FAILED,
            detail=f"Database connection failed: {exc}",
            metadata={"error": str(exc)},
        )
    return OperationalRetentionCheck(
        code="database_connectivity",
        status=RETENTION_CHECK_PASSED,
        detail="Database connection succeeded.",
    )


def _database_backed_checks(
    database_url: str | None,
    *,
    database_ready: bool,
    max_retention_days: int,
) -> tuple[OperationalRetentionCheck, ...]:
    if not database_url or not database_ready:
        detail = (
            "Retention setting check is skipped until NEX_PCX_DATABASE_URL is configured."
            if not database_url
            else "Retention setting check is skipped because database connectivity failed."
        )
        return tuple(
            OperationalRetentionCheck(code=code, status=RETENTION_CHECK_SKIPPED, detail=detail)
            for code in (
                "admin_log_retention",
                "search_log_cleanup_preview",
                "provider_route_cleanup_preview",
                "embedding_batch_run_cleanup_preview",
            )
        )

    return (
        _admin_log_retention_check(database_url, max_retention_days=max_retention_days),
        _search_log_cleanup_check(database_url, max_retention_days=max_retention_days),
        _provider_route_cleanup_check(database_url, max_retention_days=max_retention_days),
        _embedding_batch_run_cleanup_check(
            database_url,
            max_retention_days=max_retention_days,
        ),
    )


def _admin_log_retention_check(
    database_url: str,
    *,
    max_retention_days: int,
) -> OperationalRetentionCheck:
    try:
        settings = load_admin_log_retention_settings(database_url)
    except Exception as exc:
        return OperationalRetentionCheck(
            code="admin_log_retention",
            status=RETENTION_CHECK_FAILED,
            detail=f"Admin log retention settings could not be loaded: {exc}",
            metadata={"error": str(exc)},
        )

    metadata = {
        "enabled": settings.enabled,
        "min_level": settings.min_level,
        "retention_days": settings.retention_days,
        "page_size": settings.page_size,
        "max_recommended_retention_days": max_retention_days,
    }
    status, detail = _settings_status_detail(
        label="Admin log retention",
        enabled=settings.enabled,
        retention_days=settings.retention_days,
        max_retention_days=max_retention_days,
    )
    return OperationalRetentionCheck(
        code="admin_log_retention",
        status=status,
        detail=detail,
        metadata=metadata,
    )


def _search_log_cleanup_check(
    database_url: str,
    *,
    max_retention_days: int,
) -> OperationalRetentionCheck:
    try:
        settings = load_search_log_retention_settings(database_url)
        preview = cleanup_expired_search_logs(database_url, dry_run=True)
    except Exception as exc:
        return OperationalRetentionCheck(
            code="search_log_cleanup_preview",
            status=RETENTION_CHECK_FAILED,
            detail=f"Search log cleanup preview failed: {exc}",
            metadata={"error": str(exc)},
        )

    metadata = {
        **_retention_settings_metadata(settings),
        "expired_count": preview.expired_count,
        "deleted_count": preview.deleted_count,
        "cutoff_at": preview.cutoff_at.isoformat(),
        "dry_run": preview.dry_run,
        "max_recommended_retention_days": max_retention_days,
    }
    status, detail = _cleanup_status_detail(
        label="Search log cleanup",
        enabled=settings.enabled,
        retention_days=settings.retention_days,
        expired_count=preview.expired_count,
        max_retention_days=max_retention_days,
    )
    return OperationalRetentionCheck(
        code="search_log_cleanup_preview",
        status=status,
        detail=detail,
        metadata=metadata,
    )


def _provider_route_cleanup_check(
    database_url: str,
    *,
    max_retention_days: int,
) -> OperationalRetentionCheck:
    try:
        settings = load_provider_route_retention_settings(database_url)
        preview = cleanup_expired_provider_route_records(database_url, dry_run=True)
    except Exception as exc:
        return OperationalRetentionCheck(
            code="provider_route_cleanup_preview",
            status=RETENTION_CHECK_FAILED,
            detail=f"Provider route cleanup preview failed: {exc}",
            metadata={"error": str(exc)},
        )

    metadata = {
        **_retention_settings_metadata(settings),
        "expired_health_snapshot_count": preview.expired_health_snapshot_count,
        "expired_contract_snapshot_count": preview.expired_contract_snapshot_count,
        "expired_preflight_run_count": preview.expired_preflight_run_count,
        "expired_count": preview.expired_count,
        "deleted_count": preview.deleted_count,
        "cutoff_at": preview.cutoff_at.isoformat(),
        "dry_run": preview.dry_run,
        "max_recommended_retention_days": max_retention_days,
    }
    status, detail = _cleanup_status_detail(
        label="Provider route cleanup",
        enabled=settings.enabled,
        retention_days=settings.retention_days,
        expired_count=preview.expired_count,
        max_retention_days=max_retention_days,
    )
    return OperationalRetentionCheck(
        code="provider_route_cleanup_preview",
        status=status,
        detail=detail,
        metadata=metadata,
    )


def _embedding_batch_run_cleanup_check(
    database_url: str,
    *,
    max_retention_days: int,
) -> OperationalRetentionCheck:
    try:
        settings = load_embedding_batch_run_retention_settings(database_url)
        preview = cleanup_expired_embedding_batch_run_records(database_url, dry_run=True)
    except Exception as exc:
        return OperationalRetentionCheck(
            code="embedding_batch_run_cleanup_preview",
            status=RETENTION_CHECK_FAILED,
            detail=f"Embedding batch run cleanup preview failed: {exc}",
            metadata={"error": str(exc)},
        )

    metadata = {
        **_retention_settings_metadata(settings),
        "expired_batch_run_count": preview.expired_batch_run_count,
        "expired_count": preview.expired_count,
        "deleted_count": preview.deleted_count,
        "cutoff_at": preview.cutoff_at.isoformat(),
        "dry_run": preview.dry_run,
        "max_recommended_retention_days": max_retention_days,
    }
    status, detail = _cleanup_status_detail(
        label="Embedding batch run cleanup",
        enabled=settings.enabled,
        retention_days=settings.retention_days,
        expired_count=preview.expired_count,
        max_retention_days=max_retention_days,
    )
    return OperationalRetentionCheck(
        code="embedding_batch_run_cleanup_preview",
        status=status,
        detail=detail,
        metadata=metadata,
    )


def _artifacts_directory_check(
    artifacts_dir: Path,
    *,
    retention_days: int,
    checked_at: datetime | None,
) -> OperationalRetentionCheck:
    if not artifacts_dir.exists():
        return OperationalRetentionCheck(
            code="artifacts_retention_review",
            status=RETENTION_CHECK_WARNING,
            detail="Artifacts directory does not exist; evidence export paths should be prepared.",
            metadata={"path": str(artifacts_dir), "exists": False},
        )
    if not artifacts_dir.is_dir():
        return OperationalRetentionCheck(
            code="artifacts_retention_review",
            status=RETENTION_CHECK_FAILED,
            detail="Artifacts path exists but is not a directory.",
            metadata={"path": str(artifacts_dir), "exists": True, "is_dir": False},
        )

    now = checked_at or datetime.now(UTC)
    scanned_count = 0
    old_file_count = 0
    newest_mtime: float | None = None
    oldest_mtime: float | None = None
    for path in artifacts_dir.rglob("*"):
        if not path.is_file():
            continue
        scanned_count += 1
        modified_at = path.stat().st_mtime
        newest_mtime = modified_at if newest_mtime is None else max(newest_mtime, modified_at)
        oldest_mtime = modified_at if oldest_mtime is None else min(oldest_mtime, modified_at)
        age_seconds = now.timestamp() - modified_at
        if age_seconds > retention_days * 24 * 60 * 60:
            old_file_count += 1

    metadata = {
        "path": str(artifacts_dir),
        "exists": True,
        "is_dir": True,
        "retention_days": retention_days,
        "scanned_file_count": scanned_count,
        "old_file_count": old_file_count,
        "oldest_modified_at": _timestamp_label(oldest_mtime),
        "newest_modified_at": _timestamp_label(newest_mtime),
    }
    if old_file_count > 0:
        return OperationalRetentionCheck(
            code="artifacts_retention_review",
            status=RETENTION_CHECK_WARNING,
            detail=f"{old_file_count} artifact files are older than {retention_days} days.",
            metadata=metadata,
        )
    return OperationalRetentionCheck(
        code="artifacts_retention_review",
        status=RETENTION_CHECK_PASSED,
        detail=f"Artifacts directory has no files older than {retention_days} days.",
        metadata=metadata,
    )


def _settings_status_detail(
    *,
    label: str,
    enabled: bool,
    retention_days: int,
    max_retention_days: int,
) -> tuple[str, str]:
    if not enabled:
        return RETENTION_CHECK_WARNING, f"{label} is disabled."
    if retention_days > max_retention_days:
        return (
            RETENTION_CHECK_WARNING,
            f"{label} retention is {retention_days} days, above {max_retention_days} days.",
        )
    return RETENTION_CHECK_PASSED, f"{label} is enabled for {retention_days} days."


def _cleanup_status_detail(
    *,
    label: str,
    enabled: bool,
    retention_days: int,
    expired_count: int,
    max_retention_days: int,
) -> tuple[str, str]:
    status, detail = _settings_status_detail(
        label=label,
        enabled=enabled,
        retention_days=retention_days,
        max_retention_days=max_retention_days,
    )
    if status != RETENTION_CHECK_PASSED:
        return status, detail
    return (
        RETENTION_CHECK_PASSED,
        f"{label} dry-run found {expired_count} rows older than {retention_days} days.",
    )


def _retention_settings_metadata(
    settings: (
        SearchLogRetentionSettings
        | ProviderRouteRetentionSettings
        | EmbeddingBatchRunRetentionSettings
    ),
) -> dict[str, object]:
    return {
        "enabled": settings.enabled,
        "retention_days": settings.retention_days,
        "cleanup_batch_size": settings.cleanup_batch_size,
    }


def _overall_status(checks: tuple[OperationalRetentionCheck, ...]) -> str:
    if any(check.status == RETENTION_CHECK_FAILED for check in checks):
        return RETENTION_STATUS_BLOCKED
    if any(check.status == RETENTION_CHECK_WARNING for check in checks):
        return RETENTION_STATUS_WARNING
    return RETENTION_STATUS_READY


def _timestamp_label(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("|", "\\|").replace("\n", " ")
