"""Runtime configuration audit for operator startup checks."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app.core.config import Settings

CONFIG_AUDIT_VERSION = 1

CONFIG_CHECK_PASSED = "passed"
CONFIG_CHECK_WARNING = "warning"
CONFIG_CHECK_FAILED = "failed"

CONFIG_AUDIT_STATUS_READY = "ready"
CONFIG_AUDIT_STATUS_WARNING = "warning"
CONFIG_AUDIT_STATUS_BLOCKED = "blocked"

VALID_PROVIDER_MODES = {"mock", "remote"}
VALID_READINESS_FAILURE_MODES = {"fail", "defer"}


@dataclass(frozen=True)
class RuntimeConfigAuditCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class RuntimeConfigAuditReport:
    status: str
    checked_at: datetime
    checks: tuple[RuntimeConfigAuditCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(CONFIG_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(CONFIG_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(CONFIG_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


def build_runtime_config_audit_report(
    settings: Settings,
    *,
    checked_at: datetime | None = None,
    project_root: Path | None = None,
) -> RuntimeConfigAuditReport:
    root = project_root or Path.cwd()
    checks = (
        _environment_check(settings),
        _database_url_check(settings.database_url),
        _test_database_url_check(settings),
        _path_check(
            "upload_storage_dir",
            settings.upload_storage_dir,
            root=root,
            required=True,
            should_exist=True,
        ),
        _path_check(
            "embedding_models_dir",
            settings.embedding_models_dir,
            root=root,
            required=True,
            should_exist=settings.embedding_provider_mode != "remote",
        ),
        _provider_mode_check(settings),
        _provider_route_readiness_check(settings),
        _readiness_failure_mode_check(settings),
        _readiness_defer_seconds_check(settings),
    )
    return RuntimeConfigAuditReport(
        status=_overall_status(checks),
        checked_at=checked_at or datetime.now(UTC),
        checks=checks,
    )


def runtime_config_audit_report_payload(
    report: RuntimeConfigAuditReport,
) -> dict[str, object]:
    return {
        "version": CONFIG_AUDIT_VERSION,
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "check_count": report.check_count,
        "passed_count": report.passed_count,
        "warning_count": report.warning_count,
        "failed_count": report.failed_count,
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


def render_runtime_config_audit_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# NeX_PCX Runtime Configuration Audit",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Overall Status: {_text(payload.get('status'))}",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
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

    lines.extend(["", "## Metadata", ""])
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
    if lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def _overall_status(checks: tuple[RuntimeConfigAuditCheck, ...]) -> str:
    if any(check.status == CONFIG_CHECK_FAILED for check in checks):
        return CONFIG_AUDIT_STATUS_BLOCKED
    if any(check.status == CONFIG_CHECK_WARNING for check in checks):
        return CONFIG_AUDIT_STATUS_WARNING
    return CONFIG_AUDIT_STATUS_READY


def _environment_check(settings: Settings) -> RuntimeConfigAuditCheck:
    if settings.environment == "local":
        return RuntimeConfigAuditCheck(
            code="environment",
            status=CONFIG_CHECK_WARNING,
            detail="NEX_PCX_ENV is local; confirm this is not a production host.",
            metadata={"environment": settings.environment},
        )
    return RuntimeConfigAuditCheck(
        code="environment",
        status=CONFIG_CHECK_PASSED,
        detail=f"NEX_PCX_ENV is {settings.environment}.",
        metadata={"environment": settings.environment},
    )


def _database_url_check(database_url: str | None) -> RuntimeConfigAuditCheck:
    if not database_url:
        return RuntimeConfigAuditCheck(
            code="database_url",
            status=CONFIG_CHECK_FAILED,
            detail="NEX_PCX_DATABASE_URL is not configured.",
            metadata={"configured": False},
        )
    return RuntimeConfigAuditCheck(
        code="database_url",
        status=CONFIG_CHECK_PASSED,
        detail="NEX_PCX_DATABASE_URL is configured.",
        metadata={"configured": True, "masked": mask_database_url(database_url)},
    )


def _test_database_url_check(settings: Settings) -> RuntimeConfigAuditCheck:
    if settings.environment == "production" and settings.test_database_url:
        return RuntimeConfigAuditCheck(
            code="test_database_url",
            status=CONFIG_CHECK_WARNING,
            detail="NEX_PCX_TEST_DATABASE_URL is configured in production.",
            metadata={"configured": True, "masked": mask_database_url(settings.test_database_url)},
        )
    return RuntimeConfigAuditCheck(
        code="test_database_url",
        status=CONFIG_CHECK_PASSED,
        detail="Test database URL setting is acceptable.",
        metadata={"configured": bool(settings.test_database_url)},
    )


def _path_check(
    code: str,
    path: Path,
    *,
    root: Path,
    required: bool,
    should_exist: bool,
) -> RuntimeConfigAuditCheck:
    if required and not str(path):
        return RuntimeConfigAuditCheck(
            code=code,
            status=CONFIG_CHECK_FAILED,
            detail=f"{code} is not configured.",
            metadata={"configured": False},
        )

    resolved_path = path if path.is_absolute() else root / path
    metadata = {
        "configured": True,
        "path": str(path),
        "resolved_path": str(resolved_path),
        "exists": resolved_path.exists(),
        "is_dir": resolved_path.is_dir(),
        "is_absolute": path.is_absolute(),
    }
    if not path.is_absolute():
        return RuntimeConfigAuditCheck(
            code=code,
            status=CONFIG_CHECK_WARNING,
            detail=f"{code} uses a relative path; prefer an absolute path for operations.",
            metadata=metadata,
        )
    if resolved_path.exists() and not resolved_path.is_dir():
        return RuntimeConfigAuditCheck(
            code=code,
            status=CONFIG_CHECK_FAILED,
            detail=f"{code} exists but is not a directory.",
            metadata=metadata,
        )
    if should_exist and not resolved_path.exists():
        return RuntimeConfigAuditCheck(
            code=code,
            status=CONFIG_CHECK_WARNING,
            detail=f"{code} does not exist yet.",
            metadata=metadata,
        )
    return RuntimeConfigAuditCheck(
        code=code,
        status=CONFIG_CHECK_PASSED,
        detail=f"{code} path is acceptable.",
        metadata=metadata,
    )


def _provider_mode_check(settings: Settings) -> RuntimeConfigAuditCheck:
    if settings.embedding_provider_mode not in VALID_PROVIDER_MODES:
        return RuntimeConfigAuditCheck(
            code="embedding_provider_mode",
            status=CONFIG_CHECK_FAILED,
            detail=f"Unsupported embedding provider mode: {settings.embedding_provider_mode}.",
            metadata={"mode": settings.embedding_provider_mode},
        )
    if settings.environment == "production" and settings.embedding_provider_mode == "mock":
        return RuntimeConfigAuditCheck(
            code="embedding_provider_mode",
            status=CONFIG_CHECK_WARNING,
            detail="Embedding provider mode is mock in production.",
            metadata={"mode": settings.embedding_provider_mode},
        )
    return RuntimeConfigAuditCheck(
        code="embedding_provider_mode",
        status=CONFIG_CHECK_PASSED,
        detail=f"Embedding provider mode is {settings.embedding_provider_mode}.",
        metadata={
            "mode": settings.embedding_provider_mode,
            "remote_provider_url_configured": bool(settings.remote_embedding_provider_url),
        },
    )


def _provider_route_readiness_check(settings: Settings) -> RuntimeConfigAuditCheck:
    if not settings.embedding_require_route_readiness:
        return RuntimeConfigAuditCheck(
            code="embedding_route_readiness",
            status=CONFIG_CHECK_WARNING,
            detail="Embedding route readiness gate is disabled.",
            metadata={"required": False},
        )
    return RuntimeConfigAuditCheck(
        code="embedding_route_readiness",
        status=CONFIG_CHECK_PASSED,
        detail="Embedding route readiness gate is enabled.",
        metadata={"required": True},
    )


def _readiness_failure_mode_check(settings: Settings) -> RuntimeConfigAuditCheck:
    if settings.embedding_route_readiness_failure_mode not in VALID_READINESS_FAILURE_MODES:
        return RuntimeConfigAuditCheck(
            code="embedding_route_readiness_failure_mode",
            status=CONFIG_CHECK_FAILED,
            detail=(
                "Unsupported embedding route readiness failure mode: "
                f"{settings.embedding_route_readiness_failure_mode}."
            ),
            metadata={"mode": settings.embedding_route_readiness_failure_mode},
        )
    return RuntimeConfigAuditCheck(
        code="embedding_route_readiness_failure_mode",
        status=CONFIG_CHECK_PASSED,
        detail=(
            "Embedding route readiness failure mode is "
            f"{settings.embedding_route_readiness_failure_mode}."
        ),
        metadata={"mode": settings.embedding_route_readiness_failure_mode},
    )


def _readiness_defer_seconds_check(settings: Settings) -> RuntimeConfigAuditCheck:
    seconds = settings.embedding_route_readiness_defer_seconds
    if seconds <= 0:
        return RuntimeConfigAuditCheck(
            code="embedding_route_readiness_defer_seconds",
            status=CONFIG_CHECK_FAILED,
            detail="Embedding route readiness defer seconds must be greater than zero.",
            metadata={"seconds": seconds},
        )
    if settings.embedding_route_readiness_failure_mode == "defer" and seconds < 60:
        return RuntimeConfigAuditCheck(
            code="embedding_route_readiness_defer_seconds",
            status=CONFIG_CHECK_WARNING,
            detail="Embedding route readiness defer window is shorter than 60 seconds.",
            metadata={"seconds": seconds},
        )
    return RuntimeConfigAuditCheck(
        code="embedding_route_readiness_defer_seconds",
        status=CONFIG_CHECK_PASSED,
        detail=f"Embedding route readiness defer window is {seconds} seconds.",
        metadata={"seconds": seconds},
    )


def mask_database_url(database_url: str | None) -> str | None:
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


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
