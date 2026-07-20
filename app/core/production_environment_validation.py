"""Production environment final validation aggregation."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.core.config import Settings
from app.core.go_live_readiness import (
    GO_LIVE_STATUS_BLOCKED,
    GO_LIVE_STATUS_WARNING,
    build_go_live_readiness_report,
    go_live_readiness_report_payload,
)
from app.core.operations_startup_validation import (
    STARTUP_STATUS_BLOCKED,
    STARTUP_STATUS_WARNING,
    OperationsStartupValidationOptions,
    build_operations_startup_validation_report,
    operations_startup_validation_report_payload,
)
from app.core.runtime_config_audit import (
    CONFIG_AUDIT_STATUS_BLOCKED,
    CONFIG_AUDIT_STATUS_WARNING,
    build_runtime_config_audit_report,
    mask_database_url,
    runtime_config_audit_report_payload,
)

PRODUCTION_VALIDATION_VERSION = 1

PRODUCTION_CHECK_PASSED = "passed"
PRODUCTION_CHECK_WARNING = "warning"
PRODUCTION_CHECK_FAILED = "failed"

PRODUCTION_STATUS_READY = "ready"
PRODUCTION_STATUS_WARNING = "warning"
PRODUCTION_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class ProductionValidationOptions:
    app_url: str | None = None
    expected_database_name: str | None = None
    require_production_env: bool = True
    require_remote_provider: bool = True
    require_route_readiness: bool = True
    run_provider_preflight: bool = False
    profile_name: str | None = None
    include_inactive_routes: bool = False
    health_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class ProductionValidationCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ProductionValidationSection:
    code: str
    status: str
    payload: dict[str, object]


@dataclass(frozen=True)
class ProductionEnvironmentValidationReport:
    status: str
    checked_at: datetime
    guard_checks: tuple[ProductionValidationCheck, ...]
    sections: tuple[ProductionValidationSection, ...]

    @property
    def guard_check_count(self) -> int:
        return len(self.guard_checks)

    @property
    def failed_guard_count(self) -> int:
        return self._count_guard_status(PRODUCTION_CHECK_FAILED)

    @property
    def warning_guard_count(self) -> int:
        return self._count_guard_status(PRODUCTION_CHECK_WARNING)

    def _count_guard_status(self, status: str) -> int:
        return sum(1 for check in self.guard_checks if check.status == status)


def build_production_environment_validation_report(
    settings: Settings,
    *,
    options: ProductionValidationOptions | None = None,
    project_root: Path | None = None,
    checked_at: datetime | None = None,
) -> ProductionEnvironmentValidationReport:
    selected_options = options or ProductionValidationOptions()
    checked = checked_at or datetime.now(UTC)
    guard_checks = _production_guard_checks(settings, selected_options)
    runtime_audit = build_runtime_config_audit_report(
        settings,
        checked_at=checked,
        project_root=project_root,
    )
    startup_validation = build_operations_startup_validation_report(
        settings,
        checked_at=checked,
        options=OperationsStartupValidationOptions(
            app_base_url=selected_options.app_url,
            run_provider_preflight=selected_options.run_provider_preflight,
            profile_name=selected_options.profile_name,
            include_inactive_routes=selected_options.include_inactive_routes,
            health_timeout_seconds=selected_options.health_timeout_seconds,
        ),
    )
    go_live_readiness = build_go_live_readiness_report(settings, checked_at=checked)
    sections = (
        ProductionValidationSection(
            code="runtime_config_audit",
            status=_runtime_audit_status(runtime_audit.status),
            payload=runtime_config_audit_report_payload(runtime_audit),
        ),
        ProductionValidationSection(
            code="operations_startup_validation",
            status=_startup_status(startup_validation.status),
            payload=operations_startup_validation_report_payload(startup_validation),
        ),
        ProductionValidationSection(
            code="go_live_readiness",
            status=_go_live_status(go_live_readiness.status),
            payload=go_live_readiness_report_payload(go_live_readiness),
        ),
    )
    return ProductionEnvironmentValidationReport(
        status=_overall_status(guard_checks, sections),
        checked_at=checked,
        guard_checks=tuple(guard_checks),
        sections=sections,
    )


def production_environment_validation_payload(
    report: ProductionEnvironmentValidationReport,
) -> dict[str, object]:
    return {
        "version": PRODUCTION_VALIDATION_VERSION,
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "guard_check_count": report.guard_check_count,
        "failed_guard_count": report.failed_guard_count,
        "warning_guard_count": report.warning_guard_count,
        "guard_checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "metadata": dict(check.metadata or {}),
            }
            for check in report.guard_checks
        ],
        "sections": [
            {
                "code": section.code,
                "status": section.status,
                "payload": section.payload,
            }
            for section in report.sections
        ],
    }


def render_production_environment_validation_markdown(
    payload: dict[str, object],
) -> str:
    lines = [
        "# NeX_PCX Production Environment Validation",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Overall Status: `{_text(payload.get('status'))}`",
        f"- Failed Guards: {_text(payload.get('failed_guard_count'))}",
        f"- Warning Guards: {_text(payload.get('warning_guard_count'))}",
        "",
        "## Production Guards",
        "",
        "| Guard | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in payload.get("guard_checks", []):
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
            "## Validation Sections",
            "",
            "| Section | Status | Summary |",
            "| --- | --- | --- |",
        ]
    )
    for section in payload.get("sections", []):
        section_payload = _dict(section)
        nested_payload = _dict(section_payload.get("payload"))
        summary = _section_summary(nested_payload)
        lines.append(
            "| "
            f"{_md_cell(section_payload.get('code'))} | "
            f"{_md_cell(section_payload.get('status'))} | "
            f"{_md_cell(summary)} |"
        )

    lines.extend(["", "## Section Payloads", ""])
    for section in payload.get("sections", []):
        section_payload = _dict(section)
        lines.extend(
            [
                f"### {_text(section_payload.get('code'))}",
                "",
                "```json",
                json.dumps(
                    _dict(section_payload.get("payload")),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _production_guard_checks(
    settings: Settings,
    options: ProductionValidationOptions,
) -> tuple[ProductionValidationCheck, ...]:
    return (
        _environment_guard(settings, require_production=options.require_production_env),
        _database_guard(
            settings.database_url,
            expected_database_name=options.expected_database_name,
        ),
        _provider_mode_guard(settings, required=options.require_remote_provider),
        _route_readiness_guard(settings, required=options.require_route_readiness),
        _app_url_guard(options.app_url),
    )


def _environment_guard(
    settings: Settings,
    *,
    require_production: bool,
) -> ProductionValidationCheck:
    if settings.environment == "production":
        return ProductionValidationCheck(
            code="environment",
            status=PRODUCTION_CHECK_PASSED,
            detail="NEX_PCX_ENV is production.",
            metadata={"environment": settings.environment},
        )
    status = PRODUCTION_CHECK_FAILED if require_production else PRODUCTION_CHECK_WARNING
    detail = (
        "NEX_PCX_ENV must be production for final operations validation."
        if require_production
        else "NEX_PCX_ENV is not production; running rehearsal validation."
    )
    return ProductionValidationCheck(
        code="environment",
        status=status,
        detail=detail,
        metadata={"environment": settings.environment},
    )


def _database_guard(
    database_url: str | None,
    *,
    expected_database_name: str | None,
) -> ProductionValidationCheck:
    if not database_url:
        return ProductionValidationCheck(
            code="database_target",
            status=PRODUCTION_CHECK_FAILED,
            detail="NEX_PCX_DATABASE_URL is required.",
            metadata={"configured": False},
        )
    database_name = _database_name(database_url)
    metadata = {
        "configured": True,
        "database_name": database_name,
        "expected_database_name": expected_database_name,
        "masked": mask_database_url(database_url),
    }
    if expected_database_name and database_name != expected_database_name:
        return ProductionValidationCheck(
            code="database_target",
            status=PRODUCTION_CHECK_FAILED,
            detail=(f"Database name is {database_name}; expected {expected_database_name}."),
            metadata=metadata,
        )
    if database_name and database_name.endswith("_test"):
        return ProductionValidationCheck(
            code="database_target",
            status=PRODUCTION_CHECK_FAILED,
            detail="Database target appears to be a test database.",
            metadata=metadata,
        )
    if database_name and database_name.endswith("_dev"):
        return ProductionValidationCheck(
            code="database_target",
            status=PRODUCTION_CHECK_WARNING,
            detail="Database target appears to be a development database.",
            metadata=metadata,
        )
    return ProductionValidationCheck(
        code="database_target",
        status=PRODUCTION_CHECK_PASSED,
        detail="Database target is acceptable.",
        metadata=metadata,
    )


def _provider_mode_guard(
    settings: Settings,
    *,
    required: bool,
) -> ProductionValidationCheck:
    if settings.embedding_provider_mode == "remote":
        return ProductionValidationCheck(
            code="embedding_provider_mode",
            status=PRODUCTION_CHECK_PASSED,
            detail="Embedding provider mode is remote.",
            metadata={"mode": settings.embedding_provider_mode},
        )
    status = PRODUCTION_CHECK_FAILED if required else PRODUCTION_CHECK_WARNING
    return ProductionValidationCheck(
        code="embedding_provider_mode",
        status=status,
        detail="Embedding provider mode should be remote for operations.",
        metadata={"mode": settings.embedding_provider_mode},
    )


def _route_readiness_guard(
    settings: Settings,
    *,
    required: bool,
) -> ProductionValidationCheck:
    if settings.embedding_require_route_readiness:
        return ProductionValidationCheck(
            code="embedding_route_readiness",
            status=PRODUCTION_CHECK_PASSED,
            detail="Embedding route readiness gate is enabled.",
            metadata={"required": True},
        )
    status = PRODUCTION_CHECK_FAILED if required else PRODUCTION_CHECK_WARNING
    return ProductionValidationCheck(
        code="embedding_route_readiness",
        status=status,
        detail="Embedding route readiness gate should be enabled for operations.",
        metadata={"required": False},
    )


def _app_url_guard(app_url: str | None) -> ProductionValidationCheck:
    if app_url:
        return ProductionValidationCheck(
            code="app_url",
            status=PRODUCTION_CHECK_PASSED,
            detail="Application URL is configured for health validation.",
            metadata={"app_url": app_url.rstrip("/")},
        )
    return ProductionValidationCheck(
        code="app_url",
        status=PRODUCTION_CHECK_WARNING,
        detail="Application URL was not provided; /healthz validation will be skipped.",
        metadata={"configured": False},
    )


def _overall_status(
    guard_checks: tuple[ProductionValidationCheck, ...],
    sections: tuple[ProductionValidationSection, ...],
) -> str:
    if any(check.status == PRODUCTION_CHECK_FAILED for check in guard_checks):
        return PRODUCTION_STATUS_BLOCKED
    if any(section.status == PRODUCTION_STATUS_BLOCKED for section in sections):
        return PRODUCTION_STATUS_BLOCKED
    if any(check.status == PRODUCTION_CHECK_WARNING for check in guard_checks):
        return PRODUCTION_STATUS_WARNING
    if any(section.status == PRODUCTION_STATUS_WARNING for section in sections):
        return PRODUCTION_STATUS_WARNING
    return PRODUCTION_STATUS_READY


def _runtime_audit_status(status: str) -> str:
    if status == CONFIG_AUDIT_STATUS_BLOCKED:
        return PRODUCTION_STATUS_BLOCKED
    if status == CONFIG_AUDIT_STATUS_WARNING:
        return PRODUCTION_STATUS_WARNING
    return PRODUCTION_STATUS_READY


def _startup_status(status: str) -> str:
    if status == STARTUP_STATUS_BLOCKED:
        return PRODUCTION_STATUS_BLOCKED
    if status == STARTUP_STATUS_WARNING:
        return PRODUCTION_STATUS_WARNING
    return PRODUCTION_STATUS_READY


def _go_live_status(status: str) -> str:
    if status == GO_LIVE_STATUS_BLOCKED:
        return PRODUCTION_STATUS_BLOCKED
    if status == GO_LIVE_STATUS_WARNING:
        return PRODUCTION_STATUS_WARNING
    return PRODUCTION_STATUS_READY


def _database_name(database_url: str) -> str | None:
    try:
        parsed = urlsplit(database_url)
    except ValueError:
        return None
    return parsed.path.lstrip("/") or None


def _section_summary(payload: dict[str, object]) -> str:
    parts = []
    for key in ("passed_count", "warning_count", "failed_count", "skipped_count"):
        if key in payload:
            parts.append(f"{key}={payload[key]}")
    return ", ".join(parts) if parts else f"status={payload.get('status')}"


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("|", "\\|").replace("\n", " ")
