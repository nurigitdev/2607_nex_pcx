"""Executable startup validation for operations runbooks."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from alembic.script import ScriptDirectory

from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_provider_route_preflight import (
    run_embedding_provider_route_preflight,
)
from app.core.go_live_readiness import (
    GO_LIVE_STATUS_BLOCKED,
    GO_LIVE_STATUS_READY,
    build_go_live_readiness_report,
)
from app.core.migrations import make_alembic_config

STARTUP_CHECK_PASSED = "passed"
STARTUP_CHECK_WARNING = "warning"
STARTUP_CHECK_FAILED = "failed"
STARTUP_CHECK_SKIPPED = "skipped"

STARTUP_STATUS_READY = "ready"
STARTUP_STATUS_WARNING = "warning"
STARTUP_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class OperationsStartupValidationOptions:
    app_base_url: str | None = None
    run_provider_preflight: bool = False
    profile_name: str | None = None
    include_inactive_routes: bool = False
    health_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class OperationsStartupValidationCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class OperationsStartupValidationReport:
    status: str
    checked_at: datetime
    checks: tuple[OperationsStartupValidationCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(STARTUP_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(STARTUP_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(STARTUP_CHECK_FAILED)

    @property
    def skipped_count(self) -> int:
        return self._count_status(STARTUP_CHECK_SKIPPED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


def build_operations_startup_validation_report(
    settings: Settings,
    *,
    options: OperationsStartupValidationOptions | None = None,
    checked_at: datetime | None = None,
) -> OperationsStartupValidationReport:
    selected_options = options or OperationsStartupValidationOptions()
    database_url = settings.database_url
    checks = [_database_url_check(database_url)]
    database_ready = False

    if database_url:
        database_check = _database_connectivity_check(database_url)
        database_ready = database_check.status == STARTUP_CHECK_PASSED
        checks.append(database_check)
    else:
        checks.append(
            OperationsStartupValidationCheck(
                code="database_connectivity",
                status=STARTUP_CHECK_SKIPPED,
                detail="Database connectivity is skipped until NEX_PCX_DATABASE_URL is configured.",
            )
        )

    checks.append(_alembic_revision_check(database_url, database_ready=database_ready))
    checks.append(
        _app_healthz_check(
            selected_options.app_base_url,
            timeout_seconds=selected_options.health_timeout_seconds,
        )
    )
    checks.append(
        _app_identity_check(
            selected_options.app_base_url,
            expected_app_name=settings.app_name,
            timeout_seconds=selected_options.health_timeout_seconds,
        )
    )
    checks.append(_go_live_readiness_check(settings))
    checks.append(
        _provider_preflight_check(
            database_url,
            database_ready=database_ready,
            options=selected_options,
        )
    )

    return OperationsStartupValidationReport(
        status=_overall_status(tuple(checks)),
        checked_at=checked_at or datetime.now(UTC),
        checks=tuple(checks),
    )


def operations_startup_validation_report_payload(
    report: OperationsStartupValidationReport,
) -> dict[str, object]:
    return {
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
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


def _overall_status(checks: tuple[OperationsStartupValidationCheck, ...]) -> str:
    if any(check.status == STARTUP_CHECK_FAILED for check in checks):
        return STARTUP_STATUS_BLOCKED
    if any(check.status == STARTUP_CHECK_WARNING for check in checks):
        return STARTUP_STATUS_WARNING
    return STARTUP_STATUS_READY


def _database_url_check(database_url: str | None) -> OperationsStartupValidationCheck:
    if database_url:
        return OperationsStartupValidationCheck(
            code="database_url",
            status=STARTUP_CHECK_PASSED,
            detail="NEX_PCX_DATABASE_URL is configured.",
            metadata={"configured": True},
        )
    return OperationsStartupValidationCheck(
        code="database_url",
        status=STARTUP_CHECK_FAILED,
        detail="NEX_PCX_DATABASE_URL is not configured.",
        metadata={"configured": False},
    )


def _database_connectivity_check(database_url: str) -> OperationsStartupValidationCheck:
    try:
        with connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 AS ok")
                cursor.fetchone()
    except Exception as exc:
        return OperationsStartupValidationCheck(
            code="database_connectivity",
            status=STARTUP_CHECK_FAILED,
            detail=f"Database connection failed: {exc}",
            metadata={"error": str(exc)},
        )
    return OperationsStartupValidationCheck(
        code="database_connectivity",
        status=STARTUP_CHECK_PASSED,
        detail="Database connection succeeded.",
    )


def _alembic_revision_check(
    database_url: str | None,
    *,
    database_ready: bool,
) -> OperationsStartupValidationCheck:
    if not database_url:
        return OperationsStartupValidationCheck(
            code="alembic_revision",
            status=STARTUP_CHECK_SKIPPED,
            detail="Alembic revision check is skipped until NEX_PCX_DATABASE_URL is configured.",
        )
    if not database_ready:
        return OperationsStartupValidationCheck(
            code="alembic_revision",
            status=STARTUP_CHECK_SKIPPED,
            detail="Alembic revision check is skipped because database connectivity failed.",
        )
    try:
        current_revisions = _current_database_revisions(database_url)
        head_revisions = _head_revisions(database_url)
    except Exception as exc:
        return OperationsStartupValidationCheck(
            code="alembic_revision",
            status=STARTUP_CHECK_FAILED,
            detail=f"Alembic revision check failed: {exc}",
            metadata={"error": str(exc)},
        )

    up_to_date = set(current_revisions) == set(head_revisions)
    status = STARTUP_CHECK_PASSED if up_to_date else STARTUP_CHECK_FAILED
    detail = (
        "Database schema is at Alembic head."
        if up_to_date
        else "Database schema is not at Alembic head."
    )
    return OperationsStartupValidationCheck(
        code="alembic_revision",
        status=status,
        detail=detail,
        metadata={
            "current_revisions": list(current_revisions),
            "head_revisions": list(head_revisions),
        },
    )


def _current_database_revisions(database_url: str) -> tuple[str, ...]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version_num FROM alembic_version ORDER BY version_num")
            rows = cursor.fetchall()
    return tuple(str(row["version_num"]) for row in rows)


def _head_revisions(database_url: str) -> tuple[str, ...]:
    script = ScriptDirectory.from_config(make_alembic_config(database_url))
    return tuple(script.get_heads())


def _app_healthz_check(
    app_base_url: str | None,
    *,
    timeout_seconds: float,
) -> OperationsStartupValidationCheck:
    if not app_base_url:
        return OperationsStartupValidationCheck(
            code="app_healthz",
            status=STARTUP_CHECK_SKIPPED,
            detail="Application health check is skipped because --app-url was not provided.",
        )
    normalized_base_url = app_base_url.rstrip("/")
    health_url = f"{normalized_base_url}/healthz"
    try:
        payload = _fetch_json(health_url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        return OperationsStartupValidationCheck(
            code="app_healthz",
            status=STARTUP_CHECK_FAILED,
            detail=f"Application health check failed: {exc}",
            metadata={"url": health_url, "error": str(exc)},
        )

    if payload.get("status") == "ok":
        return OperationsStartupValidationCheck(
            code="app_healthz",
            status=STARTUP_CHECK_PASSED,
            detail="Application /healthz endpoint returned status ok.",
            metadata={"url": health_url, "payload": payload},
        )
    return OperationsStartupValidationCheck(
        code="app_healthz",
        status=STARTUP_CHECK_FAILED,
        detail="Application /healthz endpoint did not return status ok.",
        metadata={"url": health_url, "payload": payload},
    )


def _app_identity_check(
    app_base_url: str | None,
    *,
    expected_app_name: str,
    timeout_seconds: float,
) -> OperationsStartupValidationCheck:
    if not app_base_url:
        return OperationsStartupValidationCheck(
            code="app_identity",
            status=STARTUP_CHECK_SKIPPED,
            detail="Application identity check is skipped because --app-url was not provided.",
        )
    normalized_base_url = app_base_url.rstrip("/")
    identity_url = f"{normalized_base_url}/openapi.json"
    try:
        payload = _fetch_json(identity_url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        return OperationsStartupValidationCheck(
            code="app_identity",
            status=STARTUP_CHECK_FAILED,
            detail=f"Application identity check failed: {exc}",
            metadata={"url": identity_url, "error": str(exc)},
        )

    title = _dict(payload.get("info")).get("title")
    if title == expected_app_name:
        return OperationsStartupValidationCheck(
            code="app_identity",
            status=STARTUP_CHECK_PASSED,
            detail=f"Application identity matches {expected_app_name}.",
            metadata={
                "url": identity_url,
                "expected_app_name": expected_app_name,
                "title": title,
            },
        )
    return OperationsStartupValidationCheck(
        code="app_identity",
        status=STARTUP_CHECK_FAILED,
        detail=("Application identity did not match expected app name " f"{expected_app_name}."),
        metadata={
            "url": identity_url,
            "expected_app_name": expected_app_name,
            "title": title,
        },
    )


def _fetch_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    import httpx

    response = httpx.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        msg = "Health response was not a JSON object"
        raise RuntimeError(msg)
    return payload


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _go_live_readiness_check(settings: Settings) -> OperationsStartupValidationCheck:
    try:
        report = build_go_live_readiness_report(settings)
    except Exception as exc:
        return OperationsStartupValidationCheck(
            code="go_live_readiness",
            status=STARTUP_CHECK_FAILED,
            detail=f"Go-live readiness check failed: {exc}",
            metadata={"error": str(exc)},
        )

    if report.status == GO_LIVE_STATUS_READY:
        status = STARTUP_CHECK_PASSED
    elif report.status == GO_LIVE_STATUS_BLOCKED:
        status = STARTUP_CHECK_FAILED
    else:
        status = STARTUP_CHECK_WARNING

    return OperationsStartupValidationCheck(
        code="go_live_readiness",
        status=status,
        detail=f"Go-live readiness status is {report.status}.",
        metadata={
            "go_live_status": report.status,
            "check_count": report.check_count,
            "passed_count": report.passed_count,
            "warning_count": report.warning_count,
            "failed_count": report.failed_count,
            "skipped_count": report.skipped_count,
        },
    )


def _provider_preflight_check(
    database_url: str | None,
    *,
    database_ready: bool,
    options: OperationsStartupValidationOptions,
) -> OperationsStartupValidationCheck:
    if not options.run_provider_preflight:
        return OperationsStartupValidationCheck(
            code="provider_route_preflight",
            status=STARTUP_CHECK_SKIPPED,
            detail=(
                "Provider route preflight is skipped because "
                "--run-provider-preflight was not provided."
            ),
        )
    if not database_url:
        return OperationsStartupValidationCheck(
            code="provider_route_preflight",
            status=STARTUP_CHECK_SKIPPED,
            detail="Provider route preflight is skipped until NEX_PCX_DATABASE_URL is configured.",
        )
    if not database_ready:
        return OperationsStartupValidationCheck(
            code="provider_route_preflight",
            status=STARTUP_CHECK_SKIPPED,
            detail="Provider route preflight is skipped because database connectivity failed.",
        )

    try:
        payload = run_embedding_provider_route_preflight(
            database_url,
            profile_name=options.profile_name,
            active_only=not options.include_inactive_routes,
        )
    except Exception as exc:
        return OperationsStartupValidationCheck(
            code="provider_route_preflight",
            status=STARTUP_CHECK_FAILED,
            detail=f"Provider route preflight failed: {exc}",
            metadata={"error": str(exc)},
        )

    route_count = int(payload.get("route_count", 0))
    failed_count = int(payload.get("failed_count", 0))
    if route_count == 0:
        status = STARTUP_CHECK_FAILED
        detail = "Provider route preflight found no routes to validate."
    elif failed_count:
        status = STARTUP_CHECK_FAILED
        detail = f"Provider route preflight failed for {failed_count}/{route_count} routes."
    else:
        status = STARTUP_CHECK_PASSED
        detail = f"Provider route preflight passed for {route_count} routes."

    return OperationsStartupValidationCheck(
        code="provider_route_preflight",
        status=status,
        detail=detail,
        metadata={
            "route_count": route_count,
            "passed_count": int(payload.get("passed_count", 0)),
            "failed_count": failed_count,
            "profile_name": payload.get("profile_name"),
            "active_only": payload.get("active_only"),
        },
    )
