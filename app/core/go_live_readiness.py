"""Go-live readiness checklist aggregation."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from app.core.admin_logging import load_log_settings
from app.core.config import Settings
from app.core.dashboard_failures import get_dashboard_recent_failures
from app.core.database import connect
from app.core.embedding_jobs import (
    get_embedding_job_backlog_summary,
    list_active_embedding_profiles,
)
from app.core.embedding_model_distribution import audit_embedding_model_readiness
from app.core.embedding_provider_preflight_schedules import (
    list_embedding_provider_preflight_schedules,
)
from app.core.embedding_provider_route_readiness import (
    get_embedding_provider_route_readiness_summary,
)
from app.core.embedding_provider_route_retention import (
    load_provider_route_retention_settings,
)
from app.core.embedding_worker_batch_run_retention import (
    load_embedding_batch_run_retention_settings,
)
from app.core.pipeline_jobs import get_pipeline_queue_summary
from app.core.search_logs import load_search_log_retention_settings

GO_LIVE_CHECK_PASSED = "passed"
GO_LIVE_CHECK_WARNING = "warning"
GO_LIVE_CHECK_FAILED = "failed"
GO_LIVE_CHECK_SKIPPED = "skipped"

GO_LIVE_STATUS_READY = "ready"
GO_LIVE_STATUS_WARNING = "warning"
GO_LIVE_STATUS_BLOCKED = "blocked"

_NEGATIVE_STATUSES = {GO_LIVE_CHECK_WARNING, GO_LIVE_CHECK_FAILED}


@dataclass(frozen=True)
class GoLiveReadinessCheck:
    code: str
    status: str
    detail: str
    action_url: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GoLiveReadinessSection:
    code: str
    checks: tuple[GoLiveReadinessCheck, ...]

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if check.status == GO_LIVE_CHECK_FAILED)

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == GO_LIVE_CHECK_WARNING)


@dataclass(frozen=True)
class GoLiveReadinessReport:
    status: str
    checked_at: datetime
    sections: tuple[GoLiveReadinessSection, ...]

    @property
    def check_count(self) -> int:
        return sum(len(section.checks) for section in self.sections)

    @property
    def passed_count(self) -> int:
        return self._count_status(GO_LIVE_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(GO_LIVE_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(GO_LIVE_CHECK_FAILED)

    @property
    def skipped_count(self) -> int:
        return self._count_status(GO_LIVE_CHECK_SKIPPED)

    def _count_status(self, status: str) -> int:
        return sum(
            1 for section in self.sections for check in section.checks if check.status == status
        )


def build_go_live_readiness_report(
    settings: Settings,
    *,
    checked_at: datetime | None = None,
) -> GoLiveReadinessReport:
    database_url = settings.database_url
    database_checks = [_database_configured_check(database_url)]
    database_ready = False
    if database_url:
        database_connectivity = _database_connectivity_check(database_url)
        database_checks.append(database_connectivity)
        database_ready = database_connectivity.status == GO_LIVE_CHECK_PASSED
    else:
        database_checks.append(
            GoLiveReadinessCheck(
                code="database_connectivity",
                status=GO_LIVE_CHECK_SKIPPED,
                detail="Database connectivity is skipped until NEX_PCX_DATABASE_URL is configured.",
            )
        )

    sections = [
        GoLiveReadinessSection(code="runtime", checks=tuple(database_checks)),
        GoLiveReadinessSection(
            code="storage",
            checks=(_upload_storage_writable_check(settings.upload_storage_dir),),
        ),
        GoLiveReadinessSection(
            code="models",
            checks=(_embedding_model_distribution_check(settings.embedding_models_dir),),
        ),
    ]
    if database_url and database_ready:
        sections.extend(_database_backed_sections(database_url))
    elif database_url:
        sections.append(
            GoLiveReadinessSection(
                code="operations",
                checks=(
                    GoLiveReadinessCheck(
                        code="database_backed_checks",
                        status=GO_LIVE_CHECK_SKIPPED,
                        detail=(
                            "Database-backed operational checks are skipped because "
                            "connectivity failed."
                        ),
                    ),
                ),
            )
        )
    else:
        sections.append(
            GoLiveReadinessSection(
                code="operations",
                checks=(
                    GoLiveReadinessCheck(
                        code="database_backed_checks",
                        status=GO_LIVE_CHECK_SKIPPED,
                        detail=(
                            "Database-backed operational checks are skipped until the "
                            "database URL is configured."
                        ),
                    ),
                ),
            )
        )

    return GoLiveReadinessReport(
        status=_overall_status(sections),
        checked_at=checked_at or datetime.now(UTC),
        sections=tuple(sections),
    )


def go_live_readiness_report_payload(
    report: GoLiveReadinessReport,
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
        "sections": [
            {
                "code": section.code,
                "failed_count": section.failed_count,
                "warning_count": section.warning_count,
                "checks": [
                    {
                        "code": check.code,
                        "status": check.status,
                        "detail": check.detail,
                        "action_url": check.action_url,
                        "metadata": dict(check.metadata or {}),
                    }
                    for check in section.checks
                ],
            }
            for section in report.sections
        ],
    }


def _overall_status(sections: list[GoLiveReadinessSection]) -> str:
    statuses = [
        check.status
        for section in sections
        for check in section.checks
        if check.status in _NEGATIVE_STATUSES
    ]
    if GO_LIVE_CHECK_FAILED in statuses:
        return GO_LIVE_STATUS_BLOCKED
    if GO_LIVE_CHECK_WARNING in statuses:
        return GO_LIVE_STATUS_WARNING
    return GO_LIVE_STATUS_READY


def _database_configured_check(database_url: str | None) -> GoLiveReadinessCheck:
    if database_url:
        return GoLiveReadinessCheck(
            code="database_configured",
            status=GO_LIVE_CHECK_PASSED,
            detail="NEX_PCX_DATABASE_URL is configured.",
            metadata={"configured": True},
        )
    return GoLiveReadinessCheck(
        code="database_configured",
        status=GO_LIVE_CHECK_FAILED,
        detail="NEX_PCX_DATABASE_URL is not configured.",
        metadata={"configured": False},
    )


def _database_connectivity_check(database_url: str) -> GoLiveReadinessCheck:
    try:
        with connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 AS ok")
                cursor.fetchone()
    except Exception as exc:
        return GoLiveReadinessCheck(
            code="database_connectivity",
            status=GO_LIVE_CHECK_FAILED,
            detail=f"Database connection failed: {exc}",
            metadata={"error": str(exc)},
        )
    return GoLiveReadinessCheck(
        code="database_connectivity",
        status=GO_LIVE_CHECK_PASSED,
        detail="Database connection succeeded.",
    )


def _upload_storage_writable_check(storage_dir: Path) -> GoLiveReadinessCheck:
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            prefix=".go-live-",
            suffix=".tmp",
            dir=storage_dir,
            delete=True,
            encoding="utf-8",
        ) as temp_file:
            temp_file.write("ok")
            temp_file.flush()
    except OSError as exc:
        return GoLiveReadinessCheck(
            code="upload_storage_writable",
            status=GO_LIVE_CHECK_FAILED,
            detail=f"Upload storage is not writable: {exc}",
            action_url="/files/upload",
            metadata={"path": str(storage_dir), "error": str(exc)},
        )
    return GoLiveReadinessCheck(
        code="upload_storage_writable",
        status=GO_LIVE_CHECK_PASSED,
        detail="Upload storage directory is writable.",
        action_url="/files/upload",
        metadata={"path": str(storage_dir)},
    )


def _embedding_model_distribution_check(models_dir: Path) -> GoLiveReadinessCheck:
    readiness = audit_embedding_model_readiness(models_dir)
    ready_count = sum(1 for item in readiness if item.ready)
    model_count = len(readiness)
    status = GO_LIVE_CHECK_PASSED if ready_count == model_count else GO_LIVE_CHECK_WARNING
    detail = f"{ready_count}/{model_count} local embedding model bundles are ready."
    return GoLiveReadinessCheck(
        code="embedding_model_bundles",
        status=status,
        detail=detail,
        action_url="/admin/embedding-models",
        metadata={
            "models_dir": str(models_dir),
            "ready_count": ready_count,
            "model_count": model_count,
        },
    )


def _database_backed_sections(database_url: str) -> list[GoLiveReadinessSection]:
    return [
        GoLiveReadinessSection(
            code="profiles",
            checks=(
                _safe_check(
                    "active_embedding_profiles",
                    "/admin/embedding-jobs",
                    lambda: _active_embedding_profiles_check(database_url),
                ),
            ),
        ),
        GoLiveReadinessSection(
            code="providers",
            checks=(
                _safe_check(
                    "provider_route_readiness",
                    "/admin/embedding-provider-routes",
                    lambda: _provider_route_readiness_check(database_url),
                ),
                _safe_check(
                    "provider_preflight_schedule",
                    "/admin/embedding-provider-routes",
                    lambda: _provider_preflight_schedule_check(database_url),
                ),
            ),
        ),
        GoLiveReadinessSection(
            code="queues",
            checks=(
                _safe_check(
                    "pipeline_queue_health",
                    "/admin/jobs",
                    lambda: _pipeline_queue_health_check(database_url),
                ),
                _safe_check(
                    "embedding_queue_health",
                    "/admin/embedding-jobs",
                    lambda: _embedding_queue_health_check(database_url),
                ),
            ),
        ),
        GoLiveReadinessSection(
            code="failures",
            checks=(
                _safe_check(
                    "recent_operational_failures",
                    "/admin/logs",
                    lambda: _recent_operational_failures_check(database_url),
                ),
            ),
        ),
        GoLiveReadinessSection(
            code="retention",
            checks=(
                _safe_check(
                    "retention_settings",
                    "/admin/dashboard-settings",
                    lambda: _retention_settings_check(database_url),
                ),
            ),
        ),
    ]


def _safe_check(
    code: str,
    action_url: str,
    callback: Any,
) -> GoLiveReadinessCheck:
    try:
        return callback()
    except Exception as exc:
        return GoLiveReadinessCheck(
            code=code,
            status=GO_LIVE_CHECK_FAILED,
            detail=f"Readiness check failed: {exc}",
            action_url=action_url,
            metadata={"error": str(exc)},
        )


def _active_embedding_profiles_check(database_url: str) -> GoLiveReadinessCheck:
    profiles = list_active_embedding_profiles(database_url)
    status = GO_LIVE_CHECK_PASSED if profiles else GO_LIVE_CHECK_FAILED
    return GoLiveReadinessCheck(
        code="active_embedding_profiles",
        status=status,
        detail=f"{len(profiles)} active embedding profiles are configured.",
        action_url="/admin/embedding-jobs",
        metadata={
            "active_profile_count": len(profiles),
            "profiles": [profile.profile_name for profile in profiles],
        },
    )


def _provider_route_readiness_check(database_url: str) -> GoLiveReadinessCheck:
    readiness = get_embedding_provider_route_readiness_summary(
        database_url,
        active_only=True,
    )
    if readiness.route_count == 0:
        status = GO_LIVE_CHECK_FAILED
    elif readiness.ready_count == readiness.route_count:
        status = GO_LIVE_CHECK_PASSED
    elif readiness.ready_count > 0:
        status = GO_LIVE_CHECK_WARNING
    else:
        status = GO_LIVE_CHECK_FAILED
    return GoLiveReadinessCheck(
        code="provider_route_readiness",
        status=status,
        detail=f"{readiness.ready_count}/{readiness.route_count} active provider routes are ready.",
        action_url="/admin/embedding-provider-routes",
        metadata={
            "route_count": readiness.route_count,
            "ready_count": readiness.ready_count,
            "blocked_count": readiness.blocked_count,
            "needs_preflight_count": readiness.needs_preflight_count,
        },
    )


def _provider_preflight_schedule_check(database_url: str) -> GoLiveReadinessCheck:
    schedules = list_embedding_provider_preflight_schedules(database_url)
    enabled_count = sum(1 for schedule in schedules if schedule.is_enabled)
    status = GO_LIVE_CHECK_PASSED if enabled_count else GO_LIVE_CHECK_WARNING
    return GoLiveReadinessCheck(
        code="provider_preflight_schedule",
        status=status,
        detail=f"{enabled_count}/{len(schedules)} provider route preflight schedules are enabled.",
        action_url="/admin/embedding-provider-routes",
        metadata={"schedule_count": len(schedules), "enabled_count": enabled_count},
    )


def _pipeline_queue_health_check(database_url: str) -> GoLiveReadinessCheck:
    summary = get_pipeline_queue_summary(database_url)
    if (
        summary.stale_running_count
        or summary.exhausted_failed_count
        or summary.exhausted_canceled_count
    ):
        status = GO_LIVE_CHECK_FAILED
    elif summary.retryable_failed_count or summary.retryable_canceled_count:
        status = GO_LIVE_CHECK_WARNING
    else:
        status = GO_LIVE_CHECK_PASSED
    return GoLiveReadinessCheck(
        code="pipeline_queue_health",
        status=status,
        detail="Pipeline queue stale leases and failed jobs are within go-live tolerance.",
        action_url="/admin/jobs",
        metadata={
            "queued_count": summary.queued_count,
            "running_count": summary.running_count,
            "stale_running_count": summary.stale_running_count,
            "retryable_failed_count": summary.retryable_failed_count,
            "exhausted_failed_count": summary.exhausted_failed_count,
            "retryable_canceled_count": summary.retryable_canceled_count,
            "exhausted_canceled_count": summary.exhausted_canceled_count,
        },
    )


def _embedding_queue_health_check(database_url: str) -> GoLiveReadinessCheck:
    summary = get_embedding_job_backlog_summary(database_url)
    if summary.stale_running_count or summary.exhausted_failed_count:
        status = GO_LIVE_CHECK_FAILED
    elif summary.retryable_failed_count:
        status = GO_LIVE_CHECK_WARNING
    else:
        status = GO_LIVE_CHECK_PASSED
    return GoLiveReadinessCheck(
        code="embedding_queue_health",
        status=status,
        detail="Embedding queue stale leases and failed jobs are within go-live tolerance.",
        action_url="/admin/embedding-jobs",
        metadata={
            "pending_count": summary.pending_count,
            "running_count": summary.running_count,
            "stale_running_count": summary.stale_running_count,
            "retryable_failed_count": summary.retryable_failed_count,
            "exhausted_failed_count": summary.exhausted_failed_count,
        },
    )


def _recent_operational_failures_check(database_url: str) -> GoLiveReadinessCheck:
    summary = get_dashboard_recent_failures(database_url, limit=10)
    status = GO_LIVE_CHECK_WARNING if summary.total_count else GO_LIVE_CHECK_PASSED
    return GoLiveReadinessCheck(
        code="recent_operational_failures",
        status=status,
        detail=f"{summary.total_count} recent operational failures are visible.",
        action_url="/admin/logs",
        metadata={
            "total_count": summary.total_count,
            "provider_alert_count": summary.provider_alert_count,
            "app_error_count": summary.app_error_count,
            "parsing_failure_count": summary.parsing_failure_count,
        },
    )


def _retention_settings_check(database_url: str) -> GoLiveReadinessCheck:
    with connect(database_url) as connection:
        log_settings = load_log_settings(connection)
    provider_route = load_provider_route_retention_settings(database_url)
    embedding_batch = load_embedding_batch_run_retention_settings(database_url)
    search_log = load_search_log_retention_settings(database_url)
    enabled_count = sum(
        1
        for enabled in (
            log_settings.enabled,
            provider_route.enabled,
            embedding_batch.enabled,
            search_log.enabled,
        )
        if enabled
    )
    status = GO_LIVE_CHECK_PASSED if enabled_count == 4 else GO_LIVE_CHECK_WARNING
    return GoLiveReadinessCheck(
        code="retention_settings",
        status=status,
        detail=f"{enabled_count}/4 retention domains are enabled.",
        action_url="/admin/dashboard-settings",
        metadata={
            "admin_log_retention_days": log_settings.retention_days,
            "provider_route_retention_days": provider_route.retention_days,
            "embedding_batch_retention_days": embedding_batch.retention_days,
            "search_log_retention_days": search_log.retention_days,
            "enabled_count": enabled_count,
        },
    )
