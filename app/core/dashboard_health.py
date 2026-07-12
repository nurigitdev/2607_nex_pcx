"""Operational health summary helpers for the dashboard."""

from dataclasses import dataclass

from app.core.dashboard_failures import DashboardFailureSummary
from app.core.embedding_jobs import EmbeddingJobBacklogSummary
from app.core.pipeline_jobs import PipelineQueueSummary

DASHBOARD_HEALTH_HEALTHY = "healthy"
DASHBOARD_HEALTH_WARNING = "warning"
DASHBOARD_HEALTH_CRITICAL = "critical"

_SEVERITY_RANK = {
    DASHBOARD_HEALTH_HEALTHY: 0,
    DASHBOARD_HEALTH_WARNING: 1,
    DASHBOARD_HEALTH_CRITICAL: 2,
}


@dataclass(frozen=True)
class DashboardHealthSignal:
    code: str
    severity: str
    count: int
    action_url: str


@dataclass(frozen=True)
class DashboardOperationalHealth:
    status: str
    signal_count: int
    critical_count: int
    warning_count: int
    signals: tuple[DashboardHealthSignal, ...]


def _signal(
    signals: list[DashboardHealthSignal],
    *,
    code: str,
    severity: str,
    count: int,
    action_url: str,
) -> None:
    if count <= 0:
        return
    signals.append(
        DashboardHealthSignal(
            code=code,
            severity=severity,
            count=count,
            action_url=action_url,
        )
    )


def summarize_dashboard_operational_health(
    *,
    pipeline_queue: PipelineQueueSummary | None,
    embedding_backlog: EmbeddingJobBacklogSummary | None,
    recent_failures: DashboardFailureSummary | None,
) -> DashboardOperationalHealth:
    signals: list[DashboardHealthSignal] = []

    if pipeline_queue is not None:
        _signal(
            signals,
            code="pipeline_stale",
            severity=DASHBOARD_HEALTH_CRITICAL,
            count=pipeline_queue.stale_running_count,
            action_url="/admin/jobs",
        )
        _signal(
            signals,
            code="pipeline_exhausted",
            severity=DASHBOARD_HEALTH_CRITICAL,
            count=(
                pipeline_queue.exhausted_failed_count
                + pipeline_queue.exhausted_canceled_count
            ),
            action_url="/admin/jobs",
        )
        _signal(
            signals,
            code="pipeline_retryable",
            severity=DASHBOARD_HEALTH_WARNING,
            count=(
                pipeline_queue.retryable_failed_count
                + pipeline_queue.retryable_canceled_count
            ),
            action_url="/admin/jobs",
        )

    if embedding_backlog is not None:
        _signal(
            signals,
            code="embedding_stale",
            severity=DASHBOARD_HEALTH_CRITICAL,
            count=embedding_backlog.stale_running_count,
            action_url="/admin/embedding-jobs",
        )
        _signal(
            signals,
            code="embedding_exhausted",
            severity=DASHBOARD_HEALTH_CRITICAL,
            count=embedding_backlog.exhausted_failed_count,
            action_url="/admin/embedding-jobs",
        )
        _signal(
            signals,
            code="embedding_retryable",
            severity=DASHBOARD_HEALTH_WARNING,
            count=embedding_backlog.retryable_failed_count,
            action_url="/admin/embedding-jobs",
        )

    if recent_failures is not None:
        _signal(
            signals,
            code="provider_alert",
            severity=DASHBOARD_HEALTH_WARNING,
            count=recent_failures.provider_alert_count,
            action_url="/admin/embedding-provider-routes",
        )
        _signal(
            signals,
            code="app_error",
            severity=DASHBOARD_HEALTH_WARNING,
            count=recent_failures.app_error_count,
            action_url="/admin/logs",
        )
        _signal(
            signals,
            code="parsing_failure",
            severity=DASHBOARD_HEALTH_WARNING,
            count=recent_failures.parsing_failure_count,
            action_url="/documents?parse_status=failed",
        )

    ordered_signals = tuple(
        sorted(
            signals,
            key=lambda signal: (
                -_SEVERITY_RANK.get(signal.severity, 0),
                -signal.count,
                signal.code,
            ),
        )
    )
    critical_count = sum(
        1 for signal in ordered_signals if signal.severity == DASHBOARD_HEALTH_CRITICAL
    )
    warning_count = sum(
        1 for signal in ordered_signals if signal.severity == DASHBOARD_HEALTH_WARNING
    )
    status = DASHBOARD_HEALTH_HEALTHY
    if critical_count:
        status = DASHBOARD_HEALTH_CRITICAL
    elif warning_count:
        status = DASHBOARD_HEALTH_WARNING

    return DashboardOperationalHealth(
        status=status,
        signal_count=len(ordered_signals),
        critical_count=critical_count,
        warning_count=warning_count,
        signals=ordered_signals,
    )
