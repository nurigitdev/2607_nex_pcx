from app.core.dashboard_failures import DashboardFailureSummary
from app.core.dashboard_health import summarize_dashboard_operational_health
from app.core.embedding_jobs import EmbeddingJobBacklogSummary
from app.core.pipeline_jobs import PipelineQueueSummary


def _pipeline_queue(**overrides: int) -> PipelineQueueSummary:
    values = {
        "total_count": 0,
        "queued_count": 0,
        "running_count": 0,
        "stale_running_count": 0,
        "reclaimable_stale_running_count": 0,
        "failed_count": 0,
        "retryable_failed_count": 0,
        "exhausted_failed_count": 0,
        "canceled_count": 0,
        "retryable_canceled_count": 0,
        "exhausted_canceled_count": 0,
        "succeeded_count": 0,
        "skipped_count": 0,
    }
    values.update(overrides)
    return PipelineQueueSummary(
        stage_summaries=(),
        type_summaries=(),
        oldest_queued_at=None,
        oldest_stale_lease_expires_at=None,
        **values,
    )


def _embedding_backlog(**overrides: int) -> EmbeddingJobBacklogSummary:
    values = {
        "total_count": 0,
        "pending_count": 0,
        "running_count": 0,
        "stale_running_count": 0,
        "reclaimable_stale_running_count": 0,
        "failed_count": 0,
        "retryable_failed_count": 0,
        "exhausted_failed_count": 0,
        "succeeded_count": 0,
        "skipped_count": 0,
    }
    values.update(overrides)
    return EmbeddingJobBacklogSummary(profile_summaries=(), **values)


def _recent_failures(**overrides: int) -> DashboardFailureSummary:
    values = {
        "total_count": 0,
        "pipeline_failure_count": 0,
        "embedding_failure_count": 0,
        "parsing_failure_count": 0,
        "app_error_count": 0,
        "provider_alert_count": 0,
    }
    values.update(overrides)
    return DashboardFailureSummary(failures=(), **values)


def test_operational_health_is_healthy_without_signals() -> None:
    health = summarize_dashboard_operational_health(
        pipeline_queue=_pipeline_queue(),
        embedding_backlog=_embedding_backlog(),
        recent_failures=_recent_failures(),
    )

    assert health.status == "healthy"
    assert health.signal_count == 0
    assert health.signals == ()


def test_operational_health_promotes_critical_signals() -> None:
    health = summarize_dashboard_operational_health(
        pipeline_queue=_pipeline_queue(
            stale_running_count=1,
            retryable_failed_count=2,
        ),
        embedding_backlog=_embedding_backlog(exhausted_failed_count=1),
        recent_failures=_recent_failures(provider_alert_count=3),
    )

    assert health.status == "critical"
    assert health.critical_count == 2
    assert health.warning_count == 2
    assert [signal.code for signal in health.signals[:2]] == [
        "embedding_exhausted",
        "pipeline_stale",
    ]


def test_operational_health_uses_warning_when_no_critical_signals_exist() -> None:
    health = summarize_dashboard_operational_health(
        pipeline_queue=_pipeline_queue(retryable_failed_count=1),
        embedding_backlog=_embedding_backlog(),
        recent_failures=_recent_failures(app_error_count=1),
    )

    assert health.status == "warning"
    assert health.critical_count == 0
    assert health.warning_count == 2
