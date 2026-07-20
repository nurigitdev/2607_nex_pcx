from datetime import UTC, datetime

from app.core import shutdown_drain_check
from app.core.config import Settings
from app.core.embedding_jobs import (
    EmbeddingJobBacklogProfileSummary,
    EmbeddingJobBacklogSummary,
)
from app.core.pipeline_jobs import PipelineQueueSummary
from app.core.shutdown_drain_check import (
    DRAIN_CHECK_FAILED,
    DRAIN_CHECK_SKIPPED,
    DRAIN_STATUS_BLOCKED,
    DRAIN_STATUS_READY,
    DRAIN_STATUS_WARNING,
    build_shutdown_drain_report,
    render_shutdown_drain_markdown,
    shutdown_drain_report_payload,
)


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
        oldest_queued_at=(
            datetime(2026, 7, 17, 1, 2, 3, tzinfo=UTC) if values["queued_count"] else None
        ),
        oldest_stale_lease_expires_at=(
            datetime(2026, 7, 17, 1, 3, 4, tzinfo=UTC) if values["stale_running_count"] else None
        ),
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
    profile = EmbeddingJobBacklogProfileSummary(
        profile_name="kure_v1",
        total_count=values["total_count"],
        pending_count=values["pending_count"],
        running_count=values["running_count"],
        stale_running_count=values["stale_running_count"],
        reclaimable_stale_running_count=values["reclaimable_stale_running_count"],
        failed_count=values["failed_count"],
        retryable_failed_count=values["retryable_failed_count"],
        exhausted_failed_count=values["exhausted_failed_count"],
        succeeded_count=values["succeeded_count"],
        skipped_count=values["skipped_count"],
        oldest_pending_at=(
            datetime(2026, 7, 17, 2, 3, 4, tzinfo=UTC) if values["pending_count"] else None
        ),
        oldest_stale_lease_expires_at=(
            datetime(2026, 7, 17, 2, 4, 5, tzinfo=UTC) if values["stale_running_count"] else None
        ),
    )
    return EmbeddingJobBacklogSummary(profile_summaries=(profile,), **values)


def test_report_blocks_and_skips_queue_checks_without_database_url() -> None:
    report = build_shutdown_drain_report(Settings(database_url=None))
    payload = shutdown_drain_report_payload(report)

    assert report.status == DRAIN_STATUS_BLOCKED
    assert report.failed_count == 1
    assert report.skipped_count == 2
    assert payload["status"] == "blocked"
    assert [check.status for check in report.checks] == [
        DRAIN_CHECK_FAILED,
        DRAIN_CHECK_SKIPPED,
        DRAIN_CHECK_SKIPPED,
    ]


def test_report_is_ready_when_pipeline_and_embedding_queues_are_drained(monkeypatch) -> None:
    monkeypatch.setattr(
        shutdown_drain_check,
        "get_pipeline_queue_summary",
        lambda database_url: _pipeline_queue(),
    )
    monkeypatch.setattr(
        shutdown_drain_check,
        "get_embedding_job_backlog_summary",
        lambda database_url: _embedding_backlog(),
    )

    report = build_shutdown_drain_report(
        Settings(database_url="postgresql://example/db"),
        checked_at=datetime(2026, 7, 17, 3, 4, 5, tzinfo=UTC),
    )
    payload = shutdown_drain_report_payload(report)
    markdown = render_shutdown_drain_markdown(payload)

    assert report.status == DRAIN_STATUS_READY
    assert report.passed_count == 3
    assert payload["checked_at_label"] == "2026-07-17 03:04:05"
    assert "Pipeline queue is drained" in markdown
    assert '"total_count": 0' in markdown


def test_report_blocks_when_running_or_exhausted_work_remains(monkeypatch) -> None:
    monkeypatch.setattr(
        shutdown_drain_check,
        "get_pipeline_queue_summary",
        lambda database_url: _pipeline_queue(
            total_count=3,
            running_count=1,
            stale_running_count=1,
            exhausted_failed_count=1,
        ),
    )
    monkeypatch.setattr(
        shutdown_drain_check,
        "get_embedding_job_backlog_summary",
        lambda database_url: _embedding_backlog(
            total_count=2,
            running_count=1,
            exhausted_failed_count=1,
        ),
    )

    report = build_shutdown_drain_report(Settings(database_url="postgresql://example/db"))
    payload = shutdown_drain_report_payload(report)
    details = " ".join(str(check["detail"]) for check in payload["checks"])

    assert report.status == DRAIN_STATUS_BLOCKED
    assert report.failed_count == 2
    assert "1 running" in details
    assert "1 stale_running" in details
    assert "1 exhausted_failed" in details


def test_report_warns_when_only_claimable_or_pending_work_remains(monkeypatch) -> None:
    monkeypatch.setattr(
        shutdown_drain_check,
        "get_pipeline_queue_summary",
        lambda database_url: _pipeline_queue(
            total_count=3,
            queued_count=1,
            retryable_failed_count=1,
            retryable_canceled_count=1,
        ),
    )
    monkeypatch.setattr(
        shutdown_drain_check,
        "get_embedding_job_backlog_summary",
        lambda database_url: _embedding_backlog(
            total_count=2,
            pending_count=1,
            retryable_failed_count=1,
        ),
    )

    report = build_shutdown_drain_report(Settings(database_url="postgresql://example/db"))
    payload = shutdown_drain_report_payload(report)
    pipeline_metadata = payload["checks"][1]["metadata"]["summary"]
    embedding_metadata = payload["checks"][2]["metadata"]["summary"]

    assert report.status == DRAIN_STATUS_WARNING
    assert report.warning_count == 2
    assert pipeline_metadata["claimable_count"] == 3
    assert embedding_metadata["profile_summaries"][0]["profile_name"] == "kure_v1"
    assert embedding_metadata["profile_summaries"][0]["oldest_pending_at"]


def test_report_blocks_when_queue_summary_collection_raises(monkeypatch) -> None:
    def raise_pipeline_error(database_url: str) -> PipelineQueueSummary:
        raise RuntimeError(f"cannot query {database_url}")

    monkeypatch.setattr(
        shutdown_drain_check,
        "get_pipeline_queue_summary",
        raise_pipeline_error,
    )
    monkeypatch.setattr(
        shutdown_drain_check,
        "get_embedding_job_backlog_summary",
        lambda database_url: _embedding_backlog(),
    )

    report = build_shutdown_drain_report(Settings(database_url="postgresql://example/db"))
    payload = shutdown_drain_report_payload(report)

    assert report.status == DRAIN_STATUS_BLOCKED
    assert payload["checks"][1]["status"] == DRAIN_CHECK_FAILED
    assert "cannot query" in payload["checks"][1]["metadata"]["error"]


def test_markdown_handles_checks_without_metadata() -> None:
    report = build_shutdown_drain_report(Settings(database_url=None))
    markdown = render_shutdown_drain_markdown(shutdown_drain_report_payload(report))

    assert "# NeX_PCX Shutdown Drain Check" in markdown
    assert "| database_url | failed |" in markdown
