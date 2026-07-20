"""Shutdown drain checks for planned operations stops."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.core.embedding_jobs import (
    EmbeddingJobBacklogProfileSummary,
    EmbeddingJobBacklogSummary,
    get_embedding_job_backlog_summary,
)
from app.core.pipeline_jobs import PipelineQueueSummary, get_pipeline_queue_summary

SHUTDOWN_DRAIN_VERSION = 1

DRAIN_CHECK_PASSED = "passed"
DRAIN_CHECK_WARNING = "warning"
DRAIN_CHECK_FAILED = "failed"
DRAIN_CHECK_SKIPPED = "skipped"

DRAIN_STATUS_READY = "ready"
DRAIN_STATUS_WARNING = "warning"
DRAIN_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class ShutdownDrainCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ShutdownDrainReport:
    status: str
    checked_at: datetime
    checks: tuple[ShutdownDrainCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(DRAIN_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(DRAIN_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(DRAIN_CHECK_FAILED)

    @property
    def skipped_count(self) -> int:
        return self._count_status(DRAIN_CHECK_SKIPPED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


def build_shutdown_drain_report(
    settings: Settings,
    *,
    checked_at: datetime | None = None,
) -> ShutdownDrainReport:
    database_url = settings.database_url
    checks = [_database_url_check(database_url)]

    if not database_url:
        checks.extend(
            (
                ShutdownDrainCheck(
                    code="pipeline_queue",
                    status=DRAIN_CHECK_SKIPPED,
                    detail=(
                        "Pipeline queue drain check is skipped until "
                        "NEX_PCX_DATABASE_URL is configured."
                    ),
                ),
                ShutdownDrainCheck(
                    code="embedding_queue",
                    status=DRAIN_CHECK_SKIPPED,
                    detail=(
                        "Embedding queue drain check is skipped until "
                        "NEX_PCX_DATABASE_URL is configured."
                    ),
                ),
            )
        )
    else:
        checks.append(_safe_pipeline_queue_drain_check(database_url))
        checks.append(_safe_embedding_queue_drain_check(database_url))

    return ShutdownDrainReport(
        status=_overall_status(tuple(checks)),
        checked_at=checked_at or datetime.now(UTC),
        checks=tuple(checks),
    )


def shutdown_drain_report_payload(report: ShutdownDrainReport) -> dict[str, object]:
    return {
        "version": SHUTDOWN_DRAIN_VERSION,
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


def render_shutdown_drain_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# NeX_PCX Shutdown Drain Check",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Overall Status: {_text(payload.get('status'))}",
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


def _overall_status(checks: tuple[ShutdownDrainCheck, ...]) -> str:
    if any(check.status == DRAIN_CHECK_FAILED for check in checks):
        return DRAIN_STATUS_BLOCKED
    if any(check.status == DRAIN_CHECK_WARNING for check in checks):
        return DRAIN_STATUS_WARNING
    return DRAIN_STATUS_READY


def _database_url_check(database_url: str | None) -> ShutdownDrainCheck:
    if database_url:
        return ShutdownDrainCheck(
            code="database_url",
            status=DRAIN_CHECK_PASSED,
            detail="NEX_PCX_DATABASE_URL is configured.",
            metadata={"configured": True},
        )
    return ShutdownDrainCheck(
        code="database_url",
        status=DRAIN_CHECK_FAILED,
        detail="NEX_PCX_DATABASE_URL is not configured.",
        metadata={"configured": False},
    )


def _safe_pipeline_queue_drain_check(database_url: str) -> ShutdownDrainCheck:
    try:
        return _pipeline_queue_drain_check(get_pipeline_queue_summary(database_url))
    except Exception as exc:
        return ShutdownDrainCheck(
            code="pipeline_queue",
            status=DRAIN_CHECK_FAILED,
            detail=f"Pipeline queue drain check failed: {exc}",
            metadata={"error": str(exc)},
        )


def _pipeline_queue_drain_check(summary: PipelineQueueSummary) -> ShutdownDrainCheck:
    blockers = _nonzero_reasons(
        {
            "running": summary.running_count,
            "stale_running": summary.stale_running_count,
            "exhausted_failed": summary.exhausted_failed_count,
            "exhausted_canceled": summary.exhausted_canceled_count,
        }
    )
    warnings = _nonzero_reasons(
        {
            "queued": summary.queued_count,
            "retryable_failed": summary.retryable_failed_count,
            "retryable_canceled": summary.retryable_canceled_count,
        }
    )
    metadata = {"summary": _pipeline_summary_payload(summary)}

    if blockers:
        return ShutdownDrainCheck(
            code="pipeline_queue",
            status=DRAIN_CHECK_FAILED,
            detail=f"Pipeline queue is not drained: {', '.join(blockers)}.",
            metadata=metadata,
        )
    if warnings:
        return ShutdownDrainCheck(
            code="pipeline_queue",
            status=DRAIN_CHECK_WARNING,
            detail=(
                "Pipeline queue has remaining work but no running blockers: "
                f"{', '.join(warnings)}."
            ),
            metadata=metadata,
        )
    return ShutdownDrainCheck(
        code="pipeline_queue",
        status=DRAIN_CHECK_PASSED,
        detail="Pipeline queue is drained.",
        metadata=metadata,
    )


def _safe_embedding_queue_drain_check(database_url: str) -> ShutdownDrainCheck:
    try:
        return _embedding_queue_drain_check(get_embedding_job_backlog_summary(database_url))
    except Exception as exc:
        return ShutdownDrainCheck(
            code="embedding_queue",
            status=DRAIN_CHECK_FAILED,
            detail=f"Embedding queue drain check failed: {exc}",
            metadata={"error": str(exc)},
        )


def _embedding_queue_drain_check(summary: EmbeddingJobBacklogSummary) -> ShutdownDrainCheck:
    blockers = _nonzero_reasons(
        {
            "running": summary.running_count,
            "stale_running": summary.stale_running_count,
            "exhausted_failed": summary.exhausted_failed_count,
        }
    )
    warnings = _nonzero_reasons(
        {
            "pending": summary.pending_count,
            "retryable_failed": summary.retryable_failed_count,
        }
    )
    metadata = {"summary": _embedding_backlog_payload(summary)}

    if blockers:
        return ShutdownDrainCheck(
            code="embedding_queue",
            status=DRAIN_CHECK_FAILED,
            detail=f"Embedding queue is not drained: {', '.join(blockers)}.",
            metadata=metadata,
        )
    if warnings:
        return ShutdownDrainCheck(
            code="embedding_queue",
            status=DRAIN_CHECK_WARNING,
            detail=(
                "Embedding queue has remaining work but no running blockers: "
                f"{', '.join(warnings)}."
            ),
            metadata=metadata,
        )
    return ShutdownDrainCheck(
        code="embedding_queue",
        status=DRAIN_CHECK_PASSED,
        detail="Embedding queue is drained.",
        metadata=metadata,
    )


def _pipeline_summary_payload(summary: PipelineQueueSummary) -> dict[str, object]:
    return {
        "total_count": summary.total_count,
        "queued_count": summary.queued_count,
        "running_count": summary.running_count,
        "stale_running_count": summary.stale_running_count,
        "reclaimable_stale_running_count": summary.reclaimable_stale_running_count,
        "failed_count": summary.failed_count,
        "retryable_failed_count": summary.retryable_failed_count,
        "exhausted_failed_count": summary.exhausted_failed_count,
        "canceled_count": summary.canceled_count,
        "retryable_canceled_count": summary.retryable_canceled_count,
        "exhausted_canceled_count": summary.exhausted_canceled_count,
        "succeeded_count": summary.succeeded_count,
        "skipped_count": summary.skipped_count,
        "claimable_count": summary.claimable_count,
        "attention_count": summary.attention_count,
        "oldest_queued_at": _iso(summary.oldest_queued_at),
        "oldest_stale_lease_expires_at": _iso(summary.oldest_stale_lease_expires_at),
    }


def _embedding_backlog_payload(summary: EmbeddingJobBacklogSummary) -> dict[str, object]:
    return {
        "total_count": summary.total_count,
        "pending_count": summary.pending_count,
        "running_count": summary.running_count,
        "stale_running_count": summary.stale_running_count,
        "reclaimable_stale_running_count": summary.reclaimable_stale_running_count,
        "failed_count": summary.failed_count,
        "retryable_failed_count": summary.retryable_failed_count,
        "exhausted_failed_count": summary.exhausted_failed_count,
        "succeeded_count": summary.succeeded_count,
        "skipped_count": summary.skipped_count,
        "claimable_count": summary.claimable_count,
        "attention_count": summary.attention_count,
        "profile_summaries": [
            _embedding_profile_payload(profile) for profile in summary.profile_summaries
        ],
    }


def _embedding_profile_payload(
    profile: EmbeddingJobBacklogProfileSummary,
) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "total_count": profile.total_count,
        "pending_count": profile.pending_count,
        "running_count": profile.running_count,
        "stale_running_count": profile.stale_running_count,
        "reclaimable_stale_running_count": profile.reclaimable_stale_running_count,
        "failed_count": profile.failed_count,
        "retryable_failed_count": profile.retryable_failed_count,
        "exhausted_failed_count": profile.exhausted_failed_count,
        "succeeded_count": profile.succeeded_count,
        "skipped_count": profile.skipped_count,
        "claimable_count": profile.claimable_count,
        "attention_count": profile.attention_count,
        "oldest_pending_at": _iso(profile.oldest_pending_at),
        "oldest_stale_lease_expires_at": _iso(profile.oldest_stale_lease_expires_at),
    }


def _nonzero_reasons(values: dict[str, int]) -> list[str]:
    return [f"{value} {name}" for name, value in values.items() if value > 0]


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
