from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.foreground_go_live_summary import (
    SUMMARY_STATUS_BLOCKED,
    SUMMARY_STATUS_WARNING,
    ForegroundGoLiveEvidenceSpec,
    ForegroundGoLiveSummaryOptions,
    build_foreground_go_live_summary,
    foreground_go_live_summary_payload,
    render_foreground_go_live_summary_markdown,
)


def test_foreground_go_live_summary_accepts_foreground_warning(tmp_path: Path) -> None:
    write_json(tmp_path / "artifacts" / "foreground_operations_validation.json", "warning")
    write_json(tmp_path / "artifacts" / "production_environment_validation.json", "ready")
    write_json(tmp_path / "artifacts" / "go_live_evidence.json", "ready")
    write_json(tmp_path / "artifacts" / "go_live_smoke.json", "ready")
    write_json(tmp_path / "artifacts" / "app_host_service_restart_validation.json", "blocked")

    summary = build_foreground_go_live_summary(
        ForegroundGoLiveSummaryOptions(workdir=tmp_path),
        checked_at=datetime(2026, 7, 17, 10, 11, 12, tzinfo=UTC),
    )
    payload = foreground_go_live_summary_payload(summary)
    markdown = render_foreground_go_live_summary_markdown(payload)

    assert summary.status == SUMMARY_STATUS_WARNING
    assert payload["checked_at_label"] == "2026-07-17 10:11:12"
    assert payload["passed_count"] == 4
    assert payload["warning_count"] == 1
    assert payload["failed_count"] == 0
    assert "Optional hardening evidence" in markdown


def test_foreground_go_live_summary_blocks_missing_required_evidence(tmp_path: Path) -> None:
    write_json(tmp_path / "artifacts" / "foreground_operations_validation.json", "warning")

    summary = build_foreground_go_live_summary(ForegroundGoLiveSummaryOptions(workdir=tmp_path))

    assert summary.status == SUMMARY_STATUS_BLOCKED
    assert summary.failed_count == 3
    assert any("missing" in check.detail for check in summary.checks)


def test_foreground_go_live_summary_blocks_unaccepted_required_status(tmp_path: Path) -> None:
    write_json(tmp_path / "artifacts" / "foreground_operations_validation.json", "blocked")
    write_json(tmp_path / "artifacts" / "production_environment_validation.json", "ready")
    write_json(tmp_path / "artifacts" / "go_live_evidence.json", "ready")
    write_json(tmp_path / "artifacts" / "go_live_smoke.json", "ready")

    summary = build_foreground_go_live_summary(ForegroundGoLiveSummaryOptions(workdir=tmp_path))

    assert summary.status == SUMMARY_STATUS_BLOCKED
    assert summary.checks[0].status == "failed"


def test_foreground_go_live_summary_warns_for_invalid_optional_json(tmp_path: Path) -> None:
    write_json(tmp_path / "artifacts" / "foreground_operations_validation.json", "warning")
    write_json(tmp_path / "artifacts" / "production_environment_validation.json", "ready")
    write_json(tmp_path / "artifacts" / "go_live_evidence.json", "ready")
    write_json(tmp_path / "artifacts" / "go_live_smoke.json", "ready")
    optional_path = tmp_path / "artifacts" / "app_host_service_restart_validation.json"
    optional_path.parent.mkdir(parents=True, exist_ok=True)
    optional_path.write_text("{", encoding="utf-8")

    summary = build_foreground_go_live_summary(ForegroundGoLiveSummaryOptions(workdir=tmp_path))

    assert summary.status == SUMMARY_STATUS_WARNING
    assert summary.warning_count == 1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            ForegroundGoLiveSummaryOptions(required_evidence=()),
            "at least one required",
        ),
        (
            ForegroundGoLiveSummaryOptions(
                required_evidence=(
                    ForegroundGoLiveEvidenceSpec(
                        code="",
                        path="artifact.json",
                        accepted_statuses=("ready",),
                    ),
                )
            ),
            "evidence code",
        ),
        (
            ForegroundGoLiveSummaryOptions(
                required_evidence=(
                    ForegroundGoLiveEvidenceSpec(
                        code="x",
                        path="",
                        accepted_statuses=("ready",),
                    ),
                )
            ),
            "evidence path",
        ),
        (
            ForegroundGoLiveSummaryOptions(
                required_evidence=(
                    ForegroundGoLiveEvidenceSpec(
                        code="x",
                        path="artifact.json",
                        accepted_statuses=(),
                    ),
                )
            ),
            "accepted evidence status",
        ),
    ],
)
def test_foreground_go_live_summary_rejects_invalid_options(
    options: ForegroundGoLiveSummaryOptions,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_foreground_go_live_summary(options)


def write_json(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "{"
            f'"status": "{status}", '
            '"checked_at_label": "2026-07-17 10:00:00", '
            '"passed_count": 1, '
            '"warning_count": 0, '
            '"failed_count": 0'
            "}\n"
        ),
        encoding="utf-8",
    )
