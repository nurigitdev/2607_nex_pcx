import json
from datetime import UTC, datetime

import pytest

from app.core.foreground_final_handoff import (
    FINAL_HANDOFF_STATUS_BLOCKED,
    FINAL_HANDOFF_STATUS_READY,
    FINAL_HANDOFF_STATUS_WARNING,
    ForegroundFinalEvidenceSpec,
    ForegroundFinalHandoffOptions,
    build_foreground_final_handoff_report,
    foreground_final_handoff_payload,
    render_foreground_final_handoff_markdown,
)


def test_build_foreground_final_handoff_report_accepts_warning_mode(tmp_path) -> None:
    _write_json(tmp_path / "foreground.json", {"status": "warning", "warning_count": 1})
    _write_json(tmp_path / "summary.json", {"status": "ready", "passed_count": 4})
    _write_json(tmp_path / "worker-plan.json", _worker_plan_payload())
    _write_json(
        tmp_path / "manifest.json",
        {
            "file_count": 3,
            "included_count": 3,
            "missing_required_count": 0,
            "git_commit": "abc123",
        },
    )

    report = build_foreground_final_handoff_report(
        ForegroundFinalHandoffOptions(
            workdir=tmp_path,
            evidence_specs=(
                ForegroundFinalEvidenceSpec(
                    code="foreground",
                    path="foreground.json",
                    ready_statuses=("ready",),
                    warning_statuses=("warning",),
                ),
                ForegroundFinalEvidenceSpec(code="summary", path="summary.json"),
            ),
            worker_plan_path="worker-plan.json",
            handoff_manifest_path="manifest.json",
        ),
        generated_at=datetime(2026, 7, 17, 9, 0, tzinfo=UTC),
    )
    payload = foreground_final_handoff_payload(report)
    markdown = render_foreground_final_handoff_markdown(payload)

    assert report.status == FINAL_HANDOFF_STATUS_WARNING
    assert report.warning_count == 1
    assert report.failed_count == 0
    assert payload["generated_at_label"] == "2026-07-17 09:00:00"
    assert "Foreground Final Handoff Checklist" in markdown


def test_build_foreground_final_handoff_report_blocks_on_missing_evidence(tmp_path) -> None:
    _write_json(tmp_path / "summary.json", {"status": "ready"})
    _write_json(tmp_path / "worker-plan.json", _worker_plan_payload())
    _write_json(
        tmp_path / "manifest.json",
        {"file_count": 1, "included_count": 1, "missing_required_count": 0},
    )

    report = build_foreground_final_handoff_report(
        ForegroundFinalHandoffOptions(
            workdir=tmp_path,
            evidence_specs=(
                ForegroundFinalEvidenceSpec(code="missing", path="missing.json"),
                ForegroundFinalEvidenceSpec(code="summary", path="summary.json"),
            ),
            worker_plan_path="worker-plan.json",
            handoff_manifest_path="manifest.json",
        )
    )

    assert report.status == FINAL_HANDOFF_STATUS_BLOCKED
    assert report.failed_count == 1


def test_build_foreground_final_handoff_report_blocks_on_unbounded_worker_plan(
    tmp_path,
) -> None:
    _write_json(tmp_path / "foreground.json", {"status": "ready"})
    _write_json(tmp_path / "summary.json", {"status": "ready"})
    payload = _worker_plan_payload()
    payload["commands"][0]["bounded"] = False
    _write_json(tmp_path / "worker-plan.json", payload)
    _write_json(
        tmp_path / "manifest.json",
        {"file_count": 1, "included_count": 1, "missing_required_count": 0},
    )

    report = build_foreground_final_handoff_report(
        ForegroundFinalHandoffOptions(
            workdir=tmp_path,
            evidence_specs=(
                ForegroundFinalEvidenceSpec(code="foreground", path="foreground.json"),
                ForegroundFinalEvidenceSpec(code="summary", path="summary.json"),
            ),
            worker_plan_path="worker-plan.json",
            handoff_manifest_path="manifest.json",
        )
    )

    assert report.status == FINAL_HANDOFF_STATUS_BLOCKED
    assert "unbounded" in report.checks[2].detail


def test_build_foreground_final_handoff_report_warns_on_manifest_no_copy(tmp_path) -> None:
    _write_json(tmp_path / "foreground.json", {"status": "ready"})
    _write_json(tmp_path / "summary.json", {"status": "ready"})
    _write_json(tmp_path / "worker-plan.json", _worker_plan_payload())
    _write_json(
        tmp_path / "manifest.json",
        {"file_count": 3, "included_count": 2, "missing_required_count": 0},
    )

    report = build_foreground_final_handoff_report(
        ForegroundFinalHandoffOptions(
            workdir=tmp_path,
            evidence_specs=(
                ForegroundFinalEvidenceSpec(code="foreground", path="foreground.json"),
                ForegroundFinalEvidenceSpec(code="summary", path="summary.json"),
            ),
            worker_plan_path="worker-plan.json",
            handoff_manifest_path="manifest.json",
        )
    )

    assert report.status == FINAL_HANDOFF_STATUS_WARNING
    assert report.warning_count == 1


def test_build_foreground_final_handoff_report_ready_when_all_checks_pass(tmp_path) -> None:
    _write_json(tmp_path / "foreground.json", {"status": "ready"})
    _write_json(tmp_path / "summary.json", {"status": "ready"})
    _write_json(tmp_path / "worker-plan.json", _worker_plan_payload())
    _write_json(
        tmp_path / "manifest.json",
        {"file_count": 2, "included_count": 2, "missing_required_count": 0},
    )

    report = build_foreground_final_handoff_report(
        ForegroundFinalHandoffOptions(
            workdir=tmp_path,
            evidence_specs=(
                ForegroundFinalEvidenceSpec(code="foreground", path="foreground.json"),
                ForegroundFinalEvidenceSpec(code="summary", path="summary.json"),
            ),
            worker_plan_path="worker-plan.json",
            handoff_manifest_path="manifest.json",
        )
    )

    assert report.status == FINAL_HANDOFF_STATUS_READY


def test_build_foreground_final_handoff_report_rejects_empty_specs() -> None:
    with pytest.raises(ValueError, match="at least one evidence spec"):
        build_foreground_final_handoff_report(ForegroundFinalHandoffOptions(evidence_specs=()))


def _worker_plan_payload() -> dict[str, object]:
    return {
        "lease_seconds": 300,
        "embedding_limit": 5,
        "commands": [
            {"code": "pipeline_worker_help", "bounded": True},
            {"code": "embedding_worker_help", "bounded": True},
            {"code": "pipeline_worker_once", "bounded": True},
            {"code": "embedding_worker_batch", "bounded": True},
        ],
    }


def _write_json(path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
