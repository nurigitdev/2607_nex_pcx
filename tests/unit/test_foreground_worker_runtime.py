import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import app.core.foreground_worker_runtime as runtime
from app.core.foreground_production_shutdown import ProcessObservation
from app.core.foreground_worker_runtime import (
    FOREGROUND_WORKER_RUNTIME_BLOCKED,
    FOREGROUND_WORKER_RUNTIME_READY,
    FOREGROUND_WORKER_RUNTIME_WARNING,
    build_foreground_worker_runtime_report,
    foreground_worker_runtime_report_payload,
)


def test_runtime_report_ready_when_supervisor_running_and_processes_alive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 20, 2, 0, tzinfo=UTC)
    _write_json(
        tmp_path / "artifacts" / "foreground_app_worker_supervisor.json",
        {
            "status": "running",
            "generated_at": (now - timedelta(seconds=30)).isoformat(),
            "failed_worker_cycle_count": 0,
            "worker_cycle_count": 2,
            "message": "running",
        },
    )
    _write_json(
        tmp_path / "artifacts" / "foreground_worker_runner.json",
        {
            "status": "completed",
            "generated_at": (now - timedelta(seconds=10)).isoformat(),
            "failed_command_count": 0,
            "command_count": 2,
            "message": "completed",
        },
    )
    _write_pid(tmp_path / "artifacts" / "foreground_app_worker_supervisor.pid", 100)
    _write_pid(tmp_path / "artifacts" / "foreground_production_launch.pid", 101)
    monkeypatch.setattr(
        runtime,
        "inspect_process",
        lambda pid: ProcessObservation(
            process_id=pid,
            exists=True,
            command_line=("python", "supervisor" if pid == 100 else "uvicorn"),
        ),
    )

    report = build_foreground_worker_runtime_report(tmp_path, now=now)
    payload = foreground_worker_runtime_report_payload(report)

    assert report.status == FOREGROUND_WORKER_RUNTIME_READY
    assert payload["summary"]["supervisor_alive"] is True
    assert payload["summary"]["web_alive"] is True
    assert payload["supervisor_evidence"]["age_seconds"] == 30
    assert payload["worker_runner_evidence"]["age_seconds"] == 10


def test_runtime_report_warns_when_evidence_is_only_planned(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 2, 0, tzinfo=UTC)
    _write_json(
        tmp_path / "artifacts" / "foreground_app_worker_supervisor.json",
        {"status": "planned", "generated_at": now.isoformat(), "dry_run": True},
    )
    _write_json(
        tmp_path / "artifacts" / "foreground_worker_runner.json",
        {"status": "planned", "generated_at": now.isoformat(), "dry_run": True},
    )

    report = build_foreground_worker_runtime_report(tmp_path, now=now)
    payload = foreground_worker_runtime_report_payload(report)

    assert report.status == FOREGROUND_WORKER_RUNTIME_WARNING
    assert payload["supervisor_process"]["exists"] is None
    assert payload["summary"]["supervisor_status"] == "planned"


def test_runtime_report_blocks_when_running_evidence_has_stale_process(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 20, 2, 0, tzinfo=UTC)
    _write_json(
        tmp_path / "artifacts" / "foreground_app_worker_supervisor.json",
        {"status": "running", "generated_at": now.isoformat()},
    )
    _write_json(
        tmp_path / "artifacts" / "foreground_worker_runner.json",
        {"status": "completed", "generated_at": now.isoformat()},
    )
    _write_pid(tmp_path / "artifacts" / "foreground_app_worker_supervisor.pid", 100)
    _write_pid(tmp_path / "artifacts" / "foreground_production_launch.pid", 101)
    monkeypatch.setattr(
        runtime,
        "inspect_process",
        lambda pid: ProcessObservation(process_id=pid, exists=(pid == 101)),
    )

    report = build_foreground_worker_runtime_report(tmp_path, now=now)

    assert report.status == FOREGROUND_WORKER_RUNTIME_BLOCKED
    assert report.supervisor_process.exists is False
    assert report.web_process.exists is True


def test_runtime_report_blocks_on_failed_worker_runner(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 2, 0, tzinfo=UTC)
    _write_json(
        tmp_path / "artifacts" / "foreground_app_worker_supervisor.json",
        {"status": "planned", "generated_at": now.isoformat()},
    )
    _write_json(
        tmp_path / "artifacts" / "foreground_worker_runner.json",
        {
            "status": "completed",
            "generated_at": now.isoformat(),
            "failed_command_count": 1,
        },
    )

    report = build_foreground_worker_runtime_report(tmp_path, now=now)
    payload = foreground_worker_runtime_report_payload(report)

    assert report.status == FOREGROUND_WORKER_RUNTIME_BLOCKED
    assert payload["summary"]["failed_worker_command_count"] == 1


def test_runtime_report_handles_invalid_json_and_pid(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 2, 0, tzinfo=UTC)
    path = tmp_path / "artifacts" / "foreground_app_worker_supervisor.json"
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    _write_json(
        tmp_path / "artifacts" / "foreground_worker_runner.json",
        {"status": "planned", "generated_at": "not-a-date"},
    )
    (tmp_path / "artifacts" / "foreground_app_worker_supervisor.pid").write_text(
        "bad",
        encoding="utf-8",
    )

    report = build_foreground_worker_runtime_report(tmp_path, now=now)
    payload = foreground_worker_runtime_report_payload(report)

    assert report.status == FOREGROUND_WORKER_RUNTIME_WARNING
    assert "could not be read" in payload["supervisor_evidence"]["error"]
    assert payload["worker_runner_evidence"]["generated_at"] is None
    assert "valid integer" in payload["supervisor_process"]["error"]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")
