from datetime import UTC, datetime

import pytest

from app.core.foreground_production_shutdown import (
    SHUTDOWN_PLAN_BLOCKED,
    SHUTDOWN_PLAN_READY,
    SHUTDOWN_PLAN_WARNING,
    SHUTDOWN_STATUS_PLANNED,
    ForegroundProductionShutdownOptions,
    ProcessObservation,
    build_foreground_production_shutdown_evidence,
    build_foreground_production_shutdown_plan,
    foreground_production_shutdown_evidence_payload,
    foreground_production_shutdown_plan_payload,
    render_foreground_production_shutdown_markdown,
)


def test_build_foreground_production_shutdown_plan_ready_with_matching_process(
    tmp_path,
) -> None:
    pid_file = tmp_path / "app.pid"
    pid_file.write_text("1234\n", encoding="utf-8")
    observation = ProcessObservation(
        process_id=1234,
        exists=True,
        command_line=("python", "-m", "uvicorn", "app.main:create_app", "--factory"),
    )

    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            log_file="shutdown.log",
            port=18080,
        ),
        generated_at=datetime(2026, 7, 20, 9, 10, 11, tzinfo=UTC),
        process_observation=observation,
        port_reachable=True,
    )
    payload = foreground_production_shutdown_plan_payload(plan)

    assert plan.status == SHUTDOWN_PLAN_READY
    assert plan.failed_count == 0
    assert payload["generated_at_label"] == "2026-07-20 09:10:11"
    assert payload["process_observation"]["exists"] is True
    assert "app.main:create_app" in payload["process_observation"]["shell_command"]


def test_build_foreground_production_shutdown_plan_warns_for_dry_run_missing_pid(
    tmp_path,
) -> None:
    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="missing.pid",
            require_pid_file=False,
            check_port_reachable=False,
        )
    )

    assert plan.status == SHUTDOWN_PLAN_WARNING
    assert plan.process_id is None
    assert plan.warning_count == 4
    assert plan.failed_count == 0


def test_build_foreground_production_shutdown_plan_blocks_on_invalid_pid(
    tmp_path,
) -> None:
    (tmp_path / "app.pid").write_text("not-a-pid", encoding="utf-8")

    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            check_port_reachable=False,
        )
    )

    assert plan.status == SHUTDOWN_PLAN_BLOCKED
    assert any(check.code == "pid_file" for check in plan.checks)


def test_build_foreground_production_shutdown_plan_blocks_on_command_mismatch(
    tmp_path,
) -> None:
    (tmp_path / "app.pid").write_text("1234", encoding="utf-8")
    observation = ProcessObservation(
        process_id=1234,
        exists=True,
        command_line=("python", "-m", "http.server"),
    )

    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            check_port_reachable=False,
        ),
        process_observation=observation,
    )

    assert plan.status == SHUTDOWN_PLAN_BLOCKED
    assert any(check.code == "command_guard" for check in plan.checks)


def test_build_foreground_production_shutdown_plan_warns_on_stale_pid(
    tmp_path,
) -> None:
    (tmp_path / "app.pid").write_text("1234", encoding="utf-8")
    observation = ProcessObservation(process_id=1234, exists=False)

    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            check_port_reachable=False,
        ),
        process_observation=observation,
    )

    assert plan.status == SHUTDOWN_PLAN_WARNING
    assert plan.process_id == 1234


def test_build_foreground_production_shutdown_plan_blocks_on_same_pid_and_log_path(
    tmp_path,
) -> None:
    (tmp_path / "app.pid").write_text("1234", encoding="utf-8")

    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            log_file="app.pid",
            check_port_reachable=False,
        ),
        process_observation=ProcessObservation(process_id=1234, exists=False),
    )

    assert plan.status == SHUTDOWN_PLAN_BLOCKED
    assert any(check.code == "log_path" for check in plan.checks)


def test_foreground_production_shutdown_evidence_payload_and_markdown(tmp_path) -> None:
    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            require_pid_file=False,
            check_port_reachable=False,
        )
    )
    evidence = build_foreground_production_shutdown_evidence(
        plan,
        status=SHUTDOWN_STATUS_PLANNED,
        dry_run=True,
        generated_at=datetime(2026, 7, 20, 9, 30, tzinfo=UTC),
        message="planned",
    )
    payload = foreground_production_shutdown_evidence_payload(evidence)
    markdown = render_foreground_production_shutdown_markdown(payload)

    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert payload["generated_at_label"] == "2026-07-20 09:30:00"
    assert "Foreground Production Shutdown Evidence" in markdown


def test_foreground_production_shutdown_evidence_rejects_unknown_status(tmp_path) -> None:
    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            require_pid_file=False,
            check_port_reachable=False,
        )
    )

    with pytest.raises(ValueError, match="unsupported"):
        build_foreground_production_shutdown_evidence(plan, status="unknown", dry_run=True)


def test_build_foreground_production_shutdown_plan_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="port"):
        build_foreground_production_shutdown_plan(ForegroundProductionShutdownOptions(port=0))
    with pytest.raises(ValueError, match="pid_file"):
        build_foreground_production_shutdown_plan(ForegroundProductionShutdownOptions(pid_file=""))
