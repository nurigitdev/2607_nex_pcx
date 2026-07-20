import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.foreground_app_worker_supervisor import (
    ENV_FOREGROUND_CHECK_PORT_AVAILABLE,
    ENV_FOREGROUND_EMBEDDING_LIMIT_PER_PROFILE,
    ENV_FOREGROUND_GUARD_HEALTH_TIMEOUT_SECONDS,
    ENV_FOREGROUND_NO_DEFAULT_QWEN_TOKEN_GUARD,
    ENV_FOREGROUND_PIPELINE_LIMIT,
    ENV_FOREGROUND_PORT,
    ENV_FOREGROUND_SUPERVISOR_LOG_FILE,
    ENV_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS,
    ENV_FOREGROUND_WORKER_FAILURE_TOLERANCE,
    SUPERVISOR_PLAN_BLOCKED,
    SUPERVISOR_PLAN_READY,
    SUPERVISOR_PLAN_WARNING,
    SUPERVISOR_STATUS_PLANNED,
    ForegroundAppWorkerSupervisorOptions,
    ForegroundWorkerCycleObservation,
    build_foreground_app_worker_supervisor_evidence,
    build_foreground_app_worker_supervisor_plan,
    foreground_app_worker_supervisor_evidence_payload,
    foreground_app_worker_supervisor_options_from_environ,
    payload_to_json,
    render_foreground_app_worker_supervisor_markdown,
)


def test_supervisor_plan_ready_with_worker_command(tmp_path: Path) -> None:
    plan = build_foreground_app_worker_supervisor_plan(
        ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            python_bin="python",
            check_port_available=True,
        ),
        environ={"NEX_PCX_DATABASE_URL": "postgresql://user:secret@db/app"},
        port_available=True,
        generated_at=datetime(2026, 7, 20, 1, 2, 3, tzinfo=UTC),
    )

    assert plan.status == SUPERVISOR_PLAN_READY
    assert plan.failed_count == 0
    assert "scripts/run_foreground_workers.py" in plan.worker_command
    assert "--continue-on-command-failure" in plan.worker_command
    assert "--no-default-qwen-token-guard" not in plan.worker_command
    assert plan.launch_plan.database_url_configured is True


def test_supervisor_plan_warns_when_pipeline_cycle_disabled(tmp_path: Path) -> None:
    plan = build_foreground_app_worker_supervisor_plan(
        ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            python_bin="python",
            pipeline_limit=0,
            check_port_available=False,
            no_default_qwen_token_guard=True,
        ),
        environ={"NEX_PCX_DATABASE_URL": "postgresql://user:secret@db/app"},
    )

    assert plan.status == SUPERVISOR_PLAN_WARNING
    assert plan.warning_count == 2
    assert "--no-default-qwen-token-guard" in plan.worker_command


def test_supervisor_options_from_environment_override_defaults(tmp_path: Path) -> None:
    options = foreground_app_worker_supervisor_options_from_environ(
        {
            ENV_FOREGROUND_PORT: "18080",
            ENV_FOREGROUND_SUPERVISOR_LOG_FILE: "logs/supervisor.log",
            ENV_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS: "2.5",
            ENV_FOREGROUND_WORKER_FAILURE_TOLERANCE: "2",
            ENV_FOREGROUND_PIPELINE_LIMIT: "3",
            ENV_FOREGROUND_EMBEDDING_LIMIT_PER_PROFILE: "7",
            ENV_FOREGROUND_GUARD_HEALTH_TIMEOUT_SECONDS: "1.25",
            ENV_FOREGROUND_NO_DEFAULT_QWEN_TOKEN_GUARD: "yes",
            ENV_FOREGROUND_CHECK_PORT_AVAILABLE: "off",
        },
        defaults=ForegroundAppWorkerSupervisorOptions(workdir=tmp_path),
    )

    assert options.workdir == tmp_path
    assert options.port == 18080
    assert options.log_file == "logs/supervisor.log"
    assert options.worker_cycle_interval_seconds == 2.5
    assert options.worker_failure_tolerance == 2
    assert options.pipeline_limit == 3
    assert options.embedding_limit_per_profile == 7
    assert options.guard_health_timeout_seconds == 1.25
    assert options.no_default_qwen_token_guard is True
    assert options.check_port_available is False


def test_supervisor_options_from_environment_reject_invalid_values(
    tmp_path: Path,
) -> None:
    base = ForegroundAppWorkerSupervisorOptions(workdir=tmp_path)

    with pytest.raises(ValueError, match=ENV_FOREGROUND_PORT):
        foreground_app_worker_supervisor_options_from_environ(
            {ENV_FOREGROUND_PORT: "not-a-port"},
            defaults=base,
        )
    with pytest.raises(ValueError, match=ENV_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS):
        foreground_app_worker_supervisor_options_from_environ(
            {ENV_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS: "fast"},
            defaults=base,
        )
    blocked_plan = build_foreground_app_worker_supervisor_plan(
        ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            worker_failure_tolerance=-1,
            require_database_url=False,
            check_port_available=False,
        ),
        environ={},
    )
    assert blocked_plan.status == SUPERVISOR_PLAN_BLOCKED
    assert any(
        "worker_failure_tolerance" in (check.metadata or {}).get("invalid_fields", [])
        for check in blocked_plan.checks
    )
    with pytest.raises(ValueError, match=ENV_FOREGROUND_CHECK_PORT_AVAILABLE):
        foreground_app_worker_supervisor_options_from_environ(
            {ENV_FOREGROUND_CHECK_PORT_AVAILABLE: "maybe"},
            defaults=base,
        )


def test_supervisor_plan_blocks_on_launch_and_path_conflicts(tmp_path: Path) -> None:
    plan = build_foreground_app_worker_supervisor_plan(
        ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            python_bin="python",
            supervisor_pid_file="run/app.pid",
            web_pid_file="run/app.pid",
            worker_cycle_interval_seconds=0,
            embedding_limit_per_profile=0,
            check_port_available=False,
        ),
        environ={},
    )

    assert plan.status == SUPERVISOR_PLAN_BLOCKED
    assert plan.failed_count == 3
    assert {check.code for check in plan.checks if check.status == "failed"} == {
        "launch_plan",
        "supervisor_paths",
        "worker_cycle",
    }


def test_supervisor_evidence_payload_markdown_and_json_do_not_expose_secret(
    tmp_path: Path,
) -> None:
    plan = build_foreground_app_worker_supervisor_plan(
        ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            python_bin="python",
            check_port_available=False,
        ),
        environ={"NEX_PCX_DATABASE_URL": "postgresql://user:secret@db/app"},
    )
    cycle = ForegroundWorkerCycleObservation(
        index=1,
        command=plan.worker_command,
        exit_code=0,
        elapsed_ms=25,
        status="succeeded",
        message='{"status":"completed"}',
    )
    evidence = build_foreground_app_worker_supervisor_evidence(
        plan,
        status=SUPERVISOR_STATUS_PLANNED,
        dry_run=True,
        supervisor_process_id=10,
        web_process_id=11,
        started_at=datetime(2026, 7, 20, 1, 3, 0, tzinfo=UTC),
        worker_cycles=(cycle,),
        message="planned",
    )
    payload = foreground_app_worker_supervisor_evidence_payload(evidence)
    markdown = render_foreground_app_worker_supervisor_markdown(payload)
    json_text = payload_to_json(payload, pretty=True)

    assert payload["worker_cycle_count"] == 1
    assert payload["failed_worker_cycle_count"] == 0
    assert "Foreground App Worker Supervisor Evidence" in markdown
    assert "secret" not in json_text
    assert json.loads(json_text)["plan"]["status"] == SUPERVISOR_PLAN_WARNING


def test_supervisor_evidence_rejects_unknown_status(tmp_path: Path) -> None:
    plan = build_foreground_app_worker_supervisor_plan(
        ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            python_bin="python",
            require_database_url=False,
            check_port_available=False,
        ),
        environ={},
    )

    with pytest.raises(ValueError, match="unsupported"):
        build_foreground_app_worker_supervisor_evidence(
            plan,
            status="unknown",
            dry_run=True,
        )


def test_supervisor_plan_rejects_invalid_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="port"):
        build_foreground_app_worker_supervisor_plan(
            ForegroundAppWorkerSupervisorOptions(workdir=tmp_path, port=0),
        )
    with pytest.raises(ValueError, match="worker_json_output"):
        build_foreground_app_worker_supervisor_plan(
            ForegroundAppWorkerSupervisorOptions(
                workdir=tmp_path,
                worker_json_output=" ",
            ),
        )
