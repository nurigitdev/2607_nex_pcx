from datetime import UTC, datetime

import pytest

from app.core.foreground_production_launch import (
    LAUNCH_PLAN_BLOCKED,
    LAUNCH_PLAN_READY,
    LAUNCH_PLAN_WARNING,
    LAUNCH_STATUS_PLANNED,
    ForegroundProductionLaunchOptions,
    build_foreground_production_launch_evidence,
    build_foreground_production_launch_plan,
    foreground_production_launch_evidence_payload,
    foreground_production_launch_plan_payload,
    render_foreground_production_launch_markdown,
    resolve_launch_path,
)


def test_build_foreground_production_launch_plan_ready_with_configured_env(tmp_path) -> None:
    python_bin = tmp_path / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=tmp_path,
            python_bin=".venv/bin/python",
            host="0.0.0.0",
            port=18080,
        ),
        generated_at=datetime(2026, 7, 20, 9, 1, 2, tzinfo=UTC),
        environ={"NEX_PCX_DATABASE_URL": "postgresql://user:secret@db/nex_pcx_app"},
        port_available=True,
    )
    payload = foreground_production_launch_plan_payload(plan)

    assert plan.status == LAUNCH_PLAN_READY
    assert plan.failed_count == 0
    assert payload["health_url"] == "http://127.0.0.1:18080/healthz"
    assert payload["database_url_configured"] is True
    assert "secret" not in str(payload)
    assert ".venv/bin/python -m uvicorn" in payload["shell_command"]


def test_build_foreground_production_launch_plan_warns_when_checks_are_skipped(
    tmp_path,
) -> None:
    plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=tmp_path,
            python_bin="python",
            require_database_url=False,
            check_port_available=False,
        ),
        environ={},
    )

    assert plan.status == LAUNCH_PLAN_WARNING
    assert plan.warning_count == 2
    assert plan.failed_count == 0


def test_build_foreground_production_launch_plan_blocks_on_missing_database_url(
    tmp_path,
) -> None:
    plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=tmp_path,
            python_bin="python",
        ),
        environ={},
        port_available=True,
    )

    assert plan.status == LAUNCH_PLAN_BLOCKED
    assert any(check.code == "database_url" for check in plan.checks)


def test_build_foreground_production_launch_plan_blocks_on_occupied_port(
    tmp_path,
) -> None:
    plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=tmp_path,
            python_bin="python",
        ),
        environ={"NEX_PCX_DATABASE_URL": "configured"},
        port_available=False,
    )

    assert plan.status == LAUNCH_PLAN_BLOCKED
    assert any(check.code == "port_available" for check in plan.checks)


def test_build_foreground_production_launch_plan_rejects_invalid_options(tmp_path) -> None:
    with pytest.raises(ValueError, match="port"):
        build_foreground_production_launch_plan(
            ForegroundProductionLaunchOptions(workdir=tmp_path, port=0)
        )
    with pytest.raises(ValueError, match="pid_file"):
        build_foreground_production_launch_plan(
            ForegroundProductionLaunchOptions(workdir=tmp_path, pid_file="")
        )


def test_foreground_production_launch_evidence_payload_and_markdown(tmp_path) -> None:
    plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=tmp_path,
            python_bin="python",
            require_database_url=False,
            check_port_available=False,
        ),
        generated_at=datetime(2026, 7, 20, 9, 0, tzinfo=UTC),
        environ={},
    )
    evidence = build_foreground_production_launch_evidence(
        plan,
        status=LAUNCH_STATUS_PLANNED,
        dry_run=True,
        generated_at=datetime(2026, 7, 20, 9, 5, tzinfo=UTC),
        message="planned",
        metadata={"startup_health_status": "skipped"},
    )
    payload = foreground_production_launch_evidence_payload(evidence)
    markdown = render_foreground_production_launch_markdown(payload)

    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert payload["metadata"]["startup_health_status"] == "skipped"
    assert "Foreground Production Launch Evidence" in markdown


def test_foreground_production_launch_evidence_rejects_unknown_status(tmp_path) -> None:
    plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=tmp_path,
            python_bin="python",
            require_database_url=False,
            check_port_available=False,
        ),
        environ={},
    )

    with pytest.raises(ValueError, match="unsupported"):
        build_foreground_production_launch_evidence(plan, status="unknown", dry_run=True)


def test_resolve_launch_path_keeps_absolute_path(tmp_path) -> None:
    assert resolve_launch_path(tmp_path, "/tmp/app.pid") == resolve_launch_path("/", "/tmp/app.pid")
    assert resolve_launch_path(tmp_path, "artifacts/app.pid") == tmp_path / "artifacts/app.pid"
