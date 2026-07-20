import importlib.util
import json
import signal
import sys
from pathlib import Path

from app.core.foreground_production_shutdown import ProcessObservation


def _load_stop_foreground_production_app_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "stop_foreground_production_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "stop_foreground_production_app_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stop_foreground_production_app = _load_stop_foreground_production_app_module()


def test_main_writes_dry_run_shutdown_evidence(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "shutdown.json"
    markdown_output = tmp_path / "shutdown.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stop_foreground_production_app.py",
            "--workdir",
            str(tmp_path),
            "--pid-file",
            "missing.pid",
            "--dry-run",
            "--skip-port-check",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = stop_foreground_production_app.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["plan"]["status"] == "warning"
    assert "Foreground Production Shutdown Evidence" in markdown_output.read_text(encoding="utf-8")


def test_main_accepts_custom_expected_command_markers(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "shutdown.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stop_foreground_production_app.py",
            "--workdir",
            str(tmp_path),
            "--pid-file",
            "missing.pid",
            "--dry-run",
            "--skip-port-check",
            "--expected-command-marker",
            "run_foreground_app_worker_supervisor.py",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = stop_foreground_production_app.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["plan"]["expected_command_markers"] == [
        "run_foreground_app_worker_supervisor.py"
    ]


def test_main_returns_nonzero_when_shutdown_plan_is_blocked(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "blocked.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stop_foreground_production_app.py",
            "--workdir",
            str(tmp_path),
            "--pid-file",
            "missing.pid",
            "--skip-port-check",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = stop_foreground_production_app.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert payload["plan"]["failed_count"] == 2


def test_stop_foreground_process_sends_sigterm_and_writes_evidence(monkeypatch, tmp_path):
    (tmp_path / "app.pid").write_text("12345", encoding="utf-8")
    plan = stop_foreground_production_app.build_foreground_production_shutdown_plan(
        stop_foreground_production_app.ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            log_file="shutdown.log",
            check_port_reachable=False,
        ),
        process_observation=ProcessObservation(
            process_id=12345,
            exists=True,
            command_line=("python", "-m", "uvicorn", "app.main:create_app"),
        ),
        port_reachable=True,
    )
    json_output = tmp_path / "shutdown.json"
    markdown_output = tmp_path / "shutdown.md"
    killed = {}
    monkeypatch.setattr(
        stop_foreground_production_app.os,
        "kill",
        lambda pid, sig: killed.update({"pid": pid, "signal": sig}),
    )
    monkeypatch.setattr(
        stop_foreground_production_app,
        "_wait_for_process_exit",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        stop_foreground_production_app,
        "is_tcp_port_reachable",
        lambda **kwargs: False,
    )

    exit_code = stop_foreground_production_app._stop_foreground_process(
        plan=plan,
        json_output=str(json_output),
        markdown_output=str(markdown_output),
        wait_timeout_seconds=1,
        poll_interval_seconds=0.1,
        pretty=True,
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert killed == {"pid": 12345, "signal": signal.SIGTERM}
    assert payload["status"] == "stopped"
    assert payload["port_released_after_stop"] is True
    assert "Foreground process 12345 stopped" in (tmp_path / "shutdown.log").read_text(
        encoding="utf-8"
    )


def test_stop_foreground_process_reports_no_process(tmp_path) -> None:
    plan = stop_foreground_production_app.build_foreground_production_shutdown_plan(
        stop_foreground_production_app.ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            require_pid_file=False,
            check_port_reachable=False,
        )
    )
    json_output = tmp_path / "shutdown.json"

    exit_code = stop_foreground_production_app._stop_foreground_process(
        plan=plan,
        json_output=str(json_output),
        markdown_output=None,
        wait_timeout_seconds=1,
        poll_interval_seconds=0.1,
        pretty=True,
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "no_process"


def test_stop_foreground_process_reports_timeout(monkeypatch, tmp_path) -> None:
    (tmp_path / "app.pid").write_text("12345", encoding="utf-8")
    plan = stop_foreground_production_app.build_foreground_production_shutdown_plan(
        stop_foreground_production_app.ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            log_file="shutdown.log",
            check_port_reachable=False,
        ),
        process_observation=ProcessObservation(
            process_id=12345,
            exists=True,
            command_line=("python", "-m", "uvicorn", "app.main:create_app"),
        ),
        port_reachable=True,
    )
    monkeypatch.setattr(stop_foreground_production_app.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(
        stop_foreground_production_app,
        "_wait_for_process_exit",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        stop_foreground_production_app,
        "is_tcp_port_reachable",
        lambda **kwargs: True,
    )
    json_output = tmp_path / "shutdown.json"

    exit_code = stop_foreground_production_app._stop_foreground_process(
        plan=plan,
        json_output=str(json_output),
        markdown_output=None,
        wait_timeout_seconds=1,
        poll_interval_seconds=0.1,
        pretty=True,
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["port_released_after_stop"] is False


def test_stop_foreground_process_reports_race_when_process_already_exited(
    monkeypatch,
    tmp_path,
) -> None:
    (tmp_path / "app.pid").write_text("12345", encoding="utf-8")
    plan = stop_foreground_production_app.build_foreground_production_shutdown_plan(
        stop_foreground_production_app.ForegroundProductionShutdownOptions(
            workdir=tmp_path,
            pid_file="app.pid",
            log_file="shutdown.log",
            check_port_reachable=False,
        ),
        process_observation=ProcessObservation(
            process_id=12345,
            exists=True,
            command_line=("python", "-m", "uvicorn", "app.main:create_app"),
        ),
        port_reachable=True,
    )
    monkeypatch.setattr(
        stop_foreground_production_app.os,
        "kill",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr(
        stop_foreground_production_app,
        "is_tcp_port_reachable",
        lambda **kwargs: False,
    )
    json_output = tmp_path / "shutdown.json"

    exit_code = stop_foreground_production_app._stop_foreground_process(
        plan=plan,
        json_output=str(json_output),
        markdown_output=None,
        wait_timeout_seconds=1,
        poll_interval_seconds=0.1,
        pretty=True,
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "no_process"
