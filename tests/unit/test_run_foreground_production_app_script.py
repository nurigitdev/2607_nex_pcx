import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_run_foreground_production_app_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_foreground_production_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_foreground_production_app_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_foreground_production_app = _load_run_foreground_production_app_module()


def test_main_writes_dry_run_launch_evidence(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "launch.json"
    markdown_output = tmp_path / "launch.md"
    monkeypatch.setenv("NEX_PCX_DATABASE_URL", "postgresql://user:secret@db/app")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foreground_production_app.py",
            "--workdir",
            str(tmp_path),
            "--python-bin",
            "python",
            "--dry-run",
            "--skip-port-check",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = run_foreground_production_app.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["plan"]["status"] == "warning"
    assert "secret" not in json.dumps(payload)
    assert "Foreground Production Launch Evidence" in markdown_output.read_text(encoding="utf-8")


def test_main_returns_nonzero_when_launch_plan_is_blocked(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "blocked.json"
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foreground_production_app.py",
            "--workdir",
            str(tmp_path),
            "--python-bin",
            "python",
            "--skip-port-check",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = run_foreground_production_app.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert payload["plan"]["failed_count"] == 1


def test_main_rejects_allow_missing_database_url_without_dry_run(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foreground_production_app.py",
            "--allow-missing-database-url",
        ],
    )

    with pytest.raises(SystemExit):
        run_foreground_production_app.main()


def test_run_foreground_process_writes_pid_log_and_final_evidence(monkeypatch, tmp_path):
    class FakeProcess:
        pid = 12345
        stdout = ["ready\n"]

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

    process = FakeProcess()
    plan = run_foreground_production_app.build_foreground_production_launch_plan(
        run_foreground_production_app.ForegroundProductionLaunchOptions(
            workdir=tmp_path,
            python_bin="python",
            require_database_url=False,
            check_port_available=False,
            pid_file="run/app.pid",
            log_file="run/app.log",
        ),
        environ={},
    )
    json_output = tmp_path / "launch.json"
    markdown_output = tmp_path / "launch.md"

    monkeypatch.setattr(run_foreground_production_app.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(
        run_foreground_production_app,
        "_wait_for_health",
        lambda **kwargs: {"startup_health_status": "passed"},
    )

    exit_code = run_foreground_production_app._run_foreground_process(
        plan=plan,
        json_output=str(json_output),
        markdown_output=str(markdown_output),
        startup_timeout_seconds=0,
        pretty=True,
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "exited"
    assert payload["process_id"] == 12345
    assert (tmp_path / "run" / "app.pid").read_text(encoding="utf-8") == "12345\n"
    assert "ready" in (tmp_path / "run" / "app.log").read_text(encoding="utf-8")
