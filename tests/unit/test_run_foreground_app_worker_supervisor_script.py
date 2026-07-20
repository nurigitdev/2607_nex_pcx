import importlib.util
import json
import sys
from pathlib import Path


def _load_run_foreground_app_worker_supervisor_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_foreground_app_worker_supervisor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_foreground_app_worker_supervisor_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_foreground_app_worker_supervisor = _load_run_foreground_app_worker_supervisor_module()


def test_main_writes_dry_run_supervisor_evidence(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "supervisor.json"
    markdown_output = tmp_path / "supervisor.md"
    monkeypatch.setenv("NEX_PCX_DATABASE_URL", "postgresql://user:secret@db/app")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foreground_app_worker_supervisor.py",
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

    exit_code = run_foreground_app_worker_supervisor.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["plan"]["status"] == "warning"
    assert "secret" not in json.dumps(payload)
    assert "Foreground App Worker Supervisor Evidence" in markdown_output.read_text(
        encoding="utf-8"
    )


def test_main_returns_nonzero_when_supervisor_plan_is_blocked(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "blocked.json"
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foreground_app_worker_supervisor.py",
            "--workdir",
            str(tmp_path),
            "--python-bin",
            "python",
            "--skip-port-check",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = run_foreground_app_worker_supervisor.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert payload["plan"]["failed_count"] == 1


def test_run_supervisor_starts_web_runs_one_worker_cycle_and_exits(
    monkeypatch,
    tmp_path,
) -> None:
    json_output = tmp_path / "supervisor.json"
    markdown_output = tmp_path / "supervisor.md"
    plan = run_foreground_app_worker_supervisor.build_foreground_app_worker_supervisor_plan(
        run_foreground_app_worker_supervisor.ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            python_bin="python",
            require_database_url=False,
            check_port_available=False,
            supervisor_pid_file="run/supervisor.pid",
            web_pid_file="run/web.pid",
            log_file="run/supervisor.log",
            worker_cycle_interval_seconds=0.01,
        ),
        environ={},
    )
    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append(tuple(command))
        if len(popen_calls) == 1:
            return _FakeWebProcess()
        return _FakeWorkerProcess(exit_code=0)

    monkeypatch.setattr(
        run_foreground_app_worker_supervisor.subprocess,
        "Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        run_foreground_app_worker_supervisor,
        "_wait_for_health",
        lambda **kwargs: {"startup_health_status": "passed"},
    )
    monkeypatch.setattr(run_foreground_app_worker_supervisor.os, "getpid", lambda: 999)

    exit_code = run_foreground_app_worker_supervisor._run_supervisor(
        plan=plan,
        startup_timeout_seconds=0,
        max_worker_cycles=1,
        json_output=str(json_output),
        markdown_output=str(markdown_output),
        pretty=True,
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "exited"
    assert payload["supervisor_process_id"] == 999
    assert payload["web_process_id"] == 12345
    assert payload["worker_cycle_count"] == 1
    assert (tmp_path / "run" / "supervisor.pid").read_text(encoding="utf-8") == "999\n"
    assert (tmp_path / "run" / "web.pid").read_text(encoding="utf-8") == "12345\n"
    assert "worker ok" in (tmp_path / "run" / "supervisor.log").read_text(encoding="utf-8")


def test_run_supervisor_reports_failed_worker_cycle(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "supervisor.json"
    plan = run_foreground_app_worker_supervisor.build_foreground_app_worker_supervisor_plan(
        run_foreground_app_worker_supervisor.ForegroundAppWorkerSupervisorOptions(
            workdir=tmp_path,
            python_bin="python",
            require_database_url=False,
            check_port_available=False,
            supervisor_pid_file="run/supervisor.pid",
            web_pid_file="run/web.pid",
            log_file="run/supervisor.log",
            worker_cycle_interval_seconds=0.01,
        ),
        environ={},
    )
    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append(tuple(command))
        if len(popen_calls) == 1:
            return _FakeWebProcess()
        return _FakeWorkerProcess(exit_code=7, line="worker failed\n")

    monkeypatch.setattr(
        run_foreground_app_worker_supervisor.subprocess,
        "Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        run_foreground_app_worker_supervisor,
        "_wait_for_health",
        lambda **kwargs: {"startup_health_status": "passed"},
    )

    exit_code = run_foreground_app_worker_supervisor._run_supervisor(
        plan=plan,
        startup_timeout_seconds=0,
        max_worker_cycles=2,
        json_output=str(json_output),
        markdown_output=None,
        pretty=True,
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 7
    assert payload["status"] == "failed"
    assert payload["failed_worker_cycle_count"] == 1
    assert payload["worker_cycles"][0]["message"] == "worker failed"


class _FakeWebProcess:
    pid = 12345
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


class _FakeWorkerProcess:
    def __init__(self, *, exit_code: int, line: str = "worker ok\n") -> None:
        self.pid = 23456
        self.returncode = None
        self._exit_code = exit_code
        self.stdout = _FakeStdout(line)
        self._poll_count = 0

    def poll(self):
        self._poll_count += 1
        if self._poll_count == 1:
            return None
        self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class _FakeStdout:
    def __init__(self, line: str) -> None:
        self._line = line
        self._used = False

    def readline(self) -> str:
        if self._used:
            return ""
        self._used = True
        return self._line

    def __iter__(self):
        return iter(())
