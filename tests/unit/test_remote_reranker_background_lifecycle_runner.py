import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_script_module("run_remote_reranker_background.py")


class FakeCompletedProcess:
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@dataclass(frozen=True)
class FakeRequestSmokePreview:
    candidate_key: str = "candidate-1"
    rank: int = 1
    score: float = 8.75
    source_rank: int = 1
    score_components: dict[str, int] | None = None


@dataclass(frozen=True)
class FakeRequestSmokeObservation:
    request_elapsed_ms: int = 17
    provider_elapsed_ms: int = 11
    candidate_count: int = 3
    returned_count: int = 2
    runtime_metadata: dict[str, str] | None = None
    mismatches: tuple[str, ...] = ()
    error: str | None = None
    result_previews: tuple[FakeRequestSmokePreview, ...] = (FakeRequestSmokePreview(),)


@dataclass(frozen=True)
class FakeRequestSmokeReport:
    passed: bool = True
    total_elapsed_ms: int = 17
    observation: FakeRequestSmokeObservation = FakeRequestSmokeObservation(
        runtime_metadata={
            "service": "nex_pcx_reranker_provider_service",
            "backend": "qwen_reranker",
            "device": "cuda:0",
        }
    )


def _health_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "ready": True,
        "provider_type": "remote",
        "provider_model_id": "Qwen/Qwen3-Reranker-4B",
        "reranker_profile_name": "qwen3_reranker_4b",
        "device": "cuda:0",
        "runtime_metadata": {
            "service": "nex_pcx_reranker_provider_service",
            "backend": "qwen_reranker",
            "model_dir": "/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_4b",
            "model_dir_exists": True,
        },
    }
    payload.update(overrides)
    return payload


def _health_ok() -> runner.HealthObservation:
    return runner.HealthObservation(ok=True, status_code=200, payload=_health_payload(), error=None)


def test_build_background_lifecycle_plan_defaults_to_dgx_paths() -> None:
    plan = runner.build_background_lifecycle_plan(action="start")

    assert plan.ssh_target == "nexpcx@192.168.20.243"
    assert plan.base_url == "http://192.168.20.243:9104"
    assert plan.pid_file == "run/remote_reranker_provider_9104.pid"
    assert plan.log_file == "logs/remote_reranker_provider_9104.log"
    assert "[u]vicorn app.reranker_provider_service:app" in plan.process_pattern
    assert "nohup sh -c" in plan.remote_start_command
    assert "status=already_running" in plan.remote_start_command
    assert "status=not_running" in plan.remote_stop_command


def test_run_background_lifecycle_start_detects_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, str, str]] = []

    def fake_run(command: tuple[str, str, str], **_: Any) -> FakeCompletedProcess:
        commands.append(command)
        return FakeCompletedProcess(
            stdout=(
                "status=already_running\n"
                "pid=2401\n"
                "pid_file=run/remote_reranker_provider_9104.pid\n"
                "log_file=logs/remote_reranker_provider_9104.log\n"
            )
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_wait_for_health", lambda *_args, **_kwargs: _health_ok())
    plan = runner.build_background_lifecycle_plan(action="start")

    report = runner.run_background_lifecycle(plan)

    assert report.passed is True
    assert report.status == "already_running"
    assert report.pid == "2401"
    assert commands[0][0] == "ssh"
    assert commands[0][1] == "nexpcx@192.168.20.243"
    assert "nohup sh -c" in commands[0][2]


def test_run_background_lifecycle_smoke_runs_request_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: FakeCompletedProcess(stdout="status=running\npid=2401\n"),
    )
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: _health_ok())
    monkeypatch.setattr(
        runner,
        "_run_request_smoke",
        lambda *_args, **_kwargs: FakeRequestSmokeReport(),
    )
    plan = runner.build_background_lifecycle_plan(action="smoke")

    report = runner.run_background_lifecycle(plan)

    assert report.passed is True
    assert report.request_smoke_checked is True
    assert report.request_smoke_passed is True
    assert report.request_smoke_summary is not None
    assert report.request_smoke_summary["provider_elapsed_ms"] == 11
    assert report.request_smoke_summary["result_previews"][0]["candidate_key"] == "candidate-1"


def test_run_background_lifecycle_reports_health_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: FakeCompletedProcess(stdout="status=running\npid=2401\n"),
    )
    monkeypatch.setattr(
        runner,
        "_probe_health_once",
        lambda *_args, **_kwargs: runner.HealthObservation(
            ok=True,
            status_code=200,
            payload=_health_payload(device="cpu"),
            error=None,
        ),
    )
    plan = runner.build_background_lifecycle_plan(action="status")

    report = runner.run_background_lifecycle(plan)

    assert report.passed is False
    assert report.health_mismatches == ("device: expected 'cuda:0', got 'cpu'",)


def test_run_background_lifecycle_stop_passes_when_port_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: FakeCompletedProcess(stdout="status=stopped\npid=2401\n"),
    )
    monkeypatch.setattr(
        runner,
        "_probe_health_once",
        lambda *_args, **_kwargs: runner.HealthObservation(
            ok=False,
            status_code=None,
            payload=None,
            error="connection refused",
        ),
    )
    plan = runner.build_background_lifecycle_plan(action="stop")

    report = runner.run_background_lifecycle(plan)

    assert report.passed is True
    assert report.status == "stopped"
    assert report.health_ok is False


def test_run_background_lifecycle_reports_remote_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: FakeCompletedProcess(
            stdout="",
            stderr="permission denied",
            returncode=255,
        ),
    )
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: _health_ok())
    plan = runner.build_background_lifecycle_plan(action="status")

    report = runner.run_background_lifecycle(plan)

    assert report.passed is False
    assert report.command_observation.exit_code == 255
    assert report.status == "failed"


def test_background_lifecycle_writes_json_and_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: FakeCompletedProcess(stdout="status=running\npid=2401\n"),
    )
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: _health_ok())
    plan = runner.build_background_lifecycle_plan(action="status")
    report = runner.run_background_lifecycle(plan)
    json_output = tmp_path / "status.json"
    markdown_output = tmp_path / "status.md"

    runner.write_json_report(report, json_output)
    runner.write_markdown_report(report, markdown_output)

    assert json.loads(json_output.read_text(encoding="utf-8"))["passed"] is True
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "# Remote Reranker Background Lifecycle Result" in markdown
    assert "`status`: `running`" in markdown


def test_background_lifecycle_dry_run_cli_outputs_plan(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = runner.main(["status", "--dry-run", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["plan"]["base_url"] == "http://192.168.20.243:9104"
