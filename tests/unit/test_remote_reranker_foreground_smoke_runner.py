import importlib.util
import sys
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


runner = _load_script_module("run_remote_reranker_foreground_smoke.py")


class FakeProcess:
    def __init__(self, *, exit_code: int | None = None) -> None:
        self._exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._exit_code

    def terminate(self) -> None:
        self.terminated = True
        self._exit_code = 143

    def kill(self) -> None:
        self.killed = True
        self._exit_code = 137

    def wait(self, timeout: float | None = None) -> int | None:
        return self._exit_code


def _plan():
    return runner.build_reranker_foreground_smoke_plan()


def _health_payload(**overrides: Any) -> dict[str, Any]:
    return {
        "ready": True,
        "provider_type": "remote",
        "provider_model_id": "Qwen/Qwen3-Reranker-0.6B",
        "reranker_profile_name": "qwen3_reranker_0_6b",
        "device": "cuda:0",
        "runtime_metadata": {
            "service": "nex_pcx_reranker_provider_service",
            "backend": "qwen_reranker",
            "model_dir": "/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_0_6b",
            "model_dir_exists": True,
        },
        **overrides,
    }


def test_run_remote_reranker_foreground_smoke_passes_and_stops_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    popen_calls: list[tuple[str, ...]] = []
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(ok=True, status_code=200, payload=_health_payload(), error=None),
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="stopped"),
    ]

    def fake_popen(command: tuple[str, ...], **_: Any) -> FakeProcess:
        popen_calls.append(command)
        return fake_process

    def fake_probe(_: str, *, timeout_seconds: float) -> runner.HealthObservation:
        assert timeout_seconds > 0
        return observations.pop(0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "_probe_health_once", fake_probe)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    result = runner.run_foreground_smoke(
        _plan(),
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )

    assert result.passed is True
    assert result.pre_launch_health_reachable is False
    assert result.health_attempts == 2
    assert result.launch_command == popen_calls[0]
    assert result.launch_command[:2] == ("ssh", "nexpcx@192.168.20.243")
    assert "-t" not in result.launch_command
    assert fake_process.terminated is True
    assert result.stop_confirmed is True
    assert result.remote_stop_attempted is False


def test_run_remote_reranker_foreground_smoke_fails_on_health_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(
            ok=True,
            status_code=200,
            payload=_health_payload(provider_model_id="other"),
            error=None,
        ),
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="stopped"),
    ]

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    result = runner.run_foreground_smoke(
        _plan(),
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )

    assert result.passed is False
    assert result.health_ok is False
    assert result.error == "Health response did not match the expected reranker provider plan."
    assert result.health_mismatches == (
        "provider_model_id: expected 'Qwen/Qwen3-Reranker-0.6B', got 'other'",
    )


def test_run_remote_reranker_foreground_smoke_reports_early_process_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess(exit_code=3)
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
    ]

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(
        runner,
        "_probe_health_once",
        lambda *_args, **_kwargs: (
            observations.pop(0)
            if observations
            else pytest.fail("health should not be probed after early exit")
        ),
    )

    result = runner.run_foreground_smoke(
        _plan(),
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )

    assert result.passed is False
    assert result.health_checked is False
    assert result.process_exit_code_before_stop == 3
    assert result.process_exit_code_after_stop == 3
    assert result.error == "Remote reranker provider process exited early with code 3."


def test_run_remote_reranker_foreground_smoke_uses_remote_stop_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    stop_calls: list[tuple[str, ...]] = []
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(ok=True, status_code=200, payload=_health_payload(), error=None),
    ]
    unreachable_results = [False, True]

    class FakeCompleted:
        returncode = 0
        stdout = "stopped"
        stderr = ""

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(
        runner,
        "_wait_until_health_unreachable",
        lambda *_args, **_kwargs: unreachable_results.pop(0),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    def fake_run(command: tuple[str, ...], **_: Any) -> FakeCompleted:
        stop_calls.append(command)
        return FakeCompleted()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    result = runner.run_foreground_smoke(
        _plan(),
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )

    assert result.passed is True
    assert result.remote_stop_attempted is True
    assert result.remote_stop_exit_code == 0
    assert result.remote_stop_stdout == "stopped"
    assert stop_calls[0][:2] == ("ssh", "nexpcx@192.168.20.243")
    assert "[u]vicorn app.reranker_provider_service:app" in stop_calls[0][-1]


def test_run_remote_reranker_foreground_smoke_refuses_existing_health_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_probe_health_once",
        lambda *_args, **_kwargs: runner.HealthObservation(
            ok=True,
            status_code=200,
            payload=_health_payload(),
            error=None,
        ),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("runner should not launch over an active port"),
    )

    result = runner.run_foreground_smoke(
        _plan(),
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )

    assert result.passed is False
    assert result.pre_launch_health_reachable is True
    assert result.launched is False
    assert result.error == (
        "Health URL is already reachable before launch; "
        "stop the existing reranker provider or use another port."
    )


def test_reranker_health_mismatches_validate_expected_payload() -> None:
    assert runner._health_mismatches(_health_payload(), plan=_plan()) == ()

    mismatches = runner._health_mismatches(
        _health_payload(
            ready=False,
            device="cpu",
            runtime_metadata={
                "service": "unexpected",
                "backend": "mock",
                "model_dir": "/tmp/model",
                "model_dir_exists": False,
            },
        ),
        plan=_plan(),
    )

    assert "ready: expected True, got False" in mismatches
    assert "device: expected 'cuda:0', got 'cpu'" in mismatches
    assert (
        "runtime_metadata.service: expected "
        "'nex_pcx_reranker_provider_service', got 'unexpected'"
    ) in mismatches
    assert "runtime_metadata.backend: expected 'qwen_reranker', got 'mock'" in mismatches
    assert (
        "runtime_metadata.model_dir: expected "
        "'/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_0_6b', got '/tmp/model'"
    ) in mismatches
    assert "runtime_metadata.model_dir_exists: expected True, got False" in mismatches


def test_run_remote_reranker_foreground_smoke_rejects_invalid_timeouts() -> None:
    with pytest.raises(ValueError, match="startup_timeout_seconds"):
        runner.run_foreground_smoke(_plan(), startup_timeout_seconds=0)
