import importlib.util
import json
import subprocess
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


runner = _load_script_module("run_dgx_provider_route_preflight_verification.py")


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


def _plan(*provider_names: str):
    return runner.build_dgx_provider_route_preflight_plan(
        provider_names=provider_names or ("kure",),
        database_url="postgresql://nex_pcx_dev:secret@127.0.0.1:5432/nex_pcx_dev",
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )


def _kure_health_payload(**overrides: Any) -> dict[str, Any]:
    return {
        "ready": True,
        "provider_model_id": "local-kure-v1",
        "model_key": "kure_v1",
        "profile_names": ["kure_v1_1024"],
        "dimension": 1024,
        "device": "cuda:0",
        "runtime_metadata": {"backend": "sentence_transformers"},
        **overrides,
    }


def _qwen_health_payload() -> dict[str, Any]:
    return {
        "ready": True,
        "provider_model_id": "local-qwen3-embedding-4b",
        "model_key": "qwen3_embedding_4b",
        "profile_names": ["qwen3_4b_1000", "qwen3_4b_2560"],
        "dimension": None,
        "device": "cuda:0",
        "runtime_metadata": {
            "backend": "qwen_embedding",
            "profile_dimensions": {
                "qwen3_4b_1000": 1000,
                "qwen3_4b_2560": 2560,
            },
        },
    }


def _preflight_payload(profile_name: str, *, route_count: int = 1, failed_count: int = 0):
    passed_count = max(0, route_count - failed_count)
    return {
        "route_count": route_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "profile_name": profile_name,
        "active_only": True,
        "sample_set": {
            "sample_set_name": "default_provider_contract_samples",
            "input_type": "document",
            "sample_text_count": 1,
        },
        "results": [
            {
                "route_id": 10,
                "profile_name": profile_name,
                "provider_name": "gpu-primary",
                "provider_mode": "remote",
                "health_status": "ready",
                "health_snapshot_id": 100 + index,
                "contract_passed": failed_count == 0,
                "contract_status": "passed" if failed_count == 0 else "failed",
                "contract_snapshot_id": 200 + index,
                "provider_model_id": "remote-model",
                "dimension": 1024,
                "elapsed_ms": 10,
                "validation_errors": [],
                "error_message": None,
            }
            for index in range(route_count)
        ],
    }


def test_build_plan_defaults_to_provider_profiles_and_redacts_database_url() -> None:
    plan = runner.build_dgx_provider_route_preflight_plan(
        database_url="postgresql://user:supersecret@db.local:5432/nex_pcx_dev",
    )

    assert [provider.provider for provider in plan.providers] == ["kure", "bge", "qwen"]
    assert [provider.foreground_plan.port for provider in plan.providers] == [9101, 9102, 9103]
    assert plan.providers[2].profile_names == ("qwen3_4b_1000", "qwen3_4b_2560")

    payload = runner._plan_payload(plan)

    assert payload["database_url"] == "postgresql://user:***@db.local:5432/nex_pcx_dev"
    assert "supersecret" not in json.dumps(payload)


def test_run_provider_route_preflight_session_passes_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    preflight_calls: list[tuple[str, str, bool]] = []
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(
            ok=True, status_code=200, payload=_kure_health_payload(), error=None
        ),
    ]

    def fake_run_preflight(database_url: str, **kwargs: Any):
        preflight_calls.append((database_url, kwargs["profile_name"], kwargs["active_only"]))
        return _preflight_payload(kwargs["profile_name"])

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(runner, "run_preflight", fake_run_preflight)
    monkeypatch.setattr(
        runner,
        "_confirm_remote_stop",
        lambda *_args, **_kwargs: (False, False, None, "", "", None),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    plan = _plan()
    result = runner.run_provider_route_preflight_session(
        plan.providers[0],
        database_url=plan.database_url,
        active_only=True,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
        fail_fast=False,
    )

    assert result.passed is True
    assert result.launch_command[:2] == ("ssh", "nexpcx@192.168.20.243")
    assert result.profile_results[0].profile_name == "kure_v1_1024"
    assert result.profile_results[0].passed is True
    assert preflight_calls == [(plan.database_url, "kure_v1_1024", True)]
    assert fake_process.terminated is True
    assert result.post_stop_health_reachable is False


def test_qwen_session_runs_both_profile_preflights_from_one_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    preflight_profiles: list[str] = []
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(
            ok=True, status_code=200, payload=_qwen_health_payload(), error=None
        ),
    ]

    def fake_run_preflight(_database_url: str, **kwargs: Any):
        preflight_profiles.append(kwargs["profile_name"])
        return _preflight_payload(kwargs["profile_name"])

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(runner, "run_preflight", fake_run_preflight)
    monkeypatch.setattr(
        runner,
        "_confirm_remote_stop",
        lambda *_args, **_kwargs: (False, False, None, "", "", None),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    plan = _plan("qwen")
    result = runner.run_provider_route_preflight_session(
        plan.providers[0],
        database_url=plan.database_url,
        active_only=True,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
        fail_fast=False,
    )

    assert result.passed is True
    assert preflight_profiles == ["qwen3_4b_1000", "qwen3_4b_2560"]
    assert [profile.passed for profile in result.profile_results] == [True, True]


def test_session_fails_when_profile_has_no_active_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(
            ok=True, status_code=200, payload=_kure_health_payload(), error=None
        ),
    ]

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(
        runner,
        "run_preflight",
        lambda _database_url, **kwargs: _preflight_payload(kwargs["profile_name"], route_count=0),
    )
    monkeypatch.setattr(
        runner,
        "_confirm_remote_stop",
        lambda *_args, **_kwargs: (False, False, None, "", "", None),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    plan = _plan()
    result = runner.run_provider_route_preflight_session(
        plan.providers[0],
        database_url=plan.database_url,
        active_only=True,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
        fail_fast=False,
    )

    assert result.passed is False
    assert result.error == "One or more route preflight checks failed."
    assert result.profile_results[0].error == (
        "No active provider routes matched profile 'kure_v1_1024'."
    )


def test_suite_honors_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan("kure", "bge")

    def fake_session(provider_plan: Any, **_: Any):
        return runner.DgxProviderRoutePreflightProviderResult(
            provider=provider_plan.provider,
            provider_name=provider_plan.foreground_plan.provider_name,
            base_url=provider_plan.foreground_plan.base_url,
            health_url=provider_plan.foreground_plan.health_url,
            launch_command=("ssh",),
            profile_names=provider_plan.profile_names,
            startup_timeout_seconds=1,
            health_timeout_seconds=0.1,
            poll_interval_seconds=0.1,
            shutdown_timeout_seconds=0.1,
            pre_launch_health_reachable=False,
            launched=True,
            health_checked=True,
            health_ok=True,
            health_attempts=1,
            health_status_code=200,
            health_payload={},
            health_error=None,
            health_mismatches=(),
            profile_results=(),
            process_exit_code_before_stop=None,
            process_exit_code_after_stop=143,
            stopped=True,
            stop_confirmed=True,
            remote_stop_attempted=False,
            remote_stop_exit_code=None,
            remote_stop_stdout="",
            remote_stop_stderr="",
            post_stop_health_reachable=False,
            elapsed_seconds=0.1,
            stdout_tail="",
            stderr_tail="",
            error="preflight failed",
        )

    monkeypatch.setattr(runner, "run_provider_route_preflight_session", fake_session)

    report = runner.run_dgx_provider_route_preflight_suite(
        runner.DgxProviderRoutePreflightPlan(
            database_url=plan.database_url,
            providers=plan.providers,
            active_only=True,
            health_timeout_seconds=0.1,
            poll_interval_seconds=0.1,
            shutdown_timeout_seconds=0.1,
            fail_fast=True,
        )
    )

    assert report.passed is False
    assert [result.provider for result in report.results] == ["kure"]


def test_cli_prints_json_dry_run_plan_without_database_password() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_dgx_provider_route_preflight_verification.py",
            "--provider",
            "qwen",
            "--database-url",
            "postgresql://nex_pcx_dev:supersecret@127.0.0.1:5432/nex_pcx_dev",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert "supersecret" not in result.stdout
    assert payload["plan"]["database_url"] == (
        "postgresql://nex_pcx_dev:***@127.0.0.1:5432/nex_pcx_dev"
    )
    assert [provider["provider"] for provider in payload["plan"]["providers"]] == ["qwen"]
    assert payload["plan"]["providers"][0]["profile_names"] == [
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    ]


def test_markdown_report_includes_snapshot_ids(tmp_path: Path) -> None:
    plan = _plan()
    profile_result = runner.DgxProviderRoutePreflightProfileResult(
        profile_name="kure_v1_1024",
        route_count=1,
        passed_count=1,
        failed_count=0,
        preflight_payload=_preflight_payload("kure_v1_1024"),
        error=None,
    )
    provider_result = runner.DgxProviderRoutePreflightProviderResult(
        provider="kure",
        provider_name="kure-primary",
        base_url="http://192.168.20.243:9101",
        health_url="http://192.168.20.243:9101/healthz",
        launch_command=("ssh", "nexpcx@192.168.20.243"),
        profile_names=("kure_v1_1024",),
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
        pre_launch_health_reachable=False,
        launched=True,
        health_checked=True,
        health_ok=True,
        health_attempts=1,
        health_status_code=200,
        health_payload={},
        health_error=None,
        health_mismatches=(),
        profile_results=(profile_result,),
        process_exit_code_before_stop=None,
        process_exit_code_after_stop=143,
        stopped=True,
        stop_confirmed=True,
        remote_stop_attempted=False,
        remote_stop_exit_code=None,
        remote_stop_stdout="",
        remote_stop_stderr="",
        post_stop_health_reachable=False,
        elapsed_seconds=0.1,
        stdout_tail="",
        stderr_tail="",
        error=None,
    )
    report = runner.DgxProviderRoutePreflightReport(
        plan=plan,
        results=(provider_result,),
        total_elapsed_seconds=0.1,
    )
    output_path = tmp_path / "preflight.md"

    runner.write_markdown_report(report, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "`kure_v1_1024`" in content
    assert "`200`" in content
    assert "`100`" in content
