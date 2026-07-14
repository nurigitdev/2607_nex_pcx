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


suite = _load_script_module("run_remote_provider_embedding_smoke_suite.py")
embedding_smoke = _load_script_module("run_remote_provider_embedding_smoke.py")


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


def _health_payload(**overrides: Any) -> dict[str, Any]:
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


def _suite_plan(*provider_names: str):
    return suite.build_embedding_smoke_suite_plan(
        provider_names=provider_names or ("kure",),
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )


def _embedding_report(
    plan: Any,
    *,
    passed: bool = True,
) -> Any:
    observation = embedding_smoke.RemoteProviderEmbeddingSmokeObservation(
        case=plan.cases[0],
        request_elapsed_ms=11,
        provider_elapsed_ms=7,
        provider_model_id=plan.provider_model_id,
        provider_type="remote",
        dimension=plan.cases[0].output_dimension,
        input_count=plan.cases[0].text_count,
        embedding_count=plan.cases[0].text_count,
        embedding_preview=(0.1, 0.2, 0.3),
        runtime_metadata={"trace_id": plan.cases[0].trace_id},
        mismatches=() if passed else ("dimension: expected 1024, got 768",),
        error=None,
    )
    return embedding_smoke.RemoteProviderEmbeddingSmokeReport(
        plan=plan,
        observations=(observation,),
        total_elapsed_ms=12,
    )


def test_build_suite_plan_defaults_to_all_providers_in_order() -> None:
    plan = suite.build_embedding_smoke_suite_plan()

    assert [provider.provider for provider in plan.providers] == ["kure", "bge", "qwen"]
    assert [provider.foreground_plan.port for provider in plan.providers] == [9101, 9102, 9103]
    assert [provider.embedding_plan.provider_model_id for provider in plan.providers] == [
        "local-kure-v1",
        "local-bge-m3",
        "local-qwen3-embedding-4b",
    ]
    assert [provider.request_timeout_seconds for provider in plan.providers] == [
        120.0,
        120.0,
        300.0,
    ]
    assert [case.profile_name for case in plan.providers[2].embedding_plan.cases] == [
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    ]


def test_build_suite_plan_can_target_subset_and_override_request_timeout() -> None:
    plan = suite.build_embedding_smoke_suite_plan(
        provider_names=("qwen", "qwen", "kure"),
        request_timeout_seconds=45,
        texts=("alpha", "beta"),
        input_type="query",
    )

    assert [provider.provider for provider in plan.providers] == ["qwen", "kure"]
    assert [provider.request_timeout_seconds for provider in plan.providers] == [45, 45]
    assert plan.providers[0].embedding_plan.texts == ("alpha", "beta")
    assert {case.input_type for case in plan.providers[0].embedding_plan.cases} == {"query"}


def test_run_provider_embedding_smoke_session_passes_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    popen_calls: list[tuple[str, ...]] = []
    request_smoke_calls: list[str] = []
    observations = [
        suite.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        suite.HealthObservation(ok=True, status_code=200, payload=_health_payload(), error=None),
    ]

    def fake_popen(command: tuple[str, ...], **_: Any) -> FakeProcess:
        popen_calls.append(command)
        return fake_process

    def fake_request_smoke(provider_plan: Any) -> Any:
        request_smoke_calls.append(provider_plan.provider)
        return _embedding_report(provider_plan.embedding_plan)

    monkeypatch.setattr(suite.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(suite, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(suite, "_run_request_smoke", fake_request_smoke)
    monkeypatch.setattr(suite, "_wait_until_health_unreachable", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(suite.time, "sleep", lambda _: None)

    result = suite.run_provider_embedding_smoke_session(
        _suite_plan().providers[0],
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )

    assert result.passed is True
    assert result.launch_command == popen_calls[0]
    assert result.launch_command[:2] == ("ssh", "nexpcx@192.168.20.243")
    assert result.embedding_report is not None
    assert result.embedding_report.passed is True
    assert result.embedding_report.observations[0].embedding_preview == (0.1, 0.2, 0.3)
    assert request_smoke_calls == ["kure"]
    assert fake_process.terminated is True
    assert result.post_stop_health_reachable is False


def test_run_provider_embedding_smoke_session_reports_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    observations = [
        suite.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        suite.HealthObservation(ok=True, status_code=200, payload=_health_payload(), error=None),
    ]

    monkeypatch.setattr(suite.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(suite, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(
        suite,
        "_run_request_smoke",
        lambda provider_plan: _embedding_report(provider_plan.embedding_plan, passed=False),
    )
    monkeypatch.setattr(suite, "_wait_until_health_unreachable", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(suite.time, "sleep", lambda _: None)

    result = suite.run_provider_embedding_smoke_session(
        _suite_plan().providers[0],
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
    )

    assert result.passed is False
    assert result.error == "Embedding request smoke failed."
    assert result.embedding_report is not None
    assert result.embedding_report.observations[0].mismatches == (
        "dimension: expected 1024, got 768",
    )


def test_run_embedding_smoke_suite_honors_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _suite_plan("kure", "bge")

    def fake_session(provider_plan: Any, **_: Any) -> Any:
        embedding_plan = provider_plan.embedding_plan
        return suite.RemoteProviderEmbeddingSuiteProviderResult(
            provider=provider_plan.provider,
            provider_name=provider_plan.foreground_plan.provider_name,
            base_url=provider_plan.foreground_plan.base_url,
            health_url=provider_plan.foreground_plan.health_url,
            launch_command=("ssh",),
            startup_timeout_seconds=1,
            request_timeout_seconds=1,
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
            embedding_report=_embedding_report(embedding_plan, passed=False),
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
            error="Embedding request smoke failed.",
        )

    monkeypatch.setattr(suite, "run_provider_embedding_smoke_session", fake_session)
    report = suite.run_embedding_smoke_suite(
        suite.RemoteProviderEmbeddingSuitePlan(
            providers=plan.providers,
            health_timeout_seconds=0.1,
            poll_interval_seconds=0.1,
            shutdown_timeout_seconds=0.1,
            fail_fast=True,
        )
    )

    assert report.passed is False
    assert [result.provider for result in report.results] == ["kure"]


def test_suite_cli_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_remote_provider_embedding_smoke_suite.py",
            "--provider",
            "qwen",
            "--text",
            "sensitive smoke text",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert [provider["provider"] for provider in payload["plan"]["providers"]] == ["qwen"]
    assert payload["plan"]["providers"][0]["embedding_plan"]["texts"] == ["<text:1>"]
    assert [
        case["profile_name"] for case in payload["plan"]["providers"][0]["embedding_plan"]["cases"]
    ] == ["qwen3_4b_1000", "qwen3_4b_2560"]


def test_markdown_report_includes_vector_previews(tmp_path: Path) -> None:
    plan = _suite_plan()
    result = suite.RemoteProviderEmbeddingSuiteProviderResult(
        provider="kure",
        provider_name="kure-primary",
        base_url="http://192.168.20.243:9101",
        health_url="http://192.168.20.243:9101/healthz",
        launch_command=("ssh", "nexpcx@192.168.20.243"),
        startup_timeout_seconds=1,
        request_timeout_seconds=120,
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
        embedding_report=_embedding_report(plan.providers[0].embedding_plan),
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
    report = suite.RemoteProviderEmbeddingSuiteReport(
        plan=plan,
        results=(result,),
        total_elapsed_seconds=0.1,
    )
    output_path = tmp_path / "suite.md"

    suite.write_markdown_report(report, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "`kure_v1_1024`" in content
    assert "`[0.1, 0.2, 0.3]`" in content
