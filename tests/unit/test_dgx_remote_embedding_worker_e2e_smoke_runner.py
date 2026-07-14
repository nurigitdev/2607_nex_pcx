import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.core.embedding_jobs import EmbeddingJobRecord
from app.core.embedding_vectors import EmbeddingVectorRecord


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_script_module("run_dgx_remote_embedding_worker_e2e_smoke.py")


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
    return runner.build_dgx_remote_embedding_worker_smoke_plan(
        provider_names=provider_names or ("kure",),
        database_url="postgresql://nex_pcx_dev:secret@127.0.0.1:5432/nex_pcx_dev",
        startup_timeout_seconds=1,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
        remote_timeout_seconds=2,
    )


def _fixture() -> Any:
    return runner.DgxEmbeddingWorkerSmokeFixture(
        smoke_run_key="unit-smoke",
        file_id=11,
        document_id=22,
        chunk_id=33,
        job_ids_by_profile={
            "kure_v1_1024": 101,
            "qwen3_4b_1000": 201,
            "qwen3_4b_2560": 202,
        },
    )


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


def _kure_health_payload() -> dict[str, Any]:
    return {
        "ready": True,
        "provider_model_id": "local-kure-v1",
        "model_key": "kure_v1",
        "profile_names": ["kure_v1_1024"],
        "dimension": 1024,
        "device": "cuda:0",
        "runtime_metadata": {"backend": "sentence_transformers"},
    }


def _preflight_result(profile_name: str, *, passed: bool = True) -> Any:
    return runner.DgxProviderRoutePreflightProfileResult(
        profile_name=profile_name,
        route_count=1,
        passed_count=1 if passed else 0,
        failed_count=0 if passed else 1,
        preflight_payload={
            "route_count": 1,
            "passed_count": 1 if passed else 0,
            "failed_count": 0 if passed else 1,
            "results": [
                {
                    "health_snapshot_id": 100,
                    "contract_snapshot_id": 200,
                }
            ],
        },
        error=None if passed else "failed",
    )


def _job(profile_name: str, job_id: int, *, status: str = "succeeded") -> EmbeddingJobRecord:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    return EmbeddingJobRecord(
        job_id=job_id,
        chunk_id=33,
        profile_name=profile_name,
        status=status,
        attempts=1,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        error_code=None if status == "succeeded" else "ERROR",
        error_message=None if status == "succeeded" else "failed",
        last_error_at=None,
        runtime_metadata={
            "provider_runtime_source": "route",
            "provider_runtime_mode": "remote",
            "provider_runtime_base_url": "http://192.168.20.243:9103",
            "provider_runtime_timeout_seconds": 300,
            "provider_route_id": 4,
            "provider_route_name": "qwen-primary",
            "provider_route_readiness_status": "ready",
            "provider_route_readiness_ready": True,
            "provider_route_health_snapshot_id": 100,
            "provider_route_contract_snapshot_id": 200,
            "provider_model_id": "local-qwen3-embedding-4b",
            "provider_type": "remote",
            "provider_elapsed_ms": 17,
            "dimension": 1000,
        },
        created_at=now,
        started_at=now,
        finished_at=now,
        updated_at=now,
    )


def _vector(profile_name: str, dimension: int) -> EmbeddingVectorRecord:
    return EmbeddingVectorRecord(
        chunk_id=33,
        profile_name=profile_name,
        table_name=f"chunk_embeddings_{profile_name}",
        dimension=dimension,
        storage_type="halfvec" if dimension == 2560 else "vector",
        embedding_text="[0.1,0.2,0.3]",
        elapsed_ms=20,
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def test_build_plan_defaults_to_all_provider_profiles_and_redacts_database_url() -> None:
    plan = runner.build_dgx_remote_embedding_worker_smoke_plan(
        database_url="postgresql://user:supersecret@db.local:5432/nex_pcx_dev",
    )

    assert [provider.provider for provider in plan.providers] == ["kure", "bge", "qwen"]
    assert [provider.foreground_plan.port for provider in plan.providers] == [9101, 9102, 9103]
    assert runner._plan_profile_names(plan) == (
        "kure_v1_1024",
        "bge_m3_1024",
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    )

    payload = runner._plan_payload(plan)

    assert payload["database_url"] == "postgresql://user:***@db.local:5432/nex_pcx_dev"
    assert "supersecret" not in json.dumps(payload)
    assert payload["fixture_text"] == "<fixture_text>"


def test_worker_profile_smoke_persists_expected_remote_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, bool]] = []

    def fake_worker(database_url: str, **kwargs: Any):
        calls.append((database_url, kwargs["profile_name"], kwargs["require_route_readiness"]))
        return runner.EmbeddingWorkerResult(
            processed=True,
            job=_job("qwen3_4b_1000", 201),
            vector=None,
            elapsed_ms=29,
            message="Remote embedding stored",
        )

    monkeypatch.setattr(runner, "process_next_embedding_job_with_provider_routes", fake_worker)
    monkeypatch.setattr(
        runner,
        "get_chunk_embedding",
        lambda *_args, **_kwargs: _vector("qwen3_4b_1000", 1000),
    )

    result = runner._run_worker_profile_smoke(
        "postgresql://example/db",
        fixture=_fixture(),
        profile_name="qwen3_4b_1000",
        worker_name="worker",
        lease_seconds=30,
        remote_timeout_seconds=300,
        readiness_gate_defer_seconds=300,
    )

    assert result.passed is True
    assert result.provider_type == "remote"
    assert result.provider_model_id == "local-qwen3-embedding-4b"
    assert result.provider_runtime_base_url == "http://192.168.20.243:9103"
    assert result.vector_dimension == 1000
    assert result.readiness_health_snapshot_id == 100
    assert calls == [("postgresql://example/db", "qwen3_4b_1000", True)]


def test_worker_profile_smoke_fails_when_claimed_job_is_not_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "process_next_embedding_job_with_provider_routes",
        lambda *_args, **_kwargs: runner.EmbeddingWorkerResult(
            processed=True,
            job=_job("qwen3_4b_1000", 999),
            vector=None,
            elapsed_ms=5,
            message="Remote embedding stored",
        ),
    )
    monkeypatch.setattr(
        runner,
        "get_chunk_embedding",
        lambda *_args, **_kwargs: _vector("qwen3_4b_1000", 1000),
    )

    result = runner._run_worker_profile_smoke(
        "postgresql://example/db",
        fixture=_fixture(),
        profile_name="qwen3_4b_1000",
        worker_name="worker",
        lease_seconds=30,
        remote_timeout_seconds=300,
        readiness_gate_defer_seconds=300,
    )

    assert result.passed is False
    assert "unexpected job_id=999" in result.error


def test_qwen_session_runs_two_preflights_and_two_worker_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakeProcess()
    preflight_profiles: list[str] = []
    worker_profiles: list[str] = []
    observations = [
        runner.HealthObservation(ok=False, status_code=None, payload=None, error="refused"),
        runner.HealthObservation(
            ok=True,
            status_code=200,
            payload=_qwen_health_payload(),
            error=None,
        ),
    ]

    def fake_preflight(_database_url: str, **kwargs: Any):
        preflight_profiles.append(kwargs["profile_name"])
        return _preflight_result(kwargs["profile_name"])

    def fake_worker_profile(_database_url: str, **kwargs: Any):
        profile_name = kwargs["profile_name"]
        worker_profiles.append(profile_name)
        dimension = 2560 if profile_name.endswith("2560") else 1000
        return runner.DgxEmbeddingWorkerProfileSmokeResult(
            profile_name=profile_name,
            job_id=kwargs["fixture"].job_ids_by_profile[profile_name],
            chunk_id=kwargs["fixture"].chunk_id,
            processed=True,
            job_status="succeeded",
            vector_table_name=f"chunk_embeddings_{profile_name}",
            vector_dimension=dimension,
            vector_storage_type="halfvec" if dimension == 2560 else "vector",
            provider_route_id=4,
            provider_route_name="qwen-primary",
            provider_runtime_base_url="http://192.168.20.243:9103",
            provider_model_id="local-qwen3-embedding-4b",
            provider_type="remote",
            provider_elapsed_ms=17,
            elapsed_ms=30,
            readiness_status="ready",
            readiness_health_snapshot_id=100,
            readiness_contract_snapshot_id=200,
            message="Remote embedding stored",
        )

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(runner, "_probe_health_once", lambda *_args, **_kwargs: observations.pop(0))
    monkeypatch.setattr(runner, "_run_profile_preflight", fake_preflight)
    monkeypatch.setattr(runner, "_run_worker_profile_smoke", fake_worker_profile)
    monkeypatch.setattr(
        runner,
        "_confirm_remote_stop",
        lambda *_args, **_kwargs: (False, False, None, "", "", None),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    plan = _plan("qwen")
    result = runner.run_provider_worker_e2e_session(
        plan.providers[0],
        database_url=plan.database_url,
        fixture=_fixture(),
        preflight_before_worker=True,
        active_only_preflight=True,
        worker_name_prefix="worker",
        lease_seconds=30,
        remote_timeout_seconds=300,
        readiness_gate_defer_seconds=300,
        health_timeout_seconds=0.1,
        poll_interval_seconds=0.1,
        shutdown_timeout_seconds=0.1,
        fail_fast=False,
    )

    assert result.passed is True
    assert preflight_profiles == ["qwen3_4b_1000", "qwen3_4b_2560"]
    assert worker_profiles == ["qwen3_4b_1000", "qwen3_4b_2560"]
    assert fake_process.terminated is True


def test_smoke_report_cleans_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture()
    cleanup_calls: list[int] = []
    plan = _plan("kure")

    monkeypatch.setattr(runner, "create_smoke_fixture", lambda *_args, **_kwargs: fixture)
    monkeypatch.setattr(
        runner,
        "run_provider_worker_e2e_session",
        lambda provider_plan, **_kwargs: runner.DgxRemoteEmbeddingWorkerProviderResult(
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
            preflight_results=(_preflight_result("kure_v1_1024"),),
            profile_results=(
                runner.DgxEmbeddingWorkerProfileSmokeResult(
                    profile_name="kure_v1_1024",
                    job_id=101,
                    chunk_id=33,
                    processed=True,
                    job_status="succeeded",
                    vector_table_name="chunk_embeddings_kure_v1_1024",
                    vector_dimension=1024,
                    vector_storage_type="vector",
                    provider_route_id=2,
                    provider_route_name="kure-primary",
                    provider_runtime_base_url="http://192.168.20.243:9101",
                    provider_model_id="local-kure-v1",
                    provider_type="remote",
                    provider_elapsed_ms=5,
                    elapsed_ms=20,
                    readiness_status="ready",
                    readiness_health_snapshot_id=100,
                    readiness_contract_snapshot_id=200,
                    message="Remote embedding stored",
                ),
            ),
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
        ),
    )

    def fake_cleanup(_database_url: str, file_id: int) -> bool:
        cleanup_calls.append(file_id)
        return True

    monkeypatch.setattr(runner, "cleanup_smoke_fixture", fake_cleanup)

    report = runner.run_dgx_remote_embedding_worker_smoke(plan)

    assert report.passed is True
    assert report.cleanup_attempted is True
    assert report.cleanup_confirmed is True
    assert cleanup_calls == [11]


def test_cli_prints_json_dry_run_plan_without_database_password() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_dgx_remote_embedding_worker_e2e_smoke.py",
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
    assert payload["plan"]["providers"][0]["profile_names"] == [
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    ]


def test_markdown_report_includes_worker_runtime_evidence(tmp_path: Path) -> None:
    plan = _plan("qwen")
    profile_result = runner.DgxEmbeddingWorkerProfileSmokeResult(
        profile_name="qwen3_4b_1000",
        job_id=201,
        chunk_id=33,
        processed=True,
        job_status="succeeded",
        vector_table_name="chunk_embeddings_qwen3_4b_1000",
        vector_dimension=1000,
        vector_storage_type="vector",
        provider_route_id=4,
        provider_route_name="qwen-primary",
        provider_runtime_base_url="http://192.168.20.243:9103",
        provider_model_id="local-qwen3-embedding-4b",
        provider_type="remote",
        provider_elapsed_ms=17,
        elapsed_ms=30,
        readiness_status="ready",
        readiness_health_snapshot_id=100,
        readiness_contract_snapshot_id=200,
        message="Remote embedding stored",
    )
    provider_result = runner.DgxRemoteEmbeddingWorkerProviderResult(
        provider="qwen",
        provider_name="qwen-primary",
        base_url="http://192.168.20.243:9103",
        health_url="http://192.168.20.243:9103/healthz",
        launch_command=("ssh", "nexpcx@192.168.20.243"),
        profile_names=("qwen3_4b_1000", "qwen3_4b_2560"),
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
        preflight_results=(_preflight_result("qwen3_4b_1000"),),
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
    )
    report = runner.DgxRemoteEmbeddingWorkerSmokeReport(
        plan=plan,
        fixture=_fixture(),
        results=(provider_result,),
        cleanup_attempted=True,
        cleanup_confirmed=True,
        total_elapsed_seconds=0.1,
    )
    output_path = tmp_path / "worker-smoke.md"

    runner.write_markdown_report(report, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "`chunk_embeddings_qwen3_4b_1000`" in content
    assert "`local-qwen3-embedding-4b`" in content
    assert "`100`" in content
    assert "`200`" in content
