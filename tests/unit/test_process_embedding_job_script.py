import argparse
import importlib.util
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.embedding_providers import EmbeddingProviderRuntimeConfig
from app.core.embedding_worker import EmbeddingWorkerResult


def _load_process_embedding_job_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "process_embedding_job.py"
    spec = importlib.util.spec_from_file_location("process_embedding_job_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


process_embedding_job = _load_process_embedding_job_module()


def test_process_embedding_job_runtime_config_uses_settings_defaults() -> None:
    args = argparse.Namespace(
        provider_mode=None,
        remote_provider_url=None,
        remote_provider_timeout_seconds=None,
    )
    settings = Settings(
        embedding_provider_mode="remote",
        remote_embedding_provider_url="http://provider.local/",
        remote_embedding_provider_timeout_seconds=8.0,
    )

    config = process_embedding_job._runtime_config_from_args(args, settings)

    assert config == EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://provider.local",
        remote_timeout_seconds=8.0,
    )


def test_process_embedding_job_runtime_config_allows_cli_overrides() -> None:
    args = argparse.Namespace(
        provider_mode="remote",
        remote_provider_url="http://override.local/",
        remote_provider_timeout_seconds=2.5,
    )

    config = process_embedding_job._runtime_config_from_args(args, Settings())

    assert config == EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://override.local",
        remote_timeout_seconds=2.5,
    )


def test_process_embedding_job_invokes_provider_worker_and_closes_provider(monkeypatch) -> None:
    calls = {}
    provider = _ClosableProvider()

    def fake_build_provider(runtime_config):
        calls["runtime_config"] = runtime_config
        return provider

    def fake_process(database_url, **kwargs):
        calls["database_url"] = database_url
        calls["kwargs"] = kwargs
        return EmbeddingWorkerResult(processed=False, job=None, message="idle")

    monkeypatch.setattr(
        process_embedding_job,
        "build_embedding_provider_from_runtime_config",
        fake_build_provider,
    )
    monkeypatch.setattr(
        process_embedding_job,
        "process_next_embedding_job_with_provider",
        fake_process,
    )
    runtime_config = EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://provider.local",
        remote_timeout_seconds=4.0,
    )

    result = process_embedding_job._process_next_job(
        "postgresql://example/db",
        worker_name="worker-one",
        profile_name="kure_v1_1024",
        lease_seconds=11,
        runtime_config=runtime_config,
        provider_source=process_embedding_job.PROVIDER_SOURCE_RUNTIME,
    )

    assert result.message == "idle"
    assert provider.closed is True
    assert calls["database_url"] == "postgresql://example/db"
    assert calls["runtime_config"] == runtime_config
    assert calls["kwargs"]["worker_name"] == "worker-one"
    assert calls["kwargs"]["profile_name"] == "kure_v1_1024"
    assert calls["kwargs"]["lease_seconds"] == 11
    assert calls["kwargs"]["provider"] is provider
    assert calls["kwargs"]["success_message"] == "Remote embedding stored"


def test_process_embedding_job_uses_route_aware_worker_by_default(monkeypatch) -> None:
    calls = {}

    def fake_route_process(database_url, **kwargs):
        calls["database_url"] = database_url
        calls["kwargs"] = kwargs
        return EmbeddingWorkerResult(processed=False, job=None, message="idle")

    monkeypatch.setattr(
        process_embedding_job,
        "process_next_embedding_job_with_provider_routes",
        fake_route_process,
    )
    runtime_config = EmbeddingProviderRuntimeConfig(
        mode="mock",
        remote_base_url=None,
        remote_timeout_seconds=30.0,
    )

    result = process_embedding_job._process_next_job(
        "postgresql://example/db",
        worker_name="route-worker-one",
        profile_name=None,
        lease_seconds=13,
        runtime_config=runtime_config,
    )

    assert result.message == "idle"
    assert calls["database_url"] == "postgresql://example/db"
    assert calls["kwargs"]["worker_name"] == "route-worker-one"
    assert calls["kwargs"]["profile_name"] is None
    assert calls["kwargs"]["lease_seconds"] == 13
    assert calls["kwargs"]["fallback_runtime_config"] == runtime_config
    assert calls["kwargs"]["require_route_readiness"] is False
    assert calls["kwargs"]["readiness_gate_failure_mode"] == "fail"
    assert calls["kwargs"]["readiness_gate_defer_seconds"] == 300


def test_process_embedding_job_passes_route_readiness_gate_setting(monkeypatch) -> None:
    calls = {}

    def fake_route_process(database_url, **kwargs):
        calls["database_url"] = database_url
        calls["kwargs"] = kwargs
        return EmbeddingWorkerResult(processed=False, job=None, message="idle")

    monkeypatch.setattr(
        process_embedding_job,
        "process_next_embedding_job_with_provider_routes",
        fake_route_process,
    )
    runtime_config = EmbeddingProviderRuntimeConfig(
        mode="mock",
        remote_base_url=None,
        remote_timeout_seconds=30.0,
    )

    result = process_embedding_job._process_next_job(
        "postgresql://example/db",
        worker_name="route-worker-gated",
        profile_name="kure_v1_1024",
        lease_seconds=13,
        runtime_config=runtime_config,
        require_route_readiness=True,
        readiness_gate_failure_mode="defer",
        readiness_gate_defer_seconds=120,
    )

    assert result.message == "idle"
    assert calls["database_url"] == "postgresql://example/db"
    assert calls["kwargs"]["worker_name"] == "route-worker-gated"
    assert calls["kwargs"]["profile_name"] == "kure_v1_1024"
    assert calls["kwargs"]["require_route_readiness"] is True
    assert calls["kwargs"]["readiness_gate_failure_mode"] == "defer"
    assert calls["kwargs"]["readiness_gate_defer_seconds"] == 120


class _ClosableProvider:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True
