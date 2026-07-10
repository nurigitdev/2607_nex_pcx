import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.embedding_providers import (
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    EmbeddingProviderRuntimeConfig,
)


def _load_benchmark_embedding_provider_module():
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    script_path = scripts_dir / "benchmark_embedding_provider.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_embedding_provider_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_embedding_provider = _load_benchmark_embedding_provider_module()


def test_benchmark_runtime_config_uses_cli_remote_overrides() -> None:
    args = argparse.Namespace(
        provider_mode="remote",
        remote_provider_url="http://gpu-provider.local/",
        remote_provider_timeout_seconds=4.5,
    )

    config = benchmark_embedding_provider._runtime_config_from_args(args, Settings())

    assert config == EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://gpu-provider.local",
        remote_timeout_seconds=4.5,
    )


def test_benchmark_runs_warmup_iterations_and_reports_throughput(monkeypatch) -> None:
    provider = _FakeBenchmarkProvider()
    plan = benchmark_embedding_provider.EmbeddingProviderBenchmarkPlan(
        provider_mode="remote",
        remote_provider_url="http://provider.local",
        profile_name="kure_v1_1024",
        model_key="kure_v1",
        output_dimension=1024,
        input_type="document",
        batch_size=3,
        iterations=2,
        warmup_iterations=1,
        text_count=2,
    )
    timings = iter([0.0, 0.0, 0.0625, 0.0625, 0.1875, 0.25])
    monkeypatch.setattr(benchmark_embedding_provider, "perf_counter", lambda: next(timings))

    report = benchmark_embedding_provider.run_embedding_provider_benchmark(
        provider,
        plan=plan,
        texts=("alpha", "beta"),
    )

    assert [request.trace_id for request in provider.requests] == [
        "embedding-benchmark-warmup-1",
        "embedding-benchmark-1",
        "embedding-benchmark-2",
    ]
    assert provider.requests[1].texts == ("alpha", "beta", "alpha")
    assert provider.requests[2].texts == ("beta", "alpha", "beta")
    assert report.total_input_count == 6
    assert report.total_wall_elapsed_ms == 250
    assert report.avg_batch_wall_ms == 93.5
    assert report.p50_batch_wall_ms == 62
    assert report.p95_batch_wall_ms == 125
    assert report.texts_per_second == 24
    assert report.provider_model_id == "fake-benchmark-provider"
    assert report.provider_type == "remote"


def test_benchmark_script_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_embedding_provider.py",
            "--provider-mode",
            "mock",
            "--profile-name",
            "kure_v1_1024",
            "--batch-size",
            "2",
            "--iterations",
            "3",
            "--warmup-iterations",
            "0",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["plan"]["provider_mode"] == "mock"
    assert payload["plan"]["profile_name"] == "kure_v1_1024"
    assert payload["plan"]["model_key"] == "kure_v1"
    assert payload["plan"]["output_dimension"] == 1024
    assert payload["plan"]["batch_size"] == 2
    assert payload["plan"]["iterations"] == 3


class _FakeBenchmarkProvider:
    def __init__(self) -> None:
        self.requests: list[EmbeddingProviderRequest] = []

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        self.requests.append(request)
        return EmbeddingProviderResponse(
            embeddings=(tuple(0.01 for _ in range(request.output_dimension)),) * len(request.texts),
            dimension=request.output_dimension,
            provider_model_id="fake-benchmark-provider",
            provider_type="remote",
            elapsed_ms=12,
            input_count=len(request.texts),
            runtime_metadata={"trace_id": request.trace_id},
        )
