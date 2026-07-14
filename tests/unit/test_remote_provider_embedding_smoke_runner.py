import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.embedding_provider_presets import get_embedding_provider_preset
from app.core.embedding_providers import (
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    InvalidEmbeddingProviderError,
)


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


embedding_smoke = _load_script_module("run_remote_provider_embedding_smoke.py")


def test_build_embedding_smoke_plan_defaults_to_kure_remote_host() -> None:
    plan = embedding_smoke.build_embedding_smoke_plan(get_embedding_provider_preset("kure"))

    assert plan.provider == "kure"
    assert plan.base_url == "http://192.168.20.243:9101"
    assert plan.embeddings_url == "http://192.168.20.243:9101/v1/embeddings"
    assert plan.provider_model_id == "local-kure-v1"
    assert len(plan.cases) == 1
    assert plan.cases[0].profile_name == "kure_v1_1024"
    assert plan.cases[0].model_key == "kure_v1"
    assert plan.cases[0].output_dimension == 1024


def test_build_embedding_smoke_plan_includes_both_qwen_profiles() -> None:
    plan = embedding_smoke.build_embedding_smoke_plan(
        get_embedding_provider_preset("qwen"),
        base_url="http://gpu-provider.local:19103/",
        timeout_seconds=240,
    )

    assert plan.base_url == "http://gpu-provider.local:19103"
    assert plan.timeout_seconds == 240
    assert [case.profile_name for case in plan.cases] == [
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    ]
    assert [case.output_dimension for case in plan.cases] == [1000, 2560]
    assert {case.trace_id for case in plan.cases} == {
        "remote-embedding-smoke-qwen-qwen3_4b_1000",
        "remote-embedding-smoke-qwen-qwen3_4b_2560",
    }


def test_build_embedding_smoke_plan_can_target_one_profile() -> None:
    plan = embedding_smoke.build_embedding_smoke_plan(
        get_embedding_provider_preset("qwen"),
        profile_names=("qwen3_4b_2560",),
        texts=(" alpha ", "beta"),
        input_type="query",
    )

    assert len(plan.cases) == 1
    assert plan.cases[0].profile_name == "qwen3_4b_2560"
    assert plan.cases[0].output_dimension == 2560
    assert plan.cases[0].input_type == "query"
    assert plan.texts == ("alpha", "beta")


def test_build_embedding_smoke_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unsupported profile_name"):
        embedding_smoke.build_embedding_smoke_plan(
            get_embedding_provider_preset("kure"),
            profile_names=("bge_m3_1024",),
        )

    with pytest.raises(ValueError, match="timeout_seconds"):
        embedding_smoke.build_embedding_smoke_plan(
            get_embedding_provider_preset("kure"),
            timeout_seconds=0,
        )

    with pytest.raises(ValueError, match="texts"):
        embedding_smoke.build_embedding_smoke_plan(
            get_embedding_provider_preset("kure"),
            texts=(" ",),
        )


def test_run_embedding_request_smoke_passes_for_expected_response() -> None:
    provider = _FakeEmbeddingSmokeProvider()
    plan = embedding_smoke.build_embedding_smoke_plan(
        get_embedding_provider_preset("kure"),
        texts=("alpha", "beta"),
    )

    report = embedding_smoke.run_embedding_request_smoke(provider, plan=plan)

    assert report.passed is True
    assert len(provider.requests) == 1
    assert provider.requests[0].trace_id == "remote-embedding-smoke-kure-kure_v1_1024"
    assert provider.requests[0].texts == ("alpha", "beta")
    observation = report.observations[0]
    assert observation.passed is True
    assert observation.dimension == 1024
    assert observation.input_count == 2
    assert observation.embedding_count == 2
    assert observation.embedding_preview == (0.01, 0.01, 0.01)


def test_run_embedding_request_smoke_reports_mismatches() -> None:
    provider = _FakeEmbeddingSmokeProvider(provider_model_id="wrong-provider")
    plan = embedding_smoke.build_embedding_smoke_plan(get_embedding_provider_preset("bge"))

    report = embedding_smoke.run_embedding_request_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observations[0].mismatches == (
        "provider_model_id: expected 'local-bge-m3', got 'wrong-provider'",
    )


def test_run_embedding_request_smoke_captures_provider_errors() -> None:
    provider = _FailingEmbeddingSmokeProvider()
    plan = embedding_smoke.build_embedding_smoke_plan(get_embedding_provider_preset("kure"))

    report = embedding_smoke.run_embedding_request_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observations[0].error == "provider unavailable"
    assert report.observations[0].mismatches == ()


def test_embedding_smoke_cli_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_remote_provider_embedding_smoke.py",
            "--provider",
            "qwen",
            "--profile-name",
            "qwen3_4b_1000",
            "--text",
            "sensitive smoke text",
            "--base-url",
            "http://gpu-provider.local:9103/",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["plan"]["base_url"] == "http://gpu-provider.local:9103"
    assert payload["plan"]["embeddings_url"] == "http://gpu-provider.local:9103/v1/embeddings"
    assert payload["plan"]["texts"] == ["<text:1>"]
    assert payload["plan"]["cases"][0]["profile_name"] == "qwen3_4b_1000"
    assert payload["plan"]["cases"][0]["output_dimension"] == 1000


class _FakeEmbeddingSmokeProvider:
    def __init__(
        self,
        *,
        provider_model_id: str | None = None,
        provider_type: str = "remote",
    ) -> None:
        self.provider_model_id = provider_model_id
        self.provider_type = provider_type
        self.requests: list[EmbeddingProviderRequest] = []

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        self.requests.append(request)
        return EmbeddingProviderResponse(
            embeddings=(tuple(0.01 for _ in range(request.output_dimension)),) * len(request.texts),
            dimension=request.output_dimension,
            provider_model_id=self.provider_model_id
            or _expected_provider_model_id(request.profile_name),
            provider_type=self.provider_type,
            elapsed_ms=7,
            input_count=len(request.texts),
            runtime_metadata={
                "trace_id": request.trace_id,
                "profile_name": request.profile_name,
            },
        )


class _FailingEmbeddingSmokeProvider:
    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        raise InvalidEmbeddingProviderError("provider unavailable")


def _expected_provider_model_id(profile_name: str) -> str:
    if profile_name.startswith("bge"):
        return "local-bge-m3"
    if profile_name.startswith("qwen"):
        return "local-qwen3-embedding-4b"
    return "local-kure-v1"
