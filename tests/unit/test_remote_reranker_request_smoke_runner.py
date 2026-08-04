import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.rerankers import (
    RERANK_RETRIEVAL_STRATEGY,
    InvalidRerankerError,
    RerankRequest,
    RerankResult,
    RerankResultItem,
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


reranker_smoke = _load_script_module("run_remote_reranker_request_smoke.py")


def test_build_reranker_request_smoke_plan_defaults_to_dgx_reranker() -> None:
    plan = reranker_smoke.build_reranker_request_smoke_plan()

    assert plan.provider_name == "qwen-reranker-primary"
    assert plan.base_url == "http://192.168.20.243:9104"
    assert plan.rerank_url == "http://192.168.20.243:9104/v1/rerank"
    assert plan.provider_model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert plan.reranker_profile_name == "qwen3_reranker_0_6b"
    assert plan.expected_backend == "qwen_reranker"
    assert plan.expected_device == "cuda:0"
    assert plan.timeout_seconds == 300.0
    assert plan.request.top_k == 2
    assert len(plan.request.candidates) == 3
    assert plan.request.candidates[0].candidate_key == "candidate-1"
    assert plan.request.candidates[0].source_profile_name == "qwen3_4b_2560"


def test_build_reranker_request_smoke_plan_accepts_custom_inputs() -> None:
    plan = reranker_smoke.build_reranker_request_smoke_plan(
        base_url="http://gpu-reranker.local:19104/",
        query_text="  policy query  ",
        candidate_texts=(" alpha ", "beta"),
        source_profile_name="hybrid_keyword_vector",
        source_retrieval_strategy="hybrid_rrf",
        top_k=1,
        timeout_seconds=12.5,
        expected_device=None,
    )

    assert plan.base_url == "http://gpu-reranker.local:19104"
    assert plan.rerank_url == "http://gpu-reranker.local:19104/v1/rerank"
    assert plan.request.query_text == "policy query"
    assert plan.request.top_k == 1
    assert [candidate.text for candidate in plan.request.candidates] == ["alpha", "beta"]
    assert {candidate.source_profile_name for candidate in plan.request.candidates} == {
        "hybrid_keyword_vector",
    }
    assert {candidate.source_retrieval_strategy for candidate in plan.request.candidates} == {
        "hybrid_rrf",
    }
    assert plan.expected_device is None


def test_build_reranker_request_smoke_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        reranker_smoke.build_reranker_request_smoke_plan(timeout_seconds=0)

    with pytest.raises(ValueError, match="query_text"):
        reranker_smoke.build_reranker_request_smoke_plan(query_text=" ")

    with pytest.raises(ValueError, match="candidate_texts"):
        reranker_smoke.build_reranker_request_smoke_plan(candidate_texts=(" ",))

    with pytest.raises(InvalidRerankerError, match="top_k"):
        reranker_smoke.build_reranker_request_smoke_plan(top_k=0)


def test_run_reranker_request_smoke_passes_for_expected_response() -> None:
    provider = _FakeRerankerSmokeProvider()
    plan = reranker_smoke.build_reranker_request_smoke_plan(
        candidate_texts=("first candidate", "second candidate", "third candidate"),
    )

    report = reranker_smoke.run_reranker_request_smoke(provider, plan=plan)

    assert report.passed is True
    assert len(provider.requests) == 1
    assert provider.requests[0].query_text == "사내 문서 검색 권한과 업무 규칙"
    observation = report.observation
    assert observation.passed is True
    assert observation.provider_type == "remote"
    assert observation.reranker_model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert observation.reranker_profile_name == "qwen3_reranker_0_6b"
    assert observation.candidate_count == 3
    assert observation.returned_count == 2
    assert observation.provider_elapsed_ms == 11
    assert [preview.candidate_key for preview in observation.result_previews] == [
        "candidate-2",
        "candidate-1",
    ]
    assert [preview.score for preview in observation.result_previews] == [0.987654, 0.123457]


def test_run_reranker_request_smoke_reports_response_mismatches() -> None:
    provider = _FakeRerankerSmokeProvider(provider_model_id="wrong-model")
    plan = reranker_smoke.build_reranker_request_smoke_plan()

    report = reranker_smoke.run_reranker_request_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observation.mismatches == (
        "reranker_model_id: expected 'Qwen/Qwen3-Reranker-0.6B', got 'wrong-model'",
    )


def test_run_reranker_request_smoke_reports_runtime_metadata_mismatches() -> None:
    provider = _FakeRerankerSmokeProvider(
        runtime_metadata={
            "service": "unexpected",
            "backend": "mock",
            "device": "cpu",
            "elapsed_ms": "not-an-int",
        }
    )
    plan = reranker_smoke.build_reranker_request_smoke_plan()

    report = reranker_smoke.run_reranker_request_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observation.provider_elapsed_ms is None
    expected_service_mismatch = (
        "runtime_metadata.service: expected "
        "'nex_pcx_reranker_provider_service', got 'unexpected'"
    )
    assert expected_service_mismatch in report.observation.mismatches
    assert "runtime_metadata.backend: expected 'qwen_reranker', got 'mock'" in (
        report.observation.mismatches
    )
    assert "runtime_metadata.device: expected 'cuda:0', got 'cpu'" in (
        report.observation.mismatches
    )


def test_run_reranker_request_smoke_captures_provider_errors() -> None:
    provider = _FailingRerankerSmokeProvider()
    plan = reranker_smoke.build_reranker_request_smoke_plan()

    report = reranker_smoke.run_reranker_request_smoke(provider, plan=plan)

    assert report.passed is False
    assert report.observation.error == "provider unavailable"
    assert report.observation.mismatches == ()
    assert report.observation.result_previews == ()


def test_reranker_request_smoke_writes_markdown_report(tmp_path) -> None:
    provider = _FakeRerankerSmokeProvider()
    plan = reranker_smoke.build_reranker_request_smoke_plan()
    report = reranker_smoke.run_reranker_request_smoke(provider, plan=plan)
    output_path = tmp_path / "nested" / "reranker_smoke.md"

    reranker_smoke.write_markdown_report(report, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "# Remote Reranker Request Smoke Result" in content
    assert "`passed`: `true`" in content
    assert "`candidate-2`" in content
    assert "`0.987654`" in content
    assert '"backend": "qwen_reranker"' in content


def test_reranker_request_smoke_cli_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_remote_reranker_request_smoke.py",
            "--base-url",
            "http://gpu-reranker.local:9104/",
            "--query-text",
            "sensitive query",
            "--candidate-text",
            "sensitive candidate",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["plan"]["base_url"] == "http://gpu-reranker.local:9104"
    assert payload["plan"]["rerank_url"] == "http://gpu-reranker.local:9104/v1/rerank"
    assert payload["plan"]["request"]["query_text"] == "<query_text>"
    assert payload["plan"]["request"]["candidate_count"] == 1
    assert payload["plan"]["request"]["candidates"][0]["text"] == "<candidate_text:1>"


class _FakeRerankerSmokeProvider:
    def __init__(
        self,
        *,
        provider_model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        provider_type: str = "remote",
        runtime_metadata: dict[str, object] | None = None,
    ) -> None:
        self.provider_model_id = provider_model_id
        self.provider_type = provider_type
        self.runtime_metadata = runtime_metadata
        self.requests: list[RerankRequest] = []

    def rerank(self, request: RerankRequest) -> RerankResult:
        self.requests.append(request)
        result_candidates = (request.candidates[1], request.candidates[0])
        scores = (0.987654321, 0.123456789)
        return RerankResult(
            query_text=request.query_text,
            reranker_profile_name=request.reranker_profile_name,
            reranker_model_id=self.provider_model_id,
            provider_type=self.provider_type,
            retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
            candidate_count=len(request.candidates),
            returned_count=len(result_candidates),
            top_k=request.top_k,
            results=tuple(
                RerankResultItem(
                    candidate=candidate,
                    rank=index,
                    score=score,
                    score_components={
                        "source_rank": candidate.rank,
                        "raw_cross_encoder_score": score,
                    },
                )
                for index, (candidate, score) in enumerate(
                    zip(result_candidates, scores, strict=True),
                    start=1,
                )
            ),
            runtime_metadata=self.runtime_metadata
            or {
                "service": "nex_pcx_reranker_provider_service",
                "backend": "qwen_reranker",
                "device": "cuda:0",
                "elapsed_ms": 11,
            },
        )


class _FailingRerankerSmokeProvider:
    def rerank(self, request: RerankRequest) -> RerankResult:
        raise InvalidRerankerError("provider unavailable")
