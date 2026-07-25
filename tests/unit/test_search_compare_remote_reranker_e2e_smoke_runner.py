import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.permissions import PermissionSearchFilter
from app.core.reranked_search import RERANKED_SEARCH_PROFILE_NAME, RerankedSearchResult
from app.core.rerankers import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    RERANK_RETRIEVAL_STRATEGY,
)
from app.core.search_compare import (
    SEARCH_COMPARE_PROFILE_STATUS_FAILED,
    SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
    SearchCompareProfileResult,
    SearchCompareResult,
    SearchCompareResultItem,
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


runner = _load_script_module("run_search_compare_remote_reranker_e2e_smoke.py")


def test_build_search_compare_remote_reranker_e2e_plan_defaults_to_dgx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_URL", raising=False)
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)

    plan = runner.build_search_compare_remote_reranker_e2e_plan()

    assert plan.database_url is None
    assert plan.remote_reranker_base_url == "http://192.168.20.243:9104"
    assert plan.remote_reranker_timeout_seconds == 300.0
    assert plan.reranked_vector_profile_name == "qwen3_4b_2560"
    assert plan.expected_reranker_model_id == DEFAULT_RERANKER_MODEL_ID
    assert plan.expected_reranker_profile_name == DEFAULT_RERANKER_PROFILE_NAME
    assert plan.expected_reranker_provider_type == "remote"
    assert plan.expected_reranker_backend == "qwen_reranker"
    assert plan.expected_reranker_device == "cuda:0"
    assert plan.allow_mock_fallback is False


def test_build_search_compare_remote_reranker_e2e_plan_uses_env_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEX_PCX_DATABASE_URL", "postgresql://user:pass@db.local/nex")
    monkeypatch.setenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_URL", "http://env-reranker:9104/")
    monkeypatch.setenv("NEX_PCX_REMOTE_RERANKER_PROVIDER_TIMEOUT_SECONDS", "90")

    plan = runner.build_search_compare_remote_reranker_e2e_plan(
        query_text="  policy query ",
        remote_reranker_timeout_seconds=45,
        chunk_policy_name=None,
        expected_reranker_backend=None,
        expected_reranker_device=None,
        allow_mock_fallback=True,
    )

    assert plan.database_url == "postgresql://user:pass@db.local/nex"
    assert plan.query_text == "policy query"
    assert plan.remote_reranker_base_url == "http://env-reranker:9104"
    assert plan.remote_reranker_timeout_seconds == 45.0
    assert plan.chunk_policy_name is None
    assert plan.expected_reranker_backend is None
    assert plan.expected_reranker_device is None
    assert plan.allow_mock_fallback is True


def test_build_search_compare_remote_reranker_e2e_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="query_text"):
        runner.build_search_compare_remote_reranker_e2e_plan(query_text=" ")

    with pytest.raises(ValueError, match="actor_user_id"):
        runner.build_search_compare_remote_reranker_e2e_plan(actor_user_id=0)

    with pytest.raises(ValueError, match="top_k"):
        runner.build_search_compare_remote_reranker_e2e_plan(top_k=0)

    with pytest.raises(ValueError, match="remote_reranker_timeout_seconds"):
        runner.build_search_compare_remote_reranker_e2e_plan(
            remote_reranker_timeout_seconds=0,
        )


def test_run_search_compare_remote_reranker_e2e_smoke_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_database_urls: list[str] = []
    captured_inputs = []
    captured_reranker_configs = []

    def fake_run_search_compare(database_url, search_input, **kwargs):
        captured_database_urls.append(database_url)
        captured_inputs.append(search_input)
        captured_reranker_configs.append(kwargs["fallback_reranker_runtime_config"])
        return _search_compare_result()

    monkeypatch.setattr(runner, "run_search_compare", fake_run_search_compare)
    plan = runner.build_search_compare_remote_reranker_e2e_plan(
        database_url="postgresql://nex:secret@localhost/nex",
        remote_reranker_base_url="http://reranker.local:9104/",
        remote_reranker_timeout_seconds=77,
    )

    report = runner.run_search_compare_remote_reranker_e2e_smoke(plan)

    assert report.passed is True
    assert captured_database_urls == ["postgresql://nex:secret@localhost/nex"]
    assert captured_inputs[0].profiles == (RERANKED_SEARCH_PROFILE_NAME,)
    assert captured_inputs[0].allow_mock_fallback is False
    assert captured_inputs[0].reranked_vector_profile_name == "qwen3_4b_2560"
    assert captured_reranker_configs[0].mode == "remote"
    assert captured_reranker_configs[0].remote_base_url == "http://reranker.local:9104"
    assert captured_reranker_configs[0].remote_timeout_seconds == 77.0
    assert report.observation.search_log_id == 901
    assert report.observation.profile_status == SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED
    assert report.observation.result_count == 2
    assert [preview.chunk_id for preview in report.observation.result_previews] == [102, 101]
    assert [preview.score for preview in report.observation.result_previews] == [
        0.876543,
        0.654321,
    ]


def test_run_search_compare_remote_reranker_e2e_smoke_reports_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "run_search_compare",
        lambda *_args, **_kwargs: _search_compare_result(
            metadata_overrides={
                "provider_runtime_mode": "mock",
                "reranker_provider_type": "mock",
                "candidate_count": 0,
                "reranker_runtime_metadata": {
                    "service": "unexpected",
                    "backend": "mock",
                    "device": "cpu",
                },
            },
            result_overrides={
                "search_profile_name": "wrong-profile",
                "retrieval_strategy": "vector_cosine",
                "score_components": {},
            },
        ),
    )
    plan = runner.build_search_compare_remote_reranker_e2e_plan(
        database_url="postgresql://nex:secret@localhost/nex",
        remote_reranker_base_url="http://reranker.local:9104",
    )

    report = runner.run_search_compare_remote_reranker_e2e_smoke(plan)

    assert report.passed is False
    assert "metadata.provider_runtime_mode: expected 'remote', got 'mock'" in (
        report.observation.mismatches
    )
    assert "metadata.candidate_count: expected positive integer, got 0" in (
        report.observation.mismatches
    )
    assert (
        "metadata.reranker_runtime_metadata.service: "
        "expected 'nex_pcx_reranker_provider_service', got 'unexpected'"
    ) in report.observation.mismatches
    assert (
        "result[1].search_profile_name: expected 'reranked_vector_cosine', got 'wrong-profile'"
    ) in report.observation.mismatches
    assert (
        "result[1].score_components.source_rank: expected integer, got None"
    ) in report.observation.mismatches


def test_run_search_compare_remote_reranker_e2e_smoke_reports_profile_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "run_search_compare",
        lambda *_args, **_kwargs: _search_compare_result(
            profile_status=SEARCH_COMPARE_PROFILE_STATUS_FAILED,
            error_message="remote reranker timeout",
            results=(),
        ),
    )
    plan = runner.build_search_compare_remote_reranker_e2e_plan(
        database_url="postgresql://nex:secret@localhost/nex",
        remote_reranker_base_url="http://reranker.local:9104",
        remote_reranker_timeout_seconds=77,
    )

    report = runner.run_search_compare_remote_reranker_e2e_smoke(plan)

    assert report.passed is False
    assert "profile_status: expected 'succeeded', got 'failed'" in report.observation.mismatches
    assert "profile_error_message: remote reranker timeout" in report.observation.mismatches
    assert "result_count: expected at least 1 reranked result" in (report.observation.mismatches)


def test_run_search_compare_remote_reranker_e2e_smoke_captures_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_search_compare(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(runner, "run_search_compare", fake_run_search_compare)
    plan = runner.build_search_compare_remote_reranker_e2e_plan(
        database_url="postgresql://nex:secret@localhost/nex",
    )

    report = runner.run_search_compare_remote_reranker_e2e_smoke(plan)

    assert report.passed is False
    assert report.observation.error == "database unavailable"
    assert report.observation.mismatches == ()


def test_search_compare_remote_reranker_e2e_smoke_requires_database_url() -> None:
    plan = runner.build_search_compare_remote_reranker_e2e_plan(database_url=None)

    with pytest.raises(ValueError, match="database_url"):
        runner.run_search_compare_remote_reranker_e2e_smoke(plan)


def test_search_compare_remote_reranker_e2e_smoke_writes_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "run_search_compare",
        lambda *_args, **_kwargs: _search_compare_result(),
    )
    plan = runner.build_search_compare_remote_reranker_e2e_plan(
        database_url="postgresql://nex:secret@localhost/nex",
        remote_reranker_base_url="http://reranker.local:9104",
        remote_reranker_timeout_seconds=77,
    )
    report = runner.run_search_compare_remote_reranker_e2e_smoke(plan)
    output_path = tmp_path / "nested" / "search_compare_reranker.md"

    runner.write_markdown_report(report, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "# Search Compare Remote Reranker E2E Smoke Result" in content
    assert "`passed`: `true`" in content
    assert "postgresql://nex:***@localhost/nex" in content
    assert "`qwen3_4b_2560`" in content
    assert '"provider_runtime_mode": "remote"' in content


def test_search_compare_remote_reranker_e2e_smoke_cli_prints_json_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_search_compare_remote_reranker_e2e_smoke.py",
            "--database-url",
            "postgresql://nex:secret@localhost/nex",
            "--remote-reranker-base-url",
            "http://reranker.local:9104/",
            "--query-text",
            "sensitive query",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["plan"]["database_url"] == "postgresql://nex:***@localhost/nex"
    assert payload["plan"]["query_text"] == "<query_text>"
    assert payload["plan"]["remote_reranker_base_url"] == "http://reranker.local:9104"
    assert payload["plan"]["reranked_vector_profile_name"] == "qwen3_4b_2560"


def _permission_filter() -> PermissionSearchFilter:
    return PermissionSearchFilter(
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        where_sql="TRUE",
        params=(),
        metadata={"fixture": "search_compare_remote_reranker_e2e"},
    )


def _search_compare_result(
    *,
    metadata_overrides: dict[str, object] | None = None,
    result_overrides: dict[str, object] | None = None,
    profile_status: str = SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
    error_message: str | None = None,
    results: tuple[SearchCompareResultItem, ...] | None = None,
) -> SearchCompareResult:
    profile_results = results
    if profile_results is None:
        profile_results = (
            SearchCompareResultItem(1, _reranked_result(rank=1, chunk_id=102, score=0.87654321)),
            SearchCompareResultItem(2, _reranked_result(rank=2, chunk_id=101, score=0.65432109)),
        )
    profile_metadata = _profile_metadata(metadata_overrides)
    return SearchCompareResult(
        search_log_id=901,
        query_text="사내 문서 검색 권한과 업무 규칙",
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        permission_filter=_permission_filter(),
        top_k=3,
        profiles=(
            SearchCompareProfileResult(
                profile_name=RERANKED_SEARCH_PROFILE_NAME,
                elapsed_ms=88,
                results=tuple(
                    SearchCompareResultItem(
                        item.search_log_result_id,
                        _override_result(item.vector_result, result_overrides),
                    )
                    for item in profile_results
                ),
                status=profile_status,
                error_message=error_message,
                query_runtime_metadata=profile_metadata,
            ),
        ),
        total_elapsed_ms=99,
    )


def _profile_metadata(overrides: dict[str, object] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "provider_type": "rerank",
        "provider_model_id": DEFAULT_RERANKER_MODEL_ID,
        "runtime_source": "local_reranker_contract",
        "query_embedding_bridge": True,
        "retrieval_strategy": RERANK_RETRIEVAL_STRATEGY,
        "search_profile_name": RERANKED_SEARCH_PROFILE_NAME,
        "reranked_vector_profile_name": "qwen3_4b_2560",
        "source_vector_profile_name": "qwen3_4b_2560",
        "candidate_top_k": 12,
        "candidate_multiplier": 4,
        "provider_runtime_mode": "remote",
        "provider_runtime_base_url": "http://reranker.local:9104",
        "provider_runtime_timeout_seconds": 77.0,
        "reranker_profile_name": DEFAULT_RERANKER_PROFILE_NAME,
        "reranker_model_id": DEFAULT_RERANKER_MODEL_ID,
        "reranker_provider_type": "remote",
        "candidate_count": 2,
        "reranker_runtime_metadata": {
            "service": "nex_pcx_reranker_provider_service",
            "backend": "qwen_reranker",
            "device": "cuda:0",
            "elapsed_ms": 44,
        },
        "source_query_runtime_metadata": {"provider_runtime_source": "route"},
    }
    if overrides:
        metadata.update(overrides)
    return metadata


def _reranked_result(*, rank: int, chunk_id: int, score: float) -> RerankedSearchResult:
    return RerankedSearchResult(
        profile_name=RERANKED_SEARCH_PROFILE_NAME,
        distance=0.1,
        embedding_elapsed_ms=4,
        search_profile_name=RERANKED_SEARCH_PROFILE_NAME,
        retrieval_strategy=RERANK_RETRIEVAL_STRATEGY,
        rank=rank,
        chunk_id=chunk_id,
        document_id=201,
        file_id=301,
        score=score,
        chunk_text="사내 문서 검색 권한 관련 chunk",
        chunk_preview="사내 문서 검색 권한 관련 chunk",
        content_hash=f"hash-{chunk_id}",
        chunk_policy_name="heading_512_64",
        heading_path=("Policy",),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title="Policy",
        document_group="company",
        original_file_name="policy.md",
        file_ext=".md",
        score_components={
            "source_profile_name": "qwen3_4b_2560",
            "source_retrieval_strategy": "vector_cosine",
            "source_rank": rank + 1,
            "source_score": 0.91 - (rank * 0.01),
            "source_score_components": {"distance": 0.1},
            "reranker_profile_name": DEFAULT_RERANKER_PROFILE_NAME,
            "reranker_model_id": DEFAULT_RERANKER_MODEL_ID,
            "reranker_provider_type": "remote",
            "candidate_count": 2,
            "reranker_runtime_metadata": {
                "service": "nex_pcx_reranker_provider_service",
                "backend": "qwen_reranker",
                "device": "cuda:0",
                "elapsed_ms": 44,
            },
        },
    )


def _override_result(
    result: RerankedSearchResult,
    overrides: dict[str, object] | None,
) -> RerankedSearchResult:
    if not overrides:
        return result
    return RerankedSearchResult(
        **{
            **result.__dict__,
            **overrides,
        }
    )
