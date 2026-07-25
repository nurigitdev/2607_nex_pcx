"""Run Search Compare E2E smoke with the remote reranker profile enabled."""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.reranked_search import RERANKED_SEARCH_PROFILE_NAME  # noqa: E402
from app.core.rerankers import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    REMOTE_RERANKER_PROVIDER_MODE,
    RERANK_RETRIEVAL_STRATEGY,
    RerankerRuntimeConfig,
)
from app.core.search_compare import (  # noqa: E402
    SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
    SearchCompareInput,
    SearchCompareResult,
    run_search_compare,
)
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_GPU_HOST,
)
from scripts.plan_remote_reranker_foreground_smoke import (  # noqa: E402
    build_reranker_foreground_smoke_plan,
)
from scripts.run_reranker_provider import (  # noqa: E402
    DEFAULT_RERANKER_PROVIDER_PORT,
)

DEFAULT_QUERY_TEXT = "사내 문서 검색 권한과 업무 규칙"
DEFAULT_ACTOR_USER_ID = 1
DEFAULT_REQUESTED_SEARCH_SCOPE = "company"
DEFAULT_TOP_K = 3
DEFAULT_RERANKED_VECTOR_PROFILE_NAME = "qwen3_4b_2560"
DEFAULT_CHUNK_POLICY_NAME = "heading_512_64"
DEFAULT_TIMEOUT_SECONDS = 300.0
SCORE_PREVIEW_DIGITS = 6


@dataclass(frozen=True)
class SearchCompareRemoteRerankerE2EPlan:
    database_url: str | None
    query_text: str
    actor_user_id: int
    requested_search_scope: str
    top_k: int
    chunk_policy_name: str | None
    document_group: str | None
    file_type: str | None
    reranked_vector_profile_name: str
    remote_reranker_base_url: str
    remote_reranker_timeout_seconds: float
    expected_reranker_model_id: str
    expected_reranker_profile_name: str
    expected_reranker_provider_type: str
    expected_reranker_backend: str | None
    expected_reranker_device: str | None
    allow_mock_fallback: bool


@dataclass(frozen=True)
class SearchCompareRerankedResultPreview:
    rank: int
    chunk_id: int
    score: float
    source_rank: int | None
    source_score: float | None
    source_profile_name: str | None
    document_title: str | None
    original_file_name: str | None


@dataclass(frozen=True)
class SearchCompareRemoteRerankerE2EObservation:
    search_log_id: int | None
    total_elapsed_ms: int | None
    request_elapsed_ms: int
    profile_name: str | None
    profile_status: str | None
    profile_elapsed_ms: int | None
    result_count: int
    profile_query_runtime_metadata: dict[str, Any]
    result_previews: tuple[SearchCompareRerankedResultPreview, ...]
    mismatches: tuple[str, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.mismatches


@dataclass(frozen=True)
class SearchCompareRemoteRerankerE2EReport:
    plan: SearchCompareRemoteRerankerE2EPlan
    observation: SearchCompareRemoteRerankerE2EObservation
    total_elapsed_ms: int

    @property
    def passed(self) -> bool:
        return self.observation.passed


def build_search_compare_remote_reranker_e2e_plan(
    *,
    database_url: str | None = None,
    query_text: str = DEFAULT_QUERY_TEXT,
    actor_user_id: int = DEFAULT_ACTOR_USER_ID,
    requested_search_scope: str = DEFAULT_REQUESTED_SEARCH_SCOPE,
    top_k: int = DEFAULT_TOP_K,
    chunk_policy_name: str | None = DEFAULT_CHUNK_POLICY_NAME,
    document_group: str | None = None,
    file_type: str | None = None,
    reranked_vector_profile_name: str = DEFAULT_RERANKED_VECTOR_PROFILE_NAME,
    remote_reranker_base_url: str | None = None,
    host: str = DEFAULT_GPU_HOST,
    route_host: str | None = None,
    port: int = DEFAULT_RERANKER_PROVIDER_PORT,
    remote_reranker_timeout_seconds: float | None = None,
    expected_reranker_model_id: str = DEFAULT_RERANKER_MODEL_ID,
    expected_reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME,
    expected_reranker_backend: str | None = "qwen_reranker",
    expected_reranker_device: str | None = DEFAULT_DEVICE,
    allow_mock_fallback: bool = False,
) -> SearchCompareRemoteRerankerE2EPlan:
    settings = get_settings()
    selected_base_url = _normalize_base_url(
        remote_reranker_base_url
        or settings.remote_reranker_provider_url
        or build_reranker_foreground_smoke_plan(
            host=host,
            route_host=route_host,
            port=port,
        ).base_url
    )
    selected_timeout = (
        float(remote_reranker_timeout_seconds)
        if remote_reranker_timeout_seconds is not None
        else max(float(settings.remote_reranker_provider_timeout_seconds), DEFAULT_TIMEOUT_SECONDS)
    )
    if selected_timeout <= 0:
        raise ValueError("remote_reranker_timeout_seconds must be greater than 0")
    if actor_user_id <= 0:
        raise ValueError("actor_user_id must be greater than 0")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    return SearchCompareRemoteRerankerE2EPlan(
        database_url=database_url or settings.database_url,
        query_text=_validate_nonblank(query_text, "query_text"),
        actor_user_id=actor_user_id,
        requested_search_scope=_validate_nonblank(
            requested_search_scope,
            "requested_search_scope",
        ),
        top_k=top_k,
        chunk_policy_name=_validate_optional_nonblank(chunk_policy_name, "chunk_policy_name"),
        document_group=_validate_optional_nonblank(document_group, "document_group"),
        file_type=_validate_optional_nonblank(file_type, "file_type"),
        reranked_vector_profile_name=_validate_nonblank(
            reranked_vector_profile_name,
            "reranked_vector_profile_name",
        ),
        remote_reranker_base_url=selected_base_url,
        remote_reranker_timeout_seconds=selected_timeout,
        expected_reranker_model_id=_validate_nonblank(
            expected_reranker_model_id,
            "expected_reranker_model_id",
        ),
        expected_reranker_profile_name=_validate_nonblank(
            expected_reranker_profile_name,
            "expected_reranker_profile_name",
        ),
        expected_reranker_provider_type=REMOTE_RERANKER_PROVIDER_MODE,
        expected_reranker_backend=_validate_optional_nonblank(
            expected_reranker_backend,
            "expected_reranker_backend",
        ),
        expected_reranker_device=_validate_optional_nonblank(
            expected_reranker_device,
            "expected_reranker_device",
        ),
        allow_mock_fallback=allow_mock_fallback,
    )


def run_search_compare_remote_reranker_e2e_smoke(
    plan: SearchCompareRemoteRerankerE2EPlan,
) -> SearchCompareRemoteRerankerE2EReport:
    if not plan.database_url:
        raise ValueError("database_url is required")

    started_at = perf_counter()
    observation = _run_search_compare_request(plan)
    return SearchCompareRemoteRerankerE2EReport(
        plan=plan,
        observation=observation,
        total_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
    )


def write_markdown_report(
    report: SearchCompareRemoteRerankerE2EReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def _run_search_compare_request(
    plan: SearchCompareRemoteRerankerE2EPlan,
) -> SearchCompareRemoteRerankerE2EObservation:
    started_at = perf_counter()
    try:
        result = run_search_compare(
            plan.database_url or "",
            SearchCompareInput(
                query_text=plan.query_text,
                actor_user_id=plan.actor_user_id,
                requested_search_scope=plan.requested_search_scope,
                top_k=plan.top_k,
                profiles=(RERANKED_SEARCH_PROFILE_NAME,),
                chunk_policy_name=plan.chunk_policy_name,
                document_group=plan.document_group,
                file_type=plan.file_type,
                reranked_vector_profile_name=plan.reranked_vector_profile_name,
                allow_mock_fallback=plan.allow_mock_fallback,
            ),
            fallback_reranker_runtime_config=RerankerRuntimeConfig(
                mode=REMOTE_RERANKER_PROVIDER_MODE,
                remote_base_url=plan.remote_reranker_base_url,
                remote_timeout_seconds=plan.remote_reranker_timeout_seconds,
            ),
        )
    except Exception as exc:
        return SearchCompareRemoteRerankerE2EObservation(
            search_log_id=None,
            total_elapsed_ms=None,
            request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            profile_name=None,
            profile_status=None,
            profile_elapsed_ms=None,
            result_count=0,
            profile_query_runtime_metadata={},
            result_previews=(),
            mismatches=(),
            error=str(exc),
        )
    return _observation_from_result(
        result,
        plan=plan,
        request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
    )


def _observation_from_result(
    result: SearchCompareResult,
    *,
    plan: SearchCompareRemoteRerankerE2EPlan,
    request_elapsed_ms: int,
) -> SearchCompareRemoteRerankerE2EObservation:
    profile = result.profiles[0] if result.profiles else None
    metadata = dict(profile.query_runtime_metadata) if profile is not None else {}
    previews = _result_previews(profile.results if profile is not None else ())
    return SearchCompareRemoteRerankerE2EObservation(
        search_log_id=result.search_log_id,
        total_elapsed_ms=result.total_elapsed_ms,
        request_elapsed_ms=request_elapsed_ms,
        profile_name=profile.profile_name if profile is not None else None,
        profile_status=profile.status if profile is not None else None,
        profile_elapsed_ms=profile.elapsed_ms if profile is not None else None,
        result_count=len(profile.results) if profile is not None else 0,
        profile_query_runtime_metadata=metadata,
        result_previews=previews,
        mismatches=_response_mismatches(result, plan=plan),
        error=None,
    )


def _response_mismatches(
    result: SearchCompareResult,
    *,
    plan: SearchCompareRemoteRerankerE2EPlan,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if result.search_log_id <= 0:
        mismatches.append(f"search_log_id: expected positive value, got {result.search_log_id!r}")
    if len(result.profiles) != 1:
        mismatches.append(f"profile_count: expected 1, got {len(result.profiles)!r}")
        return tuple(mismatches)

    profile = result.profiles[0]
    if profile.profile_name != RERANKED_SEARCH_PROFILE_NAME:
        mismatches.append(
            f"profile_name: expected {RERANKED_SEARCH_PROFILE_NAME!r}, "
            f"got {profile.profile_name!r}"
        )
    if profile.status != SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED:
        mismatches.append(
            f"profile_status: expected {SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED!r}, "
            f"got {profile.status!r}"
        )
        if profile.error_message:
            mismatches.append(f"profile_error_message: {profile.error_message}")
    if not profile.results:
        mismatches.append("result_count: expected at least 1 reranked result")

    mismatches.extend(_metadata_mismatches(profile.query_runtime_metadata, plan=plan))
    for index, item in enumerate(profile.results, start=1):
        result_item = item.vector_result
        if getattr(result_item, "rank", None) != index:
            actual_rank = getattr(result_item, "rank", None)
            mismatches.append(f"result[{index}].rank: expected {index!r}, got {actual_rank!r}")
        if getattr(result_item, "search_profile_name", None) != RERANKED_SEARCH_PROFILE_NAME:
            mismatches.append(
                f"result[{index}].search_profile_name: expected "
                f"{RERANKED_SEARCH_PROFILE_NAME!r}, "
                f"got {getattr(result_item, 'search_profile_name', None)!r}"
            )
        if getattr(result_item, "retrieval_strategy", None) != RERANK_RETRIEVAL_STRATEGY:
            mismatches.append(
                f"result[{index}].retrieval_strategy: expected "
                f"{RERANK_RETRIEVAL_STRATEGY!r}, "
                f"got {getattr(result_item, 'retrieval_strategy', None)!r}"
            )
        score = getattr(result_item, "score", None)
        if not isinstance(score, int | float) or not math.isfinite(float(score)):
            mismatches.append(f"result[{index}].score: expected finite number, got {score!r}")
        mismatches.extend(_score_component_mismatches(index, result_item, plan=plan))
    return tuple(mismatches)


def _metadata_mismatches(
    metadata: dict[str, Any],
    *,
    plan: SearchCompareRemoteRerankerE2EPlan,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    expected_values: dict[str, object] = {
        "provider_runtime_mode": REMOTE_RERANKER_PROVIDER_MODE,
        "provider_runtime_base_url": plan.remote_reranker_base_url,
        "provider_runtime_timeout_seconds": plan.remote_reranker_timeout_seconds,
        "source_vector_profile_name": plan.reranked_vector_profile_name,
        "reranked_vector_profile_name": plan.reranked_vector_profile_name,
        "retrieval_strategy": RERANK_RETRIEVAL_STRATEGY,
        "search_profile_name": RERANKED_SEARCH_PROFILE_NAME,
        "reranker_model_id": plan.expected_reranker_model_id,
        "reranker_profile_name": plan.expected_reranker_profile_name,
        "reranker_provider_type": plan.expected_reranker_provider_type,
    }
    for key, expected in expected_values.items():
        actual = metadata.get(key)
        if actual != expected:
            mismatches.append(f"metadata.{key}: expected {expected!r}, got {actual!r}")

    candidate_count = metadata.get("candidate_count")
    if not isinstance(candidate_count, int) or candidate_count <= 0:
        mismatches.append(
            f"metadata.candidate_count: expected positive integer, got {candidate_count!r}"
        )

    runtime_metadata = metadata.get("reranker_runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        mismatches.append("metadata.reranker_runtime_metadata: expected JSON object")
        return tuple(mismatches)

    expected_runtime_values: dict[str, str] = {
        "service": "nex_pcx_reranker_provider_service",
    }
    if plan.expected_reranker_backend is not None:
        expected_runtime_values["backend"] = plan.expected_reranker_backend
    if plan.expected_reranker_device is not None:
        expected_runtime_values["device"] = plan.expected_reranker_device
    for key, expected in expected_runtime_values.items():
        actual = runtime_metadata.get(key)
        if actual != expected:
            mismatches.append(
                f"metadata.reranker_runtime_metadata.{key}: "
                f"expected {expected!r}, got {actual!r}"
            )
    return tuple(mismatches)


def _score_component_mismatches(
    index: int,
    result_item: object,
    *,
    plan: SearchCompareRemoteRerankerE2EPlan,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    score_components = getattr(result_item, "score_components", {})
    if not isinstance(score_components, dict):
        return (f"result[{index}].score_components: expected JSON object",)
    expected_values = {
        "source_profile_name": plan.reranked_vector_profile_name,
        "source_retrieval_strategy": "vector_cosine",
        "reranker_profile_name": plan.expected_reranker_profile_name,
        "reranker_model_id": plan.expected_reranker_model_id,
        "reranker_provider_type": plan.expected_reranker_provider_type,
    }
    for key, expected in expected_values.items():
        actual = score_components.get(key)
        if actual != expected:
            mismatches.append(
                f"result[{index}].score_components.{key}: " f"expected {expected!r}, got {actual!r}"
            )
    if not isinstance(score_components.get("source_rank"), int):
        mismatches.append(
            f"result[{index}].score_components.source_rank: expected integer, "
            f"got {score_components.get('source_rank')!r}"
        )
    if not isinstance(score_components.get("candidate_count"), int):
        mismatches.append(
            f"result[{index}].score_components.candidate_count: expected integer, "
            f"got {score_components.get('candidate_count')!r}"
        )
    return tuple(mismatches)


def _result_previews(results: tuple[object, ...]) -> tuple[SearchCompareRerankedResultPreview, ...]:
    previews: list[SearchCompareRerankedResultPreview] = []
    for item in results:
        result = item.vector_result
        score_components = getattr(result, "score_components", {})
        if not isinstance(score_components, dict):
            score_components = {}
        score = getattr(result, "score", 0.0)
        previews.append(
            SearchCompareRerankedResultPreview(
                rank=int(getattr(result, "rank", 0)),
                chunk_id=int(getattr(result, "chunk_id", 0)),
                score=round(float(score), SCORE_PREVIEW_DIGITS),
                source_rank=_metadata_int(score_components, "source_rank"),
                source_score=_metadata_float(score_components, "source_score"),
                source_profile_name=_metadata_str(score_components, "source_profile_name"),
                document_title=_optional_str(getattr(result, "document_title", None)),
                original_file_name=_optional_str(getattr(result, "original_file_name", None)),
            )
        )
    return tuple(previews)


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    return _optional_str(metadata.get(key))


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _validate_optional_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _validate_nonblank(value, field_name)


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("remote_reranker_base_url is required")
    return normalized


def _redact_database_url(database_url: str | None) -> str | None:
    if not database_url:
        return None
    parts = urlsplit(database_url)
    if "@" not in parts.netloc:
        return database_url
    userinfo, hostinfo = parts.netloc.rsplit("@", 1)
    username = userinfo.split(":", 1)[0]
    return urlunsplit((parts.scheme, f"{username}:***@{hostinfo}", parts.path, "", ""))


def _report_payload(report: SearchCompareRemoteRerankerE2EReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "observation": asdict(report.observation),
        "total_elapsed_ms": report.total_elapsed_ms,
    }


def _plan_payload(plan: SearchCompareRemoteRerankerE2EPlan) -> dict[str, Any]:
    return {
        **asdict(plan),
        "database_url": _redact_database_url(plan.database_url),
        "query_text": "<query_text>",
    }


def _print_human_report(report: SearchCompareRemoteRerankerE2EReport) -> None:
    observation = report.observation
    status = "PASS" if report.passed else "FAIL"
    print(f"Search Compare remote reranker E2E smoke: {status}")
    print(f"- database_url: {_redact_database_url(report.plan.database_url)}")
    print(f"- remote_reranker_base_url: {report.plan.remote_reranker_base_url}")
    print(f"- source_vector_profile: {report.plan.reranked_vector_profile_name}")
    print(f"- search_log_id: {observation.search_log_id}")
    print(f"- profile_status: {observation.profile_status}")
    print(f"- result_count: {observation.result_count}")
    print(f"- request_elapsed_ms: {observation.request_elapsed_ms}")
    for preview in observation.result_previews:
        print(
            f"- rank {preview.rank}: chunk={preview.chunk_id} "
            f"score={preview.score} source_rank={preview.source_rank}"
        )
    if observation.mismatches:
        print("- mismatches:")
        for mismatch in observation.mismatches:
            print(f"  - {mismatch}")
    if observation.error:
        print(f"- error: {observation.error}")


def _markdown_report(report: SearchCompareRemoteRerankerE2EReport) -> str:
    observation = report.observation
    lines = [
        "# Search Compare Remote Reranker E2E Smoke Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `database_url`: `{_redact_database_url(report.plan.database_url)}`",
        f"- `remote_reranker_base_url`: `{report.plan.remote_reranker_base_url}`",
        f"- `source_vector_profile`: `{report.plan.reranked_vector_profile_name}`",
        f"- `search_log_id`: `{observation.search_log_id}`",
        f"- `profile_status`: `{observation.profile_status}`",
        f"- `result_count`: `{observation.result_count}`",
        f"- `request_elapsed_ms`: `{observation.request_elapsed_ms}`",
        "",
        "## Reranked Result Preview",
        "",
        "| Rank | Chunk | Score | Source Rank | Source Score | Document | File |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for preview in observation.result_previews:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{preview.rank}`",
                    f"`{preview.chunk_id}`",
                    f"`{preview.score}`",
                    f"`{preview.source_rank}`",
                    f"`{preview.source_score}`",
                    f"`{preview.document_title or ''}`",
                    f"`{preview.original_file_name or ''}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Profile Runtime Metadata",
            "",
            "```json",
            json.dumps(
                observation.profile_query_runtime_metadata,
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )
    if observation.mismatches:
        lines.extend(["## Mismatches", ""])
        lines.extend(f"- `{mismatch}`" for mismatch in observation.mismatches)
        lines.append("")
    if observation.error:
        lines.extend(["## Error", "", f"`{observation.error}`", ""])
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Search Compare E2E smoke with a remote reranker provider.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--query-text", default=DEFAULT_QUERY_TEXT)
    parser.add_argument("--actor-user-id", type=int, default=DEFAULT_ACTOR_USER_ID)
    parser.add_argument("--requested-search-scope", default=DEFAULT_REQUESTED_SEARCH_SCOPE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-policy-name", default=DEFAULT_CHUNK_POLICY_NAME)
    parser.add_argument("--document-group", default=None)
    parser.add_argument("--file-type", default=None)
    parser.add_argument(
        "--reranked-vector-profile-name",
        default=DEFAULT_RERANKED_VECTOR_PROFILE_NAME,
    )
    parser.add_argument("--remote-reranker-base-url", default=None)
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_RERANKER_PROVIDER_PORT)
    parser.add_argument("--remote-reranker-timeout-seconds", type=float, default=None)
    parser.add_argument("--expected-reranker-model-id", default=DEFAULT_RERANKER_MODEL_ID)
    parser.add_argument("--expected-reranker-profile-name", default=DEFAULT_RERANKER_PROFILE_NAME)
    parser.add_argument("--expected-reranker-backend", default="qwen_reranker")
    parser.add_argument("--expected-reranker-device", default=DEFAULT_DEVICE)
    parser.add_argument("--allow-mock-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_search_compare_remote_reranker_e2e_plan(
            database_url=args.database_url,
            query_text=args.query_text,
            actor_user_id=args.actor_user_id,
            requested_search_scope=args.requested_search_scope,
            top_k=args.top_k,
            chunk_policy_name=args.chunk_policy_name,
            document_group=args.document_group,
            file_type=args.file_type,
            reranked_vector_profile_name=args.reranked_vector_profile_name,
            remote_reranker_base_url=args.remote_reranker_base_url,
            host=args.host,
            route_host=args.route_host,
            port=args.port,
            remote_reranker_timeout_seconds=args.remote_reranker_timeout_seconds,
            expected_reranker_model_id=args.expected_reranker_model_id,
            expected_reranker_profile_name=args.expected_reranker_profile_name,
            expected_reranker_backend=args.expected_reranker_backend,
            expected_reranker_device=args.expected_reranker_device,
            allow_mock_fallback=args.allow_mock_fallback,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0
    if not plan.database_url:
        parser.error("database_url is required. Set --database-url or NEX_PCX_DATABASE_URL.")

    report = run_search_compare_remote_reranker_e2e_smoke(plan)
    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
