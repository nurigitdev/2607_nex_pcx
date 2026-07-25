"""Run reranker request smoke checks against a remote reranker provider."""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rerankers import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    REMOTE_RERANKER_PROVIDER_MODE,
    REMOTE_RERANKER_RERANK_PATH,
    RERANK_RETRIEVAL_STRATEGY,
    InvalidRerankerError,
    RemoteRerankerProviderClient,
    RerankCandidate,
    RerankerProvider,
    RerankRequest,
    RerankResult,
)
from app.reranker_provider_service import RERANKER_PROVIDER_BACKEND_QWEN  # noqa: E402
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_GPU_HOST,
)
from scripts.plan_remote_reranker_foreground_smoke import (  # noqa: E402
    build_reranker_foreground_smoke_plan,
)
from scripts.run_reranker_provider import (  # noqa: E402
    DEFAULT_RERANKER_PROVIDER_NAME,
    DEFAULT_RERANKER_PROVIDER_PORT,
)

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_TOP_K = 2
DEFAULT_SOURCE_PROFILE_NAME = "qwen3_4b_2560"
DEFAULT_SOURCE_RETRIEVAL_STRATEGY = "vector_cosine"
DEFAULT_QUERY_TEXT = "사내 문서 검색 권한과 업무 규칙"
DEFAULT_CANDIDATE_TEXTS = (
    "사내 공통 업무 규칙 문서는 모든 직원에게 공개되며 검색 범위에 포함된다.",
    "개인 업로드 문서는 작성자와 권한을 가진 상위 조직 사용자에게만 검색된다.",
    "임베딩 provider 상태 점검은 운영자가 관리자 화면에서 확인할 수 있다.",
)
SCORE_PREVIEW_DIGITS = 6


@dataclass(frozen=True)
class RerankerResultPreview:
    candidate_key: str
    rank: int
    score: float
    source_rank: int | None
    score_components: dict[str, Any]


@dataclass(frozen=True)
class RemoteRerankerRequestSmokePlan:
    provider_name: str
    base_url: str
    rerank_url: str
    provider_model_id: str
    reranker_profile_name: str
    expected_provider_type: str
    expected_backend: str
    expected_device: str | None
    timeout_seconds: float
    request: RerankRequest


@dataclass(frozen=True)
class RemoteRerankerRequestSmokeObservation:
    request_elapsed_ms: int
    provider_elapsed_ms: int | None
    provider_type: str | None
    reranker_model_id: str | None
    reranker_profile_name: str | None
    retrieval_strategy: str | None
    candidate_count: int | None
    returned_count: int | None
    top_k: int | None
    runtime_metadata: dict[str, Any]
    result_previews: tuple[RerankerResultPreview, ...]
    mismatches: tuple[str, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.mismatches


@dataclass(frozen=True)
class RemoteRerankerRequestSmokeReport:
    plan: RemoteRerankerRequestSmokePlan
    observation: RemoteRerankerRequestSmokeObservation
    total_elapsed_ms: int

    @property
    def passed(self) -> bool:
        return self.observation.passed


def build_reranker_request_smoke_plan(
    *,
    base_url: str | None = None,
    host: str = DEFAULT_GPU_HOST,
    route_host: str | None = None,
    port: int = DEFAULT_RERANKER_PROVIDER_PORT,
    provider_name: str = DEFAULT_RERANKER_PROVIDER_NAME,
    provider_model_id: str = DEFAULT_RERANKER_MODEL_ID,
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME,
    expected_backend: str = RERANKER_PROVIDER_BACKEND_QWEN,
    expected_device: str | None = DEFAULT_DEVICE,
    query_text: str = DEFAULT_QUERY_TEXT,
    candidate_texts: tuple[str, ...] = DEFAULT_CANDIDATE_TEXTS,
    source_profile_name: str = DEFAULT_SOURCE_PROFILE_NAME,
    source_retrieval_strategy: str = DEFAULT_SOURCE_RETRIEVAL_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> RemoteRerankerRequestSmokePlan:
    selected_base_url = _normalize_base_url(
        base_url
        or build_reranker_foreground_smoke_plan(
            host=host,
            route_host=route_host,
            port=port,
            provider_name=provider_name,
            provider_model_id=provider_model_id,
            reranker_profile_name=reranker_profile_name,
        ).base_url
    )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    normalized_query = query_text.strip()
    if not normalized_query:
        raise ValueError("query_text is required")
    candidates = _build_candidates(
        candidate_texts,
        source_profile_name=source_profile_name,
        source_retrieval_strategy=source_retrieval_strategy,
    )
    request = RerankRequest(
        query_text=normalized_query,
        candidates=candidates,
        top_k=top_k,
        reranker_profile_name=_validate_nonblank(
            reranker_profile_name,
            "reranker_profile_name",
        ),
        reranker_model_id=_validate_nonblank(provider_model_id, "provider_model_id"),
    )
    _validate_request_shape(request)
    return RemoteRerankerRequestSmokePlan(
        provider_name=_validate_nonblank(provider_name, "provider_name"),
        base_url=selected_base_url,
        rerank_url=f"{selected_base_url}{REMOTE_RERANKER_RERANK_PATH}",
        provider_model_id=request.reranker_model_id,
        reranker_profile_name=request.reranker_profile_name,
        expected_provider_type=REMOTE_RERANKER_PROVIDER_MODE,
        expected_backend=_validate_nonblank(expected_backend, "expected_backend"),
        expected_device=expected_device.strip() if expected_device else None,
        timeout_seconds=timeout_seconds,
        request=request,
    )


def run_reranker_request_smoke(
    provider: RerankerProvider,
    *,
    plan: RemoteRerankerRequestSmokePlan,
) -> RemoteRerankerRequestSmokeReport:
    started_at = perf_counter()
    observation = _run_request(provider, plan=plan)
    return RemoteRerankerRequestSmokeReport(
        plan=plan,
        observation=observation,
        total_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
    )


def write_markdown_report(
    report: RemoteRerankerRequestSmokeReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def _run_request(
    provider: RerankerProvider,
    *,
    plan: RemoteRerankerRequestSmokePlan,
) -> RemoteRerankerRequestSmokeObservation:
    started_at = perf_counter()
    try:
        response = provider.rerank(plan.request)
    except (InvalidRerankerError, ValueError, OSError) as exc:
        return RemoteRerankerRequestSmokeObservation(
            request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            provider_elapsed_ms=None,
            provider_type=None,
            reranker_model_id=None,
            reranker_profile_name=None,
            retrieval_strategy=None,
            candidate_count=None,
            returned_count=None,
            top_k=None,
            runtime_metadata={},
            result_previews=(),
            mismatches=(),
            error=str(exc),
        )

    runtime_metadata = dict(response.runtime_metadata)
    return RemoteRerankerRequestSmokeObservation(
        request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
        provider_elapsed_ms=_metadata_int(runtime_metadata, "elapsed_ms"),
        provider_type=response.provider_type,
        reranker_model_id=response.reranker_model_id,
        reranker_profile_name=response.reranker_profile_name,
        retrieval_strategy=response.retrieval_strategy,
        candidate_count=response.candidate_count,
        returned_count=response.returned_count,
        top_k=response.top_k,
        runtime_metadata=runtime_metadata,
        result_previews=_result_previews(response),
        mismatches=_response_mismatches(response, plan=plan),
        error=None,
    )


def _response_mismatches(
    response: RerankResult,
    *,
    plan: RemoteRerankerRequestSmokePlan,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    expected_values = {
        "provider_type": plan.expected_provider_type,
        "reranker_model_id": plan.provider_model_id,
        "reranker_profile_name": plan.reranker_profile_name,
        "retrieval_strategy": RERANK_RETRIEVAL_STRATEGY,
        "candidate_count": len(plan.request.candidates),
        "returned_count": min(plan.request.top_k, len(plan.request.candidates)),
        "top_k": min(plan.request.top_k, len(plan.request.candidates)),
    }
    actual_values = {
        "provider_type": response.provider_type,
        "reranker_model_id": response.reranker_model_id,
        "reranker_profile_name": response.reranker_profile_name,
        "retrieval_strategy": response.retrieval_strategy,
        "candidate_count": response.candidate_count,
        "returned_count": response.returned_count,
        "top_k": response.top_k,
    }
    for key, expected in expected_values.items():
        actual = actual_values[key]
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")

    if len(response.results) != response.returned_count:
        mismatches.append(
            f"result_count: expected {response.returned_count!r}, got {len(response.results)!r}"
        )
    ranks = tuple(item.rank for item in response.results)
    expected_ranks = tuple(range(1, len(response.results) + 1))
    if ranks != expected_ranks:
        mismatches.append(f"result ranks: expected {expected_ranks!r}, got {ranks!r}")

    known_candidate_keys = {candidate.candidate_key for candidate in plan.request.candidates}
    seen_candidate_keys: set[str] = set()
    for item in response.results:
        if item.candidate.candidate_key not in known_candidate_keys:
            mismatches.append(f"unknown candidate_key: {item.candidate.candidate_key!r}")
        if item.candidate.candidate_key in seen_candidate_keys:
            mismatches.append(f"duplicate candidate_key: {item.candidate.candidate_key!r}")
        seen_candidate_keys.add(item.candidate.candidate_key)
        if not math.isfinite(item.score):
            mismatches.append(f"score must be finite for {item.candidate.candidate_key!r}")

    mismatches.extend(_runtime_metadata_mismatches(response.runtime_metadata, plan=plan))
    return tuple(mismatches)


def _runtime_metadata_mismatches(
    runtime_metadata: dict[str, Any],
    *,
    plan: RemoteRerankerRequestSmokePlan,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    expected_values: dict[str, str] = {
        "service": "nex_pcx_reranker_provider_service",
        "backend": plan.expected_backend,
    }
    if plan.expected_device is not None:
        expected_values["device"] = plan.expected_device
    for key, expected in expected_values.items():
        actual = runtime_metadata.get(key)
        if actual != expected:
            mismatches.append(f"runtime_metadata.{key}: expected {expected!r}, got {actual!r}")
    return tuple(mismatches)


def _result_previews(response: RerankResult) -> tuple[RerankerResultPreview, ...]:
    return tuple(
        RerankerResultPreview(
            candidate_key=item.candidate.candidate_key,
            rank=item.rank,
            score=round(float(item.score), SCORE_PREVIEW_DIGITS),
            source_rank=_metadata_int(item.score_components, "source_rank"),
            score_components=dict(item.score_components),
        )
        for item in response.results
    )


def _build_candidates(
    candidate_texts: tuple[str, ...],
    *,
    source_profile_name: str,
    source_retrieval_strategy: str,
) -> tuple[RerankCandidate, ...]:
    normalized_texts = tuple(text.strip() for text in candidate_texts if text.strip())
    if not normalized_texts:
        raise ValueError("candidate_texts must not be empty")
    source_profile = _validate_nonblank(source_profile_name, "source_profile_name")
    source_strategy = _validate_nonblank(
        source_retrieval_strategy,
        "source_retrieval_strategy",
    )
    return tuple(
        RerankCandidate(
            candidate_key=f"candidate-{index}",
            rank=index,
            text=text,
            source_profile_name=source_profile,
            source_retrieval_strategy=source_strategy,
            source_score=max(0.0, 1.0 - (index * 0.01)),
            chunk_id=1000 + index,
            metadata={"smoke": "remote_reranker_request"},
        )
        for index, text in enumerate(normalized_texts, start=1)
    )


def _validate_request_shape(request: RerankRequest) -> None:
    from app.core.rerankers import validate_rerank_request

    validate_rerank_request(request)


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url is required")
    return normalized


def _load_candidate_texts(
    candidate_text_args: list[str],
    candidate_texts_file: str | None,
) -> tuple[str, ...]:
    texts = [text.strip() for text in candidate_text_args if text.strip()]
    if candidate_texts_file:
        file_texts = [
            line.strip()
            for line in Path(candidate_texts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        texts.extend(file_texts)
    return tuple(texts or DEFAULT_CANDIDATE_TEXTS)


def _report_payload(report: RemoteRerankerRequestSmokeReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "observation": asdict(report.observation),
        "total_elapsed_ms": report.total_elapsed_ms,
    }


def _plan_payload(plan: RemoteRerankerRequestSmokePlan) -> dict[str, Any]:
    return {
        **asdict(plan),
        "request": {
            "query_text": "<query_text>",
            "top_k": plan.request.top_k,
            "reranker_profile_name": plan.request.reranker_profile_name,
            "reranker_model_id": plan.request.reranker_model_id,
            "candidate_count": len(plan.request.candidates),
            "candidates": [
                {
                    "candidate_key": candidate.candidate_key,
                    "rank": candidate.rank,
                    "source_profile_name": candidate.source_profile_name,
                    "source_retrieval_strategy": candidate.source_retrieval_strategy,
                    "source_score": candidate.source_score,
                    "chunk_id": candidate.chunk_id,
                    "text": f"<candidate_text:{index}>",
                }
                for index, candidate in enumerate(plan.request.candidates, start=1)
            ],
        },
    }


def _print_human_report(report: RemoteRerankerRequestSmokeReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    observation = report.observation
    print(f"Remote reranker request smoke: {status}")
    print(f"- provider_name: {report.plan.provider_name}")
    print(f"- rerank_url: {report.plan.rerank_url}")
    print(f"- model: {report.plan.provider_model_id}")
    print(f"- profile: {report.plan.reranker_profile_name}")
    print(f"- candidate_count: {observation.candidate_count}")
    print(f"- returned_count: {observation.returned_count}")
    print(f"- request_elapsed_ms: {observation.request_elapsed_ms}")
    print(f"- provider_elapsed_ms: {observation.provider_elapsed_ms}")
    for preview in observation.result_previews:
        print(
            f"- rank {preview.rank}: "
            f"{preview.candidate_key} score={preview.score} source_rank={preview.source_rank}"
        )
    if observation.mismatches:
        print("- mismatches:")
        for mismatch in observation.mismatches:
            print(f"  - {mismatch}")
    if observation.error:
        print(f"- error: {observation.error}")


def _markdown_report(report: RemoteRerankerRequestSmokeReport) -> str:
    observation = report.observation
    lines = [
        "# Remote Reranker Request Smoke Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `provider_name`: `{report.plan.provider_name}`",
        f"- `rerank_url`: `{report.plan.rerank_url}`",
        f"- `provider_model_id`: `{report.plan.provider_model_id}`",
        f"- `reranker_profile_name`: `{report.plan.reranker_profile_name}`",
        f"- `candidate_count`: `{observation.candidate_count}`",
        f"- `returned_count`: `{observation.returned_count}`",
        f"- `request_elapsed_ms`: `{observation.request_elapsed_ms}`",
        f"- `provider_elapsed_ms`: `{observation.provider_elapsed_ms}`",
        "",
        "## Result Score Preview",
        "",
        "| Rank | Candidate | Score | Source Rank |",
        "| ---: | --- | ---: | ---: |",
    ]
    for preview in observation.result_previews:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{preview.rank}`",
                    f"`{preview.candidate_key}`",
                    f"`{preview.score}`",
                    f"`{preview.source_rank}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Runtime Metadata",
            "",
            "```json",
            json.dumps(observation.runtime_metadata, ensure_ascii=False, indent=2),
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
        description="Run a reranker request smoke check against a remote reranker provider.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_RERANKER_PROVIDER_PORT)
    parser.add_argument("--provider-name", default=DEFAULT_RERANKER_PROVIDER_NAME)
    parser.add_argument("--provider-model-id", default=DEFAULT_RERANKER_MODEL_ID)
    parser.add_argument("--profile-name", default=DEFAULT_RERANKER_PROFILE_NAME)
    parser.add_argument("--expected-backend", default=RERANKER_PROVIDER_BACKEND_QWEN)
    parser.add_argument("--expected-device", default=DEFAULT_DEVICE)
    parser.add_argument("--query-text", default=DEFAULT_QUERY_TEXT)
    parser.add_argument("--candidate-text", action="append", default=[])
    parser.add_argument("--candidate-texts-file", default=None)
    parser.add_argument("--source-profile-name", default=DEFAULT_SOURCE_PROFILE_NAME)
    parser.add_argument(
        "--source-retrieval-strategy",
        default=DEFAULT_SOURCE_RETRIEVAL_STRATEGY,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_reranker_request_smoke_plan(
            base_url=args.base_url,
            host=args.host,
            route_host=args.route_host,
            port=args.port,
            provider_name=args.provider_name,
            provider_model_id=args.provider_model_id,
            reranker_profile_name=args.profile_name,
            expected_backend=args.expected_backend,
            expected_device=args.expected_device,
            query_text=args.query_text,
            candidate_texts=_load_candidate_texts(
                args.candidate_text,
                args.candidate_texts_file,
            ),
            source_profile_name=args.source_profile_name,
            source_retrieval_strategy=args.source_retrieval_strategy,
            top_k=args.top_k,
            timeout_seconds=args.timeout_seconds,
        )
    except (InvalidRerankerError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    provider = RemoteRerankerProviderClient(
        plan.base_url,
        timeout_seconds=plan.timeout_seconds,
    )
    try:
        report = run_reranker_request_smoke(provider, plan=plan)
    finally:
        provider.close()

    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
