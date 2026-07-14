"""Run embedding request smoke checks against a remote embedding provider."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.embedding_provider_presets import (  # noqa: E402
    EmbeddingProviderPreset,
    InvalidEmbeddingProviderPresetError,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from app.core.embedding_providers import (  # noqa: E402
    REMOTE_EMBEDDING_PROVIDER_EMBEDDINGS_PATH,
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProvider,
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    InvalidEmbeddingProviderError,
    RemoteEmbeddingProviderClient,
)
from app.core.embedding_vectors import get_embedding_vector_table  # noqa: E402
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_GPU_HOST,
    build_foreground_smoke_plan,
)

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_SMOKE_TEXT = "NeX_PCX remote embedding request smoke text."
EMBEDDING_PREVIEW_VALUES = 3


@dataclass(frozen=True)
class RemoteProviderEmbeddingSmokeCase:
    provider: str
    profile_name: str
    model_key: str
    output_dimension: int
    input_type: str
    text_count: int
    trace_id: str


@dataclass(frozen=True)
class RemoteProviderEmbeddingSmokePlan:
    provider: str
    provider_name: str
    base_url: str
    embeddings_url: str
    provider_model_id: str
    timeout_seconds: float
    cases: tuple[RemoteProviderEmbeddingSmokeCase, ...]
    texts: tuple[str, ...]


@dataclass(frozen=True)
class RemoteProviderEmbeddingSmokeObservation:
    case: RemoteProviderEmbeddingSmokeCase
    request_elapsed_ms: int
    provider_elapsed_ms: int | None
    provider_model_id: str | None
    provider_type: str | None
    dimension: int | None
    input_count: int | None
    embedding_count: int | None
    embedding_preview: tuple[float, ...]
    runtime_metadata: dict[str, Any]
    mismatches: tuple[str, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.mismatches


@dataclass(frozen=True)
class RemoteProviderEmbeddingSmokeReport:
    plan: RemoteProviderEmbeddingSmokePlan
    observations: tuple[RemoteProviderEmbeddingSmokeObservation, ...]
    total_elapsed_ms: int

    @property
    def passed(self) -> bool:
        return bool(self.observations) and all(
            observation.passed for observation in self.observations
        )


def build_embedding_smoke_plan(
    preset: EmbeddingProviderPreset,
    *,
    base_url: str | None = None,
    host: str = DEFAULT_GPU_HOST,
    route_host: str | None = None,
    port: int | None = None,
    profile_names: tuple[str, ...] | None = None,
    input_type: str = "document",
    texts: tuple[str, ...] = (DEFAULT_SMOKE_TEXT,),
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> RemoteProviderEmbeddingSmokePlan:
    selected_base_url = _normalize_base_url(
        base_url
        or build_foreground_smoke_plan(
            preset,
            host=host,
            route_host=route_host,
            port=port,
        ).base_url
    )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    selected_texts = tuple(text.strip() for text in texts if text.strip())
    if not selected_texts:
        raise ValueError("texts must not be empty")
    selected_input_type = input_type.strip().lower()
    if selected_input_type not in {"query", "document"}:
        raise ValueError("input_type must be query or document")

    selected_profiles = profile_names or preset.profile_names
    _validate_profiles(preset, selected_profiles)
    cases = tuple(
        RemoteProviderEmbeddingSmokeCase(
            provider=preset.preset_name,
            profile_name=profile_name,
            model_key=preset.model_key,
            output_dimension=get_embedding_vector_table(profile_name).dimension,
            input_type=selected_input_type,
            text_count=len(selected_texts),
            trace_id=f"remote-embedding-smoke-{preset.preset_name}-{profile_name}",
        )
        for profile_name in selected_profiles
    )
    return RemoteProviderEmbeddingSmokePlan(
        provider=preset.preset_name,
        provider_name=preset.provider_name,
        base_url=selected_base_url,
        embeddings_url=f"{selected_base_url}{REMOTE_EMBEDDING_PROVIDER_EMBEDDINGS_PATH}",
        provider_model_id=preset.provider_model_id,
        timeout_seconds=timeout_seconds,
        cases=cases,
        texts=selected_texts,
    )


def run_embedding_request_smoke(
    provider: EmbeddingProvider,
    *,
    plan: RemoteProviderEmbeddingSmokePlan,
) -> RemoteProviderEmbeddingSmokeReport:
    started_at = perf_counter()
    observations = tuple(_run_case(provider, plan=plan, case=case) for case in plan.cases)
    return RemoteProviderEmbeddingSmokeReport(
        plan=plan,
        observations=observations,
        total_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
    )


def _run_case(
    provider: EmbeddingProvider,
    *,
    plan: RemoteProviderEmbeddingSmokePlan,
    case: RemoteProviderEmbeddingSmokeCase,
) -> RemoteProviderEmbeddingSmokeObservation:
    request = EmbeddingProviderRequest(
        profile_name=case.profile_name,
        model_key=case.model_key,
        input_type=case.input_type,
        texts=plan.texts,
        output_dimension=case.output_dimension,
        normalize_embeddings=True,
        trace_id=case.trace_id,
        runtime_metadata={"smoke": "remote_embedding_request"},
    )
    started_at = perf_counter()
    try:
        response = provider.embed(request)
    except (InvalidEmbeddingProviderError, ValueError, OSError) as exc:
        return RemoteProviderEmbeddingSmokeObservation(
            case=case,
            request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            provider_elapsed_ms=None,
            provider_model_id=None,
            provider_type=None,
            dimension=None,
            input_count=None,
            embedding_count=None,
            embedding_preview=(),
            runtime_metadata={},
            mismatches=(),
            error=str(exc),
        )

    return RemoteProviderEmbeddingSmokeObservation(
        case=case,
        request_elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
        provider_elapsed_ms=response.elapsed_ms,
        provider_model_id=response.provider_model_id,
        provider_type=response.provider_type,
        dimension=response.dimension,
        input_count=response.input_count,
        embedding_count=len(response.embeddings),
        embedding_preview=_embedding_preview(response),
        runtime_metadata=dict(response.runtime_metadata),
        mismatches=_response_mismatches(response, plan=plan, case=case),
        error=None,
    )


def _response_mismatches(
    response: EmbeddingProviderResponse,
    *,
    plan: RemoteProviderEmbeddingSmokePlan,
    case: RemoteProviderEmbeddingSmokeCase,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    expected_values = {
        "provider_model_id": plan.provider_model_id,
        "provider_type": REMOTE_EMBEDDING_PROVIDER_TYPE,
        "dimension": case.output_dimension,
        "input_count": case.text_count,
    }
    actual_values = {
        "provider_model_id": response.provider_model_id,
        "provider_type": response.provider_type,
        "dimension": response.dimension,
        "input_count": response.input_count,
    }
    for key, expected in expected_values.items():
        actual = actual_values[key]
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if len(response.embeddings) != case.text_count:
        mismatches.append(
            f"embedding_count: expected {case.text_count!r}, got {len(response.embeddings)!r}"
        )
    return tuple(mismatches)


def _embedding_preview(response: EmbeddingProviderResponse) -> tuple[float, ...]:
    if not response.embeddings:
        return ()
    return tuple(float(value) for value in response.embeddings[0][:EMBEDDING_PREVIEW_VALUES])


def _validate_profiles(
    preset: EmbeddingProviderPreset,
    profile_names: tuple[str, ...],
) -> None:
    if not profile_names:
        raise ValueError("profile_names must not be empty")
    supported_profiles = set(preset.profile_names)
    unsupported_profiles = sorted(set(profile_names) - supported_profiles)
    if unsupported_profiles:
        raise ValueError(
            f"Unsupported profile_name for {preset.preset_name}: {', '.join(unsupported_profiles)}"
        )


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url is required")
    return normalized


def _load_texts(text_args: list[str], texts_file: str | None) -> tuple[str, ...]:
    texts = [text.strip() for text in text_args if text.strip()]
    if texts_file:
        file_texts = [
            line.strip()
            for line in Path(texts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        texts.extend(file_texts)
    return tuple(texts or (DEFAULT_SMOKE_TEXT,))


def _report_payload(report: RemoteProviderEmbeddingSmokeReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "observations": [asdict(observation) for observation in report.observations],
        "total_elapsed_ms": report.total_elapsed_ms,
    }


def _plan_payload(plan: RemoteProviderEmbeddingSmokePlan) -> dict[str, Any]:
    return {
        **asdict(plan),
        "texts": [f"<text:{index + 1}>" for index in range(len(plan.texts))],
    }


def _print_human_report(report: RemoteProviderEmbeddingSmokeReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"Remote provider embedding request smoke: {status}")
    print(f"- provider: {report.plan.provider}")
    print(f"- embeddings_url: {report.plan.embeddings_url}")
    print(f"- cases: {len(report.plan.cases)}")
    print(f"- total_elapsed_ms: {report.total_elapsed_ms}")
    for observation in report.observations:
        print(
            f"- {observation.case.profile_name}: "
            f"passed={observation.passed} "
            f"dimension={observation.dimension} "
            f"input_count={observation.input_count} "
            f"provider_model_id={observation.provider_model_id}"
        )
        if observation.mismatches:
            for mismatch in observation.mismatches:
                print(f"  - mismatch: {mismatch}")
        if observation.error:
            print(f"  - error: {observation.error}")


def _build_arg_parser() -> argparse.ArgumentParser:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description="Run embedding request smoke checks against a remote embedding provider.",
    )
    parser.add_argument("--provider", choices=preset_names, default="kure")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--profile-name", action="append", default=[])
    parser.add_argument("--input-type", choices=("query", "document"), default="document")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--texts-file", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        preset = get_embedding_provider_preset(args.provider)
        plan = build_embedding_smoke_plan(
            preset,
            base_url=args.base_url,
            host=args.host,
            route_host=args.route_host,
            port=args.port,
            profile_names=tuple(args.profile_name) or None,
            input_type=args.input_type,
            texts=_load_texts(args.text, args.texts_file),
            timeout_seconds=args.timeout_seconds,
        )
    except (InvalidEmbeddingProviderPresetError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    provider = RemoteEmbeddingProviderClient(
        plan.base_url,
        timeout_seconds=plan.timeout_seconds,
    )
    try:
        report = run_embedding_request_smoke(provider, plan=plan)
    finally:
        provider.close()

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
