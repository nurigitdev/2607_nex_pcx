"""Benchmark embedding provider throughput with the standard provider contract."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_model_distribution import (  # noqa: E402
    get_embedding_model_distribution_for_profile,
)
from app.core.embedding_providers import (  # noqa: E402
    MOCK_EMBEDDING_PROVIDER_TYPE,
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProvider,
    EmbeddingProviderRequest,
    EmbeddingProviderRuntimeConfig,
    build_embedding_provider_from_runtime_config,
    embedding_provider_runtime_config_from_settings,
    normalize_embedding_provider_runtime_config,
)
from app.core.embedding_vectors import get_embedding_vector_table  # noqa: E402

DEFAULT_BENCHMARK_TEXT = "NeX_PCX embedding provider throughput benchmark text."


@dataclass(frozen=True)
class EmbeddingProviderBenchmarkPlan:
    provider_mode: str
    remote_provider_url: str | None
    profile_name: str
    model_key: str
    output_dimension: int
    input_type: str
    batch_size: int
    iterations: int
    warmup_iterations: int
    text_count: int


@dataclass(frozen=True)
class EmbeddingProviderBenchmarkIteration:
    iteration: int
    input_count: int
    wall_elapsed_ms: int
    provider_elapsed_ms: int
    provider_model_id: str
    provider_type: str


@dataclass(frozen=True)
class EmbeddingProviderBenchmarkReport:
    plan: EmbeddingProviderBenchmarkPlan
    iterations: tuple[EmbeddingProviderBenchmarkIteration, ...]
    total_input_count: int
    total_wall_elapsed_ms: int
    avg_batch_wall_ms: float
    p50_batch_wall_ms: int
    p95_batch_wall_ms: int
    texts_per_second: float
    provider_model_id: str | None
    provider_type: str | None


def _runtime_config_from_args(
    args: argparse.Namespace,
    settings: object,
) -> EmbeddingProviderRuntimeConfig:
    settings_config = embedding_provider_runtime_config_from_settings(settings)
    return normalize_embedding_provider_runtime_config(
        EmbeddingProviderRuntimeConfig(
            mode=args.provider_mode or settings_config.mode,
            remote_base_url=args.remote_provider_url or settings_config.remote_base_url,
            remote_timeout_seconds=(
                args.remote_provider_timeout_seconds
                if args.remote_provider_timeout_seconds is not None
                else settings_config.remote_timeout_seconds
            ),
        )
    )


def _build_plan(
    args: argparse.Namespace,
    runtime_config: EmbeddingProviderRuntimeConfig,
    texts: tuple[str, ...],
) -> EmbeddingProviderBenchmarkPlan:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if args.iterations <= 0:
        raise ValueError("iterations must be greater than 0")
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be greater than or equal to 0")
    if not texts:
        raise ValueError("texts must not be empty")

    model_key = args.model_key
    if model_key is None:
        model_key = get_embedding_model_distribution_for_profile(args.profile_name).model_key
    output_dimension = args.output_dimension
    if output_dimension is None:
        output_dimension = get_embedding_vector_table(args.profile_name).dimension

    return EmbeddingProviderBenchmarkPlan(
        provider_mode=runtime_config.mode,
        remote_provider_url=runtime_config.remote_base_url,
        profile_name=args.profile_name,
        model_key=model_key,
        output_dimension=output_dimension,
        input_type=args.input_type,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        text_count=len(texts),
    )


def _load_texts(text_args: list[str], texts_file: str | None) -> tuple[str, ...]:
    texts = [text.strip() for text in text_args if text.strip()]
    if texts_file:
        file_texts = [
            line.strip()
            for line in Path(texts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        texts.extend(file_texts)
    return tuple(texts or (DEFAULT_BENCHMARK_TEXT,))


def run_embedding_provider_benchmark(
    provider: EmbeddingProvider,
    *,
    plan: EmbeddingProviderBenchmarkPlan,
    texts: tuple[str, ...],
) -> EmbeddingProviderBenchmarkReport:
    for warmup_index in range(plan.warmup_iterations):
        provider.embed(
            _request_for_iteration(
                plan,
                texts,
                iteration=warmup_index + 1,
                trace_prefix="embedding-benchmark-warmup",
            )
        )

    total_started_at = perf_counter()
    iterations = []
    for iteration_index in range(plan.iterations):
        request = _request_for_iteration(
            plan,
            texts,
            iteration=iteration_index + 1,
            trace_prefix="embedding-benchmark",
        )
        started_at = perf_counter()
        response = provider.embed(request)
        wall_elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
        iterations.append(
            EmbeddingProviderBenchmarkIteration(
                iteration=iteration_index + 1,
                input_count=response.input_count,
                wall_elapsed_ms=wall_elapsed_ms,
                provider_elapsed_ms=response.elapsed_ms,
                provider_model_id=response.provider_model_id,
                provider_type=response.provider_type,
            )
        )

    total_wall_elapsed_ms = max(0, int((perf_counter() - total_started_at) * 1000))
    return _report_from_iterations(plan, tuple(iterations), total_wall_elapsed_ms)


def _request_for_iteration(
    plan: EmbeddingProviderBenchmarkPlan,
    texts: tuple[str, ...],
    *,
    iteration: int,
    trace_prefix: str,
) -> EmbeddingProviderRequest:
    offset = (iteration - 1) * plan.batch_size
    batch = tuple(texts[(offset + index) % len(texts)] for index in range(plan.batch_size))
    return EmbeddingProviderRequest(
        profile_name=plan.profile_name,
        model_key=plan.model_key,
        input_type=plan.input_type,
        texts=batch,
        output_dimension=plan.output_dimension,
        trace_id=f"{trace_prefix}-{iteration}",
        runtime_metadata={
            "benchmark": "embedding_provider_throughput",
            "iteration": iteration,
        },
    )


def _report_from_iterations(
    plan: EmbeddingProviderBenchmarkPlan,
    iterations: tuple[EmbeddingProviderBenchmarkIteration, ...],
    total_wall_elapsed_ms: int,
) -> EmbeddingProviderBenchmarkReport:
    wall_elapsed_values = tuple(iteration.wall_elapsed_ms for iteration in iterations)
    total_input_count = sum(iteration.input_count for iteration in iterations)
    total_seconds = max(total_wall_elapsed_ms / 1000, 0.001)
    provider_model_id = iterations[-1].provider_model_id if iterations else None
    provider_type = iterations[-1].provider_type if iterations else None
    return EmbeddingProviderBenchmarkReport(
        plan=plan,
        iterations=iterations,
        total_input_count=total_input_count,
        total_wall_elapsed_ms=total_wall_elapsed_ms,
        avg_batch_wall_ms=(
            sum(wall_elapsed_values) / len(wall_elapsed_values) if wall_elapsed_values else 0.0
        ),
        p50_batch_wall_ms=_nearest_rank_percentile(wall_elapsed_values, 50),
        p95_batch_wall_ms=_nearest_rank_percentile(wall_elapsed_values, 95),
        texts_per_second=total_input_count / total_seconds,
        provider_model_id=provider_model_id,
        provider_type=provider_type,
    )


def _nearest_rank_percentile(values: tuple[int, ...], percentile: int) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    rank = max(1, round((percentile / 100) * len(sorted_values)))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def _report_payload(report: EmbeddingProviderBenchmarkReport) -> dict[str, object]:
    return {
        "plan": asdict(report.plan),
        "iterations": [asdict(iteration) for iteration in report.iterations],
        "total_input_count": report.total_input_count,
        "total_wall_elapsed_ms": report.total_wall_elapsed_ms,
        "avg_batch_wall_ms": round(report.avg_batch_wall_ms, 2),
        "p50_batch_wall_ms": report.p50_batch_wall_ms,
        "p95_batch_wall_ms": report.p95_batch_wall_ms,
        "texts_per_second": round(report.texts_per_second, 2),
        "provider_model_id": report.provider_model_id,
        "provider_type": report.provider_type,
    }


def _print_human_report(report: EmbeddingProviderBenchmarkReport) -> None:
    print(
        f"provider_mode={report.plan.provider_mode} profile={report.plan.profile_name} "
        f"model_key={report.plan.model_key} dimension={report.plan.output_dimension}"
    )
    print(
        f"batch_size={report.plan.batch_size} iterations={report.plan.iterations} "
        f"warmup_iterations={report.plan.warmup_iterations} "
        f"total_texts={report.total_input_count}"
    )
    print(
        f"wall_elapsed_ms={report.total_wall_elapsed_ms} "
        f"avg_batch_wall_ms={report.avg_batch_wall_ms:.2f} "
        f"p50_batch_wall_ms={report.p50_batch_wall_ms} "
        f"p95_batch_wall_ms={report.p95_batch_wall_ms}"
    )
    print(
        f"texts_per_second={report.texts_per_second:.2f} "
        f"provider_model_id={report.provider_model_id} "
        f"provider_type={report.provider_type}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark embedding provider throughput.")
    parser.add_argument(
        "--provider-mode",
        choices=(MOCK_EMBEDDING_PROVIDER_TYPE, REMOTE_EMBEDDING_PROVIDER_TYPE),
        default=None,
        help="Provider mode override. Defaults to NEX_PCX_EMBEDDING_PROVIDER_MODE.",
    )
    parser.add_argument("--remote-provider-url", default=None, help="Remote provider base URL.")
    parser.add_argument(
        "--remote-provider-timeout-seconds",
        type=float,
        default=None,
        help="Remote provider request timeout.",
    )
    parser.add_argument("--profile-name", default="kure_v1_1024")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--output-dimension", type=int, default=None)
    parser.add_argument(
        "--input-type",
        choices=("query", "document"),
        default="document",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text to embed. Repeat to provide a pool of benchmark texts.",
    )
    parser.add_argument(
        "--texts-file",
        default=None,
        help="UTF-8 text file with one text per line.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the benchmark plan only.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    settings = get_settings()
    runtime_config = _runtime_config_from_args(args, settings)
    texts = _load_texts(args.text, args.texts_file)
    plan = _build_plan(args, runtime_config, texts)

    if args.dry_run:
        payload = {"dry_run": True, "plan": asdict(plan)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    provider = build_embedding_provider_from_runtime_config(runtime_config)
    try:
        report = run_embedding_provider_benchmark(provider, plan=plan, texts=texts)
    finally:
        if hasattr(provider, "close"):
            provider.close()  # type: ignore[attr-defined]

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
