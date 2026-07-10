"""Smoke check local SentenceTransformers embedding model bundles."""

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
from app.core.embedding_adapters import (  # noqa: E402
    SENTENCE_TRANSFORMERS_ADAPTER_NAME,
    EmbeddingModelProfile,
    InvalidEmbeddingAdapterError,
    SentenceTransformersEmbeddingAdapter,
)
from app.core.embedding_model_distribution import (  # noqa: E402
    EmbeddingModelDistribution,
    audit_single_embedding_model_readiness,
    get_embedding_model_distribution,
    list_embedding_model_distributions,
    resolve_embedding_model_dir,
)
from app.core.embedding_vectors import (  # noqa: E402
    InvalidEmbeddingVectorError,
    get_embedding_vector_table,
)

DEFAULT_SMOKE_TEXTS = ("NeX_PCX local embedding model smoke test.",)
SUPPORTED_SMOKE_ADAPTERS = frozenset({SENTENCE_TRANSFORMERS_ADAPTER_NAME})


@dataclass(frozen=True)
class EmbeddingModelSmokePlan:
    model_key: str
    repo_id: str
    adapter_name: str
    profile_name: str
    dimension: int
    local_dir: str
    ready: bool
    device: str


@dataclass(frozen=True)
class EmbeddingModelSmokeResult:
    model_key: str
    repo_id: str
    adapter_name: str
    profile_name: str
    dimension: int
    local_dir: str
    ready: bool
    status: str
    ok: bool
    device: str
    input_count: int
    elapsed_ms: int | None
    message: str


def _select_distributions(model_keys: list[str]) -> tuple[EmbeddingModelDistribution, ...]:
    if model_keys:
        return tuple(get_embedding_model_distribution(model_key) for model_key in model_keys)
    return tuple(
        distribution
        for distribution in list_embedding_model_distributions()
        if distribution.adapter_name in SUPPORTED_SMOKE_ADAPTERS
    )


def _build_smoke_plan(
    distribution: EmbeddingModelDistribution,
    *,
    models_dir: Path,
    device: str,
) -> EmbeddingModelSmokePlan:
    profile_name = distribution.profile_names[0]
    table = get_embedding_vector_table(profile_name)
    readiness = audit_single_embedding_model_readiness(distribution, models_dir)
    return EmbeddingModelSmokePlan(
        model_key=distribution.model_key,
        repo_id=distribution.repo_id,
        adapter_name=distribution.adapter_name,
        profile_name=profile_name,
        dimension=table.dimension,
        local_dir=str(resolve_embedding_model_dir(distribution, models_dir)),
        ready=readiness.ready,
        device=device,
    )


def run_embedding_model_smoke(
    distribution: EmbeddingModelDistribution,
    *,
    models_dir: Path,
    device: str,
    texts: tuple[str, ...],
) -> EmbeddingModelSmokeResult:
    plan = _build_smoke_plan(distribution, models_dir=models_dir, device=device)
    if distribution.adapter_name not in SUPPORTED_SMOKE_ADAPTERS:
        return _result_from_plan(
            plan,
            status="unsupported_adapter",
            ok=False,
            input_count=len(texts),
            elapsed_ms=None,
            message=f"Unsupported local smoke adapter: {distribution.adapter_name}",
        )
    if not plan.ready:
        return _result_from_plan(
            plan,
            status="not_ready",
            ok=False,
            input_count=len(texts),
            elapsed_ms=None,
            message="Local model directory is missing config or model weight files",
        )

    started_at = perf_counter()
    try:
        adapter = SentenceTransformersEmbeddingAdapter(
            EmbeddingModelProfile(
                profile_name=plan.profile_name,
                model_name=distribution.repo_id,
                dimension=plan.dimension,
                storage_type=get_embedding_vector_table(plan.profile_name).storage_type,
                adapter_name=SENTENCE_TRANSFORMERS_ADAPTER_NAME,
                local_model_path=plan.local_dir,
                device=device,
            )
        )
        embeddings = adapter.embed_documents(texts)
    except (InvalidEmbeddingAdapterError, InvalidEmbeddingVectorError, RuntimeError) as exc:
        return _result_from_plan(
            plan,
            status="failed",
            ok=False,
            input_count=len(texts),
            elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            message=str(exc),
        )

    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    return _result_from_plan(
        plan,
        status="passed",
        ok=True,
        input_count=len(embeddings),
        elapsed_ms=elapsed_ms,
        message="Local model loaded and produced embeddings",
    )


def _result_from_plan(
    plan: EmbeddingModelSmokePlan,
    *,
    status: str,
    ok: bool,
    input_count: int,
    elapsed_ms: int | None,
    message: str,
) -> EmbeddingModelSmokeResult:
    return EmbeddingModelSmokeResult(
        model_key=plan.model_key,
        repo_id=plan.repo_id,
        adapter_name=plan.adapter_name,
        profile_name=plan.profile_name,
        dimension=plan.dimension,
        local_dir=plan.local_dir,
        ready=plan.ready,
        status=status,
        ok=ok,
        device=plan.device,
        input_count=input_count,
        elapsed_ms=elapsed_ms,
        message=message,
    )


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _print_human_results(results: tuple[EmbeddingModelSmokeResult, ...]) -> None:
    for result in results:
        elapsed = "n/a" if result.elapsed_ms is None else f"{result.elapsed_ms}ms"
        print(
            f"- {result.model_key}: {result.status} "
            f"profile={result.profile_name} dimension={result.dimension} "
            f"device={result.device} elapsed={elapsed} dir={result.local_dir}"
        )
        if result.message:
            print(f"  {result.message}")


def _print_human_plan(plans: tuple[EmbeddingModelSmokePlan, ...]) -> None:
    for plan in plans:
        print(
            f"- {plan.model_key}: profile={plan.profile_name} "
            f"dimension={plan.dimension} ready={plan.ready} "
            f"device={plan.device} dir={plan.local_dir}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke check local SentenceTransformers embedding models."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model key to smoke check. Repeat to check multiple keys. "
            "Defaults to SentenceTransformers-backed models."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Local model bundle root. Defaults to NEX_PCX_MODELS_DIR or models.",
    )
    parser.add_argument("--device", default="cpu", help="SentenceTransformer device.")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text to embed. Repeat to send multiple texts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected model smoke plan without loading models.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    settings = get_settings()
    models_dir = Path(args.models_dir) if args.models_dir else settings.embedding_models_dir
    distributions = _select_distributions(args.model)
    texts = tuple(args.text) or DEFAULT_SMOKE_TEXTS

    if args.dry_run:
        plans = tuple(
            _build_smoke_plan(distribution, models_dir=models_dir, device=args.device)
            for distribution in distributions
        )
        if args.json:
            _print_json(
                {
                    "models_dir": str(models_dir),
                    "device": args.device,
                    "dry_run": True,
                    "models": [asdict(plan) for plan in plans],
                }
            )
            return 0
        _print_human_plan(plans)
        return 0

    results = tuple(
        run_embedding_model_smoke(
            distribution,
            models_dir=models_dir,
            device=args.device,
            texts=texts,
        )
        for distribution in distributions
    )
    if args.json:
        _print_json(
            {
                "models_dir": str(models_dir),
                "device": args.device,
                "dry_run": False,
                "results": [asdict(result) for result in results],
            }
        )
    else:
        _print_human_results(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
