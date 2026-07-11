"""Process one pending embedding job through the configured embedding provider."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_providers import (  # noqa: E402
    MOCK_EMBEDDING_PROVIDER_TYPE,
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProviderRuntimeConfig,
    build_embedding_provider_from_runtime_config,
    embedding_provider_runtime_config_from_settings,
    normalize_embedding_provider_runtime_config,
)
from app.core.embedding_worker import (  # noqa: E402
    ERROR_CODE_EMBEDDING_PROVIDER_ERROR,
    ERROR_CODE_MOCK_EMBEDDING_ERROR,
    EmbeddingWorkerBatchResult,
    EmbeddingWorkerResult,
    process_embedding_worker_batch,
    process_next_embedding_job_with_provider,
    process_next_embedding_job_with_provider_routes,
)
from app.core.embedding_worker_batch_runs import (  # noqa: E402
    EmbeddingWorkerBatchRunInput,
    record_embedding_worker_batch_run,
)
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS  # noqa: E402

PROVIDER_SOURCE_ROUTE = "route"
PROVIDER_SOURCE_RUNTIME = "runtime"
PROVIDER_SOURCES = (PROVIDER_SOURCE_ROUTE, PROVIDER_SOURCE_RUNTIME)


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


def _process_next_job(
    database_url: str,
    *,
    worker_name: str,
    profile_name: str | None,
    lease_seconds: int,
    runtime_config: EmbeddingProviderRuntimeConfig,
    provider_source: str = PROVIDER_SOURCE_ROUTE,
    require_route_readiness: bool = False,
    readiness_gate_failure_mode: str = "fail",
    readiness_gate_defer_seconds: int = 300,
) -> EmbeddingWorkerResult:
    if provider_source == PROVIDER_SOURCE_ROUTE:
        return process_next_embedding_job_with_provider_routes(
            database_url,
            worker_name=worker_name,
            profile_name=profile_name,
            lease_seconds=lease_seconds,
            fallback_runtime_config=runtime_config,
            require_route_readiness=require_route_readiness,
            readiness_gate_failure_mode=readiness_gate_failure_mode,
            readiness_gate_defer_seconds=readiness_gate_defer_seconds,
        )

    provider = build_embedding_provider_from_runtime_config(runtime_config)
    try:
        return process_next_embedding_job_with_provider(
            database_url,
            worker_name=worker_name,
            provider=provider,
            profile_name=profile_name,
            lease_seconds=lease_seconds,
            success_message=_success_message(runtime_config.mode),
            provider_error_code=_provider_error_code(runtime_config.mode),
        )
    finally:
        if hasattr(provider, "close"):
            provider.close()  # type: ignore[attr-defined]


def _process_job_batch(
    database_url: str,
    *,
    worker_name: str,
    profile_name: str | None,
    lease_seconds: int,
    runtime_config: EmbeddingProviderRuntimeConfig,
    provider_source: str = PROVIDER_SOURCE_ROUTE,
    require_route_readiness: bool = False,
    readiness_gate_failure_mode: str = "fail",
    readiness_gate_defer_seconds: int = 300,
    limit: int,
) -> EmbeddingWorkerBatchResult:
    if provider_source == PROVIDER_SOURCE_RUNTIME:
        provider = build_embedding_provider_from_runtime_config(runtime_config)
        try:
            return process_embedding_worker_batch(
                lambda: process_next_embedding_job_with_provider(
                    database_url,
                    worker_name=worker_name,
                    provider=provider,
                    profile_name=profile_name,
                    lease_seconds=lease_seconds,
                    success_message=_success_message(runtime_config.mode),
                    provider_error_code=_provider_error_code(runtime_config.mode),
                ),
                limit=limit,
            )
        finally:
            if hasattr(provider, "close"):
                provider.close()  # type: ignore[attr-defined]

    return process_embedding_worker_batch(
        lambda: process_next_embedding_job_with_provider_routes(
            database_url,
            worker_name=worker_name,
            profile_name=profile_name,
            lease_seconds=lease_seconds,
            fallback_runtime_config=runtime_config,
            require_route_readiness=require_route_readiness,
            readiness_gate_failure_mode=readiness_gate_failure_mode,
            readiness_gate_defer_seconds=readiness_gate_defer_seconds,
        ),
        limit=limit,
    )


def _success_message(provider_mode: str) -> str:
    if provider_mode == MOCK_EMBEDDING_PROVIDER_TYPE:
        return "Mock embedding stored"
    return "Remote embedding stored"


def _provider_error_code(provider_mode: str) -> str:
    if provider_mode == MOCK_EMBEDDING_PROVIDER_TYPE:
        return ERROR_CODE_MOCK_EMBEDDING_ERROR
    return ERROR_CODE_EMBEDDING_PROVIDER_ERROR


def _result_payload(
    result: EmbeddingWorkerResult,
    *,
    runtime_config: EmbeddingProviderRuntimeConfig,
    provider_source: str,
    require_route_readiness: bool,
    readiness_gate_failure_mode: str,
    readiness_gate_defer_seconds: int,
) -> dict[str, object]:
    runtime_metadata = result.job.runtime_metadata if result.job else {}
    return {
        "provider_source": runtime_metadata.get("provider_runtime_source", provider_source),
        "provider_mode": runtime_metadata.get("provider_runtime_mode", runtime_config.mode),
        "remote_provider_url": runtime_metadata.get(
            "provider_runtime_base_url",
            runtime_config.remote_base_url,
        ),
        "provider_route_id": runtime_metadata.get("provider_route_id"),
        "provider_route_name": runtime_metadata.get("provider_route_name"),
        "require_route_readiness": require_route_readiness,
        "readiness_gate_failure_mode": readiness_gate_failure_mode,
        "readiness_gate_defer_seconds": readiness_gate_defer_seconds,
        "processed": result.processed,
        "job_id": result.job.job_id if result.job else None,
        "chunk_id": result.job.chunk_id if result.job else None,
        "profile_name": result.job.profile_name if result.job else None,
        "status": result.job.status if result.job else None,
        "elapsed_ms": result.elapsed_ms,
        "table_name": result.vector.table_name if result.vector else None,
        "message": result.message,
    }


def _batch_payload(
    batch: EmbeddingWorkerBatchResult,
    *,
    runtime_config: EmbeddingProviderRuntimeConfig,
    provider_source: str,
    require_route_readiness: bool,
    readiness_gate_failure_mode: str,
    readiness_gate_defer_seconds: int,
    batch_run_id: int | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    elapsed_ms: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider_source": provider_source,
        "provider_mode": runtime_config.mode,
        "remote_provider_url": runtime_config.remote_base_url,
        "require_route_readiness": require_route_readiness,
        "readiness_gate_failure_mode": readiness_gate_failure_mode,
        "readiness_gate_defer_seconds": readiness_gate_defer_seconds,
        "limit": batch.limit,
        "result_count": batch.result_count,
        "processed_count": batch.processed_count,
        "succeeded_count": batch.succeeded_count,
        "failed_count": batch.failed_count,
        "deferred_count": batch.deferred_count,
        "idle_count": batch.idle_count,
        "stopped_reason": batch.stopped_reason,
        "job_ids": list(batch.job_ids),
        "results": [
            _result_payload(
                result,
                runtime_config=runtime_config,
                provider_source=provider_source,
                require_route_readiness=require_route_readiness,
                readiness_gate_failure_mode=readiness_gate_failure_mode,
                readiness_gate_defer_seconds=readiness_gate_defer_seconds,
            )
            for result in batch.results
        ],
    }
    if batch_run_id is not None:
        payload["batch_run_id"] = batch_run_id
    if started_at is not None:
        payload["started_at"] = started_at.isoformat()
    if completed_at is not None:
        payload["completed_at"] = completed_at.isoformat()
    if elapsed_ms is not None:
        payload["elapsed_ms"] = elapsed_ms
    return payload


def _record_batch_run(
    database_url: str,
    batch: EmbeddingWorkerBatchResult,
    *,
    worker_name: str,
    profile_name: str | None,
    runtime_config: EmbeddingProviderRuntimeConfig,
    provider_source: str,
    require_route_readiness: bool,
    readiness_gate_failure_mode: str,
    readiness_gate_defer_seconds: int,
    started_at: datetime,
    completed_at: datetime,
    elapsed_ms: int,
    payload: dict[str, object],
) -> int:
    run_record = record_embedding_worker_batch_run(
        database_url,
        EmbeddingWorkerBatchRunInput(
            worker_name=worker_name,
            profile_name=profile_name,
            provider_source=provider_source,
            provider_mode=runtime_config.mode,
            remote_provider_url=runtime_config.remote_base_url,
            require_route_readiness=require_route_readiness,
            readiness_gate_failure_mode=readiness_gate_failure_mode,
            readiness_gate_defer_seconds=readiness_gate_defer_seconds,
            limit_requested=batch.limit,
            result_count=batch.result_count,
            processed_count=batch.processed_count,
            succeeded_count=batch.succeeded_count,
            failed_count=batch.failed_count,
            deferred_count=batch.deferred_count,
            idle_count=batch.idle_count,
            stopped_reason=batch.stopped_reason,
            job_ids=batch.job_ids,
            runtime_metadata={
                "script": "process_embedding_job.py",
                "provider": {
                    "source": provider_source,
                    "mode": runtime_config.mode,
                    "remote_provider_url": runtime_config.remote_base_url,
                },
                "results": payload["results"],
            },
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            completed_at=completed_at,
        ),
    )
    return run_record.batch_run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Process pending embedding jobs.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--worker-name", default="embedding-worker")
    parser.add_argument("--profile-name", default=None)
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum pending embedding jobs to process before exiting.",
    )
    parser.add_argument(
        "--provider-source",
        choices=PROVIDER_SOURCES,
        default=PROVIDER_SOURCE_ROUTE,
        help=(
            "Provider source. 'route' selects an active profile route from the database "
            "and falls back to runtime config; 'runtime' ignores route records."
        ),
    )
    readiness_group = parser.add_mutually_exclusive_group()
    readiness_group.add_argument(
        "--require-route-readiness",
        action="store_true",
        default=None,
        help="Require selected provider routes to pass the readiness gate before use.",
    )
    readiness_group.add_argument(
        "--skip-route-readiness",
        action="store_false",
        dest="require_route_readiness",
        help="Disable the provider route readiness gate for this run.",
    )
    parser.set_defaults(require_route_readiness=None)
    parser.add_argument(
        "--provider-mode",
        choices=(MOCK_EMBEDDING_PROVIDER_TYPE, REMOTE_EMBEDDING_PROVIDER_TYPE),
        default=None,
        help="Provider mode override. Defaults to NEX_PCX_EMBEDDING_PROVIDER_MODE.",
    )
    parser.add_argument(
        "--remote-provider-url",
        default=None,
        help="Remote provider base URL override.",
    )
    parser.add_argument(
        "--remote-provider-timeout-seconds",
        type=float,
        default=None,
        help="Remote provider request timeout override.",
    )
    parser.add_argument(
        "--readiness-gate-failure-mode",
        choices=("fail", "defer"),
        default=None,
        help="How route-aware workers handle jobs when no route passes readiness.",
    )
    parser.add_argument(
        "--readiness-gate-defer-seconds",
        type=int,
        default=None,
        help="Seconds to defer a claimed job when readiness-gate failure mode is defer.",
    )
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    runtime_config = _runtime_config_from_args(args, settings)
    require_route_readiness = (
        args.require_route_readiness
        if args.require_route_readiness is not None
        else settings.embedding_require_route_readiness
    )
    readiness_gate_failure_mode = (
        args.readiness_gate_failure_mode or settings.embedding_route_readiness_failure_mode
    )
    readiness_gate_defer_seconds = (
        args.readiness_gate_defer_seconds
        if args.readiness_gate_defer_seconds is not None
        else settings.embedding_route_readiness_defer_seconds
    )
    try:
        if args.limit == 1:
            result = _process_next_job(
                database_url,
                worker_name=args.worker_name,
                profile_name=args.profile_name,
                lease_seconds=args.lease_seconds,
                runtime_config=runtime_config,
                provider_source=args.provider_source,
                require_route_readiness=require_route_readiness,
                readiness_gate_failure_mode=readiness_gate_failure_mode,
                readiness_gate_defer_seconds=readiness_gate_defer_seconds,
            )
            payload = _result_payload(
                result,
                runtime_config=runtime_config,
                provider_source=args.provider_source,
                require_route_readiness=require_route_readiness,
                readiness_gate_failure_mode=readiness_gate_failure_mode,
                readiness_gate_defer_seconds=readiness_gate_defer_seconds,
            )
        else:
            started_at = datetime.now(UTC)
            started_timer = perf_counter()
            batch = _process_job_batch(
                database_url,
                worker_name=args.worker_name,
                profile_name=args.profile_name,
                lease_seconds=args.lease_seconds,
                runtime_config=runtime_config,
                provider_source=args.provider_source,
                require_route_readiness=require_route_readiness,
                readiness_gate_failure_mode=readiness_gate_failure_mode,
                readiness_gate_defer_seconds=readiness_gate_defer_seconds,
                limit=args.limit,
            )
            completed_at = datetime.now(UTC)
            elapsed_ms = int((perf_counter() - started_timer) * 1000)
            payload = _batch_payload(
                batch,
                runtime_config=runtime_config,
                provider_source=args.provider_source,
                require_route_readiness=require_route_readiness,
                readiness_gate_failure_mode=readiness_gate_failure_mode,
                readiness_gate_defer_seconds=readiness_gate_defer_seconds,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_ms=elapsed_ms,
            )
            batch_run_id = _record_batch_run(
                database_url,
                batch,
                worker_name=args.worker_name,
                profile_name=args.profile_name,
                runtime_config=runtime_config,
                provider_source=args.provider_source,
                require_route_readiness=require_route_readiness,
                readiness_gate_failure_mode=readiness_gate_failure_mode,
                readiness_gate_defer_seconds=readiness_gate_defer_seconds,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_ms=elapsed_ms,
                payload=payload,
            )
            payload["batch_run_id"] = batch_run_id
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
