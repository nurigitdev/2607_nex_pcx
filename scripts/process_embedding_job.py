"""Process one pending embedding job through the configured embedding provider."""

import argparse
import json
import sys
from pathlib import Path

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
    EmbeddingWorkerResult,
    process_next_embedding_job_with_provider,
    process_next_embedding_job_with_provider_routes,
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
) -> EmbeddingWorkerResult:
    if provider_source == PROVIDER_SOURCE_ROUTE:
        return process_next_embedding_job_with_provider_routes(
            database_url,
            worker_name=worker_name,
            profile_name=profile_name,
            lease_seconds=lease_seconds,
            fallback_runtime_config=runtime_config,
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
        "processed": result.processed,
        "job_id": result.job.job_id if result.job else None,
        "chunk_id": result.job.chunk_id if result.job else None,
        "profile_name": result.job.profile_name if result.job else None,
        "status": result.job.status if result.job else None,
        "elapsed_ms": result.elapsed_ms,
        "table_name": result.vector.table_name if result.vector else None,
        "message": result.message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Process one pending embedding job.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--worker-name", default="embedding-worker")
    parser.add_argument("--profile-name", default=None)
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument(
        "--provider-source",
        choices=PROVIDER_SOURCES,
        default=PROVIDER_SOURCE_ROUTE,
        help=(
            "Provider source. 'route' selects an active profile route from the database "
            "and falls back to runtime config; 'runtime' ignores route records."
        ),
    )
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
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    runtime_config = _runtime_config_from_args(args, settings)
    result = _process_next_job(
        database_url,
        worker_name=args.worker_name,
        profile_name=args.profile_name,
        lease_seconds=args.lease_seconds,
        runtime_config=runtime_config,
        provider_source=args.provider_source,
    )
    payload = _result_payload(
        result,
        runtime_config=runtime_config,
        provider_source=args.provider_source,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
