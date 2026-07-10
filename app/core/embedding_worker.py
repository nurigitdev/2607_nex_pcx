"""Embedding worker for provider-backed vector persistence."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobRecord,
    claim_next_embedding_job,
    get_embedding_job,
    mark_embedding_job_failed,
    mark_embedding_job_succeeded_in_connection,
)
from app.core.embedding_model_distribution import (
    InvalidEmbeddingModelDistributionError,
    get_embedding_model_distribution_for_profile,
)
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteRecord,
    list_embedding_provider_routes,
)
from app.core.embedding_providers import (
    EmbeddingProvider,
    EmbeddingProviderRequest,
    EmbeddingProviderRuntimeConfig,
    InvalidEmbeddingProviderError,
    MockEmbeddingProvider,
    build_embedding_provider_from_runtime_config,
    normalize_embedding_provider_runtime_config,
)
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    EmbeddingVectorRecord,
    InvalidEmbeddingVectorError,
    get_chunk_text_in_connection,
    get_embedding_vector_table,
    store_chunk_embedding_in_connection,
)
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS

ERROR_CODE_CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
ERROR_CODE_MOCK_EMBEDDING_ERROR = "MOCK_EMBEDDING_ERROR"
ERROR_CODE_EMBEDDING_PROVIDER_ERROR = "EMBEDDING_PROVIDER_ERROR"
ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE = "UNSUPPORTED_EMBEDDING_PROFILE"

EmbeddingProviderBuilder = Callable[[EmbeddingProviderRuntimeConfig], EmbeddingProvider]
EmbeddingProviderRouteSelector = Callable[[str, str], EmbeddingProviderRouteRecord | None]
EmbeddingProviderRouteCandidatesSelector = Callable[
    [str, str],
    Sequence[EmbeddingProviderRouteRecord],
]


@dataclass(frozen=True)
class MockEmbeddingWorkerResult:
    processed: bool
    job: EmbeddingJobRecord | None
    vector: EmbeddingVectorRecord | None = None
    elapsed_ms: int | None = None
    message: str | None = None


EmbeddingWorkerResult = MockEmbeddingWorkerResult


def process_next_mock_embedding_job(
    database_url: str,
    *,
    worker_name: str,
    profile_name: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> MockEmbeddingWorkerResult:
    """Claim and process one embedding job with a deterministic mock vector."""

    return process_next_embedding_job_with_provider(
        database_url,
        worker_name=worker_name,
        provider=MockEmbeddingProvider(),
        profile_name=profile_name,
        lease_seconds=lease_seconds,
        success_message="Mock embedding stored",
        provider_error_code=ERROR_CODE_MOCK_EMBEDDING_ERROR,
    )


def process_next_embedding_job_with_provider(
    database_url: str,
    *,
    worker_name: str,
    provider: EmbeddingProvider,
    profile_name: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    success_message: str = "Embedding stored",
    provider_error_code: str = ERROR_CODE_EMBEDDING_PROVIDER_ERROR,
) -> EmbeddingWorkerResult:
    """Claim and process one embedding job through an embedding provider."""

    job = claim_next_embedding_job(
        database_url,
        worker_name,
        profile_name=profile_name,
        lease_seconds=lease_seconds,
    )
    if job is None:
        return EmbeddingWorkerResult(
            processed=False,
            job=None,
            message="No pending embedding job is available",
        )

    return _process_claimed_embedding_job_with_provider(
        database_url,
        job=job,
        provider=provider,
        success_message=success_message,
        provider_error_code=provider_error_code,
    )


def process_next_embedding_job_with_provider_routes(
    database_url: str,
    *,
    worker_name: str,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig,
    profile_name: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    provider_builder: EmbeddingProviderBuilder = build_embedding_provider_from_runtime_config,
    route_selector: EmbeddingProviderRouteSelector | None = None,
    route_candidates_selector: EmbeddingProviderRouteCandidatesSelector | None = None,
) -> EmbeddingWorkerResult:
    """Claim and process one embedding job using a profile-specific provider route."""

    job = claim_next_embedding_job(
        database_url,
        worker_name,
        profile_name=profile_name,
        lease_seconds=lease_seconds,
    )
    if job is None:
        return EmbeddingWorkerResult(
            processed=False,
            job=None,
            message="No pending embedding job is available",
        )

    route_candidates = _route_candidates_for_job(
        database_url,
        job.profile_name,
        route_selector=route_selector,
        route_candidates_selector=(
            route_candidates_selector or _select_embedding_provider_route_candidates
        ),
    )
    if len(route_candidates) > 1:
        return _process_claimed_embedding_job_with_provider_route_failover(
            database_url,
            job=job,
            routes=route_candidates,
            fallback_runtime_config=fallback_runtime_config,
            provider_builder=provider_builder,
        )

    route = route_candidates[0] if route_candidates else None
    runtime_config = _runtime_config_for_job(
        route,
        fallback_runtime_config=fallback_runtime_config,
    )
    runtime_metadata = _runtime_metadata_for_job_provider(route, runtime_config=runtime_config)
    provider = None
    try:
        provider = provider_builder(runtime_config)
        return _process_claimed_embedding_job_with_provider(
            database_url,
            job=job,
            provider=provider,
            success_message=_success_message_for_provider_mode(runtime_config.mode),
            provider_error_code=_provider_error_code_for_provider_mode(runtime_config.mode),
            runtime_metadata=runtime_metadata,
        )
    except InvalidEmbeddingProviderError as exc:
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=_provider_error_code_for_provider_mode(runtime_config.mode),
            error_message=str(exc),
        )
        return EmbeddingWorkerResult(
            processed=True,
            job=failed_job,
            message=str(exc),
        )
    finally:
        if provider is not None and hasattr(provider, "close"):
            provider.close()  # type: ignore[attr-defined]


def _select_embedding_provider_route_candidates(
    database_url: str,
    profile_name: str,
) -> Sequence[EmbeddingProviderRouteRecord]:
    return list_embedding_provider_routes(
        database_url,
        profile_name=profile_name,
        active_only=True,
    )


def _route_candidates_for_job(
    database_url: str,
    profile_name: str,
    *,
    route_selector: EmbeddingProviderRouteSelector | None,
    route_candidates_selector: EmbeddingProviderRouteCandidatesSelector,
) -> list[EmbeddingProviderRouteRecord]:
    if route_selector is not None:
        selected = route_selector(database_url, profile_name)
        return [selected] if selected else []
    return list(route_candidates_selector(database_url, profile_name))


def _process_claimed_embedding_job_with_provider_route_failover(
    database_url: str,
    *,
    job: EmbeddingJobRecord,
    routes: Sequence[EmbeddingProviderRouteRecord],
    fallback_runtime_config: EmbeddingProviderRuntimeConfig,
    provider_builder: EmbeddingProviderBuilder,
) -> EmbeddingWorkerResult:
    failed_attempts: list[dict[str, object]] = []
    provider = None
    for attempt_index, route in enumerate(routes, start=1):
        runtime_config = _runtime_config_for_job(
            route,
            fallback_runtime_config=fallback_runtime_config,
        )
        runtime_metadata = _runtime_metadata_for_job_provider(
            route,
            runtime_config=runtime_config,
            failover_attempt_index=attempt_index,
            failover_candidate_count=len(routes),
            failed_route_attempts=failed_attempts,
        )
        provider = None
        try:
            provider = provider_builder(runtime_config)
            return _process_claimed_embedding_job_with_provider(
                database_url,
                job=job,
                provider=provider,
                success_message=_success_message_for_provider_mode(runtime_config.mode),
                provider_error_code=_provider_error_code_for_provider_mode(runtime_config.mode),
                runtime_metadata=runtime_metadata,
                defer_provider_failure=True,
            )
        except InvalidEmbeddingProviderError as exc:
            failed_attempts.append(_failed_route_attempt(route, str(exc)))
        except _EmbeddingWorkerProviderFailure as exc:
            failed_attempts.append(_failed_route_attempt(route, exc.error_message))
        finally:
            if provider is not None and hasattr(provider, "close"):
                provider.close()  # type: ignore[attr-defined]

    message = _failover_failure_message(failed_attempts)
    failed_job = _fail_claimed_embedding_job(
        database_url,
        job,
        error_code=ERROR_CODE_EMBEDDING_PROVIDER_ERROR,
        error_message=message,
    )
    return EmbeddingWorkerResult(
        processed=True,
        job=failed_job,
        message=message,
    )


def _process_claimed_embedding_job_with_provider(
    database_url: str,
    *,
    job: EmbeddingJobRecord,
    provider: EmbeddingProvider,
    success_message: str,
    provider_error_code: str,
    runtime_metadata: Mapping[str, object] | None = None,
    defer_provider_failure: bool = False,
) -> EmbeddingWorkerResult:
    """Process an already-claimed embedding job through an embedding provider."""

    started_at = perf_counter()
    try:
        table = get_embedding_vector_table(job.profile_name)
        distribution = get_embedding_model_distribution_for_profile(job.profile_name)
        with connect(database_url) as connection:
            chunk_text = get_chunk_text_in_connection(connection, job.chunk_id)
            if chunk_text is None:
                raise _EmbeddingWorkerFailure(
                    ERROR_CODE_CHUNK_NOT_FOUND,
                    f"Chunk was not found for chunk_id={job.chunk_id}",
                )

            provider_response = provider.embed(
                EmbeddingProviderRequest(
                    profile_name=job.profile_name,
                    model_key=distribution.model_key,
                    input_type="document",
                    texts=(chunk_text,),
                    output_dimension=table.dimension,
                    trace_id=f"embedding-job-{job.job_id}",
                )
            )
            embedding = provider_response.embeddings[0]
            elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
            vector = store_chunk_embedding_in_connection(
                connection,
                EmbeddingVectorInput(
                    chunk_id=job.chunk_id,
                    profile_name=job.profile_name,
                    embedding=embedding,
                    elapsed_ms=elapsed_ms,
                ),
            )
            final_job = mark_embedding_job_succeeded_in_connection(
                connection,
                job.job_id,
                runtime_metadata={
                    **dict(provider_response.runtime_metadata),
                    **dict(runtime_metadata or {}),
                    "adapter": provider_response.provider_type,
                    "provider_model_id": provider_response.provider_model_id,
                    "provider_type": provider_response.provider_type,
                    "dimension": table.dimension,
                    "storage_type": table.storage_type,
                    "table_name": table.table_name,
                    "provider_elapsed_ms": provider_response.elapsed_ms,
                    "elapsed_ms": elapsed_ms,
                },
            )

        if final_job is None:
            msg = f"Claimed embedding job disappeared before completion: {job.job_id}"
            raise RuntimeError(msg)

        return EmbeddingWorkerResult(
            processed=True,
            job=final_job,
            vector=vector,
            elapsed_ms=elapsed_ms,
            message=success_message,
        )
    except _EmbeddingWorkerFailure as exc:
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=exc.error_code,
            error_message=exc.error_message,
        )
        return EmbeddingWorkerResult(
            processed=True,
            job=failed_job,
            message=exc.error_message,
        )
    except (InvalidEmbeddingVectorError, InvalidEmbeddingModelDistributionError) as exc:
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE,
            error_message=str(exc),
        )
        return EmbeddingWorkerResult(
            processed=True,
            job=failed_job,
            message=str(exc),
        )
    except InvalidEmbeddingProviderError as exc:
        if defer_provider_failure:
            raise _EmbeddingWorkerProviderFailure(provider_error_code, str(exc)) from exc
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=provider_error_code,
            error_message=str(exc),
        )
        return EmbeddingWorkerResult(
            processed=True,
            job=failed_job,
            message=str(exc),
        )
    except Exception as exc:
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=provider_error_code,
            error_message=str(exc),
        )
        return EmbeddingWorkerResult(
            processed=True,
            job=failed_job,
            message=str(exc),
        )


def _fail_claimed_embedding_job(
    database_url: str,
    job: EmbeddingJobRecord,
    *,
    error_code: str,
    error_message: str,
) -> EmbeddingJobRecord:
    failed_job = mark_embedding_job_failed(
        database_url,
        job.job_id,
        error_code=error_code,
        error_message=error_message,
    )
    return failed_job or get_embedding_job(database_url, job.job_id) or job


def _runtime_config_for_job(
    route: EmbeddingProviderRouteRecord | None,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig,
) -> EmbeddingProviderRuntimeConfig:
    if route is None:
        return normalize_embedding_provider_runtime_config(fallback_runtime_config)
    return normalize_embedding_provider_runtime_config(
        EmbeddingProviderRuntimeConfig(
            mode=route.provider_mode,
            remote_base_url=route.provider_base_url,
            remote_timeout_seconds=route.timeout_seconds,
        )
    )


def _runtime_metadata_for_job_provider(
    route: EmbeddingProviderRouteRecord | None,
    *,
    runtime_config: EmbeddingProviderRuntimeConfig,
    failover_attempt_index: int | None = None,
    failover_candidate_count: int | None = None,
    failed_route_attempts: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if route is None:
        metadata: dict[str, object] = {
            "provider_runtime_source": "fallback_runtime_config",
            "provider_runtime_mode": runtime_config.mode,
            "provider_runtime_base_url": runtime_config.remote_base_url,
            "provider_runtime_timeout_seconds": runtime_config.remote_timeout_seconds,
        }
        _add_failover_metadata(
            metadata,
            failover_attempt_index=failover_attempt_index,
            failover_candidate_count=failover_candidate_count,
            failed_route_attempts=failed_route_attempts,
        )
        return metadata
    metadata: dict[str, object] = {
        "provider_runtime_source": "route",
        "provider_runtime_mode": route.provider_mode,
        "provider_runtime_base_url": route.provider_base_url,
        "provider_runtime_timeout_seconds": route.timeout_seconds,
        "provider_route_id": route.route_id,
        "provider_route_name": route.provider_name,
        "provider_route_priority": route.priority,
    }
    if route.runtime_metadata:
        metadata["provider_route_metadata"] = dict(route.runtime_metadata)
    _add_failover_metadata(
        metadata,
        failover_attempt_index=failover_attempt_index,
        failover_candidate_count=failover_candidate_count,
        failed_route_attempts=failed_route_attempts,
    )
    return metadata


def _add_failover_metadata(
    metadata: dict[str, object],
    *,
    failover_attempt_index: int | None,
    failover_candidate_count: int | None,
    failed_route_attempts: Sequence[Mapping[str, object]] | None,
) -> None:
    if failover_attempt_index is not None:
        metadata["provider_route_failover_attempt"] = failover_attempt_index
    if failover_candidate_count is not None:
        metadata["provider_route_failover_candidate_count"] = failover_candidate_count
    if failed_route_attempts:
        metadata["provider_route_failed_attempts"] = [
            dict(attempt) for attempt in failed_route_attempts
        ]


def _failed_route_attempt(
    route: EmbeddingProviderRouteRecord,
    error_message: str,
) -> dict[str, object]:
    return {
        "route_id": route.route_id,
        "provider_name": route.provider_name,
        "provider_mode": route.provider_mode,
        "priority": route.priority,
        "error_message": error_message,
    }


def _failover_failure_message(failed_attempts: Sequence[Mapping[str, object]]) -> str:
    if not failed_attempts:
        return "No provider route could process the embedding job"
    providers = ", ".join(str(attempt["provider_name"]) for attempt in failed_attempts)
    last_error = str(failed_attempts[-1]["error_message"])
    return f"All provider routes failed ({providers}): {last_error}"


def _success_message_for_provider_mode(provider_mode: str) -> str:
    if provider_mode == "mock":
        return "Mock embedding stored"
    return "Remote embedding stored"


def _provider_error_code_for_provider_mode(provider_mode: str) -> str:
    if provider_mode == "mock":
        return ERROR_CODE_MOCK_EMBEDDING_ERROR
    return ERROR_CODE_EMBEDDING_PROVIDER_ERROR


@dataclass(frozen=True)
class _EmbeddingWorkerFailure(Exception):
    error_code: str
    error_message: str

    def __str__(self) -> str:
        return self.error_message


@dataclass(frozen=True)
class _EmbeddingWorkerProviderFailure(Exception):
    error_code: str
    error_message: str

    def __str__(self) -> str:
        return self.error_message
