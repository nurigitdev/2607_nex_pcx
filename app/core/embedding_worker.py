"""Embedding worker for provider-backed vector persistence."""

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
from app.core.embedding_providers import (
    EmbeddingProvider,
    EmbeddingProviderRequest,
    InvalidEmbeddingProviderError,
    MockEmbeddingProvider,
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


@dataclass(frozen=True)
class _EmbeddingWorkerFailure(Exception):
    error_code: str
    error_message: str

    def __str__(self) -> str:
        return self.error_message
