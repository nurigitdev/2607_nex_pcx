"""Mock embedding worker for deterministic vector persistence."""

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
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    EmbeddingVectorRecord,
    InvalidEmbeddingVectorError,
    generate_mock_embedding,
    get_chunk_text_in_connection,
    get_embedding_vector_table,
    store_chunk_embedding_in_connection,
)
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS

ERROR_CODE_CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
ERROR_CODE_MOCK_EMBEDDING_ERROR = "MOCK_EMBEDDING_ERROR"
ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE = "UNSUPPORTED_EMBEDDING_PROFILE"


@dataclass(frozen=True)
class MockEmbeddingWorkerResult:
    processed: bool
    job: EmbeddingJobRecord | None
    vector: EmbeddingVectorRecord | None = None
    elapsed_ms: int | None = None
    message: str | None = None


def process_next_mock_embedding_job(
    database_url: str,
    *,
    worker_name: str,
    profile_name: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> MockEmbeddingWorkerResult:
    """Claim and process one embedding job with a deterministic mock vector."""

    job = claim_next_embedding_job(
        database_url,
        worker_name,
        profile_name=profile_name,
        lease_seconds=lease_seconds,
    )
    if job is None:
        return MockEmbeddingWorkerResult(
            processed=False,
            job=None,
            message="No pending embedding job is available",
        )

    started_at = perf_counter()
    try:
        table = get_embedding_vector_table(job.profile_name)
        with connect(database_url) as connection:
            chunk_text = get_chunk_text_in_connection(connection, job.chunk_id)
            if chunk_text is None:
                raise _MockEmbeddingFailure(
                    ERROR_CODE_CHUNK_NOT_FOUND,
                    f"Chunk was not found for chunk_id={job.chunk_id}",
                )

            embedding = generate_mock_embedding(
                chunk_text,
                profile_name=job.profile_name,
                dimension=table.dimension,
            )
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
                    "adapter": "mock",
                    "dimension": table.dimension,
                    "storage_type": table.storage_type,
                    "table_name": table.table_name,
                    "elapsed_ms": elapsed_ms,
                },
            )

        if final_job is None:
            msg = f"Claimed embedding job disappeared before completion: {job.job_id}"
            raise RuntimeError(msg)

        return MockEmbeddingWorkerResult(
            processed=True,
            job=final_job,
            vector=vector,
            elapsed_ms=elapsed_ms,
            message="Mock embedding stored",
        )
    except _MockEmbeddingFailure as exc:
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=exc.error_code,
            error_message=exc.error_message,
        )
        return MockEmbeddingWorkerResult(
            processed=True,
            job=failed_job,
            message=exc.error_message,
        )
    except InvalidEmbeddingVectorError as exc:
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE,
            error_message=str(exc),
        )
        return MockEmbeddingWorkerResult(
            processed=True,
            job=failed_job,
            message=str(exc),
        )
    except Exception as exc:
        failed_job = _fail_claimed_embedding_job(
            database_url,
            job,
            error_code=ERROR_CODE_MOCK_EMBEDDING_ERROR,
            error_message=str(exc),
        )
        return MockEmbeddingWorkerResult(
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
class _MockEmbeddingFailure(Exception):
    error_code: str
    error_message: str

    def __str__(self) -> str:
        return self.error_message
