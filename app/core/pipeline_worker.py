"""Pipeline worker MVP for local document ingestion."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.bm25_keyword_index import refresh_chunk_policy_keyword_index_in_connection
from app.core.chunking import (
    DEFAULT_CHUNK_POLICIES,
    ChunkPolicy,
    chunk_document_blocks,
    get_chunk_policy,
)
from app.core.chunks import replace_document_chunks_in_connection
from app.core.database import connect
from app.core.embedding_jobs import (
    create_embedding_jobs_for_chunk_in_connection,
    list_active_embedding_profiles_in_connection,
)
from app.core.extraction_runtime import ExtractionRuntimeRequest, ExtractionRuntimeResult
from app.core.file_metadata import (
    FileMetadataRecord,
    get_file_metadata,
    mark_file_parse_failed,
    mark_file_parse_running_in_connection,
    mark_file_parse_succeeded_in_connection,
)
from app.core.local_extraction import (
    LocalExtractionHandler,
    persist_extraction_runtime_result_in_connection,
    run_local_extraction,
    select_local_extraction_handler,
)
from app.core.pipeline_jobs import (
    DEFAULT_LEASE_SECONDS,
    PipelineJobRecord,
    claim_next_pipeline_job,
    get_pipeline_job,
    mark_pipeline_failed,
    mark_pipeline_succeeded_in_connection,
    update_pipeline_progress,
    update_pipeline_progress_in_connection,
)

MARKDOWN_PIPELINE_TOTAL_UNITS = 5
DEFAULT_PIPELINE_CHUNK_POLICY_NAMES = tuple(DEFAULT_CHUNK_POLICIES)
ERROR_CODE_INVALID_JOB_INPUT = "INVALID_JOB_INPUT"
ERROR_CODE_STORED_FILE_NOT_FOUND = "STORED_FILE_NOT_FOUND"
ERROR_CODE_UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
ERROR_CODE_UNSUPPORTED_JOB_TYPE = "UNSUPPORTED_JOB_TYPE"
ERROR_CODE_MARKDOWN_PIPELINE_ERROR = "MARKDOWN_PIPELINE_ERROR"


@dataclass(frozen=True)
class MarkdownPipelinePolicyResult:
    chunk_policy_name: str
    chunk_count: int
    embedding_job_count: int
    bm25_term_row_count: int = 0
    bm25_statistics_row_count: int = 0


@dataclass(frozen=True)
class MarkdownPipelineWorkerResult:
    processed: bool
    job: PipelineJobRecord | None
    chunk_count: int = 0
    embedding_job_count: int = 0
    bm25_term_row_count: int = 0
    bm25_statistics_row_count: int = 0
    policy_results: tuple[MarkdownPipelinePolicyResult, ...] = ()
    message: str | None = None


def process_next_markdown_pipeline_job(
    database_url: str,
    *,
    worker_name: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    chunk_policy_name: str | None = None,
    chunk_policy_names: Sequence[str] | None = None,
) -> MarkdownPipelineWorkerResult:
    """Claim and process one queued local document ingestion job."""

    policies = _resolve_chunk_policies(
        chunk_policy_name=chunk_policy_name,
        chunk_policy_names=chunk_policy_names,
    )
    job = claim_next_pipeline_job(database_url, worker_name, lease_seconds=lease_seconds)
    if job is None:
        return MarkdownPipelineWorkerResult(
            processed=False,
            job=None,
            policy_results=tuple(
                MarkdownPipelinePolicyResult(
                    chunk_policy_name=policy.chunk_policy_name,
                    chunk_count=0,
                    embedding_job_count=0,
                )
                for policy in policies
            ),
            message="No queued pipeline job is available",
        )

    handler: LocalExtractionHandler | None = None
    try:
        file_record = _load_claimed_job_file(database_url, job)
        handler = select_local_extraction_handler(file_record.file_ext)
        if handler is None:
            message = f"No local extraction runtime is registered for {file_record.file_ext} files"
            failed_job = _fail_claimed_job(
                database_url,
                job,
                error_code=ERROR_CODE_UNSUPPORTED_FILE_TYPE,
                error_message=message,
            )
            return MarkdownPipelineWorkerResult(
                processed=True,
                job=failed_job,
                message=message,
            )

        source_path = Path(file_record.storage_path)
        if not source_path.is_file():
            message = f"Stored upload file was not found: {source_path}"
            failed_job = _fail_claimed_job(
                database_url,
                job,
                error_code=ERROR_CODE_STORED_FILE_NOT_FOUND,
                error_message=message,
                parser_name=handler.parser_name,
                parser_version=handler.parser_version,
            )
            return MarkdownPipelineWorkerResult(
                processed=True,
                job=failed_job,
                message=message,
            )

        with connect(database_url) as connection:
            mark_file_parse_running_in_connection(
                connection,
                file_record.file_id,
                parser_name=handler.parser_name,
                parser_version=handler.parser_version,
            )
            update_pipeline_progress_in_connection(
                connection,
                job.job_id,
                processed_units=0,
                total_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
                stage="text_extraction",
                current_message="Reading source file",
            )

        extraction_request = ExtractionRuntimeRequest(
            file_id=file_record.file_id,
            document_id=file_record.document_id,
            storage_path=str(source_path),
            extraction_profile_name=handler.profile_name,
            mime_type=file_record.mime_type,
            detected_file_type=file_record.file_ext.lstrip(".") or None,
            options={
                "pipeline_worker": worker_name,
                "chunk_policy_names": [policy.chunk_policy_name for policy in policies],
            },
        )
        runtime_result = run_local_extraction(extraction_request)
        update_pipeline_progress(
            database_url,
            job.job_id,
            processed_units=1,
            total_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
            stage="text_extraction",
            current_message=(
                f"Extracted {len(runtime_result.blocks)} document blocks"
                if runtime_result.status == "succeeded"
                else "Text extraction failed"
            ),
        )
        if runtime_result.status != "succeeded":
            with connect(database_url) as connection:
                persist_extraction_runtime_result_in_connection(
                    connection,
                    extraction_request,
                    runtime_result,
                )
            error_code = str(
                runtime_result.runtime_metadata.get(
                    "error_code",
                    ERROR_CODE_MARKDOWN_PIPELINE_ERROR,
                )
            )
            message = runtime_result.errors[0] if runtime_result.errors else "Extraction failed"
            failed_job = _fail_claimed_job(
                database_url,
                job,
                error_code=error_code,
                error_message=message,
                parser_name=handler.parser_name,
                parser_version=handler.parser_version,
            )
            return MarkdownPipelineWorkerResult(
                processed=True,
                job=failed_job,
                message=message,
            )

        with connect(database_url) as connection:
            persisted_extraction = persist_extraction_runtime_result_in_connection(
                connection,
                extraction_request,
                runtime_result,
            )
            update_pipeline_progress_in_connection(
                connection,
                job.job_id,
                processed_units=2,
                total_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
                stage="parsing",
                current_message=(
                    f"Persisted {len(persisted_extraction.blocks)} extracted document blocks"
                ),
            )
            active_profiles = list_active_embedding_profiles_in_connection(connection)
            active_profile_names = [profile.profile_name for profile in active_profiles]
            chunks = []
            embedding_job_results = []
            policy_results = []
            chunk_policy_summaries = []
            bm25_term_row_count = 0
            bm25_statistics_row_count = 0

            for policy in policies:
                chunk_inputs = chunk_document_blocks(
                    list(persisted_extraction.blocks),
                    document_id=file_record.document_id,
                    policy=policy,
                    parser_name=handler.parser_name,
                    parser_version=handler.parser_version,
                )
                policy_chunks = replace_document_chunks_in_connection(
                    connection,
                    file_record.document_id,
                    chunk_inputs,
                    chunk_policy_name=policy.chunk_policy_name,
                )
                policy_embedding_job_results = [
                    result
                    for chunk in policy_chunks
                    for result in create_embedding_jobs_for_chunk_in_connection(
                        connection,
                        chunk.chunk_id,
                        profile_names=active_profile_names,
                    )
                ]
                chunks.extend(policy_chunks)
                embedding_job_results.extend(policy_embedding_job_results)
                chunk_policy_summaries.append(
                    (
                        policy.chunk_policy_name,
                        len(policy_chunks),
                        len(policy_embedding_job_results),
                    )
                )

            update_pipeline_progress_in_connection(
                connection,
                job.job_id,
                processed_units=3,
                total_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
                stage="chunking",
                current_message=(
                    f"Prepared {len(chunks)} chunks across "
                    f"{len(chunk_policy_summaries)} chunk policies"
                ),
            )
            for policy_name, chunk_count, embedding_job_count in chunk_policy_summaries:
                bm25_refresh_result = refresh_chunk_policy_keyword_index_in_connection(
                    connection,
                    chunk_policy_name=policy_name,
                )
                bm25_term_row_count += bm25_refresh_result.term_row_count
                bm25_statistics_row_count += bm25_refresh_result.statistics_row_count
                policy_results.append(
                    MarkdownPipelinePolicyResult(
                        chunk_policy_name=policy_name,
                        chunk_count=chunk_count,
                        embedding_job_count=embedding_job_count,
                        bm25_term_row_count=bm25_refresh_result.term_row_count,
                        bm25_statistics_row_count=(bm25_refresh_result.statistics_row_count),
                    )
                )
            update_pipeline_progress_in_connection(
                connection,
                job.job_id,
                processed_units=4,
                total_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
                stage="keyword_indexing",
                current_message=(
                    f"Refreshed BM25 keyword index with {bm25_term_row_count} term rows "
                    f"and {bm25_statistics_row_count} statistics rows"
                ),
            )
            mark_file_parse_succeeded_in_connection(
                connection,
                file_record.file_id,
                parser_name=handler.parser_name,
                parser_version=handler.parser_version,
                extracted_text_size=_runtime_extracted_text_size(runtime_result),
            )
            update_pipeline_progress_in_connection(
                connection,
                job.job_id,
                processed_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
                total_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
                stage="embedding",
                current_message=(
                    f"Queued {len(embedding_job_results)} embedding jobs "
                    f"for {len(chunks)} chunks across {len(policy_results)} chunk policies"
                ),
            )
            final_job = mark_pipeline_succeeded_in_connection(
                connection,
                job.job_id,
                message=(
                    f"Document ingestion completed with {len(chunks)} chunks "
                    f"and {len(embedding_job_results)} embedding jobs "
                    f"across {len(policy_results)} chunk policies; "
                    f"BM25 keyword index refreshed with {bm25_term_row_count} "
                    f"term rows"
                ),
            )

        if final_job is None:
            msg = f"Claimed pipeline job disappeared before completion: {job.job_id}"
            raise RuntimeError(msg)

        return MarkdownPipelineWorkerResult(
            processed=True,
            job=final_job,
            chunk_count=len(chunks),
            embedding_job_count=len(embedding_job_results),
            bm25_term_row_count=bm25_term_row_count,
            bm25_statistics_row_count=bm25_statistics_row_count,
            policy_results=tuple(policy_results),
            message="Document ingestion completed",
        )
    except _FailClaimedJob as exc:
        failed_job = _fail_claimed_job(
            database_url,
            job,
            error_code=exc.error_code,
            error_message=exc.error_message,
        )
        return MarkdownPipelineWorkerResult(
            processed=True,
            job=failed_job,
            message=exc.error_message,
        )
    except Exception as exc:
        failed_job = _fail_claimed_job(
            database_url,
            job,
            error_code=ERROR_CODE_MARKDOWN_PIPELINE_ERROR,
            error_message=str(exc),
            parser_name=handler.parser_name if handler else None,
            parser_version=handler.parser_version if handler else None,
        )
        return MarkdownPipelineWorkerResult(
            processed=True,
            job=failed_job,
            message=str(exc),
        )


def _load_claimed_job_file(
    database_url: str,
    job: PipelineJobRecord,
) -> FileMetadataRecord:
    if job.job_type != "document_ingestion":
        message = f"Local ingestion worker cannot process {job.job_type} jobs"
        raise _FailClaimedJob(ERROR_CODE_UNSUPPORTED_JOB_TYPE, message)
    if job.file_id is None:
        raise _FailClaimedJob(ERROR_CODE_INVALID_JOB_INPUT, "Pipeline job is missing file_id")
    if job.document_id is None:
        raise _FailClaimedJob(
            ERROR_CODE_INVALID_JOB_INPUT,
            "Pipeline job is missing document_id",
        )

    file_record = get_file_metadata(database_url, job.file_id)
    if file_record is None:
        raise _FailClaimedJob(
            ERROR_CODE_INVALID_JOB_INPUT,
            f"File metadata was not found for file_id={job.file_id}",
        )
    if file_record.document_id is None:
        raise _FailClaimedJob(
            ERROR_CODE_INVALID_JOB_INPUT,
            "File metadata is missing document_id",
        )
    if file_record.document_id != job.document_id:
        raise _FailClaimedJob(
            ERROR_CODE_INVALID_JOB_INPUT,
            "Pipeline job document_id does not match file metadata",
        )
    return file_record


def _resolve_chunk_policies(
    *,
    chunk_policy_name: str | None,
    chunk_policy_names: Sequence[str] | None,
) -> tuple[ChunkPolicy, ...]:
    if chunk_policy_names is not None:
        raw_policy_names = list(chunk_policy_names)
        if not raw_policy_names:
            raise ValueError("chunk_policy_names must not be empty")
    elif chunk_policy_name is not None:
        raw_policy_names = [chunk_policy_name]
    else:
        raw_policy_names = list(DEFAULT_PIPELINE_CHUNK_POLICY_NAMES)

    normalized_names = [name.strip() for name in raw_policy_names]
    if any(not name for name in normalized_names):
        raise ValueError("chunk_policy_names must not contain blank values")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("chunk_policy_names must be unique")

    return tuple(get_chunk_policy(name) for name in normalized_names)


def _fail_claimed_job(
    database_url: str,
    job: PipelineJobRecord,
    *,
    error_code: str,
    error_message: str,
    parser_name: str | None = None,
    parser_version: str | None = None,
) -> PipelineJobRecord:
    if job.file_id is not None:
        mark_file_parse_failed(
            database_url,
            job.file_id,
            error_message=error_message,
            parser_name=parser_name,
            parser_version=parser_version,
        )
    failed_job = mark_pipeline_failed(
        database_url,
        job.job_id,
        error_code=error_code,
        error_message=error_message,
    )
    return failed_job or get_pipeline_job(database_url, job.job_id) or job


def _runtime_extracted_text_size(runtime_result: ExtractionRuntimeResult) -> int:
    return sum(len(artifact.content_text or "") for artifact in runtime_result.artifacts)


@dataclass(frozen=True)
class _FailClaimedJob(Exception):
    error_code: str
    error_message: str

    def __str__(self) -> str:
        return self.error_message
