"""Pipeline worker MVP for local document ingestion."""

from dataclasses import dataclass
from pathlib import Path

from app.core.chunking import chunk_document_blocks, get_chunk_policy
from app.core.chunks import DEFAULT_CHUNK_POLICY_NAME, replace_document_chunks_in_connection
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
ERROR_CODE_INVALID_JOB_INPUT = "INVALID_JOB_INPUT"
ERROR_CODE_STORED_FILE_NOT_FOUND = "STORED_FILE_NOT_FOUND"
ERROR_CODE_UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
ERROR_CODE_UNSUPPORTED_JOB_TYPE = "UNSUPPORTED_JOB_TYPE"
ERROR_CODE_MARKDOWN_PIPELINE_ERROR = "MARKDOWN_PIPELINE_ERROR"


@dataclass(frozen=True)
class MarkdownPipelineWorkerResult:
    processed: bool
    job: PipelineJobRecord | None
    chunk_count: int = 0
    embedding_job_count: int = 0
    message: str | None = None


def process_next_markdown_pipeline_job(
    database_url: str,
    *,
    worker_name: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    chunk_policy_name: str = DEFAULT_CHUNK_POLICY_NAME,
) -> MarkdownPipelineWorkerResult:
    """Claim and process one queued local document ingestion job."""

    policy = get_chunk_policy(chunk_policy_name)
    job = claim_next_pipeline_job(database_url, worker_name, lease_seconds=lease_seconds)
    if job is None:
        return MarkdownPipelineWorkerResult(
            processed=False,
            job=None,
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
            options={"pipeline_worker": worker_name},
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
            chunk_inputs = chunk_document_blocks(
                list(persisted_extraction.blocks),
                document_id=file_record.document_id,
                policy=policy,
                parser_name=handler.parser_name,
                parser_version=handler.parser_version,
            )
            update_pipeline_progress_in_connection(
                connection,
                job.job_id,
                processed_units=3,
                total_units=MARKDOWN_PIPELINE_TOTAL_UNITS,
                stage="chunking",
                current_message=f"Prepared {len(chunk_inputs)} chunks",
            )
            chunks = replace_document_chunks_in_connection(
                connection,
                file_record.document_id,
                chunk_inputs,
                chunk_policy_name=policy.chunk_policy_name,
            )
            active_profiles = list_active_embedding_profiles_in_connection(connection)
            active_profile_names = [profile.profile_name for profile in active_profiles]
            embedding_job_results = [
                result
                for chunk in chunks
                for result in create_embedding_jobs_for_chunk_in_connection(
                    connection,
                    chunk.chunk_id,
                    profile_names=active_profile_names,
                )
            ]
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
                    f"for {len(chunks)} chunks"
                ),
            )
            final_job = mark_pipeline_succeeded_in_connection(
                connection,
                job.job_id,
                message=(
                    f"Document ingestion completed with {len(chunks)} chunks "
                    f"and {len(embedding_job_results)} embedding jobs"
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
