from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.chunks import list_document_chunks
from app.core.database import connect, fetch_one
from app.core.embedding_jobs import list_embedding_jobs
from app.core.file_uploads import store_upload
from app.core.ingestion_artifacts import (
    list_document_blocks,
    list_document_extraction_artifacts,
    list_document_extraction_runs,
)
from app.core.pipeline_jobs import get_pipeline_job, list_pipeline_job_events
from app.core.pipeline_worker import (
    ERROR_CODE_UNSUPPORTED_FILE_TYPE,
    process_next_markdown_pipeline_job,
)

pytestmark = pytest.mark.integration


def cleanup_checksum(database_url: str, checksum: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE sha256_checksum = %s", (checksum,))


def prioritize_job(database_url: str, job_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE pipeline_jobs SET priority = 0 WHERE job_id = %s",
                (job_id,),
            )


def test_markdown_pipeline_worker_parses_and_stores_chunks(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    unique_text = uuid4().hex
    content = f"""# Slice 017

Markdown pipeline worker integration test {unique_text}.

## Details

This section should become another heading-aware chunk.
""".encode()
    checksum = sha256(content).hexdigest()

    try:
        upload_result = store_upload(
            database_url=migrated_database_url,
            upload_stream=BytesIO(content),
            original_file_name="slice-017-worker.md",
            storage_dir=tmp_path,
            mime_type="text/markdown",
            document_group="slice-017",
            security_level="internal",
            uploaded_by="integration-test",
        )
        assert upload_result.pipeline_job is not None
        prioritize_job(migrated_database_url, upload_result.pipeline_job.job_id)

        result = process_next_markdown_pipeline_job(
            migrated_database_url,
            worker_name="slice-017-test-worker",
        )

        assert result.processed is True
        assert result.job is not None
        assert result.job.job_id == upload_result.pipeline_job.job_id
        assert result.job.status == "succeeded"
        assert result.job.stage == "completed"
        assert result.job.progress_percent == 100
        assert result.chunk_count == 2
        assert result.embedding_job_count == 8

        chunks = list_document_chunks(
            migrated_database_url,
            upload_result.file.document_id,
        )
        extraction_runs = list_document_extraction_runs(
            migrated_database_url,
            upload_result.file.document_id,
        )
        extraction_artifacts = list_document_extraction_artifacts(
            migrated_database_url,
            upload_result.file.document_id,
            artifact_type="normalized_markdown",
        )
        document_blocks = list_document_blocks(
            migrated_database_url,
            upload_result.file.document_id,
            artifact_id=extraction_artifacts[0].artifact_id,
        )
        embedding_jobs = [
            job
            for chunk in chunks
            for job in list_embedding_jobs(migrated_database_url, chunk_id=chunk.chunk_id)
        ]
        file_row = fetch_one(
            migrated_database_url,
            """
            SELECT
                parser_name,
                parser_version,
                parse_status,
                parse_error_message,
                extracted_text_size
            FROM files
            WHERE file_id = %s
            """,
            (upload_result.file.file_id,),
        )
        events = list_pipeline_job_events(
            migrated_database_url,
            upload_result.pipeline_job.job_id,
        )

        assert [chunk.chunk_seq for chunk in chunks] == [0, 1]
        assert chunks[0].chunk_text.startswith("# Slice 017")
        assert chunks[1].heading_path == ("Slice 017", "Details")
        assert extraction_runs[0].status == "succeeded"
        assert extraction_runs[0].extraction_profile_name == "local_markdown_default"
        assert extraction_artifacts[0].artifact_type == "normalized_markdown"
        assert [block.block_type for block in document_blocks] == [
            "heading",
            "paragraph",
            "heading",
            "paragraph",
        ]
        assert chunks[0].artifact_id == extraction_artifacts[0].artifact_id
        assert chunks[0].block_id == document_blocks[0].block_id
        assert chunks[0].metadata["block_ids"] == [
            document_blocks[0].block_id,
            document_blocks[1].block_id,
        ]
        assert len(embedding_jobs) == 8
        assert {job.profile_name for job in embedding_jobs} == {
            "kure_v1_1024",
            "bge_m3_1024",
            "qwen3_4b_1000",
            "qwen3_4b_2560",
        }
        assert {job.status for job in embedding_jobs} == {"pending"}
        assert file_row["parser_name"] == "markdown"
        assert file_row["parser_version"] == "0.1.0"
        assert file_row["parse_status"] == "succeeded"
        assert file_row["parse_error_message"] is None
        assert file_row["extracted_text_size"] > 0
        assert {event.event_type for event in events} >= {
            "created",
            "claimed",
            "progress",
            "stage_succeeded",
        }
    finally:
        cleanup_checksum(migrated_database_url, checksum)


def test_markdown_pipeline_worker_fails_unsupported_file_type(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    content = f"%PDF-1.7\nslice-017 unsupported {uuid4().hex}\n".encode()
    checksum = sha256(content).hexdigest()

    try:
        upload_result = store_upload(
            database_url=migrated_database_url,
            upload_stream=BytesIO(content),
            original_file_name="slice-017-worker.pdf",
            storage_dir=tmp_path,
            mime_type="application/pdf",
            document_group="slice-017",
            security_level="internal",
            uploaded_by="integration-test",
        )
        assert upload_result.pipeline_job is not None
        prioritize_job(migrated_database_url, upload_result.pipeline_job.job_id)

        result = process_next_markdown_pipeline_job(
            migrated_database_url,
            worker_name="slice-017-test-worker",
        )

        assert result.processed is True
        assert result.job is not None
        assert result.job.job_id == upload_result.pipeline_job.job_id
        assert result.job.status == "failed"
        assert result.job.error_code == ERROR_CODE_UNSUPPORTED_FILE_TYPE

        stored_job = get_pipeline_job(migrated_database_url, upload_result.pipeline_job.job_id)
        file_row = fetch_one(
            migrated_database_url,
            """
            SELECT parse_status, parse_error_message
            FROM files
            WHERE file_id = %s
            """,
            (upload_result.file.file_id,),
        )
        chunks = list_document_chunks(
            migrated_database_url,
            upload_result.file.document_id,
        )

        assert stored_job is not None
        assert stored_job.status == "failed"
        assert stored_job.error_code == ERROR_CODE_UNSUPPORTED_FILE_TYPE
        assert file_row["parse_status"] == "failed"
        assert "cannot process .pdf files" in file_row["parse_error_message"]
        assert chunks == []
    finally:
        cleanup_checksum(migrated_database_url, checksum)
