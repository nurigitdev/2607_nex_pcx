from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.core.chunks import ChunkInput
from app.core.extraction_runtime import ExtractionRuntimeArtifact, ExtractionRuntimeResult
from app.core.file_metadata import FileMetadataRecord
from app.core.pipeline_jobs import PipelineJobRecord
from app.core.pipeline_worker import (
    ERROR_CODE_INVALID_JOB_INPUT,
    ERROR_CODE_MARKDOWN_PIPELINE_ERROR,
    ERROR_CODE_STORED_FILE_NOT_FOUND,
    ERROR_CODE_UNSUPPORTED_FILE_TYPE,
    ERROR_CODE_UNSUPPORTED_JOB_TYPE,
    process_next_markdown_pipeline_job,
)


def make_job(**overrides) -> PipelineJobRecord:
    now = datetime(2026, 7, 6, tzinfo=UTC)
    values = {
        "job_id": 1,
        "job_type": "document_ingestion",
        "file_id": 10,
        "document_id": 20,
        "parent_job_id": None,
        "requested_by_user_id": None,
        "status": "running",
        "stage": "upload_saved",
        "priority": 100,
        "total_units": 0,
        "processed_units": 0,
        "progress_percent": Decimal("0.00"),
        "current_message": None,
        "attempts": 1,
        "max_attempts": 3,
        "lease_owner": "unit-test-worker",
        "lease_expires_at": now,
        "heartbeat_at": now,
        "error_code": None,
        "error_message": None,
        "metadata": {},
        "queued_at": now,
        "started_at": now,
        "finished_at": None,
        "updated_at": now,
    }
    values.update(overrides)
    return PipelineJobRecord(**values)


def make_file(**overrides) -> FileMetadataRecord:
    values = {
        "file_id": 10,
        "document_id": 20,
        "original_file_name": "example.md",
        "stored_file_name": "example.stored.md",
        "file_ext": ".md",
        "mime_type": "text/markdown",
        "file_size_bytes": 42,
        "sha256_checksum": "checksum",
        "storage_path": "/tmp/missing.md",
        "document_group": "unit",
        "security_level": "internal",
        "parse_status": "pending",
    }
    values.update(overrides)
    return FileMetadataRecord(**values)


def patch_claim(monkeypatch, job: PipelineJobRecord | None) -> None:
    monkeypatch.setattr(
        "app.core.pipeline_worker.claim_next_pipeline_job",
        lambda *args, **kwargs: job,
    )


def patch_failure(monkeypatch, claimed_job: PipelineJobRecord) -> dict[str, int]:
    calls = {"file_failed": 0}

    def fake_mark_file_parse_failed(*args, **kwargs):
        calls["file_failed"] += 1
        return None

    def fake_mark_pipeline_failed(*args, error_code: str, error_message: str, **kwargs):
        return replace(
            claimed_job,
            status="failed",
            error_code=error_code,
            error_message=error_message,
        )

    monkeypatch.setattr(
        "app.core.pipeline_worker.mark_file_parse_failed",
        fake_mark_file_parse_failed,
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.mark_pipeline_failed",
        fake_mark_pipeline_failed,
    )
    return calls


def test_process_next_markdown_pipeline_job_returns_idle_when_queue_is_empty(monkeypatch) -> None:
    patch_claim(monkeypatch, None)

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.processed is False
    assert result.job is None
    assert result.message == "No queued pipeline job is available"


def test_process_next_markdown_pipeline_job_fails_unsupported_job_type(monkeypatch) -> None:
    job = make_job(job_type="embedding", file_id=None, document_id=None)
    patch_claim(monkeypatch, job)
    calls = patch_failure(monkeypatch, job)

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_UNSUPPORTED_JOB_TYPE
    assert calls["file_failed"] == 0


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (make_job(file_id=None), "missing file_id"),
        (make_job(document_id=None), "missing document_id"),
    ],
)
def test_process_next_markdown_pipeline_job_fails_invalid_job_identity(
    monkeypatch,
    job: PipelineJobRecord,
    message: str,
) -> None:
    patch_claim(monkeypatch, job)
    patch_failure(monkeypatch, job)

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_INVALID_JOB_INPUT
    assert message in result.message


@pytest.mark.parametrize(
    ("file_record", "message"),
    [
        (None, "metadata was not found"),
        (make_file(document_id=None), "missing document_id"),
        (make_file(document_id=999), "does not match"),
    ],
)
def test_process_next_markdown_pipeline_job_fails_invalid_file_metadata(
    monkeypatch,
    file_record: FileMetadataRecord | None,
    message: str,
) -> None:
    job = make_job()
    patch_claim(monkeypatch, job)
    patch_failure(monkeypatch, job)
    monkeypatch.setattr("app.core.pipeline_worker.get_file_metadata", lambda *args: file_record)

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_INVALID_JOB_INPUT
    assert message in result.message


def test_process_next_markdown_pipeline_job_fails_unregistered_local_runtime(
    monkeypatch,
) -> None:
    job = make_job()
    patch_claim(monkeypatch, job)
    calls = patch_failure(monkeypatch, job)
    monkeypatch.setattr(
        "app.core.pipeline_worker.get_file_metadata",
        lambda *args: make_file(file_ext=".unknown"),
    )

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_UNSUPPORTED_FILE_TYPE
    assert "No local extraction runtime is registered for .unknown files" in result.message
    assert calls["file_failed"] == 1


def test_process_next_markdown_pipeline_job_fails_when_stored_file_is_missing(
    monkeypatch,
) -> None:
    job = make_job()
    patch_claim(monkeypatch, job)
    patch_failure(monkeypatch, job)
    monkeypatch.setattr(
        "app.core.pipeline_worker.get_file_metadata",
        lambda *args: make_file(storage_path="/tmp/does-not-exist.md"),
    )

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_STORED_FILE_NOT_FOUND
    assert "was not found" in result.message


@contextmanager
def fake_connection():
    yield object()


def test_process_next_markdown_pipeline_job_marks_failed_when_completion_disappears(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "example.md"
    source.write_text("# Example\n\nBody", encoding="utf-8")
    job = make_job()
    patch_claim(monkeypatch, job)
    patch_failure(monkeypatch, job)
    monkeypatch.setattr(
        "app.core.pipeline_worker.get_file_metadata",
        lambda *args: make_file(storage_path=str(source)),
    )
    monkeypatch.setattr("app.core.pipeline_worker.connect", lambda *args: fake_connection())
    monkeypatch.setattr(
        "app.core.pipeline_worker.update_pipeline_progress", lambda *args, **k: None
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.update_pipeline_progress_in_connection",
        lambda *args, **k: None,
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.mark_file_parse_running_in_connection",
        lambda *args, **k: None,
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.run_local_extraction",
        lambda *args, **k: ExtractionRuntimeResult(
            status="succeeded",
            artifacts=(
                ExtractionRuntimeArtifact(
                    artifact_type="normalized_markdown",
                    content_text="# Example",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.persist_extraction_runtime_result_in_connection",
        lambda *args, **k: SimpleNamespace(blocks=[]),
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.chunk_document_blocks",
        lambda *args, **k: [ChunkInput(document_id=20, chunk_seq=0, chunk_text="# Example")],
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.replace_document_chunks_in_connection",
        lambda *args, **k: [],
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.list_active_embedding_profiles_in_connection",
        lambda *args, **k: [],
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.create_embedding_jobs_for_chunk_in_connection",
        lambda *args, **k: [],
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.mark_file_parse_succeeded_in_connection",
        lambda *args, **k: None,
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.mark_pipeline_succeeded_in_connection",
        lambda *args, **k: None,
    )

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_MARKDOWN_PIPELINE_ERROR
    assert "disappeared before completion" in result.message


def test_process_next_markdown_pipeline_job_fails_when_local_extraction_fails(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "empty.md"
    source.write_text(" ", encoding="utf-8")
    job = make_job()
    patch_claim(monkeypatch, job)
    patch_failure(monkeypatch, job)
    persist_calls = {"count": 0}

    def fake_persist(*args, **kwargs):
        persist_calls["count"] += 1
        return SimpleNamespace(blocks=[])

    monkeypatch.setattr(
        "app.core.pipeline_worker.get_file_metadata",
        lambda *args: make_file(storage_path=str(source)),
    )
    monkeypatch.setattr("app.core.pipeline_worker.connect", lambda *args: fake_connection())
    monkeypatch.setattr(
        "app.core.pipeline_worker.update_pipeline_progress", lambda *args, **k: None
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.update_pipeline_progress_in_connection",
        lambda *args, **k: None,
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.mark_file_parse_running_in_connection",
        lambda *args, **k: None,
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.run_local_extraction",
        lambda *args, **k: ExtractionRuntimeResult(
            status="failed",
            errors=("Source file does not contain extractable text",),
            runtime_metadata={"error_code": "LOCAL_SOURCE_EMPTY"},
        ),
    )
    monkeypatch.setattr(
        "app.core.pipeline_worker.persist_extraction_runtime_result_in_connection",
        fake_persist,
    )

    result = process_next_markdown_pipeline_job(
        "postgresql://example/db",
        worker_name="unit-test-worker",
    )

    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == "LOCAL_SOURCE_EMPTY"
    assert "extractable text" in result.message
    assert persist_calls["count"] == 1
