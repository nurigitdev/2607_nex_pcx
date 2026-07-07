from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.chunks import ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    create_embedding_job,
    mark_embedding_job_failed,
)
from app.core.file_metadata import FileMetadataInput, create_file_metadata
from app.core.pipeline_jobs import PipelineJobInput, create_pipeline_job, mark_pipeline_failed
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_retry_fixture(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-051-{suffix}"
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"retry-{suffix}.md",
            stored_file_name=f"retry-{suffix}.stored.md",
            file_size_bytes=42,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/retry-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group="slice-051",
            security_level="internal",
            uploaded_by="integration-test",
            document_title=f"Retry Document {suffix}",
        ),
    )
    document_id = created.file.document_id
    assert document_id is not None
    chunk = create_chunk(
        database_url,
        ChunkInput(
            document_id=document_id,
            chunk_seq=0,
            chunk_text="Retry API chunk",
            token_count=3,
        ),
    )
    pipeline_job = create_pipeline_job(
        database_url,
        PipelineJobInput(
            job_type="document_ingestion",
            file_id=created.file.file_id,
            document_id=document_id,
        ),
    )
    failed_pipeline_job = mark_pipeline_failed(
        database_url,
        pipeline_job.job_id,
        error_code="TEST_PIPELINE",
        error_message="pipeline failed",
    )
    embedding_job = create_embedding_job(
        database_url,
        EmbeddingJobInput(chunk_id=chunk.chunk_id, profile_name="kure_v1_1024"),
    )
    failed_embedding_job = mark_embedding_job_failed(
        database_url,
        embedding_job.job.job_id,
        error_code="TEST_EMBEDDING",
        error_message="embedding failed",
    )
    assert failed_pipeline_job is not None
    assert failed_embedding_job is not None
    return {
        "checksum": checksum,
        "pipeline_job_id": failed_pipeline_job.job_id,
        "embedding_job_id": failed_embedding_job.job_id,
    }


def _cleanup_retry_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM files WHERE sha256_checksum = %s",
                (fixture["checksum"],),
            )


def test_failed_pipeline_and_embedding_jobs_can_retry_through_api_and_ui(
    migrated_database_url: str,
) -> None:
    fixture = _create_retry_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        with TestClient(app) as client:
            pipeline_page = client.get(f"/admin/jobs?job_id={fixture['pipeline_job_id']}")
            embedding_page = client.get(
                f"/admin/embedding-jobs?job_id={fixture['embedding_job_id']}",
            )
            pipeline_retry = client.post(
                f"/api/pipeline/jobs/{fixture['pipeline_job_id']}/retry",
            )
            pipeline_conflict = client.post(
                f"/api/pipeline/jobs/{fixture['pipeline_job_id']}/retry",
            )
            embedding_retry = client.post(
                f"/api/embedding/jobs/{fixture['embedding_job_id']}/retry",
            )
            embedding_conflict = client.post(
                f"/api/embedding/jobs/{fixture['embedding_job_id']}/retry",
            )
            missing_pipeline = client.post("/api/pipeline/jobs/999999999/retry")
            missing_embedding = client.post("/api/embedding/jobs/999999999/retry")

        assert pipeline_page.status_code == 200
        assert 'id="pipeline-retry-button"' in pipeline_page.text
        assert "/api/pipeline/jobs/" in pipeline_page.text
        assert embedding_page.status_code == 200
        assert 'id="embedding-retry-button"' in embedding_page.text
        assert "/api/embedding/jobs/" in embedding_page.text
        assert pipeline_retry.status_code == 200
        assert pipeline_retry.json()["job"]["status"] == "queued"
        assert pipeline_retry.json()["job"]["error_message"] is None
        assert pipeline_conflict.status_code == 409
        assert embedding_retry.status_code == 200
        assert embedding_retry.json()["job"]["status"] == "pending"
        assert embedding_retry.json()["job"]["error_message"] is None
        assert embedding_conflict.status_code == 409
        assert missing_pipeline.status_code == 404
        assert missing_embedding.status_code == 404
    finally:
        _cleanup_retry_fixture(migrated_database_url, fixture)
