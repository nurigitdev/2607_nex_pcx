from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_jobs import EmbeddingJobInput, create_embedding_job
from app.core.embedding_worker import process_next_mock_embedding_job
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_chunk(database_url: str, chunk_text: str) -> tuple[int, int]:
    checksum = f"embedding-monitor-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path
                )
                VALUES (%s, %s, '.md', 1, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Embedding monitor fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    char_count
                )
                VALUES (%s, 0, %s, %s, 'heading_512_64', %s)
                RETURNING chunk_id
                """,
                (document_id, chunk_text, f"chunk-{checksum}", len(chunk_text)),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_embedding_job_monitor_api_and_ui(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    file_id, chunk_id = _create_chunk(
        migrated_database_url,
        "Embedding monitor API and UI fixture",
    )
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name="kure_v1_1024"),
        )
        processed = process_next_mock_embedding_job(
            migrated_database_url,
            worker_name="monitor-worker",
            profile_name="kure_v1_1024",
        )
        assert processed.job is not None
        assert processed.job.job_id == created.job.job_id

        app = create_app(
            Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
        )

        with TestClient(app) as client:
            list_response = client.get(
                "/api/embedding/jobs",
                params={"status": "succeeded", "profile_name": "kure_v1_1024"},
            )
            detail_response = client.get(f"/api/embedding/jobs/{created.job.job_id}")
            page_response = client.get(f"/admin/embedding-jobs?job_id={created.job.job_id}")

        list_payload = list_response.json()
        detail_payload = detail_response.json()

        assert list_response.status_code == 200
        assert any(job["job_id"] == created.job.job_id for job in list_payload["jobs"])
        assert detail_response.status_code == 200
        assert detail_payload["job"]["status"] == "succeeded"
        assert detail_payload["embedding"]["dimension"] == 1024
        assert detail_payload["embedding"]["table_name"] == "chunk_embeddings_kure_v1_1024"
        assert page_response.status_code == 200
        assert "Embedding Job Monitor" in page_response.text
        assert f"#{created.job.job_id}" in page_response.text
        assert "kure_v1_1024" in page_response.text
        assert "chunk_embeddings_kure_v1_1024" in page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_embedding_job_monitor_api_handles_invalid_requests(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path))

    with TestClient(app) as client:
        invalid_status = client.get("/api/embedding/jobs", params={"status": "queued"})
        missing_job = client.get("/api/embedding/jobs/999999999")

    assert invalid_status.status_code == 400
    assert "Unsupported embedding job status" in invalid_status.json()["detail"]
    assert missing_job.status_code == 404
