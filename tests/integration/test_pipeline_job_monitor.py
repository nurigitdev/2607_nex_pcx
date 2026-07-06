from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.pipeline_jobs import PipelineJobInput, create_pipeline_job, update_pipeline_progress
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_monitor_document(database_url: str) -> tuple[int, int, int, str]:
    suffix = uuid4().hex
    original_file_name = f"monitor-fixture-{suffix}.md"
    checksum = f"pipeline-monitor-{suffix}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    uploaded_by_user_id
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    original_file_name,
                    f"{suffix}.stored.md",
                    checksum,
                    f"/tmp/{original_file_name}",
                    user_id,
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    owner_user_id,
                    access_scope
                )
                VALUES (%s, %s, %s, 'personal')
                RETURNING document_id
                """,
                (file_id, f"Monitor fixture {suffix}", user_id),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id, user_id, original_file_name


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_pipeline_job_monitor_api_and_ui(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    file_id, document_id, user_id, original_file_name = _create_monitor_document(
        migrated_database_url,
    )
    try:
        job = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
                total_units=4,
            ),
        )
        update_pipeline_progress(
            migrated_database_url,
            job.job_id,
            processed_units=2,
            total_units=4,
            stage="parsing",
            current_message="monitor fixture halfway",
        )
        app = create_app(
            Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
        )

        with TestClient(app) as client:
            list_response = client.get("/api/pipeline/jobs", params={"status": "queued"})
            detail_response = client.get(f"/api/pipeline/jobs/{job.job_id}")
            page_response = client.get(f"/admin/jobs?job_id={job.job_id}")

        list_payload = list_response.json()
        detail_payload = detail_response.json()

        assert list_response.status_code == 200
        assert any(
            item["job"]["job_id"] == job.job_id and item["original_file_name"] == original_file_name
            for item in list_payload["jobs"]
        )
        assert detail_response.status_code == 200
        assert detail_payload["job"]["progress_percent"] == "50.00"
        assert [event["event_type"] for event in detail_payload["events"]] == [
            "created",
            "progress",
        ]
        assert page_response.status_code == 200
        assert "Pipeline Job Monitor" in page_response.text
        assert f"#{job.job_id}" in page_response.text
        assert original_file_name in page_response.text
        assert "monitor fixture halfway" in page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_pipeline_job_monitor_api_handles_invalid_requests(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path))

    with TestClient(app) as client:
        invalid_status = client.get("/api/pipeline/jobs", params={"status": "unknown"})
        missing_job = client.get("/api/pipeline/jobs/999999999")

    assert invalid_status.status_code == 400
    assert "Unsupported pipeline job status" in invalid_status.json()["detail"]
    assert missing_job.status_code == 404
