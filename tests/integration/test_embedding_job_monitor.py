from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Json

from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    create_embedding_job,
    get_embedding_job_backlog_summary,
    mark_embedding_job_failed,
)
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


def _cleanup_files_and_profile(
    database_url: str,
    file_ids: list[int],
    profile_name: str,
) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = ANY(%s)", (file_ids,))
            cursor.execute(
                "DELETE FROM embedding_profiles WHERE profile_name = %s",
                (profile_name,),
            )


def _create_test_embedding_profile(database_url: str) -> str:
    profile_name = f"slice_145_{uuid4().hex}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_profiles (
                    profile_name,
                    model_name,
                    dimension,
                    storage_type,
                    is_active
                )
                VALUES (%s, 'example/slice-145-model', 3, 'vector', true)
                """,
                (profile_name,),
            )
    return profile_name


def _update_embedding_job_runtime_metadata(
    database_url: str,
    job_id: int,
    runtime_metadata: dict[str, object],
) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET runtime_metadata = %s
                WHERE job_id = %s
                """,
                (Json(runtime_metadata), job_id),
            )


def test_embedding_job_backlog_summary_api_and_ui(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    profile_name = _create_test_embedding_profile(migrated_database_url)
    file_ids: list[int] = []
    try:
        job_ids: list[int] = []
        for label in ["pending", "stale", "running", "retryable", "exhausted"]:
            file_id, chunk_id = _create_chunk(
                migrated_database_url,
                f"Embedding backlog {label} fixture",
            )
            file_ids.append(file_id)
            created = create_embedding_job(
                migrated_database_url,
                EmbeddingJobInput(
                    chunk_id=chunk_id,
                    profile_name=profile_name,
                    max_attempts=3,
                ),
            )
            job_ids.append(created.job.job_id)

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_jobs
                    SET status = 'running',
                        attempts = 1,
                        lease_owner = 'stale-worker',
                        lease_expires_at = now() - interval '5 minutes',
                        started_at = now() - interval '10 minutes',
                        updated_at = now()
                    WHERE job_id = %s
                    """,
                    (job_ids[1],),
                )
                cursor.execute(
                    """
                    UPDATE embedding_jobs
                    SET status = 'running',
                        attempts = 1,
                        lease_owner = 'active-worker',
                        lease_expires_at = now() + interval '5 minutes',
                        started_at = now(),
                        updated_at = now()
                    WHERE job_id = %s
                    """,
                    (job_ids[2],),
                )

        retryable_failed = mark_embedding_job_failed(
            migrated_database_url,
            job_ids[3],
            error_code="SLICE_145_RETRYABLE",
            error_message="retryable provider failure",
        )
        exhausted_failed = mark_embedding_job_failed(
            migrated_database_url,
            job_ids[4],
            error_code="SLICE_145_EXHAUSTED",
            error_message="exhausted provider failure",
        )
        assert retryable_failed is not None
        assert exhausted_failed is not None
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_jobs
                    SET attempts = max_attempts
                    WHERE job_id = %s
                    """,
                    (exhausted_failed.job_id,),
                )

        summary = get_embedding_job_backlog_summary(migrated_database_url)
        profile_summary = next(
            item for item in summary.profile_summaries if item.profile_name == profile_name
        )
        app = create_app(
            Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
        )

        with TestClient(app) as client:
            api_response = client.get("/api/admin/embedding-jobs/backlog-summary")
            stale_response = client.get(
                "/api/admin/embedding-jobs/stale-leases",
                params={"profile_name": profile_name, "reclaimable_only": True},
            )
            page_response = client.get(
                "/admin/embedding-jobs",
                params={"profile_name": profile_name},
            )
            bulk_retry_response = client.post(
                "/api/admin/embedding-jobs/retry-failed",
                json={"profile_name": profile_name, "limit": 100},
            )
            second_bulk_retry_response = client.post(
                "/api/admin/embedding-jobs/retry-failed",
                json={"profile_name": profile_name, "limit": 100},
            )
            release_response = client.post(
                f"/api/admin/embedding-jobs/{job_ids[1]}/release-stale-lease"
            )
            second_release_response = client.post(
                f"/api/admin/embedding-jobs/{job_ids[1]}/release-stale-lease"
            )

        api_profile = next(
            item
            for item in api_response.json()["backlog"]["profiles"]
            if item["profile_name"] == profile_name
        )

        assert profile_summary.total_count == 5
        assert profile_summary.pending_count == 1
        assert profile_summary.running_count == 2
        assert profile_summary.stale_running_count == 1
        assert profile_summary.reclaimable_stale_running_count == 1
        assert profile_summary.failed_count == 2
        assert profile_summary.retryable_failed_count == 1
        assert profile_summary.exhausted_failed_count == 1
        assert profile_summary.claimable_count == 2
        assert profile_summary.attention_count == 3
        assert api_response.status_code == 200
        assert api_response.json()["backlog"]["claimable_count"] >= 2
        assert api_profile["claimable_count"] == 2
        assert api_profile["attention_count"] == 3
        assert api_profile["oldest_pending_at"] is not None
        assert stale_response.status_code == 200
        assert stale_response.json()["stale_job_count"] == 1
        assert stale_response.json()["jobs"][0]["job_id"] == job_ids[1]
        assert bulk_retry_response.status_code == 200
        assert bulk_retry_response.json()["profile_name"] == profile_name
        assert bulk_retry_response.json()["failed_job_count"] == 2
        assert bulk_retry_response.json()["retried_count"] == 1
        assert bulk_retry_response.json()["retried_jobs"][0]["job_id"] == job_ids[3]
        assert bulk_retry_response.json()["retried_jobs"][0]["status"] == "pending"
        assert bulk_retry_response.json()["skipped_count"] == 1
        assert bulk_retry_response.json()["skipped_jobs"][0]["job_id"] == job_ids[4]
        assert bulk_retry_response.json()["skipped_jobs"][0]["reason"] == "max_attempts_reached"
        assert second_bulk_retry_response.status_code == 200
        assert second_bulk_retry_response.json()["retried_count"] == 0
        assert second_bulk_retry_response.json()["skipped_count"] == 1
        assert release_response.status_code == 200
        assert release_response.json()["job"]["status"] == "pending"
        assert release_response.json()["job"]["lease_owner"] is None
        assert second_release_response.status_code == 409
        assert page_response.status_code == 200
        assert "임베딩 Queue Backlog" in page_response.text
        assert "Failed Job Bulk Retry" in page_response.text
        assert "/api/admin/embedding-jobs/retry-failed" in page_response.text
        assert "data-failed-bulk-retry-button" in page_response.text
        assert f'data-profile-name="{profile_name}"' in page_response.text
        assert "Stale Lease Recovery" in page_response.text
        assert "/api/admin/embedding-jobs/stale-leases" in page_response.text
        assert f'data-job-id="{job_ids[1]}"' in page_response.text
        assert "release-stale-lease" in page_response.text
        assert "/api/admin/embedding-jobs/backlog-summary" in page_response.text
        assert profile_name in page_response.text
    finally:
        _cleanup_files_and_profile(migrated_database_url, file_ids, profile_name)


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
        _update_embedding_job_runtime_metadata(
            migrated_database_url,
            created.job.job_id,
            {
                "provider_runtime_source": "route",
                "provider_route_id": 11,
                "provider_route_name": "gpu-ready",
                "provider_route_priority": 1,
                "provider_route_failover_candidate_count": 2,
                "provider_route_failover_attempt": 2,
                "provider_route_failed_attempts": [
                    {
                        "route_id": 10,
                        "provider_name": "gpu-down",
                        "provider_mode": "remote",
                        "priority": 0,
                        "error_message": "gpu-down unavailable",
                    }
                ],
            },
        )

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
        listed_job = next(
            job for job in list_payload["jobs"] if job["job_id"] == created.job.job_id
        )

        assert list_response.status_code == 200
        assert any(job["job_id"] == created.job.job_id for job in list_payload["jobs"])
        assert listed_job["provider_route_failover"]["candidate_count"] == 2
        assert listed_job["provider_route_failover"]["succeeded_attempt"] == 2
        assert listed_job["provider_route_failover"]["selected_provider_name"] == "gpu-ready"
        assert detail_response.status_code == 200
        assert detail_payload["job"]["status"] == "succeeded"
        assert detail_payload["job"]["provider_route_failover"]["failed_attempts"] == [
            {
                "route_id": 10,
                "provider_name": "gpu-down",
                "provider_mode": "remote",
                "priority": 0,
                "error_message": "gpu-down unavailable",
            }
        ]
        assert detail_payload["embedding"]["dimension"] == 1024
        assert detail_payload["embedding"]["table_name"] == "chunk_embeddings_kure_v1_1024"
        assert page_response.status_code == 200
        assert "Embedding Job Monitor" in page_response.text
        assert f"#{created.job.job_id}" in page_response.text
        assert "kure_v1_1024" in page_response.text
        assert "chunk_embeddings_kure_v1_1024" in page_response.text
        assert "Provider Route Failover" in page_response.text
        assert "gpu-ready" in page_response.text
        assert "gpu-down unavailable" in page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_embedding_job_monitor_shows_readiness_gate_failure(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    file_id, chunk_id = _create_chunk(
        migrated_database_url,
        "Embedding monitor readiness gate fixture",
    )
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name="kure_v1_1024"),
        )
        failed = mark_embedding_job_failed(
            migrated_database_url,
            created.job.job_id,
            error_code="EMBEDDING_PROVIDER_ROUTE_NOT_READY",
            error_message=(
                "No provider route passed the readiness gate " "(gpu-blocked:needs_contract)"
            ),
            runtime_metadata={
                "provider_route_readiness_gate": "blocked_all_routes",
                "provider_route_readiness_blocked_count": 1,
                "provider_route_readiness_blocked_routes": [
                    {
                        "route_id": 40,
                        "provider_name": "gpu-blocked",
                        "profile_name": "kure_v1_1024",
                        "status": "needs_contract",
                        "reasons": ["contract_snapshot_missing"],
                        "health_snapshot_id": 7,
                        "contract_snapshot_id": None,
                    }
                ],
            },
        )
        assert failed is not None

        app = create_app(
            Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
        )

        with TestClient(app) as client:
            detail_response = client.get(f"/api/embedding/jobs/{created.job.job_id}")
            page_response = client.get(f"/admin/embedding-jobs?job_id={created.job.job_id}")

        assert detail_response.status_code == 200
        gate = detail_response.json()["job"]["provider_route_readiness_gate"]
        assert gate["gate"] == "blocked_all_routes"
        assert gate["blocked_count"] == 1
        assert gate["blocked_routes"][0]["provider_name"] == "gpu-blocked"
        assert gate["blocked_routes"][0]["status"] == "needs_contract"
        assert page_response.status_code == 200
        assert "Provider Route Readiness Gate" in page_response.text
        assert "EMBEDDING_PROVIDER_ROUTE_NOT_READY" in page_response.text
        assert "gpu-blocked" in page_response.text
        assert "contract_snapshot_missing" in page_response.text
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
