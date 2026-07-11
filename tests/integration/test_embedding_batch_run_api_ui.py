from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.chunks import ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    create_embedding_job,
    get_embedding_job,
    mark_embedding_job_failed,
)
from app.core.embedding_worker_batch_runs import (
    EmbeddingWorkerBatchRunInput,
    record_embedding_worker_batch_run,
)
from app.core.file_metadata import FileMetadataInput, create_file_metadata
from app.main import create_app

pytestmark = pytest.mark.integration


def _cleanup_batch_runs(database_url: str, batch_run_ids: list[int]) -> None:
    if not batch_run_ids:
        return
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_worker_batch_runs WHERE batch_run_id = ANY(%s)",
                (batch_run_ids,),
            )


def _cleanup_file(database_url: str, checksum: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM files WHERE sha256_checksum = %s",
                (checksum,),
            )


def test_embedding_batch_run_api_and_ui(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    worker_name = f"ui-batch-worker-{suffix}"
    started_at = datetime(2026, 7, 11, 16, 0, tzinfo=UTC)
    batch_run_ids: list[int] = []

    try:
        run = record_embedding_worker_batch_run(
            migrated_database_url,
            EmbeddingWorkerBatchRunInput(
                worker_name=worker_name,
                profile_name="kure_v1_1024",
                provider_source="route",
                provider_mode="mock",
                require_route_readiness=True,
                readiness_gate_failure_mode="defer",
                readiness_gate_defer_seconds=60,
                limit_requested=3,
                result_count=2,
                processed_count=1,
                succeeded_count=1,
                failed_count=0,
                deferred_count=0,
                idle_count=1,
                stopped_reason="queue_empty",
                job_ids=(91001,),
                runtime_metadata={
                    "script": "process_embedding_job.py",
                    "provider": {"source": "route", "mode": "mock"},
                    "results": [
                        {
                            "provider_source": "route",
                            "provider_mode": "mock",
                            "processed": True,
                            "job_id": 91001,
                            "chunk_id": 81001,
                            "profile_name": "kure_v1_1024",
                            "status": "succeeded",
                            "elapsed_ms": 7,
                            "message": "Mock embedding stored",
                        },
                        {
                            "provider_source": "route",
                            "provider_mode": "mock",
                            "processed": False,
                            "job_id": None,
                            "message": "No pending embedding job is available",
                        },
                    ],
                },
                elapsed_ms=15,
                started_at=started_at,
                completed_at=started_at + timedelta(milliseconds=15),
            ),
        )
        batch_run_ids.append(run.batch_run_id)
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            list_response = client.get(
                "/api/admin/embedding-batch-runs",
                params={
                    "worker_name": worker_name,
                    "profile_name": "kure_v1_1024",
                    "stopped_reason": "queue_empty",
                    "limit": 20,
                },
            )
            detail_response = client.get(
                f"/api/admin/embedding-batch-runs/{run.batch_run_id}"
            )
            throughput_response = client.get(
                "/api/admin/embedding-batch-runs/throughput-summary",
                params={
                    "worker_name": worker_name,
                    "profile_name": "kure_v1_1024",
                    "limit": 20,
                },
            )
            page_response = client.get(
                "/admin/embedding-batch-runs",
                params={
                    "batch_run_id": run.batch_run_id,
                    "worker_name": worker_name,
                    "profile_name": "kure_v1_1024",
                    "limit": 20,
                },
            )
            invalid_response = client.get(
                "/api/admin/embedding-batch-runs",
                params={"stopped_reason": "stopped"},
            )
            missing_response = client.get("/api/admin/embedding-batch-runs/999999999")

        list_payload = list_response.json()
        detail_payload = detail_response.json()
        throughput_payload = throughput_response.json()

        assert list_response.status_code == 200
        assert list_payload["batch_run_count"] == 1
        assert list_payload["summary"]["processed_count"] == 1
        assert list_payload["batch_runs"][0]["batch_run_id"] == run.batch_run_id
        assert list_payload["batch_runs"][0]["stopped_reason"] == "queue_empty"
        assert detail_response.status_code == 200
        assert detail_payload["batch_run"]["worker_name"] == worker_name
        assert detail_payload["batch_run"]["runtime_metadata"]["results"][0]["job_id"] == 91001
        assert throughput_response.status_code == 200
        assert throughput_payload["batch_run_count"] == 1
        assert throughput_payload["throughput"]["overall"]["run_count"] == 1
        assert throughput_payload["throughput"]["overall"]["processed_count"] == 1
        assert throughput_payload["throughput"]["overall"]["success_rate_pct"] == 100
        assert throughput_payload["throughput"]["overall"]["throughput_per_second"] == 66.67
        assert throughput_payload["throughput"]["groups"][0]["profile_name"] == "kure_v1_1024"
        assert throughput_payload["throughput"]["groups"][0]["provider_source"] == "route"
        assert page_response.status_code == 200
        assert "임베딩 Batch 실행 이력" in page_response.text
        assert "data-embedding-batch-runs-page" in page_response.text
        assert "/api/admin/embedding-batch-runs" in page_response.text
        assert "Embedding Throughput Trend" in page_response.text
        assert "/api/admin/embedding-batch-runs/throughput-summary" in page_response.text
        assert "Jobs/sec" in page_response.text
        assert "66.67" in page_response.text
        assert worker_name in page_response.text
        assert f"#{run.batch_run_id}" in page_response.text
        assert "Mock embedding stored" in page_response.text
        assert "/admin/embedding-jobs?job_id=91001" in page_response.text
        assert "batch-run-json-viewer" in page_response.text
        assert invalid_response.status_code == 400
        assert "Unsupported stopped_reason" in invalid_response.json()["detail"]
        assert missing_response.status_code == 404
    finally:
        _cleanup_batch_runs(migrated_database_url, batch_run_ids)


def test_embedding_batch_run_failed_jobs_can_retry_from_api_and_ui(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    checksum = f"slice-144-{suffix}"
    started_at = datetime(2026, 7, 11, 17, 0, tzinfo=UTC)
    batch_run_ids: list[int] = []

    try:
        created_file = create_file_metadata(
            migrated_database_url,
            FileMetadataInput(
                original_file_name=f"batch-retry-{suffix}.md",
                stored_file_name=f"batch-retry-{suffix}.stored.md",
                file_size_bytes=128,
                sha256_checksum=checksum,
                storage_path=f"/tmp/nex_pcx/batch-retry-{suffix}.stored.md",
                mime_type="text/markdown",
                document_group="slice-144",
                security_level="internal",
                uploaded_by="integration-test",
                document_title=f"Batch Retry {suffix}",
            ),
        )
        document_id = created_file.file.document_id
        assert document_id is not None
        chunk = create_chunk(
            migrated_database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=0,
                chunk_text="Batch retry failed embedding job chunk.",
                token_count=6,
            ),
        )
        created_job = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk.chunk_id, profile_name="kure_v1_1024"),
        )
        failed_job = mark_embedding_job_failed(
            migrated_database_url,
            created_job.job.job_id,
            error_code="SLICE_144_TEST",
            error_message="batch retry fixture failed",
        )
        assert failed_job is not None

        run = record_embedding_worker_batch_run(
            migrated_database_url,
            EmbeddingWorkerBatchRunInput(
                worker_name=f"retry-batch-worker-{suffix}",
                profile_name="kure_v1_1024",
                provider_source="route",
                provider_mode="mock",
                limit_requested=1,
                result_count=1,
                processed_count=1,
                succeeded_count=0,
                failed_count=1,
                deferred_count=0,
                idle_count=0,
                stopped_reason="limit_reached",
                job_ids=(failed_job.job_id,),
                runtime_metadata={
                    "results": [
                        {
                            "provider_source": "route",
                            "provider_mode": "mock",
                            "processed": True,
                            "job_id": failed_job.job_id,
                            "chunk_id": chunk.chunk_id,
                            "profile_name": "kure_v1_1024",
                            "status": "failed",
                            "elapsed_ms": 9,
                            "message": "batch retry fixture failed",
                        }
                    ]
                },
                elapsed_ms=9,
                started_at=started_at,
                completed_at=started_at + timedelta(milliseconds=9),
            ),
        )
        batch_run_ids.append(run.batch_run_id)
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            page_response = client.get(
                f"/admin/embedding-batch-runs?batch_run_id={run.batch_run_id}"
            )
            retry_response = client.post(
                f"/api/admin/embedding-batch-runs/{run.batch_run_id}/retry-failed"
            )
            second_retry_response = client.post(
                f"/api/admin/embedding-batch-runs/{run.batch_run_id}/retry-failed"
            )
            missing_response = client.post(
                "/api/admin/embedding-batch-runs/999999999/retry-failed"
            )

        retried_job = get_embedding_job(migrated_database_url, failed_job.job_id)

        assert page_response.status_code == 200
        assert "실패 Job 재시도" in page_response.text
        assert (
            f"/api/admin/embedding-batch-runs/{run.batch_run_id}/retry-failed"
            in page_response.text
        )
        assert retry_response.status_code == 200
        assert retry_response.json()["batch_run_id"] == run.batch_run_id
        assert retry_response.json()["failed_job_ids"] == [failed_job.job_id]
        assert retry_response.json()["retried_count"] == 1
        assert retry_response.json()["skipped_count"] == 0
        assert retry_response.json()["retried_jobs"][0]["status"] == "pending"
        assert retried_job is not None
        assert retried_job.status == "pending"
        assert retried_job.error_message is None
        assert second_retry_response.status_code == 200
        assert second_retry_response.json()["retried_count"] == 0
        assert second_retry_response.json()["skipped_count"] == 1
        assert second_retry_response.json()["skipped_jobs"][0]["reason"] == "not_failed"
        assert missing_response.status_code == 404
    finally:
        _cleanup_batch_runs(migrated_database_url, batch_run_ids)
        _cleanup_file(migrated_database_url, checksum)
