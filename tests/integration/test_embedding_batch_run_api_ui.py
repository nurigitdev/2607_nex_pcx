from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_worker_batch_runs import (
    EmbeddingWorkerBatchRunInput,
    record_embedding_worker_batch_run,
)
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
            page_response = client.get(
                f"/admin/embedding-batch-runs?batch_run_id={run.batch_run_id}"
            )
            invalid_response = client.get(
                "/api/admin/embedding-batch-runs",
                params={"stopped_reason": "stopped"},
            )
            missing_response = client.get("/api/admin/embedding-batch-runs/999999999")

        list_payload = list_response.json()
        detail_payload = detail_response.json()

        assert list_response.status_code == 200
        assert list_payload["batch_run_count"] == 1
        assert list_payload["summary"]["processed_count"] == 1
        assert list_payload["batch_runs"][0]["batch_run_id"] == run.batch_run_id
        assert list_payload["batch_runs"][0]["stopped_reason"] == "queue_empty"
        assert detail_response.status_code == 200
        assert detail_payload["batch_run"]["worker_name"] == worker_name
        assert detail_payload["batch_run"]["runtime_metadata"]["results"][0]["job_id"] == 91001
        assert page_response.status_code == 200
        assert "임베딩 Batch 실행 이력" in page_response.text
        assert "data-embedding-batch-runs-page" in page_response.text
        assert "/api/admin/embedding-batch-runs" in page_response.text
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
