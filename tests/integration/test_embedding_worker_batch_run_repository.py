from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.core.database import connect, fetch_one
from app.core.embedding_worker_batch_runs import (
    EmbeddingWorkerBatchRunInput,
    get_embedding_worker_batch_run,
    list_embedding_worker_batch_runs,
    record_embedding_worker_batch_run,
)

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


def test_embedding_worker_batch_runs_table_exists(
    migrated_database_url: str,
) -> None:
    table_name = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.embedding_worker_batch_runs'
        ) AS table_name
        """,
    )

    assert table_name["table_name"] == "embedding_worker_batch_runs"


def test_embedding_worker_batch_run_repository_records_and_filters_history(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    worker_name = f"embedding-worker-{suffix}"
    profile_name = "kure_v1_1024"
    started_at = datetime(2026, 7, 11, 14, 30, tzinfo=UTC)
    batch_run_ids: list[int] = []

    try:
        first_run = record_embedding_worker_batch_run(
            migrated_database_url,
            EmbeddingWorkerBatchRunInput(
                worker_name=worker_name,
                profile_name=profile_name,
                provider_source="route",
                provider_mode="mock",
                remote_provider_url=None,
                require_route_readiness=True,
                readiness_gate_failure_mode="defer",
                readiness_gate_defer_seconds=90,
                limit_requested=5,
                result_count=3,
                processed_count=2,
                succeeded_count=1,
                failed_count=0,
                deferred_count=1,
                idle_count=1,
                stopped_reason="queue_empty",
                job_ids=(101, 102),
                runtime_metadata={
                    "script": "process_embedding_job.py",
                    "results": [{"job_id": 101, "status": "succeeded"}],
                },
                started_at=started_at,
                completed_at=started_at + timedelta(seconds=2),
            ),
        )
        second_run = record_embedding_worker_batch_run(
            migrated_database_url,
            EmbeddingWorkerBatchRunInput(
                worker_name=f"{worker_name}-other",
                profile_name=None,
                provider_source="runtime",
                provider_mode="remote",
                remote_provider_url="http://provider.local",
                limit_requested=2,
                result_count=2,
                processed_count=2,
                succeeded_count=2,
                failed_count=0,
                deferred_count=0,
                idle_count=0,
                stopped_reason="limit_reached",
                job_ids=(201, 202),
                elapsed_ms=1500,
                started_at=started_at + timedelta(minutes=5),
                completed_at=started_at + timedelta(minutes=5, seconds=1),
            ),
        )
        batch_run_ids.extend([first_run.batch_run_id, second_run.batch_run_id])

        latest_runs = list_embedding_worker_batch_runs(
            migrated_database_url,
            limit=200,
        )
        worker_runs = list_embedding_worker_batch_runs(
            migrated_database_url,
            worker_name=worker_name,
            limit=10,
        )
        profile_runs = list_embedding_worker_batch_runs(
            migrated_database_url,
            profile_name=profile_name,
            limit=10,
        )
        fetched = get_embedding_worker_batch_run(
            migrated_database_url,
            first_run.batch_run_id,
        )

        assert first_run.batch_run_id in [run.batch_run_id for run in latest_runs]
        assert second_run.batch_run_id in [run.batch_run_id for run in latest_runs]
        assert [run.batch_run_id for run in worker_runs] == [first_run.batch_run_id]
        assert first_run.batch_run_id in [run.batch_run_id for run in profile_runs]
        assert fetched == first_run
        assert first_run.elapsed_ms == 2000
        assert first_run.processed_count == 2
        assert first_run.deferred_count == 1
        assert first_run.job_ids == (101, 102)
        assert first_run.runtime_metadata["script"] == "process_embedding_job.py"
        assert second_run.remote_provider_url == "http://provider.local"
        assert second_run.elapsed_ms == 1500
    finally:
        _cleanup_batch_runs(migrated_database_url, batch_run_ids)
