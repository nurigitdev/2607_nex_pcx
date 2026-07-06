from decimal import Decimal
from uuid import uuid4

import pytest

from app.core.database import connect, fetch_one
from app.core.pipeline_jobs import (
    InvalidPipelineJobError,
    PipelineJobInput,
    claim_next_pipeline_job,
    create_pipeline_job,
    get_pipeline_job,
    heartbeat_pipeline_job,
    list_pipeline_job_events,
    list_pipeline_jobs,
    mark_pipeline_failed,
    mark_pipeline_succeeded,
    record_pipeline_event,
    retry_pipeline_job,
    update_pipeline_progress,
)

pytestmark = pytest.mark.integration


def _create_document(database_url: str) -> tuple[int, int, int]:
    checksum = f"pipeline-repository-{uuid4()}"
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
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
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
                (file_id, f"Pipeline repository fixture {checksum}", user_id),
            )
            document_id = cursor.fetchone()["document_id"]

    return file_id, document_id, user_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _event_count(database_url: str, job_id: int, event_type: str) -> int:
    row = fetch_one(
        database_url,
        """
        SELECT count(*) AS count
        FROM pipeline_job_events
        WHERE job_id = %s
          AND event_type = %s
        """,
        (job_id, event_type),
    )
    return int(row["count"])


def test_create_pipeline_job_records_created_event(
    migrated_database_url: str,
) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        job = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
                priority=5,
                total_units=4,
                metadata={"source": "repository-test"},
            ),
        )
        stored = get_pipeline_job(migrated_database_url, job.job_id)

        assert stored == job
        assert job.status == "queued"
        assert job.stage == "upload_saved"
        assert job.priority == 5
        assert job.total_units == 4
        assert job.processed_units == 0
        assert job.progress_percent == Decimal("0.00")
        assert job.metadata == {"source": "repository-test"}
        assert _event_count(migrated_database_url, job.job_id, "created") == 1
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_claim_heartbeat_progress_and_success_lifecycle(
    migrated_database_url: str,
) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        created = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
                priority=0,
                total_units=4,
            ),
        )

        claimed = claim_next_pipeline_job(
            migrated_database_url,
            "worker-one",
            lease_seconds=60,
        )
        assert claimed is not None
        assert claimed.job_id == created.job_id
        assert claimed.status == "running"
        assert claimed.lease_owner == "worker-one"
        assert claimed.attempts == 1
        assert claimed.started_at is not None
        assert claimed.lease_expires_at is not None
        assert claimed.heartbeat_at is not None

        heartbeat = heartbeat_pipeline_job(
            migrated_database_url,
            claimed.job_id,
            "worker-one",
            lease_seconds=120,
        )
        wrong_heartbeat = heartbeat_pipeline_job(
            migrated_database_url,
            claimed.job_id,
            "worker-two",
        )
        progress = update_pipeline_progress(
            migrated_database_url,
            claimed.job_id,
            processed_units=2,
            total_units=4,
            stage="parsing",
            current_message="parsed half of fixture",
        )
        succeeded = mark_pipeline_succeeded(
            migrated_database_url,
            claimed.job_id,
            message="fixture complete",
        )

        assert heartbeat is not None
        assert heartbeat.lease_owner == "worker-one"
        assert wrong_heartbeat is None
        assert progress is not None
        assert progress.stage == "parsing"
        assert progress.processed_units == 2
        assert progress.progress_percent == Decimal("50.00")
        assert progress.current_message == "parsed half of fixture"
        assert succeeded is not None
        assert succeeded.status == "succeeded"
        assert succeeded.stage == "completed"
        assert succeeded.processed_units == 4
        assert succeeded.progress_percent == Decimal("100.00")
        assert succeeded.lease_owner is None
        assert succeeded.finished_at is not None
        assert _event_count(migrated_database_url, created.job_id, "claimed") == 1
        assert _event_count(migrated_database_url, created.job_id, "heartbeat") == 1
        assert _event_count(migrated_database_url, created.job_id, "progress") == 1
        assert _event_count(migrated_database_url, created.job_id, "stage_succeeded") == 1
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_failed_job_can_be_queued_for_retry(migrated_database_url: str) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        created = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
                priority=0,
            ),
        )
        claimed = claim_next_pipeline_job(migrated_database_url, "worker-one")
        assert claimed is not None
        assert claimed.job_id == created.job_id

        failed = mark_pipeline_failed(
            migrated_database_url,
            claimed.job_id,
            error_code="PARSER_ERROR",
            error_message="parser failed",
        )
        retried = retry_pipeline_job(
            migrated_database_url,
            claimed.job_id,
            message="manual retry",
        )

        assert failed is not None
        assert failed.status == "failed"
        assert failed.error_code == "PARSER_ERROR"
        assert failed.error_message == "parser failed"
        assert failed.lease_owner is None
        assert failed.finished_at is not None
        assert retried is not None
        assert retried.status == "queued"
        assert retried.error_code is None
        assert retried.error_message is None
        assert retried.finished_at is None
        assert retried.attempts == 1
        assert _event_count(migrated_database_url, created.job_id, "failed") == 1
        assert _event_count(migrated_database_url, created.job_id, "retried") == 1
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_claim_reclaims_expired_running_job(migrated_database_url: str) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        created = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
                priority=0,
            ),
        )
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE pipeline_jobs
                    SET status = 'running',
                        lease_owner = 'stale-worker',
                        lease_expires_at = now() - interval '1 minute',
                        heartbeat_at = now() - interval '2 minutes'
                    WHERE job_id = %s
                    """,
                    (created.job_id,),
                )

        claimed = claim_next_pipeline_job(
            migrated_database_url,
            "recovery-worker",
            lease_seconds=30,
        )

        assert claimed is not None
        assert claimed.job_id == created.job_id
        assert claimed.status == "running"
        assert claimed.lease_owner == "recovery-worker"
        assert claimed.attempts == 1
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_retry_returns_none_when_job_is_not_retryable(
    migrated_database_url: str,
) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        job = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
                priority=0,
            ),
        )

        assert retry_pipeline_job(migrated_database_url, job.job_id) is None
        assert get_pipeline_job(migrated_database_url, job.job_id) == job
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_pipeline_progress_rejects_invalid_totals(
    migrated_database_url: str,
) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        job = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
            ),
        )

        with pytest.raises(InvalidPipelineJobError, match="total_units"):
            update_pipeline_progress(
                migrated_database_url,
                job.job_id,
                processed_units=0,
                total_units=-1,
            )

        with pytest.raises(InvalidPipelineJobError, match="processed_units"):
            update_pipeline_progress(
                migrated_database_url,
                job.job_id,
                processed_units=2,
                total_units=1,
            )
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_pipeline_repository_returns_none_for_missing_jobs(
    migrated_database_url: str,
) -> None:
    missing_job_id = 999_999_999

    assert get_pipeline_job(migrated_database_url, missing_job_id) is None
    assert (
        update_pipeline_progress(
            migrated_database_url,
            missing_job_id,
            processed_units=0,
            total_units=0,
        )
        is None
    )
    assert mark_pipeline_succeeded(migrated_database_url, missing_job_id) is None
    assert (
        mark_pipeline_failed(
            migrated_database_url,
            missing_job_id,
            error_code="MISSING",
            error_message="missing job",
        )
        is None
    )


def test_record_pipeline_event_wrapper_persists_metadata(
    migrated_database_url: str,
) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        job = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
            ),
        )

        event = record_pipeline_event(
            migrated_database_url,
            job.job_id,
            "stage_started",
            stage="parsing",
            status="running",
            message="manual stage event",
            event_metadata={"stage_index": 2},
        )

        assert event.job_id == job.job_id
        assert event.event_type == "stage_started"
        assert event.stage == "parsing"
        assert event.status == "running"
        assert event.message == "manual stage event"
        assert event.event_metadata == {"stage_index": 2}
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_list_pipeline_jobs_filters_status_and_includes_context(
    migrated_database_url: str,
) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    try:
        job = create_pipeline_job(
            migrated_database_url,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=file_id,
                document_id=document_id,
                requested_by_user_id=user_id,
                total_units=3,
            ),
        )
        failed = mark_pipeline_failed(
            migrated_database_url,
            job.job_id,
            error_code="MONITOR_TEST",
            error_message="monitor fixture failure",
        )

        failed_jobs = list_pipeline_jobs(migrated_database_url, status="failed", limit=20)
        events = list_pipeline_job_events(migrated_database_url, job.job_id)

        assert failed is not None
        assert any(item.job.job_id == job.job_id for item in failed_jobs)
        list_item = next(item for item in failed_jobs if item.job.job_id == job.job_id)
        assert list_item.original_file_name is not None
        assert list_item.document_title is not None
        assert list_item.requested_by_login_id == "alice.member"
        assert [event.event_type for event in events] == ["created", "failed"]
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_pipeline_job_list_rejects_invalid_filters() -> None:
    with pytest.raises(InvalidPipelineJobError, match="Unsupported pipeline job status"):
        list_pipeline_jobs("postgresql://example/db", status="unknown")

    with pytest.raises(InvalidPipelineJobError, match="limit"):
        list_pipeline_job_events("postgresql://example/db", 1, limit=0)
