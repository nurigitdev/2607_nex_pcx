from decimal import Decimal
from uuid import uuid4

import pytest
from psycopg import Connection, errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_document(database_url: str) -> tuple[int, int, int]:
    checksum = f"pipeline-job-{uuid4()}"
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
                (file_id, f"Pipeline fixture {checksum}", user_id),
            )
            document_id = cursor.fetchone()["document_id"]

    return file_id, document_id, user_id


def _create_pipeline_job(
    database_url: str,
    *,
    priority: int = 100,
    metadata: str = "{}",
) -> tuple[int, int]:
    file_id, document_id, user_id = _create_document(database_url)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO pipeline_jobs (
                    job_type,
                    file_id,
                    document_id,
                    requested_by_user_id,
                    priority,
                    metadata
                )
                VALUES (
                    'document_ingestion',
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb
                )
                RETURNING job_id
                """,
                (file_id, document_id, user_id, priority, metadata),
            )
            job_id = cursor.fetchone()["job_id"]

    return job_id, file_id


def _claim_fixture_job(connection: Connection, worker_name: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT job_id
            FROM pipeline_jobs
            WHERE status = 'queued'
              AND metadata->>'fixture' = 'skip_locked'
            ORDER BY priority ASC, queued_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """)
        job_id = cursor.fetchone()["job_id"]
        cursor.execute(
            """
            UPDATE pipeline_jobs
            SET status = 'running',
                lease_owner = %s,
                lease_expires_at = now() + interval '5 minutes',
                heartbeat_at = now(),
                attempts = attempts + 1
            WHERE job_id = %s
            RETURNING job_id
            """,
            (worker_name, job_id),
        )
        return cursor.fetchone()["job_id"]


def test_pipeline_queue_tables_and_indexes(migrated_database_url: str) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('pipeline_jobs', 'pipeline_job_events')
        """,
    )
    index_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname IN (
              'idx_pipeline_jobs_claim',
              'idx_pipeline_jobs_file',
              'idx_pipeline_jobs_document',
              'idx_pipeline_jobs_lease',
              'idx_pipeline_job_events_job'
          )
        """,
    )
    claim_index = fetch_one(
        migrated_database_url,
        """
        SELECT indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = 'idx_pipeline_jobs_claim'
        """,
    )

    assert table_count["count"] == 2
    assert index_count["count"] == 5
    assert "WHERE (status = 'queued'::text)" in claim_index["indexdef"]


def test_pipeline_job_defaults_and_event_cascade(
    migrated_database_url: str,
) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO pipeline_jobs (
                    job_type,
                    file_id,
                    document_id,
                    requested_by_user_id
                )
                VALUES ('document_ingestion', %s, %s, %s)
                RETURNING
                    job_id,
                    status,
                    stage,
                    priority,
                    total_units,
                    processed_units,
                    progress_percent,
                    attempts,
                    max_attempts,
                    metadata
                """,
                (file_id, document_id, user_id),
            )
            job_row = cursor.fetchone()
            cursor.execute(
                """
                INSERT INTO pipeline_job_events (
                    job_id,
                    event_type,
                    stage,
                    status,
                    message,
                    event_metadata
                )
                VALUES (
                    %s,
                    'created',
                    'upload_saved',
                    'queued',
                    'queued after upload',
                    '{"source": "test"}'
                )
                RETURNING event_id
                """,
                (job_row["job_id"],),
            )
            event_id = cursor.fetchone()["event_id"]
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            cursor.execute(
                """
                SELECT count(*) AS count
                FROM pipeline_jobs
                WHERE job_id = %s
                """,
                (job_row["job_id"],),
            )
            job_count = cursor.fetchone()
            cursor.execute(
                """
                SELECT count(*) AS count
                FROM pipeline_job_events
                WHERE event_id = %s
                """,
                (event_id,),
            )
            event_count = cursor.fetchone()

    assert job_row["status"] == "queued"
    assert job_row["stage"] == "upload_saved"
    assert job_row["priority"] == 100
    assert job_row["total_units"] == 0
    assert job_row["processed_units"] == 0
    assert job_row["progress_percent"] == Decimal("0.00")
    assert job_row["attempts"] == 0
    assert job_row["max_attempts"] == 3
    assert job_row["metadata"] == {}
    assert job_count["count"] == 0
    assert event_count["count"] == 0


def test_pipeline_job_check_constraints(migrated_database_url: str) -> None:
    file_id, document_id, user_id = _create_document(migrated_database_url)
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO pipeline_jobs (
                        job_type,
                        file_id,
                        document_id,
                        requested_by_user_id
                    )
                    VALUES ('unknown_stage', %s, %s, %s)
                    """,
                    (file_id, document_id, user_id),
                )
        connection.rollback()

        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO pipeline_jobs (
                        job_type,
                        file_id,
                        document_id,
                        requested_by_user_id,
                        status
                    )
                    VALUES ('document_ingestion', %s, %s, %s, 'running')
                    """,
                    (file_id, document_id, user_id),
                )
        connection.rollback()

        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO pipeline_jobs (
                        job_type,
                        file_id,
                        document_id,
                        requested_by_user_id,
                        total_units,
                        processed_units
                    )
                    VALUES ('document_ingestion', %s, %s, %s, 1, 2)
                    """,
                    (file_id, document_id, user_id),
                )
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO pipeline_jobs (
                    job_type,
                    file_id,
                    document_id,
                    requested_by_user_id
                )
                VALUES ('document_ingestion', %s, %s, %s)
                RETURNING job_id
                """,
                (file_id, document_id, user_id),
            )
            job_id = cursor.fetchone()["job_id"]
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO pipeline_job_events (
                        job_id,
                        event_type,
                        stage,
                        status
                    )
                    VALUES (%s, 'unknown_event', 'upload_saved', 'queued')
                    """,
                    (job_id,),
                )
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_pipeline_job_claim_query_skips_locked_rows(
    migrated_database_url: str,
) -> None:
    first_job_id, first_file_id = _create_pipeline_job(
        migrated_database_url,
        priority=1,
        metadata='{"fixture": "skip_locked"}',
    )
    second_job_id, second_file_id = _create_pipeline_job(
        migrated_database_url,
        priority=2,
        metadata='{"fixture": "skip_locked"}',
    )
    with connect(migrated_database_url) as first_connection:
        with connect(migrated_database_url) as second_connection:
            claimed_first_id = _claim_fixture_job(first_connection, "worker-one")
            claimed_second_id = _claim_fixture_job(second_connection, "worker-two")
            first_connection.rollback()
            second_connection.rollback()

    with connect(migrated_database_url) as cleanup_connection:
        with cleanup_connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM files WHERE file_id IN (%s, %s)",
                (first_file_id, second_file_id),
            )

    assert claimed_first_id == first_job_id
    assert claimed_second_id == second_job_id
