from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    claim_next_embedding_job,
    create_embedding_job,
    create_embedding_jobs_for_chunk,
    defer_embedding_job,
    get_embedding_job,
    heartbeat_embedding_job,
    list_active_embedding_profiles,
    list_embedding_jobs,
    list_stale_embedding_jobs,
    mark_embedding_job_failed,
    mark_embedding_job_skipped,
    mark_embedding_job_succeeded,
    release_stale_embedding_job_lease,
    retry_embedding_job,
)

pytestmark = pytest.mark.integration


def _create_chunk(database_url: str) -> tuple[int, int]:
    checksum = f"embedding-repository-{uuid4()}"
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
                (file_id, f"Embedding repository fixture {checksum}"),
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
                VALUES (%s, 0, 'Embedding repository chunk', %s, 'heading_512_64', 26)
                RETURNING chunk_id
                """,
                (document_id, f"chunk-{checksum}"),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id


def _create_inactive_profile(database_url: str) -> str:
    profile_name = f"test_profile_{uuid4().hex}"
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
                VALUES (%s, 'example/test-model', 3, 'vector', false)
                """,
                (profile_name,),
            )
    return profile_name


def _cleanup_fixture(database_url: str, file_id: int, profile_name: str | None = None) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            if profile_name is not None:
                cursor.execute(
                    "DELETE FROM embedding_profiles WHERE profile_name = %s",
                    (profile_name,),
                )


def test_list_active_embedding_profiles_returns_seed_profiles(
    migrated_database_url: str,
) -> None:
    profiles = list_active_embedding_profiles(migrated_database_url)

    by_name = {profile.profile_name: profile for profile in profiles}
    assert set(by_name) >= {
        "kure_v1_1024",
        "bge_m3_1024",
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    }
    assert by_name["qwen3_4b_2560"].dimension == 2560
    assert by_name["qwen3_4b_2560"].storage_type == "halfvec"
    assert by_name["kure_v1_1024"].is_active is True


def test_create_embedding_job_is_idempotent_and_listable(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    profile_name = _create_inactive_profile(migrated_database_url)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(
                chunk_id=chunk_id,
                profile_name=profile_name,
                max_attempts=5,
                runtime_metadata={"source": "repository-test"},
            ),
        )
        duplicate = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
        )
        stored = get_embedding_job(migrated_database_url, created.job.job_id)
        listed = list_embedding_jobs(
            migrated_database_url,
            chunk_id=chunk_id,
            profile_name=profile_name,
            status="pending",
        )

        assert created.created is True
        assert duplicate.created is False
        assert duplicate.job.job_id == created.job.job_id
        assert stored == created.job
        assert listed == [created.job]
        assert created.job.status == "pending"
        assert created.job.attempts == 0
        assert created.job.max_attempts == 5
        assert created.job.runtime_metadata == {"source": "repository-test"}
    finally:
        _cleanup_fixture(migrated_database_url, file_id, profile_name)


def test_create_embedding_jobs_for_chunk_uses_active_profiles(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    try:
        created = create_embedding_jobs_for_chunk(migrated_database_url, chunk_id)
        duplicate = create_embedding_jobs_for_chunk(migrated_database_url, chunk_id)
        stored_jobs = list_embedding_jobs(migrated_database_url, chunk_id=chunk_id, limit=10)

        assert len(created) == 4
        assert all(result.created for result in created)
        assert len(duplicate) == 4
        assert not any(result.created for result in duplicate)
        assert {result.job.profile_name for result in created} == {
            "kure_v1_1024",
            "bge_m3_1024",
            "qwen3_4b_1000",
            "qwen3_4b_2560",
        }
        assert {job.job_id for job in stored_jobs} == {result.job.job_id for result in created}
    finally:
        _cleanup_fixture(migrated_database_url, file_id)


def test_claim_heartbeat_and_success_lifecycle(migrated_database_url: str) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    profile_name = _create_inactive_profile(migrated_database_url)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(
                chunk_id=chunk_id,
                profile_name=profile_name,
                runtime_metadata={"adapter": "mock"},
            ),
        )

        claimed = claim_next_embedding_job(
            migrated_database_url,
            "embedding-worker-one",
            profile_name=profile_name,
            lease_seconds=60,
        )
        empty_claim = claim_next_embedding_job(
            migrated_database_url,
            "embedding-worker-two",
            profile_name=profile_name,
        )
        wrong_heartbeat = heartbeat_embedding_job(
            migrated_database_url,
            created.job.job_id,
            "other-worker",
        )
        heartbeat = heartbeat_embedding_job(
            migrated_database_url,
            created.job.job_id,
            "embedding-worker-one",
            lease_seconds=120,
        )
        succeeded = mark_embedding_job_succeeded(
            migrated_database_url,
            created.job.job_id,
            runtime_metadata={"elapsed_ms": 12},
        )

        assert claimed is not None
        assert claimed.job_id == created.job.job_id
        assert claimed.status == "running"
        assert claimed.lease_owner == "embedding-worker-one"
        assert claimed.lease_expires_at is not None
        assert claimed.started_at is not None
        assert claimed.attempts == 1
        assert empty_claim is None
        assert wrong_heartbeat is None
        assert heartbeat is not None
        assert heartbeat.lease_owner == "embedding-worker-one"
        assert succeeded is not None
        assert succeeded.status == "succeeded"
        assert succeeded.lease_owner is None
        assert succeeded.error_code is None
        assert succeeded.error_message is None
        assert succeeded.finished_at is not None
        assert succeeded.runtime_metadata == {"adapter": "mock", "elapsed_ms": 12}
    finally:
        _cleanup_fixture(migrated_database_url, file_id, profile_name)


def test_failed_job_can_retry_and_expired_running_job_can_be_reclaimed(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    profile_name = _create_inactive_profile(migrated_database_url)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
        )
        claimed = claim_next_embedding_job(
            migrated_database_url,
            "embedding-worker-one",
            profile_name=profile_name,
        )
        assert claimed is not None

        failed = mark_embedding_job_failed(
            migrated_database_url,
            claimed.job_id,
            error_code="EMBEDDING_ERROR",
            error_message="model failed",
            runtime_metadata={"provider_route_readiness_gate": "blocked_all_routes"},
        )
        retried = retry_embedding_job(migrated_database_url, claimed.job_id)

        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_jobs
                    SET status = 'running',
                        lease_owner = 'stale-worker',
                        lease_expires_at = now() - interval '1 minute'
                    WHERE job_id = %s
                    """,
                    (created.job.job_id,),
                )

        reclaimed = claim_next_embedding_job(
            migrated_database_url,
            "recovery-worker",
            profile_name=profile_name,
            lease_seconds=30,
        )

        assert failed is not None
        assert failed.status == "failed"
        assert failed.error_code == "EMBEDDING_ERROR"
        assert failed.error_message == "model failed"
        assert failed.last_error_at is not None
        assert failed.runtime_metadata["provider_route_readiness_gate"] == "blocked_all_routes"
        assert retried is not None
        assert retried.status == "pending"
        assert retried.error_code is None
        assert retried.error_message is None
        assert retried.finished_at is None
        assert reclaimed is not None
        assert reclaimed.job_id == created.job.job_id
        assert reclaimed.status == "running"
        assert reclaimed.lease_owner == "recovery-worker"
        assert reclaimed.attempts == 2
    finally:
        _cleanup_fixture(migrated_database_url, file_id, profile_name)


def test_stale_embedding_job_lease_can_be_released_to_pending(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    profile_name = _create_inactive_profile(migrated_database_url)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
        )
        claimed = claim_next_embedding_job(
            migrated_database_url,
            "lease-release-worker",
            profile_name=profile_name,
        )
        assert claimed is not None
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_jobs
                    SET lease_expires_at = now() - interval '2 minutes',
                        error_code = 'WORKER_STALE',
                        error_message = 'worker lease expired',
                        last_error_at = now()
                    WHERE job_id = %s
                    """,
                    (created.job.job_id,),
                )

        stale_jobs = list_stale_embedding_jobs(
            migrated_database_url,
            profile_name=profile_name,
            reclaimable_only=True,
        )
        released = release_stale_embedding_job_lease(
            migrated_database_url,
            created.job.job_id,
        )
        stale_after_release = list_stale_embedding_jobs(
            migrated_database_url,
            profile_name=profile_name,
        )
        second_release = release_stale_embedding_job_lease(
            migrated_database_url,
            created.job.job_id,
        )

        assert [job.job_id for job in stale_jobs] == [created.job.job_id]
        assert released is not None
        assert released.status == "pending"
        assert released.lease_owner is None
        assert released.lease_expires_at is None
        assert released.error_code is None
        assert released.error_message is None
        assert released.last_error_at is None
        assert released.attempts == 1
        assert stale_after_release == []
        assert second_release is None
    finally:
        _cleanup_fixture(migrated_database_url, file_id, profile_name)


def test_embedding_job_can_be_deferred_until_readiness_gate_recovers(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    profile_name = _create_inactive_profile(migrated_database_url)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
        )
        claimed = claim_next_embedding_job(
            migrated_database_url,
            "embedding-worker-one",
            profile_name=profile_name,
        )
        assert claimed is not None

        deferred = defer_embedding_job(
            migrated_database_url,
            claimed.job_id,
            lease_owner="readiness-gate",
            defer_seconds=300,
            error_code="EMBEDDING_PROVIDER_ROUTE_WAITING",
            error_message="No provider route passed the readiness gate",
            runtime_metadata={"provider_route_readiness_gate": "blocked_all_routes"},
        )
        unavailable = claim_next_embedding_job(
            migrated_database_url,
            "another-worker",
            profile_name=profile_name,
        )

        assert deferred is not None
        assert deferred.job_id == created.job.job_id
        assert deferred.status == "running"
        assert deferred.lease_owner == "readiness-gate"
        assert deferred.lease_expires_at is not None
        assert deferred.error_code == "EMBEDDING_PROVIDER_ROUTE_WAITING"
        assert deferred.runtime_metadata["provider_route_readiness_gate"] == "blocked_all_routes"
        assert unavailable is None
    finally:
        _cleanup_fixture(migrated_database_url, file_id, profile_name)


def test_embedding_job_can_be_skipped_and_retry_ignores_non_failed_jobs(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url)
    profile_name = _create_inactive_profile(migrated_database_url)
    try:
        created = create_embedding_job(
            migrated_database_url,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
        )
        skipped = mark_embedding_job_skipped(
            migrated_database_url,
            created.job.job_id,
            reason="profile disabled",
        )
        retried = retry_embedding_job(migrated_database_url, created.job.job_id)

        assert skipped is not None
        assert skipped.status == "skipped"
        assert skipped.error_message == "profile disabled"
        assert skipped.finished_at is not None
        assert retried is None
    finally:
        _cleanup_fixture(migrated_database_url, file_id, profile_name)
