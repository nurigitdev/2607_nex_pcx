import pytest

from app.core.embedding_jobs import (
    EmbeddingJobInput,
    InvalidEmbeddingJobError,
    claim_next_embedding_job_in_connection,
    get_embedding_job_in_connection,
    heartbeat_embedding_job_in_connection,
    list_embedding_jobs,
    mark_embedding_job_failed_in_connection,
    mark_embedding_job_skipped_in_connection,
    retry_embedding_job_in_connection,
    validate_embedding_job_input,
)


def test_validate_embedding_job_input_accepts_valid_input() -> None:
    assert (
        validate_embedding_job_input(
            EmbeddingJobInput(
                chunk_id=1,
                profile_name="kure_v1_1024",
                runtime_metadata={"source": "unit"},
            )
        )
        == "kure_v1_1024"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_id": 0}, "chunk_id"),
        ({"profile_name": " "}, "profile_name"),
        ({"max_attempts": 0}, "max_attempts"),
    ],
)
def test_validate_embedding_job_input_rejects_invalid_values(overrides, message: str) -> None:
    values = {"chunk_id": 1, "profile_name": "kure_v1_1024"}
    values.update(overrides)

    with pytest.raises(InvalidEmbeddingJobError, match=message):
        validate_embedding_job_input(EmbeddingJobInput(**values))


def test_get_embedding_job_rejects_non_positive_id_before_db() -> None:
    with pytest.raises(InvalidEmbeddingJobError, match="job_id"):
        get_embedding_job_in_connection(None, 0)  # type: ignore[arg-type]


def test_claim_embedding_job_validates_worker_and_lease_before_db() -> None:
    with pytest.raises(InvalidEmbeddingJobError, match="worker_name"):
        claim_next_embedding_job_in_connection(None, " ")  # type: ignore[arg-type]

    with pytest.raises(InvalidEmbeddingJobError, match="lease_seconds"):
        claim_next_embedding_job_in_connection(
            None,  # type: ignore[arg-type]
            "worker-one",
            lease_seconds=0,
        )


def test_heartbeat_embedding_job_validates_inputs_before_db() -> None:
    with pytest.raises(InvalidEmbeddingJobError, match="job_id"):
        heartbeat_embedding_job_in_connection(
            None,  # type: ignore[arg-type]
            0,
            "worker-one",
        )

    with pytest.raises(InvalidEmbeddingJobError, match="worker_name"):
        heartbeat_embedding_job_in_connection(
            None,  # type: ignore[arg-type]
            1,
            " ",
        )


def test_mark_embedding_job_failed_requires_error_details_before_db() -> None:
    with pytest.raises(InvalidEmbeddingJobError, match="error_code"):
        mark_embedding_job_failed_in_connection(
            None,  # type: ignore[arg-type]
            1,
            error_code=" ",
            error_message="failed",
        )

    with pytest.raises(InvalidEmbeddingJobError, match="error_message"):
        mark_embedding_job_failed_in_connection(
            None,  # type: ignore[arg-type]
            1,
            error_code="EMBEDDING_ERROR",
            error_message=" ",
        )


def test_mark_embedding_job_skipped_rejects_blank_reason_before_db() -> None:
    with pytest.raises(InvalidEmbeddingJobError, match="reason"):
        mark_embedding_job_skipped_in_connection(
            None,  # type: ignore[arg-type]
            1,
            reason=" ",
        )


def test_retry_embedding_job_rejects_non_positive_id_before_db() -> None:
    with pytest.raises(InvalidEmbeddingJobError, match="job_id"):
        retry_embedding_job_in_connection(None, 0)  # type: ignore[arg-type]


def test_list_embedding_jobs_validates_filters_before_db() -> None:
    with pytest.raises(InvalidEmbeddingJobError, match="chunk_id"):
        list_embedding_jobs("postgresql://example/db", chunk_id=0)

    with pytest.raises(InvalidEmbeddingJobError, match="profile_name"):
        list_embedding_jobs("postgresql://example/db", profile_name=" ")

    with pytest.raises(InvalidEmbeddingJobError, match="Unsupported embedding job status"):
        list_embedding_jobs("postgresql://example/db", status="queued")

    with pytest.raises(InvalidEmbeddingJobError, match="limit"):
        list_embedding_jobs("postgresql://example/db", limit=0)
