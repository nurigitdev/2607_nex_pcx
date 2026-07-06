import pytest

from app.core.pipeline_jobs import (
    InvalidPipelineJobError,
    PipelineJobInput,
    claim_next_pipeline_job_in_connection,
    mark_pipeline_failed_in_connection,
    record_pipeline_event_in_connection,
    update_pipeline_progress_in_connection,
    validate_pipeline_job_input,
)


def test_validate_pipeline_job_input_accepts_minimal_job() -> None:
    validate_pipeline_job_input(PipelineJobInput(job_type="document_ingestion"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"job_type": "unknown"}, "Unsupported pipeline job type"),
        ({"file_id": 0}, "file_id"),
        ({"document_id": -1}, "document_id"),
        ({"parent_job_id": 0}, "parent_job_id"),
        ({"requested_by_user_id": -1}, "requested_by_user_id"),
        ({"priority": -1}, "priority"),
        ({"total_units": -1}, "total_units"),
    ],
)
def test_validate_pipeline_job_input_rejects_invalid_values(
    overrides,
    message: str,
) -> None:
    values = {"job_type": "document_ingestion"}
    values.update(overrides)

    with pytest.raises(InvalidPipelineJobError, match=message):
        validate_pipeline_job_input(PipelineJobInput(**values))


def test_record_pipeline_event_rejects_invalid_stage_before_db() -> None:
    with pytest.raises(InvalidPipelineJobError, match="Unsupported pipeline stage"):
        record_pipeline_event_in_connection(
            None,  # type: ignore[arg-type]
            1,
            "created",
            stage="unknown",
        )


def test_record_pipeline_event_rejects_invalid_status_before_db() -> None:
    with pytest.raises(InvalidPipelineJobError, match="Unsupported pipeline job status"):
        record_pipeline_event_in_connection(
            None,  # type: ignore[arg-type]
            1,
            "created",
            status="unknown",
        )


def test_record_pipeline_event_rejects_invalid_event_type_before_db() -> None:
    with pytest.raises(InvalidPipelineJobError, match="Unsupported pipeline event type"):
        record_pipeline_event_in_connection(
            None,  # type: ignore[arg-type]
            1,
            "unknown",
        )


def test_claim_pipeline_job_rejects_blank_worker_before_db() -> None:
    with pytest.raises(InvalidPipelineJobError, match="worker_name"):
        claim_next_pipeline_job_in_connection(
            None,  # type: ignore[arg-type]
            " ",
        )


def test_claim_pipeline_job_rejects_non_positive_lease_before_db() -> None:
    with pytest.raises(InvalidPipelineJobError, match="lease_seconds"):
        claim_next_pipeline_job_in_connection(
            None,  # type: ignore[arg-type]
            "worker-one",
            lease_seconds=0,
        )


def test_update_pipeline_progress_rejects_invalid_counts_before_db() -> None:
    with pytest.raises(InvalidPipelineJobError, match="processed_units"):
        update_pipeline_progress_in_connection(
            None,  # type: ignore[arg-type]
            1,
            processed_units=-1,
        )


def test_mark_pipeline_failed_requires_error_details_before_db() -> None:
    with pytest.raises(InvalidPipelineJobError, match="error_code"):
        mark_pipeline_failed_in_connection(
            None,  # type: ignore[arg-type]
            1,
            error_code=" ",
            error_message="failed",
        )

    with pytest.raises(InvalidPipelineJobError, match="error_message"):
        mark_pipeline_failed_in_connection(
            None,  # type: ignore[arg-type]
            1,
            error_code="PARSER_ERROR",
            error_message=" ",
        )
