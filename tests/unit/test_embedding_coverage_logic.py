from decimal import Decimal

from app.core.embedding_coverage import (
    _coverage_percent,
    _overall_coverage_percent,
    _status_for_cell,
)


def test_embedding_coverage_status_for_cell_variants() -> None:
    assert (
        _status_for_cell(
            chunk_count=0,
            job_count=0,
            pending_count=0,
            running_count=0,
            failed_count=0,
            succeeded_job_count=0,
            skipped_count=0,
            embedded_chunk_count=0,
        )
        == "not_chunked"
    )
    assert (
        _status_for_cell(
            chunk_count=2,
            job_count=2,
            pending_count=0,
            running_count=0,
            failed_count=0,
            succeeded_job_count=2,
            skipped_count=0,
            embedded_chunk_count=2,
        )
        == "complete"
    )
    assert (
        _status_for_cell(
            chunk_count=2,
            job_count=1,
            pending_count=0,
            running_count=1,
            failed_count=0,
            succeeded_job_count=0,
            skipped_count=0,
            embedded_chunk_count=0,
        )
        == "running"
    )
    assert (
        _status_for_cell(
            chunk_count=2,
            job_count=1,
            pending_count=0,
            running_count=0,
            failed_count=1,
            succeeded_job_count=0,
            skipped_count=0,
            embedded_chunk_count=0,
        )
        == "failed"
    )
    assert (
        _status_for_cell(
            chunk_count=2,
            job_count=1,
            pending_count=1,
            running_count=0,
            failed_count=0,
            succeeded_job_count=0,
            skipped_count=0,
            embedded_chunk_count=0,
        )
        == "pending"
    )
    assert (
        _status_for_cell(
            chunk_count=2,
            job_count=1,
            pending_count=0,
            running_count=0,
            failed_count=0,
            succeeded_job_count=1,
            skipped_count=0,
            embedded_chunk_count=1,
        )
        == "partial"
    )
    assert (
        _status_for_cell(
            chunk_count=2,
            job_count=1,
            pending_count=0,
            running_count=0,
            failed_count=0,
            succeeded_job_count=0,
            skipped_count=1,
            embedded_chunk_count=0,
        )
        == "skipped"
    )
    assert (
        _status_for_cell(
            chunk_count=2,
            job_count=0,
            pending_count=0,
            running_count=0,
            failed_count=0,
            succeeded_job_count=0,
            skipped_count=0,
            embedded_chunk_count=0,
        )
        == "missing"
    )


def test_embedding_coverage_percent_helpers_handle_zero_denominators() -> None:
    assert _coverage_percent(embedded_chunk_count=0, chunk_count=0) == Decimal("0.00")
    assert _coverage_percent(embedded_chunk_count=1, chunk_count=4) == Decimal("25.00")
    assert _overall_coverage_percent(
        embedded_chunk_count=0,
        expected_embedding_count=0,
    ) == Decimal("0.00")
    assert _overall_coverage_percent(
        embedded_chunk_count=3,
        expected_embedding_count=4,
    ) == Decimal("75.00")
