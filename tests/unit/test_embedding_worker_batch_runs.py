from datetime import UTC, datetime, timedelta

import pytest

from app.core.embedding_worker_batch_runs import (
    EmbeddingWorkerBatchRunInput,
    InvalidEmbeddingWorkerBatchRunError,
    list_embedding_worker_batch_runs,
    validate_embedding_worker_batch_run_input,
)


def make_batch_run_input(**overrides) -> EmbeddingWorkerBatchRunInput:
    now = datetime(2026, 7, 11, 14, 0, tzinfo=UTC)
    values = {
        "worker_name": " batch-worker ",
        "profile_name": " kure_v1_1024 ",
        "provider_source": " ROUTE ",
        "provider_mode": " MOCK ",
        "remote_provider_url": None,
        "require_route_readiness": True,
        "readiness_gate_failure_mode": " DEFER ",
        "readiness_gate_defer_seconds": 120,
        "limit_requested": 5,
        "result_count": 3,
        "processed_count": 2,
        "succeeded_count": 1,
        "failed_count": 0,
        "deferred_count": 1,
        "idle_count": 1,
        "stopped_reason": " QUEUE_EMPTY ",
        "job_ids": (10, 11),
        "runtime_metadata": {"results": [{"job_id": 10}]},
        "elapsed_ms": 25,
        "started_at": now,
        "completed_at": now + timedelta(milliseconds=25),
    }
    values.update(overrides)
    return EmbeddingWorkerBatchRunInput(**values)


def test_validate_embedding_worker_batch_run_input_normalizes_values() -> None:
    validated = validate_embedding_worker_batch_run_input(make_batch_run_input())

    assert validated.worker_name == "batch-worker"
    assert validated.profile_name == "kure_v1_1024"
    assert validated.provider_source == "route"
    assert validated.provider_mode == "mock"
    assert validated.readiness_gate_failure_mode == "defer"
    assert validated.stopped_reason == "queue_empty"
    assert validated.job_ids == (10, 11)
    assert validated.runtime_metadata == {"results": [{"job_id": 10}]}


@pytest.mark.parametrize(
    ("run_input", "message"),
    [
        (make_batch_run_input(worker_name=" "), "worker_name is required"),
        (make_batch_run_input(provider_source="unknown"), "Unsupported provider_source"),
        (
            make_batch_run_input(readiness_gate_failure_mode="skip"),
            "Unsupported readiness_gate_failure_mode",
        ),
        (
            make_batch_run_input(readiness_gate_defer_seconds=0),
            "readiness_gate_defer_seconds",
        ),
        (make_batch_run_input(limit_requested=101), "limit_requested"),
        (make_batch_run_input(result_count=6), "result_count"),
        (make_batch_run_input(processed_count=1), "processed_count plus idle_count"),
        (
            make_batch_run_input(succeeded_count=2, failed_count=1, deferred_count=1),
            "terminal and deferred counts",
        ),
        (make_batch_run_input(job_ids=(10, 0)), "job_ids"),
        (make_batch_run_input(stopped_reason="stopped"), "Unsupported stopped_reason"),
        (make_batch_run_input(elapsed_ms=-1), "elapsed_ms"),
        (
            make_batch_run_input(
                started_at=datetime(2026, 7, 11, 14, 0, tzinfo=UTC),
                completed_at=datetime(2026, 7, 11, 13, 59, tzinfo=UTC),
            ),
            "completed_at",
        ),
    ],
)
def test_validate_embedding_worker_batch_run_input_rejects_invalid_values(
    run_input: EmbeddingWorkerBatchRunInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingWorkerBatchRunError, match=message):
        validate_embedding_worker_batch_run_input(run_input)


@pytest.mark.parametrize("limit", [0, 201])
def test_list_embedding_worker_batch_runs_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(InvalidEmbeddingWorkerBatchRunError, match="limit"):
        list_embedding_worker_batch_runs("postgresql://example/db", limit=limit)


def test_list_embedding_worker_batch_runs_rejects_invalid_stopped_reason() -> None:
    with pytest.raises(InvalidEmbeddingWorkerBatchRunError, match="Unsupported stopped_reason"):
        list_embedding_worker_batch_runs(
            "postgresql://example/db",
            stopped_reason="stopped",
        )
