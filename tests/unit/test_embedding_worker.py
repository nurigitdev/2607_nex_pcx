from dataclasses import replace
from datetime import UTC, datetime

from app.core.embedding_jobs import EmbeddingJobRecord
from app.core.embedding_worker import (
    ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE,
    process_next_mock_embedding_job,
)


def make_job(**overrides) -> EmbeddingJobRecord:
    now = datetime(2026, 7, 6, tzinfo=UTC)
    values = {
        "job_id": 1,
        "chunk_id": 10,
        "profile_name": "kure_v1_1024",
        "status": "running",
        "attempts": 1,
        "max_attempts": 3,
        "lease_owner": "unit-worker",
        "lease_expires_at": now,
        "error_code": None,
        "error_message": None,
        "last_error_at": None,
        "runtime_metadata": {},
        "created_at": now,
        "started_at": now,
        "finished_at": None,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingJobRecord(**values)


def test_process_next_mock_embedding_job_returns_idle_when_queue_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: None,
    )

    result = process_next_mock_embedding_job(
        "postgresql://example/db",
        worker_name="unit-worker",
    )

    assert result.processed is False
    assert result.job is None
    assert result.message == "No pending embedding job is available"


def test_process_next_mock_embedding_job_fails_unsupported_profile(monkeypatch) -> None:
    job = make_job(profile_name="unknown_profile")
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: job,
    )

    def fake_mark_failed(*args, error_code: str, error_message: str, **kwargs):
        return replace(
            job,
            status="failed",
            error_code=error_code,
            error_message=error_message,
        )

    monkeypatch.setattr("app.core.embedding_worker.mark_embedding_job_failed", fake_mark_failed)

    result = process_next_mock_embedding_job(
        "postgresql://example/db",
        worker_name="unit-worker",
    )

    assert result.processed is True
    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE
    assert "Unsupported embedding profile" in result.message
