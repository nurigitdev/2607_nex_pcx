from dataclasses import replace
from datetime import UTC, datetime

from app.core.embedding_jobs import EmbeddingJobRecord
from app.core.embedding_providers import (
    EmbeddingProviderRuntimeConfig,
    InvalidEmbeddingProviderError,
)
from app.core.embedding_worker import (
    ERROR_CODE_EMBEDDING_PROVIDER_ERROR,
    ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE,
    process_next_embedding_job_with_provider_routes,
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


def test_route_aware_embedding_worker_returns_idle_when_queue_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: None,
    )

    result = process_next_embedding_job_with_provider_routes(
        "postgresql://example/db",
        worker_name="unit-route-worker",
        fallback_runtime_config=EmbeddingProviderRuntimeConfig(mode="mock"),
    )

    assert result.processed is False
    assert result.job is None
    assert result.message == "No pending embedding job is available"


def test_route_aware_embedding_worker_marks_job_failed_when_provider_build_fails(
    monkeypatch,
) -> None:
    job = make_job()
    captured_config = None
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: job,
    )

    def fake_provider_builder(runtime_config: EmbeddingProviderRuntimeConfig):
        nonlocal captured_config
        captured_config = runtime_config
        raise InvalidEmbeddingProviderError("provider unavailable")

    def fake_mark_failed(*args, error_code: str, error_message: str, **kwargs):
        return replace(
            job,
            status="failed",
            error_code=error_code,
            error_message=error_message,
        )

    monkeypatch.setattr("app.core.embedding_worker.mark_embedding_job_failed", fake_mark_failed)

    result = process_next_embedding_job_with_provider_routes(
        "postgresql://example/db",
        worker_name="unit-route-worker",
        fallback_runtime_config=EmbeddingProviderRuntimeConfig(
            mode="remote",
            remote_base_url="http://provider.local/",
            remote_timeout_seconds=5.0,
        ),
        provider_builder=fake_provider_builder,
        route_selector=lambda _database_url, _profile_name: None,
    )

    assert captured_config == EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://provider.local",
        remote_timeout_seconds=5.0,
    )
    assert result.processed is True
    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_EMBEDDING_PROVIDER_ERROR
    assert result.message == "provider unavailable"
