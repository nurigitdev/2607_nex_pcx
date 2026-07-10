from dataclasses import replace
from datetime import UTC, datetime

from app.core.embedding_jobs import EmbeddingJobRecord
from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord
from app.core.embedding_providers import (
    EmbeddingProviderRuntimeConfig,
    InvalidEmbeddingProviderError,
)
from app.core.embedding_worker import (
    ERROR_CODE_EMBEDDING_PROVIDER_ERROR,
    ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE,
    EmbeddingWorkerResult,
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


def make_route(**overrides) -> EmbeddingProviderRouteRecord:
    now = datetime(2026, 7, 6, tzinfo=UTC)
    values = {
        "route_id": 1,
        "profile_name": "kure_v1_1024",
        "provider_name": "gpu-a",
        "provider_mode": "remote",
        "provider_base_url": "http://gpu-a.local",
        "timeout_seconds": 5.0,
        "priority": 1,
        "is_active": True,
        "health_check_enabled": True,
        "runtime_metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingProviderRouteRecord(**values)


class FakeClosableProvider:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


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


def test_route_aware_embedding_worker_fails_over_to_next_route(monkeypatch) -> None:
    job = make_job()
    first_route = make_route(
        route_id=10,
        provider_name="gpu-down",
        provider_base_url="http://gpu-down.local",
        priority=0,
    )
    second_route = make_route(
        route_id=11,
        provider_name="gpu-ready",
        provider_base_url="http://gpu-ready.local",
        priority=1,
    )
    built_urls: list[str | None] = []
    captured_metadata: dict[str, object] = {}
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: job,
    )

    def fake_provider_builder(runtime_config: EmbeddingProviderRuntimeConfig):
        built_urls.append(runtime_config.remote_base_url)
        if runtime_config.remote_base_url == "http://gpu-down.local":
            raise InvalidEmbeddingProviderError("gpu-down unavailable")
        return FakeClosableProvider()

    def fake_process_claimed_job(*args, runtime_metadata=None, **kwargs):
        nonlocal captured_metadata
        captured_metadata = dict(runtime_metadata or {})
        return EmbeddingWorkerResult(
            processed=True,
            job=replace(job, status="succeeded", runtime_metadata=captured_metadata),
            message=kwargs["success_message"],
        )

    monkeypatch.setattr(
        "app.core.embedding_worker._process_claimed_embedding_job_with_provider",
        fake_process_claimed_job,
    )

    result = process_next_embedding_job_with_provider_routes(
        "postgresql://example/db",
        worker_name="unit-route-worker",
        fallback_runtime_config=EmbeddingProviderRuntimeConfig(mode="mock"),
        provider_builder=fake_provider_builder,
        route_candidates_selector=lambda _database_url, _profile_name: [
            first_route,
            second_route,
        ],
    )

    assert result.processed is True
    assert result.job is not None
    assert result.job.status == "succeeded"
    assert built_urls == ["http://gpu-down.local", "http://gpu-ready.local"]
    assert captured_metadata["provider_route_id"] == second_route.route_id
    assert captured_metadata["provider_route_failover_attempt"] == 2
    assert captured_metadata["provider_route_failover_candidate_count"] == 2
    assert captured_metadata["provider_route_failed_attempts"] == [
        {
            "route_id": first_route.route_id,
            "provider_name": first_route.provider_name,
            "provider_mode": first_route.provider_mode,
            "priority": first_route.priority,
            "error_message": "gpu-down unavailable",
        }
    ]


def test_route_aware_embedding_worker_marks_failed_when_all_routes_fail(
    monkeypatch,
) -> None:
    job = make_job()
    first_route = make_route(route_id=20, provider_name="gpu-a", priority=0)
    second_route = make_route(route_id=21, provider_name="gpu-b", priority=1)
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: job,
    )

    def fake_provider_builder(runtime_config: EmbeddingProviderRuntimeConfig):
        raise InvalidEmbeddingProviderError(f"{runtime_config.remote_base_url} unavailable")

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
        fallback_runtime_config=EmbeddingProviderRuntimeConfig(mode="mock"),
        provider_builder=fake_provider_builder,
        route_candidates_selector=lambda _database_url, _profile_name: [
            first_route,
            second_route,
        ],
    )

    assert result.processed is True
    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_EMBEDDING_PROVIDER_ERROR
    assert "All provider routes failed" in result.message
    assert "gpu-a" in result.message
    assert "gpu-b" in result.message
