from dataclasses import replace
from datetime import UTC, datetime

from app.core.embedding_jobs import EmbeddingJobRecord
from app.core.embedding_provider_route_readiness import (
    EmbeddingProviderRouteReadinessItem,
    EmbeddingProviderRouteReadinessSummary,
)
from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord
from app.core.embedding_providers import (
    EmbeddingProviderRuntimeConfig,
    InvalidEmbeddingProviderError,
)
from app.core.embedding_worker import (
    ERROR_CODE_EMBEDDING_PROVIDER_ERROR,
    ERROR_CODE_EMBEDDING_PROVIDER_ROUTE_NOT_READY,
    ERROR_CODE_EMBEDDING_PROVIDER_ROUTE_WAITING,
    ERROR_CODE_UNSUPPORTED_EMBEDDING_PROFILE,
    EMBEDDING_WORKER_BATCH_STOP_LIMIT_REACHED,
    EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY,
    EmbeddingWorkerResult,
    process_embedding_worker_batch,
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


def make_readiness_item(
    route: EmbeddingProviderRouteRecord,
    *,
    ready: bool,
    status: str,
) -> EmbeddingProviderRouteReadinessItem:
    return EmbeddingProviderRouteReadinessItem(
        route=route,
        ready=ready,
        status=status,
        reasons=() if ready else (f"{status}_reason",),
        latest_health_snapshot=None,
        latest_contract_snapshot=None,
    )


class FakeClosableProvider:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_embedding_worker_batch_stops_when_queue_is_empty() -> None:
    first_job = make_job(job_id=1, status="succeeded")
    second_job = make_job(job_id=2, status="failed")
    results = iter(
        (
            EmbeddingWorkerResult(processed=True, job=first_job, message="stored"),
            EmbeddingWorkerResult(processed=True, job=second_job, message="failed"),
            EmbeddingWorkerResult(processed=False, job=None, message="idle"),
        )
    )

    batch = process_embedding_worker_batch(lambda: next(results), limit=5)

    assert batch.stopped_reason == EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY
    assert batch.result_count == 3
    assert batch.processed_count == 2
    assert batch.succeeded_count == 1
    assert batch.failed_count == 1
    assert batch.idle_count == 1
    assert batch.job_ids == (1, 2)


def test_embedding_worker_batch_stops_at_limit_and_counts_deferred_jobs() -> None:
    first_job = make_job(job_id=1, status="succeeded")
    deferred_job = make_job(
        job_id=2,
        status="running",
        error_code=ERROR_CODE_EMBEDDING_PROVIDER_ROUTE_WAITING,
    )
    results = iter(
        (
            EmbeddingWorkerResult(processed=True, job=first_job, message="stored"),
            EmbeddingWorkerResult(processed=True, job=deferred_job, message="deferred"),
            EmbeddingWorkerResult(processed=False, job=None, message="idle"),
        )
    )

    batch = process_embedding_worker_batch(lambda: next(results), limit=2)

    assert batch.stopped_reason == EMBEDDING_WORKER_BATCH_STOP_LIMIT_REACHED
    assert batch.result_count == 2
    assert batch.processed_count == 2
    assert batch.deferred_count == 1
    assert batch.idle_count == 0
    assert batch.job_ids == (1, 2)


def test_embedding_worker_batch_rejects_invalid_limits() -> None:
    for limit in (0, 101):
        try:
            process_embedding_worker_batch(
                lambda: EmbeddingWorkerResult(processed=False, job=None),
                limit=limit,
            )
        except ValueError as exc:
            assert "limit" in str(exc)
        else:
            raise AssertionError("Expected invalid batch limit to raise ValueError")


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

    def fake_mark_failed(
        *args,
        error_code: str,
        error_message: str,
        runtime_metadata: dict[str, object] | None = None,
        **kwargs,
    ):
        return replace(
            job,
            status="failed",
            error_code=error_code,
            error_message=error_message,
            runtime_metadata=runtime_metadata or {},
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

    def fake_mark_failed(
        *args,
        error_code: str,
        error_message: str,
        runtime_metadata: dict[str, object] | None = None,
        **kwargs,
    ):
        return replace(
            job,
            status="failed",
            error_code=error_code,
            error_message=error_message,
            runtime_metadata=runtime_metadata or {},
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


def test_route_aware_embedding_worker_skips_not_ready_routes_when_gate_enabled(
    monkeypatch,
) -> None:
    job = make_job()
    blocked_route = make_route(
        route_id=30,
        provider_name="gpu-blocked",
        provider_base_url="http://gpu-blocked.local",
        priority=0,
    )
    ready_route = make_route(
        route_id=31,
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
        return FakeClosableProvider()

    def fake_process_claimed_job(*args, runtime_metadata=None, **kwargs):
        nonlocal captured_metadata
        captured_metadata = dict(runtime_metadata or {})
        return EmbeddingWorkerResult(
            processed=True,
            job=replace(job, status="succeeded", runtime_metadata=captured_metadata),
            message=kwargs["success_message"],
        )

    def fake_readiness_summary(_database_url: str, _profile_name: str):
        return EmbeddingProviderRouteReadinessSummary(
            routes=(
                make_readiness_item(blocked_route, ready=False, status="contract_failed"),
                make_readiness_item(ready_route, ready=True, status="ready"),
            )
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
            blocked_route,
            ready_route,
        ],
        require_route_readiness=True,
        route_readiness_summary_getter=fake_readiness_summary,
    )

    assert result.processed is True
    assert result.job is not None
    assert result.job.status == "succeeded"
    assert built_urls == ["http://gpu-ready.local"]
    assert captured_metadata["provider_route_id"] == ready_route.route_id
    assert captured_metadata["provider_route_readiness_status"] == "ready"
    assert captured_metadata["provider_route_readiness_ready"] is True
    assert captured_metadata["provider_route_readiness_reasons"] == []


def test_route_aware_embedding_worker_fails_when_readiness_gate_blocks_all_routes(
    monkeypatch,
) -> None:
    job = make_job()
    blocked_route = make_route(
        route_id=40,
        provider_name="gpu-blocked",
        provider_base_url="http://gpu-blocked.local",
    )
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: job,
    )

    def fake_mark_failed(
        *args,
        error_code: str,
        error_message: str,
        runtime_metadata: dict[str, object] | None = None,
        **kwargs,
    ):
        return replace(
            job,
            status="failed",
            error_code=error_code,
            error_message=error_message,
            runtime_metadata=runtime_metadata or {},
        )

    def fake_readiness_summary(_database_url: str, _profile_name: str):
        return EmbeddingProviderRouteReadinessSummary(
            routes=(make_readiness_item(blocked_route, ready=False, status="needs_contract"),)
        )

    monkeypatch.setattr("app.core.embedding_worker.mark_embedding_job_failed", fake_mark_failed)

    result = process_next_embedding_job_with_provider_routes(
        "postgresql://example/db",
        worker_name="unit-route-worker",
        fallback_runtime_config=EmbeddingProviderRuntimeConfig(mode="mock"),
        route_candidates_selector=lambda _database_url, _profile_name: [blocked_route],
        require_route_readiness=True,
        route_readiness_summary_getter=fake_readiness_summary,
    )

    assert result.processed is True
    assert result.job is not None
    assert result.job.status == "failed"
    assert result.job.error_code == ERROR_CODE_EMBEDDING_PROVIDER_ROUTE_NOT_READY
    assert "No provider route passed the readiness gate" in result.message
    assert "gpu-blocked:needs_contract" in result.message
    assert result.job.runtime_metadata["provider_route_readiness_gate"] == "blocked_all_routes"
    assert result.job.runtime_metadata["provider_route_readiness_blocked_count"] == 1
    assert result.job.runtime_metadata["provider_route_readiness_blocked_routes"] == [
        {
            "route_id": blocked_route.route_id,
            "provider_name": "gpu-blocked",
            "profile_name": blocked_route.profile_name,
            "status": "needs_contract",
            "reasons": ["needs_contract_reason"],
        }
    ]


def test_route_aware_embedding_worker_defers_when_readiness_gate_blocks_all_routes(
    monkeypatch,
) -> None:
    job = make_job()
    blocked_route = make_route(
        route_id=41,
        provider_name="gpu-warming",
        provider_base_url="http://gpu-warming.local",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "app.core.embedding_worker.claim_next_embedding_job",
        lambda *args, **kwargs: job,
    )

    def fake_defer_job(*args, **kwargs):
        captured.update(kwargs)
        return replace(
            job,
            status="running",
            lease_owner=kwargs["lease_owner"],
            error_code=kwargs["error_code"],
            error_message=kwargs["error_message"],
            runtime_metadata=kwargs["runtime_metadata"],
        )

    def fake_readiness_summary(_database_url: str, _profile_name: str):
        return EmbeddingProviderRouteReadinessSummary(
            routes=(make_readiness_item(blocked_route, ready=False, status="health_not_ready"),)
        )

    monkeypatch.setattr("app.core.embedding_worker.defer_embedding_job", fake_defer_job)

    result = process_next_embedding_job_with_provider_routes(
        "postgresql://example/db",
        worker_name="unit-route-worker",
        fallback_runtime_config=EmbeddingProviderRuntimeConfig(mode="mock"),
        route_candidates_selector=lambda _database_url, _profile_name: [blocked_route],
        require_route_readiness=True,
        readiness_gate_failure_mode="defer",
        readiness_gate_defer_seconds=45,
        route_readiness_summary_getter=fake_readiness_summary,
    )

    assert result.processed is True
    assert result.job is not None
    assert result.job.status == "running"
    assert result.job.lease_owner == "readiness-gate"
    assert result.job.error_code == ERROR_CODE_EMBEDDING_PROVIDER_ROUTE_WAITING
    assert captured["defer_seconds"] == 45
    assert result.job.runtime_metadata["provider_route_readiness_blocked_routes"][0][
        "provider_name"
    ] == "gpu-warming"


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
