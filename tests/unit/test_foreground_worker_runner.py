import json
from datetime import UTC, datetime

import pytest

from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord
from app.core.foreground_worker_runner import (
    GUARD_DECISION_ALLOWED,
    GUARD_DECISION_SKIPPED,
    WORKER_RUN_STATUS_PLANNED,
    PendingEmbeddingProfileSummary,
    ProviderHealthProbe,
    build_foreground_worker_runner_evidence,
    build_provider_resource_guard_decisions,
    default_profile_token_limits,
    foreground_worker_runner_evidence_payload,
    merge_profile_token_limits,
    parse_profile_token_limits,
    probe_provider_route_health,
    render_foreground_worker_runner_markdown,
)


def test_provider_resource_guard_allows_ready_remote_route() -> None:
    pending = PendingEmbeddingProfileSummary(
        profile_name="bge_m3_1024",
        pending_count=2,
        max_token_count=610,
        max_char_count=2907,
        oldest_job_id=1,
        newest_job_id=2,
    )

    decisions = build_provider_resource_guard_decisions(
        [pending],
        [_route(profile_name="bge_m3_1024")],
        profile_token_limits={},
        health_probe=lambda route, timeout: _health(ready=True, elapsed_ms=20),
    )

    assert len(decisions) == 1
    assert decisions[0].decision == GUARD_DECISION_ALLOWED
    assert decisions[0].route_id == 1
    assert decisions[0].health is not None
    assert decisions[0].health.status == "ready"


def test_provider_resource_guard_skips_large_qwen_chunk_before_health_probe() -> None:
    pending = PendingEmbeddingProfileSummary(
        profile_name="qwen3_4b_1000",
        pending_count=1,
        max_token_count=1486,
        max_char_count=7302,
        oldest_job_id=35,
        newest_job_id=35,
    )
    called = False

    def health_probe(route, timeout):
        nonlocal called
        called = True
        return _health(ready=True)

    decisions = build_provider_resource_guard_decisions(
        [pending],
        [_route(profile_name="qwen3_4b_1000")],
        profile_token_limits=default_profile_token_limits(),
        health_probe=health_probe,
    )

    assert decisions[0].decision == GUARD_DECISION_SKIPPED
    assert "exceeds guard limit" in decisions[0].reason
    assert called is False


def test_provider_resource_guard_skips_slow_or_mock_routes() -> None:
    pending = [
        PendingEmbeddingProfileSummary(
            profile_name="kure_v1_1024",
            pending_count=1,
            max_token_count=200,
            max_char_count=900,
            oldest_job_id=1,
            newest_job_id=1,
        ),
        PendingEmbeddingProfileSummary(
            profile_name="mock_profile",
            pending_count=1,
            max_token_count=100,
            max_char_count=400,
            oldest_job_id=2,
            newest_job_id=2,
        ),
    ]

    decisions = build_provider_resource_guard_decisions(
        pending,
        [
            _route(profile_name="kure_v1_1024"),
            _route(profile_name="mock_profile", route_id=2, provider_mode="mock"),
        ],
        profile_token_limits={},
        max_health_elapsed_ms=100,
        health_probe=lambda route, timeout: _health(ready=True, elapsed_ms=250),
    )

    assert [decision.decision for decision in decisions] == [
        GUARD_DECISION_SKIPPED,
        GUARD_DECISION_SKIPPED,
    ]
    assert "latency" in decisions[0].reason
    assert "Mock provider" in decisions[1].reason


def test_provider_resource_guard_skips_missing_route_and_not_ready_health() -> None:
    pending = [
        PendingEmbeddingProfileSummary(
            profile_name="missing_profile",
            pending_count=1,
            max_token_count=200,
            max_char_count=900,
            oldest_job_id=1,
            newest_job_id=1,
        ),
        PendingEmbeddingProfileSummary(
            profile_name="bge_m3_1024",
            pending_count=1,
            max_token_count=200,
            max_char_count=900,
            oldest_job_id=2,
            newest_job_id=2,
        ),
    ]

    decisions = build_provider_resource_guard_decisions(
        pending,
        [_route(profile_name="bge_m3_1024")],
        profile_token_limits={},
        health_probe=lambda route, timeout: _health(ready=False),
    )

    assert [decision.decision for decision in decisions] == [
        GUARD_DECISION_SKIPPED,
        GUARD_DECISION_SKIPPED,
    ]
    assert "No active provider route" in decisions[0].reason
    assert "not ready" in decisions[1].reason


def test_foreground_worker_runner_evidence_payload_and_markdown() -> None:
    pending = PendingEmbeddingProfileSummary(
        profile_name="bge_m3_1024",
        pending_count=1,
        max_token_count=500,
        max_char_count=1000,
        oldest_job_id=1,
        newest_job_id=1,
    )
    decisions = build_provider_resource_guard_decisions(
        [pending],
        [_route(profile_name="bge_m3_1024")],
        profile_token_limits={},
        health_probe=lambda route, timeout: _health(ready=True, elapsed_ms=5),
    )
    from app.core.foreground_worker_runner import ForegroundWorkerRunnerPlan

    plan = ForegroundWorkerRunnerPlan(
        status="ready",
        generated_at=datetime(2026, 7, 20, 1, 2, 3, tzinfo=UTC),
        workdir="/repo",
        pipeline_limit=1,
        embedding_limit_per_profile=5,
        lease_seconds=300,
        worker_name_prefix="fg",
        health_timeout_seconds=5,
        max_health_elapsed_ms=5000,
        profile_token_limits={},
        excluded_profiles=(),
        pending_profiles=(pending,),
        guard_decisions=decisions,
    )
    evidence = build_foreground_worker_runner_evidence(
        plan,
        status=WORKER_RUN_STATUS_PLANNED,
        dry_run=True,
        generated_at=datetime(2026, 7, 20, 1, 3, 0, tzinfo=UTC),
        message="planned",
    )
    payload = foreground_worker_runner_evidence_payload(evidence)
    markdown = render_foreground_worker_runner_markdown(payload)

    assert payload["status"] == "planned"
    assert payload["plan"]["allowed_profiles"] == ["bge_m3_1024"]
    assert "Foreground Worker Runner Evidence" in markdown
    assert json.loads(json.dumps(payload))["dry_run"] is True


def test_parse_profile_token_limits_rejects_invalid_values() -> None:
    assert parse_profile_token_limits(["qwen3_4b_1000=0"]) == {}
    assert merge_profile_token_limits(["qwen3_4b_1000=0", "custom_profile=99"]) == {
        "qwen3_4b_2560": 1200,
        "custom_profile": 99,
    }
    with pytest.raises(ValueError, match="PROFILE=LIMIT"):
        parse_profile_token_limits(["qwen3_4b_1000"])
    with pytest.raises(ValueError, match="integer"):
        parse_profile_token_limits(["qwen3_4b_1000=abc"])


def test_probe_provider_route_health_detects_profile_mismatch(monkeypatch) -> None:
    import app.core.foreground_worker_runner as runner

    def fake_urlopen(url, timeout):
        assert url == "http://provider.local/healthz"
        assert timeout == 3
        return _FakeResponse(
            {
                "ready": True,
                "provider_model_id": "bge-provider",
                "provider_type": "remote",
                "model_key": "bge",
                "profile_names": ["other_profile"],
            }
        )

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    health = probe_provider_route_health(_route(profile_name="bge_m3_1024"), 3)

    assert health.ready is False
    assert health.status == "profile_mismatch"
    assert health.error_message == "Profile bge_m3_1024 is missing from provider health."


def test_probe_provider_route_health_blocks_unsupported_modes() -> None:
    health = probe_provider_route_health(
        _route(profile_name="bge_m3_1024", provider_mode="local"),
        3,
    )

    assert health.ready is False
    assert health.status == "unsupported"
    assert health.provider_type == "local"


def _route(
    *,
    profile_name: str,
    route_id: int = 1,
    provider_mode: str = "remote",
) -> EmbeddingProviderRouteRecord:
    return EmbeddingProviderRouteRecord(
        route_id=route_id,
        profile_name=profile_name,
        provider_name=f"{profile_name}-provider",
        provider_mode=provider_mode,
        provider_base_url="http://provider.local",
        timeout_seconds=30,
        priority=100,
        is_active=True,
        health_check_enabled=True,
        runtime_metadata={},
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
        updated_at=datetime(2026, 7, 20, tzinfo=UTC),
    )


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _health(*, ready: bool, elapsed_ms: int = 10) -> ProviderHealthProbe:
    return ProviderHealthProbe(
        checked=True,
        ready=ready,
        status="ready" if ready else "not_ready",
        elapsed_ms=elapsed_ms,
        provider_type="remote",
        provider_model_id="provider",
        model_key="model",
        profile_names=("bge_m3_1024", "kure_v1_1024", "qwen3_4b_1000"),
    )
