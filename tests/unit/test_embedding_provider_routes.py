from datetime import UTC, datetime

import pytest

from app.core.embedding_provider_route_readiness import EmbeddingProviderRouteReadinessItem
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteInput,
    EmbeddingProviderRouteRecord,
    InvalidEmbeddingProviderRouteError,
    get_embedding_provider_route,
    validate_embedding_provider_route_input,
)
from app.main import embedding_provider_route_readiness_recovery_action


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
    *,
    ready: bool,
    status: str,
) -> EmbeddingProviderRouteReadinessItem:
    return EmbeddingProviderRouteReadinessItem(
        route=make_route(is_active=status != "inactive"),
        ready=ready,
        status=status,
        reasons=() if ready else (f"{status}_reason",),
        latest_health_snapshot=None,
        latest_contract_snapshot=None,
    )


def test_embedding_provider_route_input_normalizes_remote_route() -> None:
    route_input = validate_embedding_provider_route_input(
        EmbeddingProviderRouteInput(
            profile_name=" kure_v1_1024 ",
            provider_name=" gpu-a ",
            provider_mode=" REMOTE ",
            provider_base_url="http://provider.local/",
            timeout_seconds=5.5,
            priority=10,
            runtime_metadata={"device": "cuda:0"},
        )
    )

    assert route_input.profile_name == "kure_v1_1024"
    assert route_input.provider_name == "gpu-a"
    assert route_input.provider_mode == "remote"
    assert route_input.provider_base_url == "http://provider.local"
    assert route_input.runtime_metadata == {"device": "cuda:0"}


@pytest.mark.parametrize(
    ("route_input", "message"),
    [
        (EmbeddingProviderRouteInput(profile_name="", provider_name="gpu"), "profile_name"),
        (EmbeddingProviderRouteInput(profile_name="p", provider_name=""), "provider_name"),
        (
            EmbeddingProviderRouteInput(
                profile_name="p",
                provider_name="gpu",
                provider_mode="local",
            ),
            "Unsupported provider_mode",
        ),
        (
            EmbeddingProviderRouteInput(profile_name="p", provider_name="gpu"),
            "provider_base_url",
        ),
        (
            EmbeddingProviderRouteInput(
                profile_name="p",
                provider_name="gpu",
                provider_base_url="http://p",
                timeout_seconds=0,
            ),
            "timeout_seconds",
        ),
        (
            EmbeddingProviderRouteInput(
                profile_name="p",
                provider_name="gpu",
                provider_base_url="http://p",
                priority=-1,
            ),
            "priority",
        ),
    ],
)
def test_embedding_provider_route_input_rejects_invalid_values(
    route_input: EmbeddingProviderRouteInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteError, match=message):
        validate_embedding_provider_route_input(route_input)


def test_get_embedding_provider_route_rejects_invalid_route_id() -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteError, match="route_id"):
        get_embedding_provider_route("postgresql://example.invalid/db", 0)


@pytest.mark.parametrize(
    ("ready", "status", "expected_action"),
    [
        (True, "ready", "ready_for_worker"),
        (False, "inactive", "activate_route"),
        (False, "needs_contract", "run_preflight"),
        (False, "contract_failed", "review_contract_snapshot"),
        (False, "health_not_ready", "check_provider_health"),
        (False, "readiness_unknown", "run_preflight"),
    ],
)
def test_embedding_provider_route_readiness_recovery_action_maps_status(
    ready: bool,
    status: str,
    expected_action: str,
) -> None:
    item = make_readiness_item(ready=ready, status=status)

    assert embedding_provider_route_readiness_recovery_action(item) == expected_action
