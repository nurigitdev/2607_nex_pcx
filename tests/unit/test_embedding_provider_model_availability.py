from datetime import UTC, datetime
from pathlib import Path

from app.core.embedding_model_distribution import (
    EmbeddingModelDistribution,
    EmbeddingModelReadiness,
)
from app.core.embedding_provider_model_availability import (
    build_provider_model_availability_matrix,
)
from app.core.embedding_provider_route_readiness import (
    EmbeddingProviderRouteReadinessItem,
    EmbeddingProviderRouteReadinessSummary,
)
from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord


def make_model_readiness(
    *,
    model_key: str = "kure_v1",
    profile_names: tuple[str, ...] = ("kure_v1_1024",),
    exists: bool = True,
    ready: bool = True,
) -> EmbeddingModelReadiness:
    distribution = EmbeddingModelDistribution(
        model_key=model_key,
        repo_id=f"example/{model_key}",
        local_dir_name=model_key,
        profile_names=profile_names,
    )
    return EmbeddingModelReadiness(
        distribution=distribution,
        local_dir=Path("models") / model_key,
        exists=exists,
        ready=ready,
        has_config=ready,
        has_tokenizer=False,
        has_model_weights=ready,
        file_count=2 if ready else 0,
        total_size_bytes=128 if ready else 0,
    )


def make_route_item(
    *,
    profile_name: str = "kure_v1_1024",
    provider_name: str = "gpu-a",
    ready: bool = False,
    status: str = "needs_contract",
    is_active: bool = True,
) -> EmbeddingProviderRouteReadinessItem:
    now = datetime(2026, 7, 13, tzinfo=UTC)
    route = EmbeddingProviderRouteRecord(
        route_id=1,
        profile_name=profile_name,
        provider_name=provider_name,
        provider_mode="remote",
        provider_base_url="http://gpu-a.local",
        timeout_seconds=5.0,
        priority=1,
        is_active=is_active,
        health_check_enabled=True,
        runtime_metadata={},
        created_at=now,
        updated_at=now,
    )
    return EmbeddingProviderRouteReadinessItem(
        route=route,
        ready=ready,
        status=status,
        reasons=(),
        latest_health_snapshot=None,
        latest_contract_snapshot=None,
    )


def test_provider_model_availability_matrix_summarizes_ready_profile() -> None:
    matrix = build_provider_model_availability_matrix(
        (make_model_readiness(),),
        EmbeddingProviderRouteReadinessSummary(
            routes=(make_route_item(ready=True, status="ready"),)
        ),
    )

    row = matrix.rows[0]
    assert matrix.profile_count == 1
    assert matrix.ready_count == 1
    assert row.status == "ready"
    assert row.severity == "ok"
    assert row.action_code == "ready"
    assert row.model_status == "ready"
    assert row.active_route_count == 1
    assert row.ready_route_count == 1
    assert row.provider_names == ("gpu-a",)
    assert row.route_status_counts == {"ready": 1}


def test_provider_model_availability_matrix_marks_missing_model_first() -> None:
    matrix = build_provider_model_availability_matrix(
        (make_model_readiness(exists=False, ready=False),),
        EmbeddingProviderRouteReadinessSummary(routes=(make_route_item(),)),
    )

    row = matrix.rows[0]
    assert row.status == "model_missing"
    assert row.severity == "blocked"
    assert row.action_code == "download_model"
    assert row.model_status == "model_missing"
    assert row.blocked_route_count == 1


def test_provider_model_availability_matrix_marks_route_gaps() -> None:
    matrix = build_provider_model_availability_matrix(
        (
            make_model_readiness(model_key="ready_no_route", profile_names=("missing_route",)),
            make_model_readiness(model_key="inactive", profile_names=("inactive_profile",)),
        ),
        EmbeddingProviderRouteReadinessSummary(
            routes=(
                make_route_item(
                    profile_name="inactive_profile",
                    provider_name="gpu-inactive",
                    is_active=False,
                    status="inactive",
                ),
            )
        ),
    )

    rows = {row.profile_name: row for row in matrix.rows}
    assert rows["missing_route"].status == "missing_route"
    assert rows["missing_route"].severity == "warning"
    assert rows["missing_route"].action_code == "register_route"
    assert rows["inactive_profile"].status == "route_inactive"
    assert rows["inactive_profile"].severity == "warning"
    assert rows["inactive_profile"].action_code == "activate_route"
    assert matrix.status_counts == {"missing_route": 1, "route_inactive": 1}


def test_provider_model_availability_matrix_recommends_preflight_for_unready_route() -> None:
    matrix = build_provider_model_availability_matrix(
        (make_model_readiness(),),
        EmbeddingProviderRouteReadinessSummary(
            routes=(make_route_item(ready=False, status="health_not_ready"),)
        ),
    )

    row = matrix.rows[0]
    assert row.status == "route_not_ready"
    assert row.severity == "warning"
    assert row.action_code == "run_preflight"
