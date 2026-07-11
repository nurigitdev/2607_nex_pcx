from datetime import UTC, datetime

from app.core.embedding_provider_contract_sample_sets import (
    EmbeddingProviderContractSampleSetRecord,
)
from app.core.embedding_provider_route_contract_snapshots import (
    EmbeddingProviderRouteContractSnapshotRecord,
)
from app.core.embedding_provider_route_contracts import EmbeddingProviderRouteContractResult
from app.core.embedding_provider_route_preflight import run_embedding_provider_route_preflight
from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord


def test_route_preflight_returns_empty_payload_when_no_routes(monkeypatch) -> None:
    import app.core.embedding_provider_route_preflight as route_preflight

    monkeypatch.setattr(
        route_preflight,
        "list_embedding_provider_routes",
        lambda database_url, **kwargs: [],
    )
    monkeypatch.setattr(
        route_preflight,
        "get_default_embedding_provider_contract_sample_set",
        lambda database_url: make_sample_set(),
    )

    payload = run_embedding_provider_route_preflight(
        "postgresql://example/db",
        profile_name="kure_v1_1024",
    )

    assert payload["route_count"] == 0
    assert payload["passed_count"] == 0
    assert payload["failed_count"] == 0
    assert payload["sample_set"]["sample_set_name"] == "unit_contract_samples"
    assert payload["results"] == []


def test_route_preflight_records_contract_when_health_is_absent(monkeypatch) -> None:
    import app.core.embedding_provider_route_preflight as route_preflight

    route = make_route()
    calls = {}

    def fake_record_health_snapshot(*args, **kwargs):
        raise AssertionError("health snapshot should not be recorded without health")

    def fake_record_contract_snapshot(database_url, contract):
        calls["contract_snapshot"] = (database_url, contract)
        return make_contract_snapshot(contract)

    monkeypatch.setattr(
        route_preflight,
        "list_embedding_provider_routes",
        lambda database_url, **kwargs: [route],
    )
    monkeypatch.setattr(
        route_preflight,
        "get_default_embedding_provider_contract_sample_set",
        lambda database_url: make_sample_set(),
    )
    monkeypatch.setattr(
        route_preflight,
        "check_embedding_provider_route_contract",
        lambda checked_route, **kwargs: make_contract(checked_route),
    )
    monkeypatch.setattr(
        route_preflight,
        "record_embedding_provider_route_health_snapshot",
        fake_record_health_snapshot,
    )
    monkeypatch.setattr(
        route_preflight,
        "record_embedding_provider_route_contract_snapshot",
        fake_record_contract_snapshot,
    )

    payload = run_embedding_provider_route_preflight("postgresql://example/db")

    assert calls["contract_snapshot"][0] == "postgresql://example/db"
    assert payload["route_count"] == 1
    assert payload["passed_count"] == 0
    assert payload["failed_count"] == 1
    assert payload["results"][0]["health_status"] is None
    assert payload["results"][0]["health_snapshot_id"] is None
    assert payload["results"][0]["contract_snapshot_id"] == 202


def make_route(**overrides) -> EmbeddingProviderRouteRecord:
    now = datetime(2026, 7, 11, tzinfo=UTC)
    values = {
        "route_id": 7,
        "profile_name": "kure_v1_1024",
        "provider_name": "gpu-primary",
        "provider_mode": "mock",
        "provider_base_url": None,
        "timeout_seconds": 30.0,
        "priority": 1,
        "is_active": True,
        "health_check_enabled": True,
        "runtime_metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingProviderRouteRecord(**values)


def make_sample_set(**overrides) -> EmbeddingProviderContractSampleSetRecord:
    now = datetime(2026, 7, 11, tzinfo=UTC)
    values = {
        "sample_set_name": "unit_contract_samples",
        "description": "Unit sample set",
        "input_type": "document",
        "sample_texts": ("contract sample one",),
        "is_active": True,
        "is_default": True,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingProviderContractSampleSetRecord(**values)


def make_contract(route: EmbeddingProviderRouteRecord) -> EmbeddingProviderRouteContractResult:
    return EmbeddingProviderRouteContractResult(
        route=route,
        passed=False,
        status="health_unreachable",
        elapsed_ms=1,
        health=None,
        input_type="document",
        sample_text_count=1,
        expected_dimension=1024,
        provider_type=None,
        provider_model_id=None,
        model_key=None,
        dimension=None,
        input_count=None,
        runtime_metadata={},
        validation_errors=("provider not reachable",),
        error_message="provider not reachable",
    )


def make_contract_snapshot(
    contract: EmbeddingProviderRouteContractResult,
) -> EmbeddingProviderRouteContractSnapshotRecord:
    now = datetime(2026, 7, 11, tzinfo=UTC)
    return EmbeddingProviderRouteContractSnapshotRecord(
        snapshot_id=202,
        route_id=contract.route.route_id,
        profile_name=contract.route.profile_name,
        provider_name=contract.route.provider_name,
        provider_mode=contract.route.provider_mode,
        passed=contract.passed,
        status=contract.status,
        elapsed_ms=contract.elapsed_ms,
        input_type=contract.input_type,
        sample_text_count=contract.sample_text_count,
        expected_dimension=contract.expected_dimension,
        provider_type=contract.provider_type,
        provider_model_id=contract.provider_model_id,
        model_key=contract.model_key,
        dimension=contract.dimension,
        input_count=contract.input_count,
        runtime_metadata=contract.runtime_metadata,
        validation_errors=contract.validation_errors,
        error_message=contract.error_message,
        checked_at=now,
    )
