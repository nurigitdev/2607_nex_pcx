import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.embedding_provider_route_contract_snapshots import (
    EmbeddingProviderRouteContractSnapshotRecord,
)
from app.core.embedding_provider_route_contracts import EmbeddingProviderRouteContractResult
from app.core.embedding_provider_route_health import EmbeddingProviderRouteHealthResult
from app.core.embedding_provider_route_health_snapshots import (
    EmbeddingProviderRouteHealthSnapshotRecord,
)
from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord


def _load_preflight_provider_routes_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "preflight_provider_routes.py"
    spec = importlib.util.spec_from_file_location("preflight_provider_routes_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight_provider_routes = _load_preflight_provider_routes_module()


def test_run_preflight_records_health_and_contract_snapshots(monkeypatch) -> None:
    route = make_route()
    calls = {}

    def fake_list_embedding_provider_routes(database_url: str, **kwargs):
        calls["list_call"] = (database_url, kwargs)
        return [route]

    monkeypatch.setattr(
        preflight_provider_routes,
        "list_embedding_provider_routes",
        fake_list_embedding_provider_routes,
    )
    monkeypatch.setattr(
        preflight_provider_routes,
        "check_embedding_provider_route_contract",
        lambda checked_route: make_contract(checked_route),
    )
    monkeypatch.setattr(
        preflight_provider_routes,
        "record_embedding_provider_route_health_snapshot",
        lambda database_url, health: make_health_snapshot(health),
    )
    monkeypatch.setattr(
        preflight_provider_routes,
        "record_embedding_provider_route_contract_snapshot",
        lambda database_url, contract: make_contract_snapshot(contract),
    )

    payload = preflight_provider_routes.run_preflight(
        "postgresql://example/db",
        profile_name="kure_v1_1024",
    )

    assert calls["list_call"] == (
        "postgresql://example/db",
        {"profile_name": "kure_v1_1024", "active_only": True},
    )
    assert payload["route_count"] == 1
    assert payload["passed_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["results"][0]["provider_name"] == "gpu-primary"
    assert payload["results"][0]["health_snapshot_id"] == 101
    assert payload["results"][0]["contract_snapshot_id"] == 202


def test_main_returns_failure_exit_code_when_contract_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        preflight_provider_routes,
        "get_settings",
        lambda: type("SettingsStub", (), {"database_url": "postgresql://example/db"})(),
    )
    monkeypatch.setattr(
        preflight_provider_routes,
        "run_preflight",
        lambda *args, **kwargs: {
            "route_count": 1,
            "passed_count": 0,
            "failed_count": 1,
            "profile_name": None,
            "active_only": True,
            "results": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight_provider_routes.py"],
    )

    exit_code = preflight_provider_routes.main()

    assert exit_code == 1
    assert '"failed_count": 1' in capsys.readouterr().out


def test_main_can_allow_failed_contracts(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight_provider_routes,
        "get_settings",
        lambda: type("SettingsStub", (), {"database_url": "postgresql://example/db"})(),
    )
    monkeypatch.setattr(
        preflight_provider_routes,
        "run_preflight",
        lambda *args, **kwargs: {
            "route_count": 1,
            "passed_count": 0,
            "failed_count": 1,
            "profile_name": None,
            "active_only": True,
            "results": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight_provider_routes.py", "--allow-failures"],
    )

    assert preflight_provider_routes.main() == 0


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


def make_contract(route: EmbeddingProviderRouteRecord) -> EmbeddingProviderRouteContractResult:
    health = EmbeddingProviderRouteHealthResult(
        route=route,
        checked=True,
        ready=True,
        status="ready",
        elapsed_ms=0,
        provider_type="mock",
        provider_model_id="mock-provider",
        model_key="kure_v1",
        profile_names=("kure_v1_1024",),
        dimension=1024,
        device=None,
        runtime_metadata={},
    )
    return EmbeddingProviderRouteContractResult(
        route=route,
        passed=True,
        status="passed",
        elapsed_ms=1,
        health=health,
        input_type="document",
        sample_text_count=1,
        expected_dimension=1024,
        provider_type="mock",
        provider_model_id="mock-provider",
        model_key="kure_v1",
        dimension=1024,
        input_count=1,
        runtime_metadata={},
    )


def make_health_snapshot(
    health: EmbeddingProviderRouteHealthResult,
) -> EmbeddingProviderRouteHealthSnapshotRecord:
    return EmbeddingProviderRouteHealthSnapshotRecord(
        snapshot_id=101,
        route_id=health.route.route_id,
        profile_name=health.route.profile_name,
        provider_name=health.route.provider_name,
        provider_mode=health.route.provider_mode,
        checked=health.checked,
        ready=health.ready,
        status=health.status,
        elapsed_ms=health.elapsed_ms,
        provider_type=health.provider_type,
        provider_model_id=health.provider_model_id,
        model_key=health.model_key,
        profile_names=health.profile_names,
        dimension=health.dimension,
        device=health.device,
        runtime_metadata=health.runtime_metadata,
        validation_errors=health.validation_errors,
        error_message=health.error_message,
        checked_at=datetime(2026, 7, 11, tzinfo=UTC),
    )


def make_contract_snapshot(
    contract: EmbeddingProviderRouteContractResult,
) -> EmbeddingProviderRouteContractSnapshotRecord:
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
        checked_at=datetime(2026, 7, 11, tzinfo=UTC),
    )
