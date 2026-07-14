import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.embedding_provider_routes import EmbeddingProviderRouteRecord


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_script_module("verify_dgx_provider_route_registration.py")


def _record_from_plan(plan: Any, **overrides: Any) -> EmbeddingProviderRouteRecord:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    values = {
        "route_id": 1,
        "profile_name": plan.profile_name,
        "provider_name": plan.provider_name,
        "provider_mode": plan.provider_mode,
        "provider_base_url": plan.provider_base_url,
        "timeout_seconds": plan.timeout_seconds,
        "priority": plan.priority,
        "is_active": plan.is_active,
        "health_check_enabled": plan.health_check_enabled,
        "runtime_metadata": dict(plan.runtime_metadata),
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return EmbeddingProviderRouteRecord(**values)


def test_build_dgx_route_plans_defaults_to_dgx_host_and_provider_timeouts() -> None:
    plans = verifier.build_dgx_route_plans()

    assert [
        (plan.profile_name, plan.provider_base_url, plan.timeout_seconds) for plan in plans
    ] == [
        ("kure_v1_1024", "http://192.168.20.243:9101", 120.0),
        ("bge_m3_1024", "http://192.168.20.243:9102", 120.0),
        ("qwen3_4b_1000", "http://192.168.20.243:9103", 300.0),
        ("qwen3_4b_2560", "http://192.168.20.243:9103", 300.0),
    ]
    assert [plan.provider_name for plan in plans] == [
        "kure-primary",
        "bge-primary",
        "qwen-primary",
        "qwen-primary",
    ]
    assert all(
        plan.runtime_metadata["script"] == "register_embedding_provider_routes.py" for plan in plans
    )


def test_build_dgx_route_plans_can_target_subset_and_override_timeout() -> None:
    plans = verifier.build_dgx_route_plans(
        provider_names=("qwen", "qwen", "kure"),
        host="gpu.local",
        timeout_seconds=45,
        priority=7,
        is_active=False,
        health_check_enabled=False,
    )

    assert [plan.profile_name for plan in plans] == [
        "qwen3_4b_1000",
        "qwen3_4b_2560",
        "kure_v1_1024",
    ]
    assert {plan.timeout_seconds for plan in plans} == {45.0}
    assert {plan.priority for plan in plans} == {7}
    assert {plan.is_active for plan in plans} == {False}
    assert {plan.health_check_enabled for plan in plans} == {False}
    assert plans[0].provider_base_url == "http://gpu.local:9103"


def test_verify_dgx_route_registration_passes_for_matching_records(
    monkeypatch,
) -> None:
    plans = verifier.build_dgx_route_plans(provider_names=("kure",))
    records = [_record_from_plan(plans[0])]

    monkeypatch.setattr(
        verifier, "list_embedding_provider_routes", lambda *_args, **_kwargs: records
    )

    report = verifier.verify_dgx_route_registration("postgresql://unit", plans)

    assert report.passed is True
    assert report.verified_count == 1
    assert report.missing_count == 0
    assert report.mismatched_count == 0
    assert report.results[0].mismatches == ()


def test_verify_dgx_route_registration_reports_missing_and_mismatched_records(
    monkeypatch,
) -> None:
    plans = verifier.build_dgx_route_plans(provider_names=("kure", "bge"))
    records = [
        _record_from_plan(
            plans[0],
            provider_base_url="http://wrong-host:9101",
            runtime_metadata={"preset_name": "wrong"},
        )
    ]

    monkeypatch.setattr(
        verifier, "list_embedding_provider_routes", lambda *_args, **_kwargs: records
    )

    report = verifier.verify_dgx_route_registration("postgresql://unit", plans)

    assert report.passed is False
    assert report.verified_count == 0
    assert report.missing_count == 1
    assert report.mismatched_count == 1
    assert "provider_base_url: expected 'http://192.168.20.243:9101'" in (
        report.results[0].mismatches[0]
    )
    assert "runtime_metadata.preset_name: expected 'kure', got 'wrong'" in (
        report.results[0].mismatches
    )
    assert report.results[1].mismatches == ("route is missing",)


def test_verify_dgx_route_registration_can_apply_before_verifying(monkeypatch) -> None:
    plans = verifier.build_dgx_route_plans(provider_names=("bge",))
    applied = []

    def fake_register(database_url, route_plans):
        applied.append((database_url, route_plans))
        return ()

    monkeypatch.setattr(verifier, "register_route_plans", fake_register)
    monkeypatch.setattr(
        verifier,
        "list_embedding_provider_routes",
        lambda *_args, **_kwargs: [_record_from_plan(plans[0])],
    )

    report = verifier.verify_dgx_route_registration("postgresql://unit", plans, apply=True)

    assert report.passed is True
    assert report.applied is True
    assert applied == [("postgresql://unit", plans)]


def test_verify_dgx_provider_route_registration_cli_prints_json_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_dgx_provider_route_registration.py",
            "--provider",
            "qwen",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert [route["profile_name"] for route in payload["routes"]] == [
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    ]
    assert {route["provider_base_url"] for route in payload["routes"]} == {
        "http://192.168.20.243:9103"
    }
    assert {route["timeout_seconds"] for route in payload["routes"]} == {300.0}
