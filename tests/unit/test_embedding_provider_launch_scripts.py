import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.embedding_provider_presets import (
    InvalidEmbeddingProviderPresetError,
    build_embedding_provider_launch_plan,
    build_embedding_provider_preset_route_plans,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
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


run_embedding_provider = _load_script_module("run_embedding_provider.py")
register_embedding_provider_routes = _load_script_module("register_embedding_provider_routes.py")


def test_embedding_provider_presets_define_expected_ports_and_profiles() -> None:
    presets = {preset.preset_name: preset for preset in list_embedding_provider_presets()}

    assert presets["kure"].default_port == 9101
    assert presets["bge"].default_port == 9102
    assert presets["qwen"].default_port == 9103
    assert presets["qwen"].profile_names == ("qwen3_4b_1000", "qwen3_4b_2560")


def test_embedding_provider_preset_route_plans_cover_shared_qwen_profiles() -> None:
    plans = build_embedding_provider_preset_route_plans(
        get_embedding_provider_preset("qwen"),
        host="gpu-qwen.local",
        port=19103,
        provider_name="qwen-gpu-primary",
        timeout_seconds=12.5,
        priority=11,
        runtime_metadata={"operator": "slice-179"},
        metadata_source="unit_test",
    )

    assert [plan.profile_name for plan in plans] == ["qwen3_4b_1000", "qwen3_4b_2560"]
    assert {plan.provider_base_url for plan in plans} == {"http://gpu-qwen.local:19103"}
    assert {plan.provider_port for plan in plans} == {19103}
    assert {plan.provider_name for plan in plans} == {"qwen-gpu-primary"}
    assert {plan.timeout_seconds for plan in plans} == {12.5}
    assert {plan.priority for plan in plans} == {11}
    assert all(plan.runtime_metadata["source"] == "unit_test" for plan in plans)
    assert all(plan.runtime_metadata["operator"] == "slice-179" for plan in plans)

    route_input = plans[0].to_route_input()
    assert route_input.profile_name == "qwen3_4b_1000"
    assert route_input.provider_base_url == "http://gpu-qwen.local:19103"


def test_embedding_provider_preset_route_plans_reject_invalid_base_url() -> None:
    with pytest.raises(InvalidEmbeddingProviderPresetError, match="absolute http"):
        build_embedding_provider_preset_route_plans(
            get_embedding_provider_preset("kure"),
            base_url="gpu-provider.local:9101",
        )


@pytest.mark.parametrize(
    ("preset_name", "message"),
    [
        (" ", "preset_name is required"),
        ("missing", "Unsupported embedding provider preset"),
    ],
)
def test_embedding_provider_preset_lookup_rejects_invalid_names(
    preset_name: str,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderPresetError, match=message):
        get_embedding_provider_preset(preset_name)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"provider_name": " "}, "provider_name is required"),
        ({"timeout_seconds": 0}, "timeout_seconds must be greater than 0"),
        ({"priority": -1}, "priority must be greater than or equal to 0"),
        ({"base_url": "   "}, "base_url is required"),
        ({"host": " "}, "host is required"),
        ({"port": 70000}, "port must be between 1 and 65535"),
    ],
)
def test_embedding_provider_preset_route_plans_reject_invalid_operator_inputs(
    kwargs,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderPresetError, match=message):
        build_embedding_provider_preset_route_plans(
            get_embedding_provider_preset("kure"),
            **kwargs,
        )


def test_embedding_provider_preset_route_plans_normalize_absolute_base_url() -> None:
    plans = build_embedding_provider_preset_route_plans(
        get_embedding_provider_preset("bge"),
        base_url="https://gpu-bge.local:9443/",
    )

    assert len(plans) == 1
    assert plans[0].provider_base_url == "https://gpu-bge.local:9443"
    assert plans[0].provider_port == 9443


def test_embedding_provider_launch_plan_builds_shell_command_from_core_helper() -> None:
    plan = build_embedding_provider_launch_plan(
        get_embedding_provider_preset("qwen"),
        python_bin="./.venv/bin/python",
        host="0.0.0.0",
        port=19103,
        device="cuda:0",
        models_dir="/srv/nex_pcx/models",
        provider_model_id="gpu-qwen3-4b",
        reload=True,
    )

    assert plan.base_url == "http://0.0.0.0:19103"
    assert plan.command[-1] == "--reload"
    assert plan.environment["NEX_PCX_PROVIDER_BACKEND"] == "qwen_embedding"
    assert plan.environment["NEX_PCX_PROVIDER_PROFILE_NAMES"] == "qwen3_4b_1000,qwen3_4b_2560"
    assert "NEX_PCX_PROVIDER_DEVICE=cuda:0" in plan.shell_command
    assert "./.venv/bin/python -m uvicorn" in plan.shell_command


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"python_bin": " "}, "python_bin is required"),
        ({"device": " "}, "device is required"),
        ({"models_dir": " "}, "models_dir is required"),
        ({"provider_model_id": " "}, "provider_model_id is required"),
        ({"port": 0}, "port must be between 1 and 65535"),
    ],
)
def test_embedding_provider_launch_plan_rejects_invalid_operator_inputs(
    kwargs,
    message: str,
) -> None:
    base_kwargs = {
        "python_bin": "./.venv/bin/python",
        "models_dir": "/srv/nex_pcx/models",
    }
    base_kwargs.update(kwargs)
    with pytest.raises(InvalidEmbeddingProviderPresetError, match=message):
        build_embedding_provider_launch_plan(
            get_embedding_provider_preset("kure"),
            **base_kwargs,
        )


def test_run_embedding_provider_builds_qwen_launch_plan() -> None:
    plan = run_embedding_provider.build_launch_plan(
        get_embedding_provider_preset("qwen"),
        python_bin="/opt/nex-pcx/.venv/bin/python",
        host="0.0.0.0",
        port=19103,
        device="cuda:0",
        models_dir="/srv/nex_pcx/models",
        provider_model_id="gpu-qwen3-4b",
        reload=True,
    )

    assert plan.base_url == "http://0.0.0.0:19103"
    assert plan.backend == "qwen_embedding"
    assert plan.profile_names == ("qwen3_4b_1000", "qwen3_4b_2560")
    assert plan.command == (
        "/opt/nex-pcx/.venv/bin/python",
        "-m",
        "uvicorn",
        "app.embedding_provider_service:app",
        "--host",
        "0.0.0.0",
        "--port",
        "19103",
        "--reload",
    )
    assert plan.environment == {
        "NEX_PCX_PROVIDER_BACKEND": "qwen_embedding",
        "NEX_PCX_PROVIDER_MODEL_KEY": "qwen3_embedding_4b",
        "NEX_PCX_PROVIDER_PROFILE_NAMES": "qwen3_4b_1000,qwen3_4b_2560",
        "NEX_PCX_PROVIDER_MODEL_ID": "gpu-qwen3-4b",
        "NEX_PCX_PROVIDER_DEVICE": "cuda:0",
        "NEX_PCX_PROVIDER_MODELS_DIR": "/srv/nex_pcx/models",
    }


def test_run_embedding_provider_script_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_embedding_provider.py",
            "--provider",
            "qwen",
            "--port",
            "9109",
            "--models-dir",
            "/tmp/models",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["plan"]["preset_name"] == "qwen"
    assert payload["plan"]["base_url"] == "http://127.0.0.1:9109"
    assert payload["plan"]["environment"]["NEX_PCX_PROVIDER_BACKEND"] == "qwen_embedding"
    assert (
        payload["plan"]["environment"]["NEX_PCX_PROVIDER_PROFILE_NAMES"]
        == "qwen3_4b_1000,qwen3_4b_2560"
    )


def test_register_embedding_provider_routes_builds_shared_qwen_route_plans() -> None:
    plans = register_embedding_provider_routes.build_route_plans(
        register_embedding_provider_routes.select_presets("qwen"),
        base_url="http://gpu-qwen.local:9103/",
        priority=20,
        timeout_seconds=45,
    )

    assert [plan.profile_name for plan in plans] == ["qwen3_4b_1000", "qwen3_4b_2560"]
    assert {plan.provider_base_url for plan in plans} == {"http://gpu-qwen.local:9103"}
    assert {plan.provider_port for plan in plans} == {9103}
    assert {plan.provider_name for plan in plans} == {"qwen-primary"}
    assert all(plan.priority == 20 for plan in plans)
    assert all(plan.timeout_seconds == 45 for plan in plans)
    assert all(plan.runtime_metadata["model_key"] == "qwen3_embedding_4b" for plan in plans)


def test_register_embedding_provider_routes_builds_all_default_ports() -> None:
    plans = register_embedding_provider_routes.build_route_plans(
        register_embedding_provider_routes.select_presets("all"),
        host="127.0.0.1",
    )

    assert [(plan.profile_name, plan.provider_port) for plan in plans] == [
        ("kure_v1_1024", 9101),
        ("bge_m3_1024", 9102),
        ("qwen3_4b_1000", 9103),
        ("qwen3_4b_2560", 9103),
    ]


def test_register_embedding_provider_routes_rejects_base_url_for_all_presets() -> None:
    with pytest.raises(ValueError, match="single provider"):
        register_embedding_provider_routes.build_route_plans(
            register_embedding_provider_routes.select_presets("all"),
            base_url="http://provider.local:9100",
        )


def test_register_embedding_provider_routes_upserts_route_inputs(monkeypatch) -> None:
    seen_inputs = []

    def fake_upsert(database_url, route_input):
        seen_inputs.append((database_url, route_input))
        now = datetime(2026, 7, 13, tzinfo=UTC)
        return EmbeddingProviderRouteRecord(
            route_id=len(seen_inputs),
            profile_name=route_input.profile_name,
            provider_name=route_input.provider_name,
            provider_mode=route_input.provider_mode,
            provider_base_url=route_input.provider_base_url,
            timeout_seconds=route_input.timeout_seconds,
            priority=route_input.priority,
            is_active=route_input.is_active,
            health_check_enabled=route_input.health_check_enabled,
            runtime_metadata=route_input.runtime_metadata or {},
            created_at=now,
            updated_at=now,
        )

    monkeypatch.setattr(
        register_embedding_provider_routes, "upsert_embedding_provider_route", fake_upsert
    )
    plans = register_embedding_provider_routes.build_route_plans(
        register_embedding_provider_routes.select_presets("kure"),
        port=19101,
        provider_name="kure-gpu-a",
        is_active=False,
        health_check_enabled=False,
    )

    records = register_embedding_provider_routes.register_route_plans(
        "postgresql://unit-test",
        plans,
    )

    assert len(records) == 1
    assert seen_inputs[0][0] == "postgresql://unit-test"
    route_input = seen_inputs[0][1]
    assert route_input.profile_name == "kure_v1_1024"
    assert route_input.provider_name == "kure-gpu-a"
    assert route_input.provider_base_url == "http://127.0.0.1:19101"
    assert route_input.is_active is False
    assert route_input.health_check_enabled is False
