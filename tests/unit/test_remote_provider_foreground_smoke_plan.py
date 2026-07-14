import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.embedding_provider_presets import get_embedding_provider_preset


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


foreground_smoke = _load_script_module("plan_remote_provider_foreground_smoke.py")


def test_foreground_smoke_plan_defaults_to_kure_on_dgx_spark() -> None:
    plan = foreground_smoke.build_foreground_smoke_plan(get_embedding_provider_preset("kure"))

    assert plan.provider == "kure"
    assert plan.ssh_target == "nexpcx@192.168.20.243"
    assert plan.workdir == "/home/nexpcx/2607_nex_pcx"
    assert plan.port == 9101
    assert plan.base_url == "http://192.168.20.243:9101"
    assert plan.health_url == "http://192.168.20.243:9101/healthz"
    assert plan.profile_names == ("kure_v1_1024",)
    assert plan.readiness_command == (
        "./.venv/bin/python",
        "scripts/check_remote_gpu_provider_host.py",
        "--host",
        "192.168.20.243",
        "--ssh-user",
        "nexpcx",
        "--provider",
        "kure",
        "--timeout-seconds",
        "12",
    )
    assert plan.health_check_command == ("curl", "-fsS", "http://192.168.20.243:9101/healthz")
    assert plan.ssh_launch_command[:3] == ("ssh", "-t", "nexpcx@192.168.20.243")
    assert "NEX_PCX_PROVIDER_BACKEND=sentence_transformers" in plan.remote_launch_command


def test_foreground_smoke_plan_supports_shared_qwen_profiles() -> None:
    plan = foreground_smoke.build_foreground_smoke_plan(
        get_embedding_provider_preset("qwen"),
        route_host="gpu-provider.internal",
        provider_model_id="dgx-qwen-2026-07",
        readiness_timeout_seconds=21,
    )

    assert plan.provider == "qwen"
    assert plan.port == 9103
    assert plan.base_url == "http://gpu-provider.internal:9103"
    assert plan.profile_names == ("qwen3_4b_1000", "qwen3_4b_2560")
    assert plan.provider_model_id == "dgx-qwen-2026-07"
    assert plan.readiness_command[-1] == "21"
    assert "NEX_PCX_PROVIDER_BACKEND=qwen_embedding" in plan.remote_launch_command
    assert "NEX_PCX_PROVIDER_PROFILE_NAMES=qwen3_4b_1000,qwen3_4b_2560" in (
        plan.remote_launch_command
    )


def test_foreground_smoke_plan_can_override_port_and_paths() -> None:
    plan = foreground_smoke.build_foreground_smoke_plan(
        get_embedding_provider_preset("bge"),
        workdir="/srv/nex-pcx",
        models_dir="/srv/models",
        python_bin="/srv/nex-pcx/.venv/bin/python",
        port=19102,
        route_host="192.168.20.244",
    )

    assert plan.port == 19102
    assert plan.workdir == "/srv/nex-pcx"
    assert plan.models_dir == "/srv/models"
    assert plan.python_bin == "/srv/nex-pcx/.venv/bin/python"
    assert plan.base_url == "http://192.168.20.244:19102"
    assert plan.health_check_command[-1] == "http://192.168.20.244:19102/healthz"


def test_foreground_smoke_plan_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="readiness_timeout_seconds"):
        foreground_smoke.build_foreground_smoke_plan(
            get_embedding_provider_preset("kure"),
            readiness_timeout_seconds=0,
        )


def test_foreground_smoke_cli_prints_json_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_remote_provider_foreground_smoke.py",
            "--provider",
            "qwen",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    plan = payload["plan"]

    assert plan["provider"] == "qwen"
    assert plan["base_url"] == "http://192.168.20.243:9103"
    assert plan["health_check_shell_command"] == "curl -fsS http://192.168.20.243:9103/healthz"
    assert plan["profile_names"] == ["qwen3_4b_1000", "qwen3_4b_2560"]
    assert "ssh -t nexpcx@192.168.20.243" in plan["ssh_launch_shell_command"]
