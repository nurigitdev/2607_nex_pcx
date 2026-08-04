import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


foreground_plan = _load_script_module("plan_remote_reranker_foreground_smoke.py")


def test_remote_reranker_foreground_plan_defaults_to_dgx_spark() -> None:
    plan = foreground_plan.build_reranker_foreground_smoke_plan()

    assert plan.provider_name == "qwen-reranker-primary"
    assert plan.ssh_target == "nexpcx@192.168.20.243"
    assert plan.workdir == "/home/nexpcx/2607_nex_pcx"
    assert plan.port == 9104
    assert plan.base_url == "http://192.168.20.243:9104"
    assert plan.health_url == "http://192.168.20.243:9104/healthz"
    assert plan.backend == "qwen_reranker"
    assert plan.reranker_profile_name == "qwen3_reranker_0_6b"
    assert plan.provider_model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert plan.health_check_command == ("curl", "-fsS", "http://192.168.20.243:9104/healthz")
    assert plan.ssh_launch_command[:3] == ("ssh", "-t", "nexpcx@192.168.20.243")
    assert "NEX_PCX_RERANKER_PROVIDER_BACKEND=qwen_reranker" in plan.remote_launch_command
    assert "app.reranker_provider_service:app" in plan.remote_launch_command
    assert plan.readiness_command[:2] == ("ssh", "nexpcx@192.168.20.243")
    assert "test -f app/reranker_provider_service.py" in plan.readiness_command[-1]
    assert "test -d /home/nexpcx/2607_nex_pcx/models/qwen3_reranker_0_6b" in (
        plan.readiness_command[-1]
    )


def test_remote_reranker_foreground_plan_can_override_port_and_paths() -> None:
    plan = foreground_plan.build_reranker_foreground_smoke_plan(
        workdir="/srv/nex-pcx",
        models_dir="/srv/models",
        model_dir_name="Qwen3-Reranker-4B",
        python_bin="/srv/nex-pcx/.venv/bin/python",
        port=19104,
        route_host="gpu-reranker.internal",
        provider_model_id="local-qwen-reranker",
    )

    assert plan.workdir == "/srv/nex-pcx"
    assert plan.models_dir == "/srv/models"
    assert plan.model_dir_name == "Qwen3-Reranker-4B"
    assert plan.python_bin == "/srv/nex-pcx/.venv/bin/python"
    assert plan.port == 19104
    assert plan.base_url == "http://gpu-reranker.internal:19104"
    assert plan.provider_model_id == "local-qwen-reranker"
    assert plan.health_check_command[-1] == "http://gpu-reranker.internal:19104/healthz"


def test_remote_reranker_foreground_plan_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="host is required"):
        foreground_plan.build_reranker_foreground_smoke_plan(host=" ")


def test_remote_reranker_foreground_plan_cli_prints_json_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_remote_reranker_foreground_smoke.py",
            "--port",
            "19104",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    plan = payload["plan"]

    assert plan["provider_name"] == "qwen-reranker-primary"
    assert plan["base_url"] == "http://192.168.20.243:19104"
    assert plan["health_check_shell_command"] == "curl -fsS http://192.168.20.243:19104/healthz"
    assert "ssh -t nexpcx@192.168.20.243" in plan["ssh_launch_shell_command"]
    assert "app.reranker_provider_service:app" in plan["launch_shell_command"]
