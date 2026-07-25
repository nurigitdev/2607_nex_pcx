import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.rerankers import DEFAULT_RERANKER_MODEL_ID, DEFAULT_RERANKER_PROFILE_NAME


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_reranker_provider = _load_script_module("run_reranker_provider.py")


def test_reranker_provider_launch_plan_builds_qwen_runtime_command() -> None:
    plan = run_reranker_provider.build_launch_plan(
        python_bin="/opt/nex-pcx/.venv/bin/python",
        host="0.0.0.0",
        port=19104,
        device="cuda:0",
        models_dir="/srv/nex_pcx/models",
        provider_model_id="Qwen/Qwen3-Reranker-4B",
        reload=True,
    )

    assert plan.provider_name == "qwen-reranker-primary"
    assert plan.backend == "qwen_reranker"
    assert plan.reranker_profile_name == DEFAULT_RERANKER_PROFILE_NAME
    assert plan.provider_model_id == DEFAULT_RERANKER_MODEL_ID
    assert plan.base_url == "http://0.0.0.0:19104"
    assert plan.command == (
        "/opt/nex-pcx/.venv/bin/python",
        "-m",
        "uvicorn",
        "app.reranker_provider_service:app",
        "--host",
        "0.0.0.0",
        "--port",
        "19104",
        "--reload",
    )
    assert plan.environment == {
        "NEX_PCX_RERANKER_PROVIDER_BACKEND": "qwen_reranker",
        "NEX_PCX_RERANKER_PROVIDER_MODEL_ID": "Qwen/Qwen3-Reranker-4B",
        "NEX_PCX_RERANKER_PROVIDER_PROFILE_NAME": "qwen3_reranker_4b",
        "NEX_PCX_RERANKER_PROVIDER_DEVICE": "cuda:0",
        "NEX_PCX_RERANKER_PROVIDER_MODELS_DIR": "/srv/nex_pcx/models",
        "NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME": "qwen3_reranker_4b",
    }
    assert "NEX_PCX_RERANKER_PROVIDER_BACKEND=qwen_reranker" in plan.shell_command
    assert "app.reranker_provider_service:app" in plan.shell_command


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"python_bin": " "}, "python_bin is required"),
        ({"provider_name": " "}, "provider_name is required"),
        ({"backend": "missing"}, "Unsupported reranker provider backend"),
        ({"host": " "}, "host is required"),
        ({"port": 0}, "port must be between 1 and 65535"),
        ({"device": " "}, "device is required"),
        ({"models_dir": " "}, "models_dir is required"),
        ({"model_dir_name": " "}, "model_dir_name is required"),
        ({"provider_model_id": " "}, "provider_model_id is required"),
        ({"reranker_profile_name": " "}, "reranker_profile_name is required"),
    ],
)
def test_reranker_provider_launch_plan_rejects_invalid_inputs(kwargs, message: str) -> None:
    base_kwargs = {"python_bin": "./.venv/bin/python"}
    base_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run_reranker_provider.build_launch_plan(**base_kwargs)


def test_run_reranker_provider_script_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_reranker_provider.py",
            "--port",
            "19104",
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
    assert payload["plan"]["provider_name"] == "qwen-reranker-primary"
    assert payload["plan"]["base_url"] == "http://127.0.0.1:19104"
    assert payload["plan"]["environment"]["NEX_PCX_RERANKER_PROVIDER_BACKEND"] == "qwen_reranker"
    assert payload["plan"]["environment"]["NEX_PCX_RERANKER_PROVIDER_MODELS_DIR"] == "/tmp/models"
