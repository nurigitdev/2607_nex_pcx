import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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


setup_remote_gpu_provider = _load_script_module("setup_remote_gpu_provider.py")


def test_remote_gpu_provider_setup_plan_matches_dgx_spark_defaults() -> None:
    plan = setup_remote_gpu_provider.build_setup_plan(
        get_embedding_provider_preset("qwen"),
        route_host="192.168.20.243",
        provider_model_id="dgx-spark-qwen3-embedding-4b-2026-07",
    )

    assert plan.workdir == "/home/nexpcx/2607_nex_pcx"
    assert plan.python_bin == "/home/nexpcx/2607_nex_pcx/.venv/bin/python"
    assert plan.models_dir == "/home/nexpcx/2607_nex_pcx/models"
    assert plan.service_name == "nex-pcx-embedding-provider-qwen"
    assert plan.route_base_url == "http://192.168.20.243:9103"
    assert plan.health_url == "http://192.168.20.243:9103/healthz"
    assert plan.launch_plan.host == "0.0.0.0"
    assert plan.launch_plan.port == 9103
    assert plan.launch_plan.environment["NEX_PCX_PROVIDER_BACKEND"] == "qwen_embedding"
    assert (
        plan.launch_plan.environment["NEX_PCX_PROVIDER_PROFILE_NAMES"]
        == "qwen3_4b_1000,qwen3_4b_2560"
    )
    assert plan.route_registration_command == (
        "./.venv/bin/python",
        "scripts/register_embedding_provider_routes.py",
        "--provider",
        "qwen",
        "--base-url",
        "http://192.168.20.243:9103",
        "--database-url",
        "$NEX_PCX_DATABASE_URL",
    )


def test_remote_gpu_provider_setup_renders_env_without_database_secret() -> None:
    plan = setup_remote_gpu_provider.build_setup_plan(
        get_embedding_provider_preset("kure"),
        route_host="192.168.20.243",
        provider_model_id="dgx-spark-kure-v1-2026-07",
    )

    env_text = setup_remote_gpu_provider.render_env_file(plan)

    assert "NEX_PCX_PROVIDER_BACKEND=sentence_transformers" in env_text
    assert "NEX_PCX_PROVIDER_MODEL_KEY=kure_v1" in env_text
    assert "NEX_PCX_PROVIDER_DEVICE=cuda:0" in env_text
    assert "NEX_PCX_DATABASE_URL" not in env_text
    assert "nuri1004" not in env_text


def test_remote_gpu_provider_setup_renders_systemd_unit() -> None:
    plan = setup_remote_gpu_provider.build_setup_plan(
        get_embedding_provider_preset("bge"),
        route_host="192.168.20.243",
        provider_model_id="dgx-spark-bge-m3-2026-07",
    )

    unit_text = setup_remote_gpu_provider.render_systemd_unit(plan)

    assert "Description=NeX-PCX embedding provider (bge)" in unit_text
    assert "User=nexpcx" in unit_text
    assert "Group=nexpcx" in unit_text
    assert "WorkingDirectory=/home/nexpcx/2607_nex_pcx" in unit_text
    assert (
        "EnvironmentFile=/home/nexpcx/2607_nex_pcx/deployment/env/"
        "nex-pcx-embedding-provider-bge.env"
    ) in unit_text
    assert (
        "ExecStart=/home/nexpcx/2607_nex_pcx/.venv/bin/python -m uvicorn "
        "app.embedding_provider_service:app --host 0.0.0.0 --port 9102"
    ) in unit_text
    assert "NoNewPrivileges=true" in unit_text


def test_remote_gpu_provider_setup_renders_user_systemd_unit() -> None:
    plan = setup_remote_gpu_provider.build_setup_plan(
        get_embedding_provider_preset("bge"),
        route_host="192.168.20.243",
        provider_model_id="dgx-spark-bge-m3-2026-07",
    )

    unit_text = setup_remote_gpu_provider.render_systemd_unit(plan, user_systemd=True)

    assert "Description=NeX-PCX embedding provider (bge)" in unit_text
    assert "User=nexpcx" not in unit_text
    assert "Group=nexpcx" not in unit_text
    assert "network-online.target" not in unit_text
    assert "NoNewPrivileges=true" not in unit_text
    assert "PrivateTmp=true" not in unit_text
    assert "WantedBy=default.target" in unit_text
    assert "WorkingDirectory=/home/nexpcx/2607_nex_pcx" in unit_text
    assert (
        "ExecStart=/home/nexpcx/2607_nex_pcx/.venv/bin/python -m uvicorn "
        "app.embedding_provider_service:app --host 0.0.0.0 --port 9102"
    ) in unit_text


def test_remote_gpu_provider_setup_writes_files(tmp_path: Path) -> None:
    plan = setup_remote_gpu_provider.build_setup_plan(
        get_embedding_provider_preset("qwen"),
        workdir=str(tmp_path / "2607_nex_pcx"),
        env_dir=str(tmp_path / "env"),
        systemd_dir=str(tmp_path / "systemd"),
        route_host="192.168.20.243",
        provider_model_id="dgx-spark-qwen3-embedding-4b-2026-07",
    )

    env_file, unit_file = setup_remote_gpu_provider.write_plan_files(plan)

    assert env_file.exists()
    assert unit_file.exists()
    assert env_file.read_text(encoding="utf-8").startswith(
        "# Generated by scripts/setup_remote_gpu_provider.py"
    )
    assert "qwen3_4b_1000,qwen3_4b_2560" in env_file.read_text(encoding="utf-8")
    assert "nex-pcx-embedding-provider-qwen" in str(unit_file)


def test_remote_gpu_provider_setup_writes_user_systemd_file(tmp_path: Path) -> None:
    plan = setup_remote_gpu_provider.build_setup_plan(
        get_embedding_provider_preset("qwen"),
        workdir=str(tmp_path / "2607_nex_pcx"),
        env_dir=str(tmp_path / "env"),
        systemd_dir=str(tmp_path / "systemd-user"),
        route_host="192.168.20.243",
        provider_model_id="dgx-spark-qwen3-embedding-4b-2026-07",
    )

    _, unit_file = setup_remote_gpu_provider.write_plan_files(plan, user_systemd=True)

    unit_text = unit_file.read_text(encoding="utf-8")
    assert "User=" not in unit_text
    assert "Group=" not in unit_text
    assert "PrivateTmp=true" not in unit_text


def test_remote_gpu_provider_setup_cli_prints_json_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/setup_remote_gpu_provider.py",
            "--provider",
            "qwen",
            "--route-host",
            "192.168.20.243",
            "--provider-model-id",
            "dgx-spark-qwen3-embedding-4b-2026-07",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["wrote_files"] is False
    assert payload["plan"]["provider"] == "qwen"
    assert payload["plan"]["route_base_url"] == "http://192.168.20.243:9103"
    assert payload["plan"]["systemd_unit_name"] == "nex-pcx-embedding-provider-qwen.service"
    assert payload["plan"]["launch_command"] == [
        "/home/nexpcx/2607_nex_pcx/.venv/bin/python",
        "-m",
        "uvicorn",
        "app.embedding_provider_service:app",
        "--host",
        "0.0.0.0",
        "--port",
        "9103",
    ]


def test_remote_gpu_provider_setup_cli_writes_user_systemd(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    systemd_dir = tmp_path / "systemd-user"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/setup_remote_gpu_provider.py",
            "--provider",
            "qwen",
            "--route-host",
            "192.168.20.243",
            "--provider-model-id",
            "dgx-spark-qwen3-embedding-4b-2026-07",
            "--env-dir",
            str(env_dir),
            "--systemd-dir",
            str(systemd_dir),
            "--user-systemd",
            "--write-files",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    unit_file = systemd_dir / "nex-pcx-embedding-provider-qwen.service"

    assert payload["wrote_files"] is True
    assert unit_file.exists()
    assert "User=" not in unit_file.read_text(encoding="utf-8")


def test_remote_gpu_provider_setup_cli_rejects_invalid_route_url() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/setup_remote_gpu_provider.py",
            "--provider",
            "qwen",
            "--route-base-url",
            "192.168.20.243:9103",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "route_base_url must be an absolute http(s) URL" in result.stderr
