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


check_remote_gpu_provider_host = _load_script_module("check_remote_gpu_provider_host.py")


def test_remote_gpu_readiness_plan_uses_dgx_spark_defaults() -> None:
    plan = check_remote_gpu_provider_host.build_readiness_plan()

    assert plan.host == "192.168.20.243"
    assert plan.ssh_user == "nexpcx"
    assert plan.ssh_target == "nexpcx@192.168.20.243"
    assert plan.workdir == "/home/nexpcx/2607_nex_pcx"
    assert plan.python_bin == "/home/nexpcx/2607_nex_pcx/.venv/bin/python"
    assert plan.models_dir == "/home/nexpcx/2607_nex_pcx/models"
    assert plan.providers == ("kure", "bge", "qwen")

    check_names = [check.name for check in plan.checks]
    assert "ssh_identity" in check_names
    assert "runtime_dependency_import" in check_names
    assert "source_tree_shape" in check_names
    assert "provider_service_import" in check_names
    assert "nvidia_smi" in check_names
    assert "setup_script_exists" in check_names
    assert "model_dir_kure_v1" in check_names
    assert "model_dir_bge_m3" in check_names
    assert "model_dir_qwen3_embedding_4b" in check_names
    assert "setup_dry_run_qwen" in check_names


def test_remote_gpu_readiness_plan_can_target_shared_qwen_only() -> None:
    plan = check_remote_gpu_provider_host.build_readiness_plan(provider="qwen")

    check_names = [check.name for check in plan.checks]

    assert plan.providers == ("qwen",)
    assert "model_dir_qwen3_embedding_4b" in check_names
    assert "model_dir_kure_v1" not in check_names
    assert "setup_dry_run_qwen" in check_names
    assert "setup_dry_run_kure" not in check_names
    assert "9103" in next(
        check.remote_command for check in plan.checks if check.name == "port_listener_snapshot"
    )


def test_remote_gpu_readiness_builds_batchmode_ssh_command() -> None:
    plan = check_remote_gpu_provider_host.build_readiness_plan(timeout_seconds=7)

    command = check_remote_gpu_provider_host.build_ssh_command(plan, "id -un")

    assert command == (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=7",
        "nexpcx@192.168.20.243",
        "id -un",
    )


def test_remote_gpu_readiness_dry_run_report_keeps_checks_planned() -> None:
    plan = check_remote_gpu_provider_host.build_readiness_plan(provider="qwen")

    report = check_remote_gpu_provider_host.build_dry_run_report(plan)
    payload = check_remote_gpu_provider_host.report_payload(report)

    assert report.ready is False
    assert {check.status for check in report.checks} == {"planned"}
    assert payload["checks"][0]["ssh_shell_command"].startswith("ssh -o BatchMode=yes")


def test_remote_gpu_readiness_report_allows_optional_port_snapshot_failure() -> None:
    plan = check_remote_gpu_provider_host.build_readiness_plan(provider="qwen")

    def fake_runner(args, *, capture_output, text, timeout, check):
        assert capture_output is True
        assert text is True
        assert timeout == plan.timeout_seconds + 5
        assert check is False
        remote_command = args[-1]
        if "ss -ltnH" in remote_command:
            return subprocess.CompletedProcess(args, 1, "", "ss failed")
        return subprocess.CompletedProcess(args, 0, "ok", "")

    report = check_remote_gpu_provider_host.run_readiness_plan(plan, runner=fake_runner)

    assert report.ready is True
    assert report.required_failed == 0
    assert report.optional_failed == 1
    assert any(
        check.name == "port_listener_snapshot" and check.status == "failed"
        for check in report.checks
    )


def test_remote_gpu_readiness_report_fails_when_required_check_fails() -> None:
    plan = check_remote_gpu_provider_host.build_readiness_plan(provider="qwen")

    def fake_runner(args, *, capture_output, text, timeout, check):
        remote_command = args[-1]
        if "nvidia-smi" in remote_command:
            return subprocess.CompletedProcess(args, 127, "", "nvidia-smi: not found")
        return subprocess.CompletedProcess(args, 0, "ok", "")

    report = check_remote_gpu_provider_host.run_readiness_plan(plan, runner=fake_runner)

    assert report.ready is False
    assert report.required_failed == 1
    assert any(check.name == "nvidia_smi" and check.status == "failed" for check in report.checks)


def test_remote_gpu_readiness_cli_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_remote_gpu_provider_host.py",
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

    assert payload["ssh_target"] == "nexpcx@192.168.20.243"
    assert payload["providers"] == ["qwen"]
    assert payload["ready"] is False
    assert {check["status"] for check in payload["checks"]} == {"planned"}
    assert "source_tree_shape" in {check["name"] for check in payload["checks"]}


def test_remote_gpu_readiness_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        check_remote_gpu_provider_host.build_readiness_plan(timeout_seconds=0)
