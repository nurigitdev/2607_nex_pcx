import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "probe_provider_resources.py"
    spec = importlib.util.spec_from_file_location("probe_provider_resources_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe_script = _load_script_module()


def test_probe_provider_resources_dry_run_writes_outputs(tmp_path, capsys) -> None:
    json_output = tmp_path / "provider-resources.json"
    markdown_output = tmp_path / "provider-resources.md"

    exit_code = probe_script.main(
        [
            "--dry-run",
            "--provider",
            "vllm",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert payload["status"] == "dry_run"
    assert payload["targets"][0]["provider_name"] == "dgx_vllm_qwen36_27b_nvfp4"
    assert payload["targets"][0]["port"] == 12000
    assert "nvidia-smi --query-compute-apps" in payload["commands"][2]
    assert "# DGX Provider Resource Probe Dry Run" in markdown
    assert "| dgx_vllm_qwen36_27b_nvfp4 | vllm | 12000 | vllm |" in markdown


def test_probe_provider_resources_builds_remote_command() -> None:
    args = probe_script.argparse.Namespace(
        host="192.168.20.243",
        ssh_user="nexpcx",
        remote_workdir="/home/nexpcx/2607_nex_pcx",
        remote_python_bin="/home/nexpcx/2607_nex_pcx/.venv/bin/python",
        provider=["qwen", "reranker"],
    )

    command = probe_script.build_remote_probe_command(args)

    assert command[0] == "ssh"
    assert command[1] == "nexpcx@192.168.20.243"
    assert "scripts/probe_provider_resources.py" in command[2]
    assert "--local-only" in command[2]
    assert "--provider qwen" in command[2]
    assert "--provider reranker" in command[2]


def test_probe_provider_resources_remote_mode_parses_json(monkeypatch, capsys) -> None:
    def fake_run(command, *, capture_output, text, timeout, check):
        assert command[0] == "ssh"
        assert capture_output is True
        assert text is True
        assert timeout == 15
        assert check is False
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"status": "ok", "host": "192.168.20.243", "snapshots": []}),
            "",
        )

    monkeypatch.setattr(probe_script.subprocess, "run", fake_run)

    exit_code = probe_script.main(["--ssh-user", "nexpcx", "--provider", "vllm", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["host"] == "192.168.20.243"


def test_probe_provider_resources_reports_bad_selector(capsys) -> None:
    exit_code = probe_script.main(["--provider", "missing"])

    assert exit_code == 2
    assert "provider resource probe plan failed" in capsys.readouterr().err
