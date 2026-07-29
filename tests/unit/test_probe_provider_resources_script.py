import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from app.core.provider_resource_probe import (
    ProviderResourceTarget,
    build_provider_resource_probe_report,
)
from app.core.provider_resource_snapshots import ProviderResourceSnapshotRecord


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


def test_probe_provider_resources_persists_local_snapshot(monkeypatch, capsys) -> None:
    report = build_provider_resource_probe_report(
        targets=(
            ProviderResourceTarget(
                provider_name="pytest-vllm",
                provider_type="vllm",
                host="192.168.20.243",
                port=12000,
                process_match="vllm",
            ),
        ),
        ps_text="9001 1 nexpcx 2048 4096 3.5 90 /home/nexpcx/.venv/bin/python -m vllm",
        ss_text='LISTEN 0 2048 0.0.0.0:12000 0.0.0.0:* users:(("python",pid=9001,fd=3))',
        meminfo_text="""
        MemTotal:       100000 kB
        MemAvailable:    80000 kB
        SwapTotal:           0 kB
        SwapFree:            0 kB
        """,
        collected_at=datetime(2026, 7, 29, 14, tzinfo=UTC),
    )
    calls = {}

    def fake_collect_local_report(args, targets):
        calls["target_count"] = len(targets)
        return report

    def fake_record(database_url, payload, *, runtime_metadata=None):
        calls["database_url"] = database_url
        calls["provider_name"] = payload["snapshots"][0]["provider_name"]
        calls["runtime_metadata"] = runtime_metadata
        return [_snapshot_record(provider_name=payload["snapshots"][0]["provider_name"])]

    monkeypatch.setattr(probe_script, "collect_local_report", fake_collect_local_report)
    monkeypatch.setattr(probe_script, "record_provider_resource_probe_payload", fake_record)

    exit_code = probe_script.main(
        [
            "--provider",
            "generation",
            "--database-url",
            "postgresql://pytest/db",
            "--persist",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert calls == {
        "target_count": 1,
        "database_url": "postgresql://pytest/db",
        "provider_name": "pytest-vllm",
        "runtime_metadata": {
            "source": "scripts/probe_provider_resources.py",
            "remote": False,
        },
    }
    assert payload["snapshot_records"][0]["provider_name"] == "pytest-vllm"


def test_probe_provider_resources_rejects_invalid_persistence_options(monkeypatch, capsys) -> None:
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)

    dry_run_exit_code = probe_script.main(["--dry-run", "--persist"])
    missing_db_exit_code = probe_script.main(["--provider", "vllm", "--persist"])

    captured = capsys.readouterr()
    assert dry_run_exit_code == 2
    assert missing_db_exit_code == 1
    assert "dry-run cannot be persisted" in captured.err
    assert "database URL is required" in captured.err


def _snapshot_record(**overrides: object) -> ProviderResourceSnapshotRecord:
    values = {
        "snapshot_id": 7,
        "probe_run_id": UUID("33333333-3333-4333-8333-333333333333"),
        "host": "192.168.20.243",
        "provider_name": "pytest-vllm",
        "provider_type": "vllm",
        "model_id": "/models/qwen",
        "port": 12000,
        "status": "ok",
        "reason_codes": (),
        "match_confidence": "port",
        "process_pid": 9001,
        "process_ppid": 1,
        "process_user": "nexpcx",
        "process_rss_bytes": 2048 * 1024,
        "process_vms_bytes": 4096 * 1024,
        "process_cpu_percent": 3.5,
        "process_uptime_seconds": 90,
        "process_command_preview": "python -m vllm",
        "process_command_hash": "hash",
        "listener_process_name": "python",
        "listener_raw_line": "LISTEN 0 2048 0.0.0.0:12000",
        "gpu_process_name": None,
        "gpu_memory_used_bytes": None,
        "system_total_ram_bytes": 100000 * 1024,
        "system_available_ram_bytes": 80000 * 1024,
        "system_memory_available_percent": 80.0,
        "system_swap_total_bytes": 0,
        "system_swap_used_bytes": 0,
        "system_swap_used_percent": None,
        "collector_error": None,
        "collector_errors": (),
        "report_status": "ok",
        "report_target_count": 1,
        "report_ok_count": 1,
        "report_warning_count": 0,
        "report_critical_count": 0,
        "report_unknown_count": 0,
        "runtime_metadata": {"source": "pytest"},
        "raw_snapshot": {"provider_name": "pytest-vllm"},
        "collected_at": datetime(2026, 7, 29, 14, tzinfo=UTC),
        "created_at": datetime(2026, 7, 29, 14, tzinfo=UTC),
    }
    values.update(overrides)
    return ProviderResourceSnapshotRecord(**values)
