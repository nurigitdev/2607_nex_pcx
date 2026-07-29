import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "collect_dgx_snapshots.py"
    spec = importlib.util.spec_from_file_location("collect_dgx_snapshots_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect_script = _load_script_module()


def test_collect_dgx_snapshots_dry_run_writes_plan_without_database_secret(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    json_output = tmp_path / "collection.json"
    markdown_output = tmp_path / "collection.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dgx_snapshots.py",
            "--database-url",
            "postgresql://user:secret@db/app",
            "--workdir",
            str(tmp_path),
            "--dry-run",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = collect_script.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert printed["status"] == "planned"
    assert payload["plan"]["status"] == "ready"
    assert len(payload["plan"]["commands"]) == 2
    assert "secret" not in json.dumps(payload)
    assert "DGX Snapshot Collection Evidence" in markdown_output.read_text(encoding="utf-8")


def test_collect_dgx_snapshots_live_run_persists_both_components_and_reports_attention(
    monkeypatch,
    tmp_path,
) -> None:
    json_output = tmp_path / "collection.json"
    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": tuple(command), "env": dict(kwargs["env"])})
        if any("scrape_vllm_runtime_metrics.py" in part for part in command):
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"provider_name": "vllm", "snapshot_record": {"snapshot_id": 10}}),
                "",
            )
        return subprocess.CompletedProcess(
            command,
            1,
            json.dumps(
                {
                    "status": "critical",
                    "snapshot_records": [{"snapshot_id": 20, "provider_name": "qwen"}],
                }
            ),
            "",
        )

    monkeypatch.setattr(collect_script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dgx_snapshots.py",
            "--database-url",
            "postgresql://user:secret@db/app",
            "--workdir",
            str(tmp_path),
            "--max-cycles",
            "1",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = collect_script.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "attention"
    assert payload["cycles"][0]["status"] == "attention"
    assert [result["status"] for result in payload["cycles"][0]["results"]] == [
        "collected",
        "attention",
    ]
    assert all(
        call["env"]["NEX_PCX_DATABASE_URL"] == "postgresql://user:secret@db/app" for call in calls
    )
    assert "secret" not in json.dumps(payload)


def test_collect_dgx_snapshots_live_run_can_collect_multiple_cycles(
    monkeypatch,
    tmp_path,
) -> None:
    json_output = tmp_path / "collection.json"
    run_count = 0
    sleeps = []

    def fake_run(command, **kwargs):
        nonlocal run_count
        run_count += 1
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"snapshot_record": {"snapshot_id": run_count}}),
            "",
        )

    monkeypatch.setattr(collect_script.subprocess, "run", fake_run)
    monkeypatch.setattr(collect_script.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dgx_snapshots.py",
            "--component",
            "vllm",
            "--database-url",
            "postgresql://db/app",
            "--workdir",
            str(tmp_path),
            "--max-cycles",
            "2",
            "--interval-seconds",
            "0.25",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = collect_script.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "completed"
    assert payload["cycle_count"] == 2
    assert run_count == 2
    assert sleeps == [0.25]


def test_collect_dgx_snapshots_blocks_live_run_without_database_url(
    monkeypatch,
    tmp_path,
) -> None:
    json_output = tmp_path / "blocked.json"
    monkeypatch.delenv("NEX_PCX_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dgx_snapshots.py",
            "--workdir",
            str(tmp_path),
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = collect_script.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert payload["plan"]["failed_count"] == 1


def test_collect_dgx_snapshots_command_errors_are_redacted_and_reported(
    monkeypatch,
    tmp_path,
) -> None:
    json_output = tmp_path / "failed.json"

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=0.01, output="partial")

    monkeypatch.setattr(collect_script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dgx_snapshots.py",
            "--component",
            "vllm",
            "--database-url",
            "postgresql://user:secret@db/app",
            "--workdir",
            str(tmp_path),
            "--command-timeout-seconds",
            "0.01",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = collect_script.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    result = payload["cycles"][0]["results"][0]

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert result["status"] == "failed"
    assert "timed out" in result["error_message"]
    assert "secret" not in json.dumps(payload)
