import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.emergency_recovery_index import (
    EmergencyRecoveryIndex,
    RecoveryCommand,
    RecoveryScenario,
)


def _load_render_emergency_recovery_index_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "render_emergency_recovery_index.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_emergency_recovery_index_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render_emergency_recovery_index = _load_render_emergency_recovery_index_module()


def make_index() -> EmergencyRecoveryIndex:
    return EmergencyRecoveryIndex(
        generated_at=datetime(2026, 7, 17, 12, 13, 14, tzinfo=UTC),
        workdir="/tmp/project",
        app_url="http://app",
        provider_host="provider-host",
        commands=(
            RecoveryCommand(
                code="script_command",
                title="Script Command",
                description="script command",
                command=("echo", "ok"),
            ),
        ),
        scenarios=(
            RecoveryScenario(
                code="script_scenario",
                title="Script Scenario",
                severity="medium",
                first_check="first",
                command_codes=("script_command",),
                checklist=("check one",),
                stop_condition="stop",
            ),
        ),
    )


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "recovery" / "index.json"
    markdown_output = tmp_path / "recovery" / "index.md"

    def fake_build_index(*, workdir, app_url, provider_host, artifacts_dir):
        captured["workdir"] = workdir
        captured["app_url"] = app_url
        captured["provider_host"] = provider_host
        captured["artifacts_dir"] = artifacts_dir
        return make_index()

    monkeypatch.setattr(
        render_emergency_recovery_index,
        "build_emergency_recovery_index",
        fake_build_index,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_emergency_recovery_index.py",
            "--workdir",
            str(tmp_path),
            "--app-url",
            "http://app.local",
            "--provider-host",
            "provider.local",
            "--artifacts-dir",
            "ops-artifacts",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    assert render_emergency_recovery_index.main() == 0
    assert captured["workdir"] == tmp_path
    assert captured["app_url"] == "http://app.local"
    assert captured["provider_host"] == "provider.local"
    assert captured["artifacts_dir"] == Path("ops-artifacts")
    assert json.loads(json_output.read_text(encoding="utf-8"))["command_count"] == 1
    assert "Script Scenario" in markdown_output.read_text(encoding="utf-8")


def test_main_prints_json_or_markdown(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        render_emergency_recovery_index,
        "build_emergency_recovery_index",
        lambda *, workdir, app_url, provider_host, artifacts_dir: make_index(),
    )

    monkeypatch.setattr(sys, "argv", ["render_emergency_recovery_index.py"])
    assert render_emergency_recovery_index.main() == 0
    assert '"command_count": 1' in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["render_emergency_recovery_index.py", "--print-markdown"],
    )
    assert render_emergency_recovery_index.main() == 0
    assert "# NeX_PCX Emergency Recovery Command Index" in capsys.readouterr().out
