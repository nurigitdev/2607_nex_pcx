import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.foreground_go_live_summary import (
    ForegroundGoLiveSummary,
    ForegroundGoLiveSummaryCheck,
)


def _load_summarize_foreground_go_live_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "summarize_foreground_go_live.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_foreground_go_live_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summarize_foreground_go_live = _load_summarize_foreground_go_live_module()


def test_main_writes_summary_outputs(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_builder(options):
        captured["options"] = options
        return ForegroundGoLiveSummary(
            status="warning",
            checked_at=datetime(2026, 7, 17, 10, 20, 21, tzinfo=UTC),
            workdir=str(options.workdir),
            checks=(
                ForegroundGoLiveSummaryCheck(
                    code="foreground_operations",
                    status="passed",
                    detail="ok",
                    path="artifacts/foreground_operations_validation.json",
                    required=True,
                    evidence_status="warning",
                ),
            ),
        )

    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"
    monkeypatch.setattr(
        summarize_foreground_go_live,
        "build_foreground_go_live_summary",
        fake_builder,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_foreground_go_live.py",
            "--workdir",
            str(tmp_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = summarize_foreground_go_live.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "warning"
    assert "Foreground Go-Live Evidence Summary" in markdown_output.read_text(encoding="utf-8")
    assert captured["options"].workdir == str(tmp_path)
