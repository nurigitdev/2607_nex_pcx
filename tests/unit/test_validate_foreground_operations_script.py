import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.foreground_operations_validation import (
    ForegroundOperationsValidationCheck,
    ForegroundOperationsValidationReport,
)


def _load_validate_foreground_operations_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "validate_foreground_operations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_foreground_operations_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_foreground_operations = _load_validate_foreground_operations_module()


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_builder(options):
        captured["options"] = options
        return ForegroundOperationsValidationReport(
            status="warning",
            checked_at=datetime(2026, 7, 17, 9, 20, 21, tzinfo=UTC),
            app_base_url=options.app_base_url,
            checks=(
                ForegroundOperationsValidationCheck(
                    code="foreground_no_auto_restart_ack",
                    status="warning",
                    detail="acknowledged",
                ),
            ),
        )

    json_output = tmp_path / "foreground.json"
    markdown_output = tmp_path / "foreground.md"
    monkeypatch.setattr(
        validate_foreground_operations,
        "build_foreground_operations_validation_report",
        fake_builder,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_foreground_operations.py",
            "--app-url",
            "http://127.0.0.1:8000",
            "--acknowledge-no-auto-restart",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = validate_foreground_operations.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "warning"
    assert "Foreground Operations Validation" in markdown_output.read_text(encoding="utf-8")
    assert captured["options"].acknowledge_no_auto_restart is True
    assert captured["options"].app_base_url == "http://127.0.0.1:8000"
