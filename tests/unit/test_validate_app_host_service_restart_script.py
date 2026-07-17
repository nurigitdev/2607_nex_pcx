import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.app_host_service_restart_validation import (
    APP_HOST_RESTART_STATUS_READY,
    AppHostServiceRestartValidationCheck,
    AppHostServiceRestartValidationReport,
)


def _load_validate_app_host_service_restart_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "validate_app_host_service_restart.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_app_host_service_restart_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_app_host_service_restart = _load_validate_app_host_service_restart_module()


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_builder(options):
        captured["options"] = options
        return AppHostServiceRestartValidationReport(
            status=APP_HOST_RESTART_STATUS_READY,
            checked_at=datetime(2026, 7, 17, 2, 3, 4, tzinfo=UTC),
            scope=options.scope,
            checks=(
                AppHostServiceRestartValidationCheck(
                    code="systemctl_available",
                    status="passed",
                    detail="systemctl is available.",
                ),
            ),
        )

    json_output = tmp_path / "restart.json"
    markdown_output = tmp_path / "restart.md"
    monkeypatch.setattr(
        validate_app_host_service_restart,
        "build_app_host_service_restart_validation_report",
        fake_builder,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_app_host_service_restart.py",
            "--scope",
            "system",
            "--app-url",
            "http://127.0.0.1:8000",
            "--restart-web",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = validate_app_host_service_restart.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "ready"
    assert "App Host Service Restart Validation" in markdown_output.read_text(encoding="utf-8")
    assert captured["options"].scope == "system"
    assert captured["options"].restart_web is True
    assert captured["options"].app_base_url == "http://127.0.0.1:8000"
