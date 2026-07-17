import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.operations_startup_validation import (
    STARTUP_CHECK_PASSED,
    STARTUP_CHECK_WARNING,
    STARTUP_STATUS_BLOCKED,
    STARTUP_STATUS_WARNING,
    OperationsStartupValidationCheck,
    OperationsStartupValidationReport,
)


def _load_validate_operations_startup_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "validate_operations_startup.py"
    spec = importlib.util.spec_from_file_location("validate_operations_startup_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_operations_startup = _load_validate_operations_startup_module()


def make_report(status: str) -> OperationsStartupValidationReport:
    check_status = (
        STARTUP_CHECK_WARNING if status == STARTUP_STATUS_WARNING else STARTUP_CHECK_PASSED
    )
    return OperationsStartupValidationReport(
        status=status,
        checked_at=datetime(2026, 7, 17, 1, 2, 3, tzinfo=UTC),
        checks=(
            OperationsStartupValidationCheck(
                code="script_test",
                status=check_status,
                detail="script test detail",
            ),
        ),
    )


def test_main_passes_cli_options_to_runner(monkeypatch, capsys) -> None:
    captured = {}

    def fake_build_report(settings, *, options):
        captured["settings"] = settings
        captured["options"] = options
        return make_report(STARTUP_STATUS_WARNING)

    monkeypatch.setattr(
        validate_operations_startup,
        "get_settings",
        lambda: Settings(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(
        validate_operations_startup,
        "build_operations_startup_validation_report",
        fake_build_report,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_operations_startup.py",
            "--database-url",
            "postgresql://override/db",
            "--app-url",
            "http://127.0.0.1:8000",
            "--health-timeout-seconds",
            "1.5",
            "--run-provider-preflight",
            "--profile-name",
            "kure_v1",
            "--include-inactive-routes",
            "--strict",
            "--pretty",
        ],
    )

    exit_code = validate_operations_startup.main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert captured["settings"].database_url == "postgresql://override/db"
    assert captured["options"].app_base_url == "http://127.0.0.1:8000"
    assert captured["options"].health_timeout_seconds == 1.5
    assert captured["options"].run_provider_preflight is True
    assert captured["options"].profile_name == "kure_v1"
    assert captured["options"].include_inactive_routes is True
    assert '"status": "warning"' in output
    assert "\n  " in output


def test_main_allows_warning_without_strict(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_operations_startup,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        validate_operations_startup,
        "build_operations_startup_validation_report",
        lambda settings, *, options: make_report(STARTUP_STATUS_WARNING),
    )
    monkeypatch.setattr(sys, "argv", ["validate_operations_startup.py"])

    assert validate_operations_startup.main() == 0


def test_main_blocks_failed_report(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_operations_startup,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        validate_operations_startup,
        "build_operations_startup_validation_report",
        lambda settings, *, options: make_report(STARTUP_STATUS_BLOCKED),
    )
    monkeypatch.setattr(sys, "argv", ["validate_operations_startup.py"])

    assert validate_operations_startup.main() == 1
