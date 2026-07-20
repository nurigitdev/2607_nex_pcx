import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.production_environment_validation import (
    PRODUCTION_CHECK_PASSED,
    PRODUCTION_STATUS_BLOCKED,
    PRODUCTION_STATUS_WARNING,
    ProductionEnvironmentValidationReport,
    ProductionValidationCheck,
)


def _load_validate_production_environment_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "validate_production_environment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_production_environment_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_production_environment = _load_validate_production_environment_module()


def make_report(status: str) -> ProductionEnvironmentValidationReport:
    return ProductionEnvironmentValidationReport(
        status=status,
        checked_at=datetime(2026, 7, 17, 18, 19, 20, tzinfo=UTC),
        guard_checks=(
            ProductionValidationCheck(
                code="script_test",
                status=PRODUCTION_CHECK_PASSED,
                detail="script test detail",
            ),
        ),
        sections=(),
    )


def test_main_writes_outputs_and_passes_options(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "production" / "validation.json"
    markdown_output = tmp_path / "production" / "validation.md"

    def fake_build_report(settings, *, project_root, options):
        captured["settings"] = settings
        captured["project_root"] = project_root
        captured["options"] = options
        return make_report(PRODUCTION_STATUS_WARNING)

    monkeypatch.setattr(
        validate_production_environment,
        "get_settings",
        lambda: Settings(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(
        validate_production_environment,
        "build_production_environment_validation_report",
        fake_build_report,
    )
    monkeypatch.setattr(
        validate_production_environment,
        "render_production_environment_validation_markdown",
        lambda payload: "# production\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_production_environment.py",
            "--database-url",
            "postgresql://override/db",
            "--app-url",
            "http://app",
            "--project-root",
            str(tmp_path),
            "--expected-database-name",
            "nex_pcx_app",
            "--run-provider-preflight",
            "--profile-name",
            "kure_v1",
            "--include-inactive-routes",
            "--allow-non-production",
            "--allow-non-remote-provider",
            "--allow-route-readiness-disabled",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    assert validate_production_environment.main() == 0
    assert captured["settings"].database_url == "postgresql://override/db"
    assert captured["project_root"] == tmp_path
    assert captured["options"].app_url == "http://app"
    assert captured["options"].expected_database_name == "nex_pcx_app"
    assert captured["options"].run_provider_preflight is True
    assert captured["options"].include_inactive_routes is True
    assert captured["options"].require_production_env is False
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "warning"
    assert markdown_output.read_text(encoding="utf-8") == "# production\n"


def test_main_handles_blocked_and_strict_warning(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        validate_production_environment,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        validate_production_environment,
        "build_production_environment_validation_report",
        lambda settings, *, project_root, options: make_report(PRODUCTION_STATUS_WARNING),
    )
    monkeypatch.setattr(sys, "argv", ["validate_production_environment.py"])

    assert validate_production_environment.main() == 0
    assert '"status": "warning"' in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["validate_production_environment.py", "--strict"])
    assert validate_production_environment.main() == 1

    monkeypatch.setattr(
        validate_production_environment,
        "build_production_environment_validation_report",
        lambda settings, *, project_root, options: make_report(PRODUCTION_STATUS_BLOCKED),
    )
    monkeypatch.setattr(sys, "argv", ["validate_production_environment.py"])
    assert validate_production_environment.main() == 1
