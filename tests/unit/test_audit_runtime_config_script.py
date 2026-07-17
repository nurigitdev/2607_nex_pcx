import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.runtime_config_audit import (
    CONFIG_AUDIT_STATUS_BLOCKED,
    CONFIG_AUDIT_STATUS_WARNING,
    CONFIG_CHECK_PASSED,
    CONFIG_CHECK_WARNING,
    RuntimeConfigAuditCheck,
    RuntimeConfigAuditReport,
)


def _load_audit_runtime_config_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "audit_runtime_config.py"
    spec = importlib.util.spec_from_file_location("audit_runtime_config_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_runtime_config = _load_audit_runtime_config_module()


def make_report(status: str) -> RuntimeConfigAuditReport:
    check_status = (
        CONFIG_CHECK_WARNING if status == CONFIG_AUDIT_STATUS_WARNING else CONFIG_CHECK_PASSED
    )
    return RuntimeConfigAuditReport(
        status=status,
        checked_at=datetime(2026, 7, 17, 6, 7, 8, tzinfo=UTC),
        checks=(
            RuntimeConfigAuditCheck(
                code="script_test",
                status=check_status,
                detail="script test detail",
            ),
        ),
    )


def test_settings_from_args_applies_overrides(tmp_path) -> None:
    args = type(
        "Args",
        (),
        {
            "database_url": "postgresql://override/db",
            "test_database_url": "postgresql://test/db",
            "environment": "production",
            "upload_storage_dir": str(tmp_path / "uploads"),
            "models_dir": str(tmp_path / "models"),
            "embedding_provider_mode": "remote",
            "remote_provider_url": "http://provider",
            "require_route_readiness": True,
            "route_readiness_failure_mode": "defer",
            "route_readiness_defer_seconds": 120,
        },
    )()

    settings = audit_runtime_config._settings_from_args(args)

    assert settings.database_url == "postgresql://override/db"
    assert settings.test_database_url == "postgresql://test/db"
    assert settings.environment == "production"
    assert settings.upload_storage_dir == tmp_path / "uploads"
    assert settings.embedding_models_dir == tmp_path / "models"
    assert settings.embedding_provider_mode == "remote"
    assert settings.remote_embedding_provider_url == "http://provider"
    assert settings.embedding_require_route_readiness is True
    assert settings.embedding_route_readiness_failure_mode == "defer"
    assert settings.embedding_route_readiness_defer_seconds == 120


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "audit" / "runtime.json"
    markdown_output = tmp_path / "audit" / "runtime.md"

    def fake_build_report(settings, *, project_root):
        captured["settings"] = settings
        captured["project_root"] = project_root
        return make_report(CONFIG_AUDIT_STATUS_WARNING)

    monkeypatch.setattr(
        audit_runtime_config,
        "get_settings",
        lambda: Settings(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(
        audit_runtime_config,
        "build_runtime_config_audit_report",
        fake_build_report,
    )
    monkeypatch.setattr(
        audit_runtime_config,
        "render_runtime_config_audit_markdown",
        lambda payload: "# audit\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_runtime_config.py",
            "--database-url",
            "postgresql://override/db",
            "--environment",
            "production",
            "--project-root",
            str(tmp_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = audit_runtime_config.main()

    assert exit_code == 0
    assert captured["settings"].database_url == "postgresql://override/db"
    assert captured["settings"].environment == "production"
    assert captured["project_root"] == tmp_path
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "warning"
    assert markdown_output.read_text(encoding="utf-8") == "# audit\n"


def test_main_prints_json_and_handles_nonzero_exit_codes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        audit_runtime_config,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        audit_runtime_config,
        "build_runtime_config_audit_report",
        lambda settings, *, project_root: make_report(CONFIG_AUDIT_STATUS_WARNING),
    )
    monkeypatch.setattr(sys, "argv", ["audit_runtime_config.py"])

    assert audit_runtime_config.main() == 0
    assert '"status": "warning"' in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["audit_runtime_config.py", "--strict"])
    assert audit_runtime_config.main() == 1

    monkeypatch.setattr(
        audit_runtime_config,
        "build_runtime_config_audit_report",
        lambda settings, *, project_root: make_report(CONFIG_AUDIT_STATUS_BLOCKED),
    )
    monkeypatch.setattr(sys, "argv", ["audit_runtime_config.py"])
    assert audit_runtime_config.main() == 1
