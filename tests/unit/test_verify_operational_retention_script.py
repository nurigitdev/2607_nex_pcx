import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.operational_retention_verification import (
    RETENTION_CHECK_PASSED,
    RETENTION_CHECK_WARNING,
    RETENTION_STATUS_BLOCKED,
    RETENTION_STATUS_WARNING,
    OperationalRetentionCheck,
    OperationalRetentionVerificationReport,
)


def _load_verify_operational_retention_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "verify_operational_retention.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_operational_retention_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_operational_retention = _load_verify_operational_retention_module()


def make_report(status: str) -> OperationalRetentionVerificationReport:
    check_status = (
        RETENTION_CHECK_WARNING
        if status == RETENTION_STATUS_WARNING
        else RETENTION_CHECK_PASSED
    )
    return OperationalRetentionVerificationReport(
        status=status,
        checked_at=datetime(2026, 7, 17, 10, 11, 12, tzinfo=UTC),
        project_root="/tmp/project",
        checks=(
            OperationalRetentionCheck(
                code="script_test",
                status=check_status,
                detail="script test detail",
            ),
        ),
    )


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "retention" / "retention.json"
    markdown_output = tmp_path / "retention" / "retention.md"

    def fake_build_report(
        database_url,
        *,
        project_root,
        max_retention_days,
        artifact_retention_days,
    ):
        captured["database_url"] = database_url
        captured["project_root"] = project_root
        captured["max_retention_days"] = max_retention_days
        captured["artifact_retention_days"] = artifact_retention_days
        return make_report(RETENTION_STATUS_WARNING)

    monkeypatch.setattr(
        verify_operational_retention,
        "get_settings",
        lambda: Settings(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(
        verify_operational_retention,
        "build_operational_retention_verification_report",
        fake_build_report,
    )
    monkeypatch.setattr(
        verify_operational_retention,
        "render_operational_retention_verification_markdown",
        lambda payload: "# retention\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_operational_retention.py",
            "--database-url",
            "postgresql://override/db",
            "--project-root",
            str(tmp_path),
            "--max-retention-days",
            "60",
            "--artifact-retention-days",
            "14",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = verify_operational_retention.main()

    assert exit_code == 0
    assert captured["database_url"] == "postgresql://override/db"
    assert captured["project_root"] == tmp_path
    assert captured["max_retention_days"] == 60
    assert captured["artifact_retention_days"] == 14
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "warning"
    assert markdown_output.read_text(encoding="utf-8") == "# retention\n"


def test_main_prints_json_and_handles_nonzero_exit_codes(monkeypatch, capsys) -> None:
    def fake_warning_report(
        database_url,
        *,
        project_root,
        max_retention_days,
        artifact_retention_days,
    ):
        return make_report(RETENTION_STATUS_WARNING)

    def fake_blocked_report(
        database_url,
        *,
        project_root,
        max_retention_days,
        artifact_retention_days,
    ):
        return make_report(RETENTION_STATUS_BLOCKED)

    monkeypatch.setattr(
        verify_operational_retention,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        verify_operational_retention,
        "build_operational_retention_verification_report",
        fake_warning_report,
    )
    monkeypatch.setattr(sys, "argv", ["verify_operational_retention.py"])

    assert verify_operational_retention.main() == 0
    assert '"status": "warning"' in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["verify_operational_retention.py", "--strict"])
    assert verify_operational_retention.main() == 1

    monkeypatch.setattr(
        verify_operational_retention,
        "build_operational_retention_verification_report",
        fake_blocked_report,
    )
    monkeypatch.setattr(sys, "argv", ["verify_operational_retention.py"])
    assert verify_operational_retention.main() == 1
