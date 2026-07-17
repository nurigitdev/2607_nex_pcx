import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.backup_restore_smoke import (
    BACKUP_CHECK_PASSED,
    BACKUP_CHECK_WARNING,
    BACKUP_SMOKE_STATUS_BLOCKED,
    BACKUP_SMOKE_STATUS_WARNING,
    BackupRestoreCommand,
    BackupRestoreSmokeCheck,
    BackupRestoreSmokeReport,
)
from app.core.config import Settings


def _load_run_backup_restore_smoke_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_backup_restore_smoke.py"
    spec = importlib.util.spec_from_file_location("run_backup_restore_smoke_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_backup_restore_smoke = _load_run_backup_restore_smoke_module()


def make_report(status: str) -> BackupRestoreSmokeReport:
    check_status = (
        BACKUP_CHECK_WARNING if status == BACKUP_SMOKE_STATUS_WARNING else BACKUP_CHECK_PASSED
    )
    return BackupRestoreSmokeReport(
        status=status,
        checked_at=datetime(2026, 7, 17, 8, 9, 10, tzinfo=UTC),
        backup_dir="/tmp/backups/latest",
        checks=(
            BackupRestoreSmokeCheck(
                code="script_test",
                status=check_status,
                detail="script test detail",
            ),
        ),
        commands=(
            BackupRestoreCommand(
                code="database_backup",
                description="backup",
                command=("pg_dump", "--file", "db.dump", "${NEX_PCX_DATABASE_URL}"),
            ),
        ),
    )


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "smoke" / "backup.json"
    markdown_output = tmp_path / "smoke" / "backup.md"

    def fake_build_report(settings, *, backup_dir, restore_database_url, project_root):
        captured["settings"] = settings
        captured["backup_dir"] = backup_dir
        captured["restore_database_url"] = restore_database_url
        captured["project_root"] = project_root
        return make_report(BACKUP_SMOKE_STATUS_WARNING)

    monkeypatch.setattr(
        run_backup_restore_smoke,
        "get_settings",
        lambda: Settings(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(
        run_backup_restore_smoke,
        "build_backup_restore_smoke_report",
        fake_build_report,
    )
    monkeypatch.setattr(
        run_backup_restore_smoke,
        "render_backup_restore_smoke_markdown",
        lambda payload: "# backup\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backup_restore_smoke.py",
            "--database-url",
            "postgresql://override/db",
            "--restore-database-url",
            "postgresql://restore/db",
            "--backup-dir",
            str(tmp_path / "backups"),
            "--project-root",
            str(tmp_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = run_backup_restore_smoke.main()

    assert exit_code == 0
    assert captured["settings"].database_url == "postgresql://override/db"
    assert captured["restore_database_url"] == "postgresql://restore/db"
    assert captured["project_root"] == tmp_path
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "warning"
    assert markdown_output.read_text(encoding="utf-8") == "# backup\n"


def test_main_prints_json_and_handles_nonzero_exit_codes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_backup_restore_smoke,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        run_backup_restore_smoke,
        "build_backup_restore_smoke_report",
        lambda settings, *, backup_dir, restore_database_url, project_root: make_report(
            BACKUP_SMOKE_STATUS_WARNING
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_backup_restore_smoke.py"])

    assert run_backup_restore_smoke.main() == 0
    assert '"status": "warning"' in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["run_backup_restore_smoke.py", "--strict"])
    assert run_backup_restore_smoke.main() == 1

    monkeypatch.setattr(
        run_backup_restore_smoke,
        "build_backup_restore_smoke_report",
        lambda settings, *, backup_dir, restore_database_url, project_root: make_report(
            BACKUP_SMOKE_STATUS_BLOCKED
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_backup_restore_smoke.py"])
    assert run_backup_restore_smoke.main() == 1
