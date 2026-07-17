from datetime import UTC, datetime
from pathlib import Path

from app.core import backup_restore_smoke
from app.core.backup_restore_smoke import (
    BACKUP_CHECK_FAILED,
    BACKUP_SMOKE_STATUS_BLOCKED,
    BACKUP_SMOKE_STATUS_READY,
    BACKUP_SMOKE_STATUS_WARNING,
    backup_restore_smoke_report_payload,
    build_backup_restore_smoke_report,
    mask_database_url,
    render_backup_restore_smoke_markdown,
)
from app.core.config import Settings


def _settings(tmp_path: Path, **overrides) -> Settings:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    values = {
        "database_url": "postgresql://app:secret@127.0.0.1:5432/nex_pcx_app",
        "upload_storage_dir": upload_dir,
    }
    values.update(overrides)
    return Settings(**values)


def test_backup_restore_smoke_ready_with_tools_and_distinct_restore_url(
    monkeypatch,
    tmp_path,
    ) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "backups").mkdir()
    monkeypatch.setattr(backup_restore_smoke.shutil, "which", lambda tool: f"/usr/bin/{tool}")

    report = build_backup_restore_smoke_report(
        _settings(tmp_path),
        backup_dir=tmp_path / "backups" / "latest",
        restore_database_url="postgresql://restore:secret@127.0.0.1:5432/nex_pcx_restore",
        project_root=tmp_path,
        checked_at=datetime(2026, 7, 17, 7, 8, 9, tzinfo=UTC),
    )
    payload = backup_restore_smoke_report_payload(report)
    markdown = render_backup_restore_smoke_markdown(payload)

    assert report.status == BACKUP_SMOKE_STATUS_READY
    assert payload["checked_at_label"] == "2026-07-17 07:08:09"
    assert payload["commands"][0]["code"] == "database_backup"
    assert "${NEX_PCX_DATABASE_URL}" in payload["commands"][0]["shell_command"]
    assert "${NEX_PCX_RESTORE_DATABASE_URL}" in payload["commands"][-1]["shell_command"]
    assert "restore:secret" not in str(payload["commands"])
    assert "restore_connection_smoke" in markdown
    assert "postgresql://app:***@127.0.0.1:5432/nex_pcx_app" in str(payload)


def test_backup_restore_smoke_blocks_missing_source_and_same_restore_url(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(backup_restore_smoke.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    database_url = "postgresql://app:secret@127.0.0.1:5432/nex_pcx_app"

    report = build_backup_restore_smoke_report(
        _settings(tmp_path, database_url=None),
        backup_dir=tmp_path / "backups",
        restore_database_url=None,
        project_root=tmp_path,
    )
    assert report.status == BACKUP_SMOKE_STATUS_BLOCKED

    report = build_backup_restore_smoke_report(
        _settings(tmp_path, database_url=database_url),
        backup_dir=tmp_path / "backups",
        restore_database_url=database_url,
        project_root=tmp_path,
    )
    payload = backup_restore_smoke_report_payload(report)
    failed_codes = {
        check["code"] for check in payload["checks"] if check["status"] == BACKUP_CHECK_FAILED
    }

    assert report.status == BACKUP_SMOKE_STATUS_BLOCKED
    assert "restore_database_url" in failed_codes


def test_backup_restore_smoke_warns_for_missing_tools_optional_artifacts_and_restore(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(backup_restore_smoke.shutil, "which", lambda tool: None)

    report = build_backup_restore_smoke_report(
        _settings(tmp_path),
        backup_dir=tmp_path / "missing-parent" / "latest",
        project_root=tmp_path,
    )
    payload = backup_restore_smoke_report_payload(report)
    warning_codes = {
        check["code"] for check in payload["checks"] if check["status"] == "warning"
    }

    assert report.status == BACKUP_SMOKE_STATUS_WARNING
    assert "pg_dump_available" in warning_codes
    assert "restore_database_url" in warning_codes
    assert "backup_dir" in warning_codes
    assert "artifacts_dir" in warning_codes


def test_backup_restore_smoke_blocks_invalid_directories(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(backup_restore_smoke.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    backup_file = tmp_path / "backup-file"
    backup_file.write_text("not a directory", encoding="utf-8")
    upload_file = tmp_path / "upload-file"
    upload_file.write_text("not a directory", encoding="utf-8")

    report = build_backup_restore_smoke_report(
        _settings(tmp_path, upload_storage_dir=upload_file),
        backup_dir=backup_file,
        restore_database_url="postgresql://restore:secret@example/restore",
        project_root=tmp_path,
    )

    assert report.status == BACKUP_SMOKE_STATUS_BLOCKED
    assert report.failed_count == 2


def test_mask_database_url_handles_edge_cases() -> None:
    assert mask_database_url(None) is None
    assert mask_database_url("not-a-url") == "***"
    assert mask_database_url("postgresql://example/db") == "postgresql://***@example/db"
