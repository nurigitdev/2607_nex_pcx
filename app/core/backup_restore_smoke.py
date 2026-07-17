"""Backup and restore smoke planning for NeX_PCX operations."""

import shlex
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app.core.config import Settings

BACKUP_RESTORE_SMOKE_VERSION = 1

BACKUP_CHECK_PASSED = "passed"
BACKUP_CHECK_WARNING = "warning"
BACKUP_CHECK_FAILED = "failed"

BACKUP_SMOKE_STATUS_READY = "ready"
BACKUP_SMOKE_STATUS_WARNING = "warning"
BACKUP_SMOKE_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class BackupRestoreSmokeCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class BackupRestoreCommand:
    code: str
    description: str
    command: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


@dataclass(frozen=True)
class BackupRestoreSmokeReport:
    status: str
    checked_at: datetime
    backup_dir: str
    checks: tuple[BackupRestoreSmokeCheck, ...]
    commands: tuple[BackupRestoreCommand, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(BACKUP_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(BACKUP_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(BACKUP_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


def build_backup_restore_smoke_report(
    settings: Settings,
    *,
    backup_dir: Path,
    restore_database_url: str | None = None,
    project_root: Path | None = None,
    checked_at: datetime | None = None,
) -> BackupRestoreSmokeReport:
    root = project_root or Path.cwd()
    resolved_backup_dir = backup_dir if backup_dir.is_absolute() else root / backup_dir
    checks = (
        _tool_check("pg_dump"),
        _tool_check("pg_restore"),
        _tool_check("psql"),
        _database_url_check(settings.database_url),
        _restore_database_url_check(settings.database_url, restore_database_url),
        _backup_dir_check(resolved_backup_dir),
        _directory_check("upload_storage_dir", settings.upload_storage_dir, root=root),
        _directory_check("artifacts_dir", Path("artifacts"), root=root, required=False),
    )
    commands = _build_commands(
        settings,
        backup_dir=resolved_backup_dir,
        restore_database_url=restore_database_url,
        root=root,
    )
    return BackupRestoreSmokeReport(
        status=_overall_status(checks),
        checked_at=checked_at or datetime.now(UTC),
        backup_dir=str(resolved_backup_dir),
        checks=checks,
        commands=commands,
    )


def backup_restore_smoke_report_payload(
    report: BackupRestoreSmokeReport,
) -> dict[str, object]:
    return {
        "version": BACKUP_RESTORE_SMOKE_VERSION,
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "backup_dir": report.backup_dir,
        "check_count": report.check_count,
        "passed_count": report.passed_count,
        "warning_count": report.warning_count,
        "failed_count": report.failed_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "metadata": dict(check.metadata or {}),
            }
            for check in report.checks
        ],
        "commands": [
            {
                "code": command.code,
                "description": command.description,
                "command": list(command.command),
                "shell_command": command.shell_command,
            }
            for command in report.commands
        ],
    }


def render_backup_restore_smoke_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# NeX_PCX Backup + Restore Smoke Report",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Overall Status: {_text(payload.get('status'))}",
        f"- Backup Dir: `{_text(payload.get('backup_dir'))}`",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} |"
        )

    lines.extend(["", "## Command Manifest", ""])
    for command in payload.get("commands", []):
        command_payload = _dict(command)
        lines.extend(
            [
                f"### {_text(command_payload.get('code'))}",
                "",
                _text(command_payload.get("description")),
                "",
                "```bash",
                _text(command_payload.get("shell_command")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _overall_status(checks: tuple[BackupRestoreSmokeCheck, ...]) -> str:
    if any(check.status == BACKUP_CHECK_FAILED for check in checks):
        return BACKUP_SMOKE_STATUS_BLOCKED
    if any(check.status == BACKUP_CHECK_WARNING for check in checks):
        return BACKUP_SMOKE_STATUS_WARNING
    return BACKUP_SMOKE_STATUS_READY


def _tool_check(tool_name: str) -> BackupRestoreSmokeCheck:
    resolved = shutil.which(tool_name)
    if not resolved:
        return BackupRestoreSmokeCheck(
            code=f"{tool_name}_available",
            status=BACKUP_CHECK_WARNING,
            detail=f"{tool_name} was not found on PATH.",
            metadata={"available": False},
        )
    return BackupRestoreSmokeCheck(
        code=f"{tool_name}_available",
        status=BACKUP_CHECK_PASSED,
        detail=f"{tool_name} is available.",
        metadata={"available": True, "path": resolved},
    )


def _database_url_check(database_url: str | None) -> BackupRestoreSmokeCheck:
    if not database_url:
        return BackupRestoreSmokeCheck(
            code="database_url",
            status=BACKUP_CHECK_FAILED,
            detail="NEX_PCX_DATABASE_URL is required for database backup.",
            metadata={"configured": False},
        )
    return BackupRestoreSmokeCheck(
        code="database_url",
        status=BACKUP_CHECK_PASSED,
        detail="Source database URL is configured.",
        metadata={"configured": True, "masked": mask_database_url(database_url)},
    )


def _restore_database_url_check(
    database_url: str | None,
    restore_database_url: str | None,
) -> BackupRestoreSmokeCheck:
    if not restore_database_url:
        return BackupRestoreSmokeCheck(
            code="restore_database_url",
            status=BACKUP_CHECK_WARNING,
            detail="Restore database URL is not configured; restore smoke will be manual.",
            metadata={"configured": False},
        )
    if database_url and _normalized_url(database_url) == _normalized_url(restore_database_url):
        return BackupRestoreSmokeCheck(
            code="restore_database_url",
            status=BACKUP_CHECK_FAILED,
            detail="Restore database URL must not match the source database URL.",
            metadata={
                "configured": True,
                "source_masked": mask_database_url(database_url),
                "restore_masked": mask_database_url(restore_database_url),
            },
        )
    return BackupRestoreSmokeCheck(
        code="restore_database_url",
        status=BACKUP_CHECK_PASSED,
        detail="Restore database URL is configured and distinct from source.",
        metadata={"configured": True, "masked": mask_database_url(restore_database_url)},
    )


def _backup_dir_check(backup_dir: Path) -> BackupRestoreSmokeCheck:
    if backup_dir.exists() and not backup_dir.is_dir():
        return BackupRestoreSmokeCheck(
            code="backup_dir",
            status=BACKUP_CHECK_FAILED,
            detail="Backup output path exists but is not a directory.",
            metadata={"path": str(backup_dir), "exists": True, "is_dir": False},
        )
    parent = backup_dir if backup_dir.exists() else backup_dir.parent
    writable = parent.exists() and parent.is_dir()
    status = BACKUP_CHECK_PASSED if writable else BACKUP_CHECK_WARNING
    detail = (
        "Backup output directory is writable or can be created."
        if writable
        else "Backup output directory parent does not exist yet."
    )
    return BackupRestoreSmokeCheck(
        code="backup_dir",
        status=status,
        detail=detail,
        metadata={
            "path": str(backup_dir),
            "exists": backup_dir.exists(),
            "parent": str(parent),
            "parent_exists": parent.exists(),
        },
    )


def _directory_check(
    code: str,
    path: Path,
    *,
    root: Path,
    required: bool = True,
) -> BackupRestoreSmokeCheck:
    resolved_path = path if path.is_absolute() else root / path
    metadata = {
        "path": str(path),
        "resolved_path": str(resolved_path),
        "exists": resolved_path.exists(),
        "is_dir": resolved_path.is_dir(),
    }
    if not resolved_path.exists():
        status = BACKUP_CHECK_WARNING if not required else BACKUP_CHECK_FAILED
        return BackupRestoreSmokeCheck(
            code=code,
            status=status,
            detail=f"{code} does not exist.",
            metadata=metadata,
        )
    if not resolved_path.is_dir():
        return BackupRestoreSmokeCheck(
            code=code,
            status=BACKUP_CHECK_FAILED,
            detail=f"{code} exists but is not a directory.",
            metadata=metadata,
        )
    return BackupRestoreSmokeCheck(
        code=code,
        status=BACKUP_CHECK_PASSED,
        detail=f"{code} is available for backup.",
        metadata=metadata,
    )


def _build_commands(
    settings: Settings,
    *,
    backup_dir: Path,
    restore_database_url: str | None,
    root: Path,
) -> tuple[BackupRestoreCommand, ...]:
    db_dump = backup_dir / "nex_pcx_db.dump"
    uploads_archive = backup_dir / "uploads.tar.gz"
    artifacts_archive = backup_dir / "artifacts.tar.gz"
    upload_dir = (
        settings.upload_storage_dir
        if settings.upload_storage_dir.is_absolute()
        else root / settings.upload_storage_dir
    )
    artifacts_dir = root / "artifacts"
    commands = [
        BackupRestoreCommand(
            code="database_backup",
            description="Create a PostgreSQL custom-format database dump.",
            command=(
                "pg_dump",
                "--format=custom",
                "--file",
                str(db_dump),
                "${NEX_PCX_DATABASE_URL}",
            ),
        ),
        BackupRestoreCommand(
            code="upload_storage_backup",
            description="Archive uploaded source files.",
            command=(
                "tar",
                "-czf",
                str(uploads_archive),
                "-C",
                str(upload_dir.parent),
                upload_dir.name,
            ),
        ),
        BackupRestoreCommand(
            code="artifacts_backup",
            description="Archive operator evidence and generated artifacts.",
            command=(
                "tar",
                "-czf",
                str(artifacts_archive),
                "-C",
                str(artifacts_dir.parent),
                artifacts_dir.name,
            ),
        ),
        BackupRestoreCommand(
            code="dump_list_smoke",
            description="Verify the dump catalog can be read.",
            command=("pg_restore", "--list", str(db_dump)),
        ),
    ]
    if restore_database_url:
        commands.append(
            BackupRestoreCommand(
                code="restore_connection_smoke",
                description="Verify the restore target connection before any restore.",
                command=(
                    "psql",
                    "${NEX_PCX_RESTORE_DATABASE_URL}",
                    "-c",
                    "SELECT 1 AS restore_target_ok",
                ),
            )
        )
    return tuple(commands)


def mask_database_url(database_url: str | None) -> str | None:
    if not database_url:
        return None
    try:
        parsed = urlsplit(database_url)
        if not parsed.scheme or not parsed.netloc:
            return "***"
        username = parsed.username or ""
        userinfo = f"{username}:***@" if username else "***@"
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme, f"{userinfo}{host}{port}", parsed.path, "", ""))
    except ValueError:
        return "***"


def _normalized_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
