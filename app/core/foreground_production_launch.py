"""Foreground production launch planning and evidence rendering."""

from __future__ import annotations

import json
import shlex
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.service_startup_templates import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT

FOREGROUND_PRODUCTION_LAUNCH_VERSION = 1

LAUNCH_CHECK_PASSED = "passed"
LAUNCH_CHECK_WARNING = "warning"
LAUNCH_CHECK_FAILED = "failed"

LAUNCH_PLAN_READY = "ready"
LAUNCH_PLAN_WARNING = "warning"
LAUNCH_PLAN_BLOCKED = "blocked"

LAUNCH_STATUS_PLANNED = "planned"
LAUNCH_STATUS_RUNNING = "running"
LAUNCH_STATUS_EXITED = "exited"
LAUNCH_STATUS_FAILED = "failed"
LAUNCH_STATUS_BLOCKED = "blocked"

DEFAULT_APP_TARGET = "app.main:create_app"
DEFAULT_PYTHON_BIN = "./.venv/bin/python"
DEFAULT_DATABASE_URL_ENV = "NEX_PCX_DATABASE_URL"
DEFAULT_PID_FILE = "artifacts/foreground_production_launch.pid"
DEFAULT_LOG_FILE = "artifacts/foreground_production_launch.log"


@dataclass(frozen=True)
class ForegroundProductionLaunchOptions:
    workdir: str | Path = "."
    python_bin: str = DEFAULT_PYTHON_BIN
    app_target: str = DEFAULT_APP_TARGET
    host: str = DEFAULT_WEB_HOST
    port: int = DEFAULT_WEB_PORT
    database_url_env: str = DEFAULT_DATABASE_URL_ENV
    require_database_url: bool = True
    pid_file: str = DEFAULT_PID_FILE
    log_file: str = DEFAULT_LOG_FILE
    check_port_available: bool = True
    port_check_timeout_seconds: float = 1.0


@dataclass(frozen=True)
class ForegroundProductionLaunchCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ForegroundProductionLaunchPlan:
    status: str
    generated_at: datetime
    workdir: str
    command: tuple[str, ...]
    host: str
    port: int
    health_url: str
    database_url_env: str
    database_url_configured: bool
    pid_file: str
    log_file: str
    checks: tuple[ForegroundProductionLaunchCheck, ...]

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command)

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(LAUNCH_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(LAUNCH_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(LAUNCH_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


@dataclass(frozen=True)
class ForegroundProductionLaunchEvidence:
    status: str
    dry_run: bool
    generated_at: datetime
    plan: ForegroundProductionLaunchPlan
    process_id: int | None = None
    returncode: int | None = None
    started_at: datetime | None = None
    exited_at: datetime | None = None
    message: str = ""
    metadata: dict[str, object] | None = None


def build_foreground_production_launch_plan(
    options: ForegroundProductionLaunchOptions | None = None,
    *,
    generated_at: datetime | None = None,
    environ: Mapping[str, str] | None = None,
    port_available: bool | None = None,
) -> ForegroundProductionLaunchPlan:
    selected_options = _validate_options(options or ForegroundProductionLaunchOptions())
    root = Path(selected_options.workdir)
    selected_environ = environ or {}
    command = (
        selected_options.python_bin,
        "-m",
        "uvicorn",
        selected_options.app_target,
        "--factory",
        "--host",
        selected_options.host,
        "--port",
        str(selected_options.port),
    )
    database_url_configured = bool(selected_environ.get(selected_options.database_url_env))
    checks = (
        _workdir_check(root),
        _python_bin_check(root, selected_options.python_bin),
        _database_url_check(
            env_name=selected_options.database_url_env,
            configured=database_url_configured,
            required=selected_options.require_database_url,
        ),
        _port_check(selected_options, port_available=port_available),
        _pid_log_path_check(
            pid_file=selected_options.pid_file,
            log_file=selected_options.log_file,
        ),
    )
    return ForegroundProductionLaunchPlan(
        status=_overall_plan_status(checks),
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(root),
        command=command,
        host=selected_options.host,
        port=selected_options.port,
        health_url=_health_url(selected_options.host, selected_options.port),
        database_url_env=selected_options.database_url_env,
        database_url_configured=database_url_configured,
        pid_file=selected_options.pid_file,
        log_file=selected_options.log_file,
        checks=checks,
    )


def build_foreground_production_launch_evidence(
    plan: ForegroundProductionLaunchPlan,
    *,
    status: str,
    dry_run: bool,
    process_id: int | None = None,
    returncode: int | None = None,
    started_at: datetime | None = None,
    exited_at: datetime | None = None,
    message: str = "",
    metadata: dict[str, object] | None = None,
    generated_at: datetime | None = None,
) -> ForegroundProductionLaunchEvidence:
    selected_status = status.strip()
    if selected_status not in {
        LAUNCH_STATUS_PLANNED,
        LAUNCH_STATUS_RUNNING,
        LAUNCH_STATUS_EXITED,
        LAUNCH_STATUS_FAILED,
        LAUNCH_STATUS_BLOCKED,
    }:
        raise ValueError("unsupported launch evidence status")
    return ForegroundProductionLaunchEvidence(
        status=selected_status,
        dry_run=dry_run,
        generated_at=generated_at or datetime.now(UTC),
        plan=plan,
        process_id=process_id,
        returncode=returncode,
        started_at=started_at,
        exited_at=exited_at,
        message=message,
        metadata=dict(metadata or {}),
    )


def foreground_production_launch_plan_payload(
    plan: ForegroundProductionLaunchPlan,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_PRODUCTION_LAUNCH_VERSION,
        "status": plan.status,
        "generated_at": plan.generated_at.isoformat(),
        "generated_at_label": plan.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": plan.workdir,
        "command": list(plan.command),
        "shell_command": plan.shell_command,
        "host": plan.host,
        "port": plan.port,
        "health_url": plan.health_url,
        "database_url_env": plan.database_url_env,
        "database_url_configured": plan.database_url_configured,
        "pid_file": plan.pid_file,
        "log_file": plan.log_file,
        "check_count": plan.check_count,
        "passed_count": plan.passed_count,
        "warning_count": plan.warning_count,
        "failed_count": plan.failed_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "metadata": dict(check.metadata or {}),
            }
            for check in plan.checks
        ],
    }


def foreground_production_launch_evidence_payload(
    evidence: ForegroundProductionLaunchEvidence,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_PRODUCTION_LAUNCH_VERSION,
        "status": evidence.status,
        "dry_run": evidence.dry_run,
        "generated_at": evidence.generated_at.isoformat(),
        "generated_at_label": evidence.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "process_id": evidence.process_id,
        "returncode": evidence.returncode,
        "started_at": _iso_or_none(evidence.started_at),
        "started_at_label": _label_or_none(evidence.started_at),
        "exited_at": _iso_or_none(evidence.exited_at),
        "exited_at_label": _label_or_none(evidence.exited_at),
        "message": evidence.message,
        "metadata": dict(evidence.metadata or {}),
        "plan": foreground_production_launch_plan_payload(evidence.plan),
    }


def render_foreground_production_launch_markdown(payload: dict[str, object]) -> str:
    plan = _dict(payload.get("plan"))
    lines = [
        "# Foreground Production Launch Evidence",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Dry Run: {_text(payload.get('dry_run'))}",
        f"- Process ID: {_text(payload.get('process_id'))}",
        f"- Return Code: {_text(payload.get('returncode'))}",
        f"- Message: {_text(payload.get('message'))}",
        "",
        "## Launch Plan",
        "",
        f"- Plan Status: `{_text(plan.get('status'))}`",
        f"- Workdir: `{_text(plan.get('workdir'))}`",
        f"- Host: `{_text(plan.get('host'))}`",
        f"- Port: {_text(plan.get('port'))}",
        f"- Health URL: `{_text(plan.get('health_url'))}`",
        f"- PID File: `{_text(plan.get('pid_file'))}`",
        f"- Log File: `{_text(plan.get('log_file'))}`",
        f"- Database URL Env: `{_text(plan.get('database_url_env'))}`",
        f"- Database URL Configured: {_text(plan.get('database_url_configured'))}",
        f"- Command: `{_text(plan.get('shell_command'))}`",
        "",
        "## Pre-Launch Checks",
        "",
        "| Code | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in plan.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} |"
        )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- Run without `--dry-run` in a supervised terminal to start the app.",
            "- Keep the terminal alive while foreground operation is active.",
            "- Use the PID file and log file paths above for stop and incident evidence.",
            "- Database URL values are never written to this evidence file.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def resolve_launch_path(workdir: str | Path, path: str) -> Path:
    selected_path = Path(path).expanduser()
    if selected_path.is_absolute():
        return selected_path
    return Path(workdir) / selected_path


def _validate_options(
    options: ForegroundProductionLaunchOptions,
) -> ForegroundProductionLaunchOptions:
    workdir = _require_non_empty(str(options.workdir), name="workdir")
    python_bin = _require_non_empty(options.python_bin, name="python_bin")
    app_target = _require_non_empty(options.app_target, name="app_target")
    host = _require_non_empty(options.host, name="host")
    port = _validate_positive_int(options.port, name="port")
    database_url_env = _require_non_empty(options.database_url_env, name="database_url_env")
    pid_file = _require_non_empty(options.pid_file, name="pid_file")
    log_file = _require_non_empty(options.log_file, name="log_file")
    if options.port_check_timeout_seconds <= 0:
        raise ValueError("port_check_timeout_seconds must be greater than zero")
    return ForegroundProductionLaunchOptions(
        workdir=Path(workdir),
        python_bin=python_bin,
        app_target=app_target,
        host=host,
        port=port,
        database_url_env=database_url_env,
        require_database_url=options.require_database_url,
        pid_file=pid_file,
        log_file=log_file,
        check_port_available=options.check_port_available,
        port_check_timeout_seconds=options.port_check_timeout_seconds,
    )


def _workdir_check(root: Path) -> ForegroundProductionLaunchCheck:
    if root.exists() and root.is_dir():
        return ForegroundProductionLaunchCheck(
            code="workdir",
            status=LAUNCH_CHECK_PASSED,
            detail="Working directory exists.",
            metadata={"workdir": str(root)},
        )
    return ForegroundProductionLaunchCheck(
        code="workdir",
        status=LAUNCH_CHECK_FAILED,
        detail="Working directory is missing or is not a directory.",
        metadata={"workdir": str(root)},
    )


def _python_bin_check(root: Path, python_bin: str) -> ForegroundProductionLaunchCheck:
    if "/" not in python_bin and "\\" not in python_bin:
        return ForegroundProductionLaunchCheck(
            code="python_bin",
            status=LAUNCH_CHECK_PASSED,
            detail="Python executable will be resolved from PATH.",
            metadata={"python_bin": python_bin},
        )
    resolved = resolve_launch_path(root, python_bin)
    if resolved.exists() and resolved.is_file():
        return ForegroundProductionLaunchCheck(
            code="python_bin",
            status=LAUNCH_CHECK_PASSED,
            detail="Python executable exists.",
            metadata={"python_bin": python_bin},
        )
    return ForegroundProductionLaunchCheck(
        code="python_bin",
        status=LAUNCH_CHECK_FAILED,
        detail="Python executable path is missing.",
        metadata={"python_bin": python_bin},
    )


def _database_url_check(
    *,
    env_name: str,
    configured: bool,
    required: bool,
) -> ForegroundProductionLaunchCheck:
    if configured:
        return ForegroundProductionLaunchCheck(
            code="database_url",
            status=LAUNCH_CHECK_PASSED,
            detail=f"{env_name} is configured.",
            metadata={"env_name": env_name, "configured": True},
        )
    status = LAUNCH_CHECK_FAILED if required else LAUNCH_CHECK_WARNING
    detail = f"{env_name} is required before launch." if required else f"{env_name} is not set."
    return ForegroundProductionLaunchCheck(
        code="database_url",
        status=status,
        detail=detail,
        metadata={"env_name": env_name, "configured": False},
    )


def _port_check(
    options: ForegroundProductionLaunchOptions,
    *,
    port_available: bool | None,
) -> ForegroundProductionLaunchCheck:
    if not options.check_port_available:
        return ForegroundProductionLaunchCheck(
            code="port_available",
            status=LAUNCH_CHECK_WARNING,
            detail="Port availability check was skipped.",
            metadata={"host": options.host, "port": options.port},
        )
    available = (
        is_tcp_port_available(
            host=options.host,
            port=options.port,
            timeout_seconds=options.port_check_timeout_seconds,
        )
        if port_available is None
        else port_available
    )
    if available:
        return ForegroundProductionLaunchCheck(
            code="port_available",
            status=LAUNCH_CHECK_PASSED,
            detail="Target web port appears available.",
            metadata={"host": options.host, "port": options.port},
        )
    return ForegroundProductionLaunchCheck(
        code="port_available",
        status=LAUNCH_CHECK_FAILED,
        detail="Target web port is already reachable before launch.",
        metadata={"host": options.host, "port": options.port},
    )


def _pid_log_path_check(*, pid_file: str, log_file: str) -> ForegroundProductionLaunchCheck:
    if Path(pid_file) == Path(log_file):
        return ForegroundProductionLaunchCheck(
            code="pid_log_paths",
            status=LAUNCH_CHECK_FAILED,
            detail="PID file and log file must be different paths.",
        )
    return ForegroundProductionLaunchCheck(
        code="pid_log_paths",
        status=LAUNCH_CHECK_PASSED,
        detail="PID file and log file paths are distinct.",
        metadata={"pid_file": pid_file, "log_file": log_file},
    )


def is_tcp_port_available(*, host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout_seconds)
        return probe.connect_ex((probe_host, port)) != 0


def _overall_plan_status(checks: tuple[ForegroundProductionLaunchCheck, ...]) -> str:
    if any(check.status == LAUNCH_CHECK_FAILED for check in checks):
        return LAUNCH_PLAN_BLOCKED
    if any(check.status == LAUNCH_CHECK_WARNING for check in checks):
        return LAUNCH_PLAN_WARNING
    return LAUNCH_PLAN_READY


def _health_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{display_host}:{port}/healthz"


def _validate_positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _require_non_empty(value: str, *, name: str) -> str:
    selected_value = value.strip()
    if not selected_value:
        raise ValueError(f"{name} is required")
    return selected_value


def _quote_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _label_or_none(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else None


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
