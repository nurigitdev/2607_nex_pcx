"""Foreground production shutdown planning and evidence rendering."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.foreground_production_launch import (
    DEFAULT_PID_FILE,
    is_tcp_port_available,
    resolve_launch_path,
)
from app.core.service_startup_templates import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT

FOREGROUND_PRODUCTION_SHUTDOWN_VERSION = 1

SHUTDOWN_CHECK_PASSED = "passed"
SHUTDOWN_CHECK_WARNING = "warning"
SHUTDOWN_CHECK_FAILED = "failed"

SHUTDOWN_PLAN_READY = "ready"
SHUTDOWN_PLAN_WARNING = "warning"
SHUTDOWN_PLAN_BLOCKED = "blocked"

SHUTDOWN_STATUS_PLANNED = "planned"
SHUTDOWN_STATUS_STOPPING = "stopping"
SHUTDOWN_STATUS_STOPPED = "stopped"
SHUTDOWN_STATUS_NO_PROCESS = "no_process"
SHUTDOWN_STATUS_FAILED = "failed"
SHUTDOWN_STATUS_BLOCKED = "blocked"

DEFAULT_SHUTDOWN_LOG_FILE = "artifacts/foreground_production_shutdown.log"
DEFAULT_EXPECTED_COMMAND_MARKERS = ("uvicorn", "app.main:create_app")
DEFAULT_SIGNAL_NAME = "SIGTERM"


@dataclass(frozen=True)
class ProcessObservation:
    process_id: int
    exists: bool
    command_line: tuple[str, ...] = ()
    error: str | None = None

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command_line)


@dataclass(frozen=True)
class ForegroundProductionShutdownOptions:
    workdir: str | Path = "."
    pid_file: str = DEFAULT_PID_FILE
    log_file: str = DEFAULT_SHUTDOWN_LOG_FILE
    host: str = DEFAULT_WEB_HOST
    port: int = DEFAULT_WEB_PORT
    require_pid_file: bool = True
    check_port_reachable: bool = True
    port_check_timeout_seconds: float = 1.0
    expected_command_markers: tuple[str, ...] = DEFAULT_EXPECTED_COMMAND_MARKERS
    signal_name: str = DEFAULT_SIGNAL_NAME


@dataclass(frozen=True)
class ForegroundProductionShutdownCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ForegroundProductionShutdownPlan:
    status: str
    generated_at: datetime
    workdir: str
    pid_file: str
    log_file: str
    host: str
    port: int
    signal_name: str
    process_id: int | None
    process_observation: ProcessObservation | None
    expected_command_markers: tuple[str, ...]
    port_reachable_before_stop: bool | None
    checks: tuple[ForegroundProductionShutdownCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(SHUTDOWN_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(SHUTDOWN_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(SHUTDOWN_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


@dataclass(frozen=True)
class ForegroundProductionShutdownEvidence:
    status: str
    dry_run: bool
    generated_at: datetime
    plan: ForegroundProductionShutdownPlan
    process_id: int | None = None
    signal_name: str | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    port_released_after_stop: bool | None = None
    message: str = ""
    metadata: dict[str, object] | None = None


def build_foreground_production_shutdown_plan(
    options: ForegroundProductionShutdownOptions | None = None,
    *,
    generated_at: datetime | None = None,
    process_observation: ProcessObservation | None = None,
    port_reachable: bool | None = None,
) -> ForegroundProductionShutdownPlan:
    selected_options = _validate_options(options or ForegroundProductionShutdownOptions())
    root = Path(selected_options.workdir)
    pid_file_path = resolve_launch_path(root, selected_options.pid_file)
    process_id, pid_check = _pid_file_check(
        pid_file_path=pid_file_path,
        display_path=selected_options.pid_file,
        required=selected_options.require_pid_file,
    )
    observation = (
        process_observation
        if process_observation is not None
        else inspect_process(process_id) if process_id is not None else None
    )
    port_reachable_before_stop = (
        is_tcp_port_reachable(
            host=selected_options.host,
            port=selected_options.port,
            timeout_seconds=selected_options.port_check_timeout_seconds,
        )
        if selected_options.check_port_reachable and port_reachable is None
        else port_reachable if selected_options.check_port_reachable else None
    )
    checks = (
        _workdir_check(root),
        pid_check,
        _process_check(
            observation=observation,
            required=selected_options.require_pid_file,
        ),
        _command_guard_check(
            observation=observation,
            expected_markers=selected_options.expected_command_markers,
        ),
        _port_reachable_check(
            reachable=port_reachable_before_stop,
            enabled=selected_options.check_port_reachable,
            host=selected_options.host,
            port=selected_options.port,
        ),
        _log_path_check(
            pid_file=selected_options.pid_file,
            log_file=selected_options.log_file,
        ),
    )
    return ForegroundProductionShutdownPlan(
        status=_overall_plan_status(checks),
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(root),
        pid_file=selected_options.pid_file,
        log_file=selected_options.log_file,
        host=selected_options.host,
        port=selected_options.port,
        signal_name=selected_options.signal_name,
        process_id=process_id,
        process_observation=observation,
        expected_command_markers=selected_options.expected_command_markers,
        port_reachable_before_stop=port_reachable_before_stop,
        checks=checks,
    )


def build_foreground_production_shutdown_evidence(
    plan: ForegroundProductionShutdownPlan,
    *,
    status: str,
    dry_run: bool,
    process_id: int | None = None,
    signal_name: str | None = None,
    started_at: datetime | None = None,
    stopped_at: datetime | None = None,
    port_released_after_stop: bool | None = None,
    message: str = "",
    metadata: dict[str, object] | None = None,
    generated_at: datetime | None = None,
) -> ForegroundProductionShutdownEvidence:
    selected_status = status.strip()
    if selected_status not in {
        SHUTDOWN_STATUS_PLANNED,
        SHUTDOWN_STATUS_STOPPING,
        SHUTDOWN_STATUS_STOPPED,
        SHUTDOWN_STATUS_NO_PROCESS,
        SHUTDOWN_STATUS_FAILED,
        SHUTDOWN_STATUS_BLOCKED,
    }:
        raise ValueError("unsupported shutdown evidence status")
    return ForegroundProductionShutdownEvidence(
        status=selected_status,
        dry_run=dry_run,
        generated_at=generated_at or datetime.now(UTC),
        plan=plan,
        process_id=process_id,
        signal_name=signal_name,
        started_at=started_at,
        stopped_at=stopped_at,
        port_released_after_stop=port_released_after_stop,
        message=message,
        metadata=dict(metadata or {}),
    )


def foreground_production_shutdown_plan_payload(
    plan: ForegroundProductionShutdownPlan,
) -> dict[str, object]:
    observation = plan.process_observation
    return {
        "version": FOREGROUND_PRODUCTION_SHUTDOWN_VERSION,
        "status": plan.status,
        "generated_at": plan.generated_at.isoformat(),
        "generated_at_label": plan.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": plan.workdir,
        "pid_file": plan.pid_file,
        "log_file": plan.log_file,
        "host": plan.host,
        "port": plan.port,
        "signal_name": plan.signal_name,
        "process_id": plan.process_id,
        "expected_command_markers": list(plan.expected_command_markers),
        "port_reachable_before_stop": plan.port_reachable_before_stop,
        "process_observation": _process_observation_payload(observation),
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


def foreground_production_shutdown_evidence_payload(
    evidence: ForegroundProductionShutdownEvidence,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_PRODUCTION_SHUTDOWN_VERSION,
        "status": evidence.status,
        "dry_run": evidence.dry_run,
        "generated_at": evidence.generated_at.isoformat(),
        "generated_at_label": evidence.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "process_id": evidence.process_id,
        "signal_name": evidence.signal_name,
        "started_at": _iso_or_none(evidence.started_at),
        "started_at_label": _label_or_none(evidence.started_at),
        "stopped_at": _iso_or_none(evidence.stopped_at),
        "stopped_at_label": _label_or_none(evidence.stopped_at),
        "port_released_after_stop": evidence.port_released_after_stop,
        "message": evidence.message,
        "metadata": dict(evidence.metadata or {}),
        "plan": foreground_production_shutdown_plan_payload(evidence.plan),
    }


def render_foreground_production_shutdown_markdown(payload: dict[str, object]) -> str:
    plan = _dict(payload.get("plan"))
    observation = _dict(plan.get("process_observation"))
    lines = [
        "# Foreground Production Shutdown Evidence",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Dry Run: {_text(payload.get('dry_run'))}",
        f"- Process ID: {_text(payload.get('process_id'))}",
        f"- Signal: `{_text(payload.get('signal_name'))}`",
        f"- Port Released After Stop: {_text(payload.get('port_released_after_stop'))}",
        f"- Message: {_text(payload.get('message'))}",
        "",
        "## Shutdown Plan",
        "",
        f"- Plan Status: `{_text(plan.get('status'))}`",
        f"- Workdir: `{_text(plan.get('workdir'))}`",
        f"- PID File: `{_text(plan.get('pid_file'))}`",
        f"- Log File: `{_text(plan.get('log_file'))}`",
        f"- Host: `{_text(plan.get('host'))}`",
        f"- Port: {_text(plan.get('port'))}",
        f"- Port Reachable Before Stop: {_text(plan.get('port_reachable_before_stop'))}",
        f"- Observed Process Exists: {_text(observation.get('exists'))}",
        f"- Observed Command: `{_text(observation.get('shell_command'))}`",
        "",
        "## Pre-Stop Checks",
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
            "- Run with `--dry-run` first to verify the PID file and command guard.",
            "- Actual stop sends `SIGTERM` only after the process guard passes.",
            "- A stale PID file is recorded as `no_process`; inspect before relaunch.",
            "- Database URL values are never written to this evidence file.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def inspect_process(process_id: int | None) -> ProcessObservation | None:
    if process_id is None:
        return None
    proc_cmdline = Path("/proc") / str(process_id) / "cmdline"
    try:
        raw = proc_cmdline.read_bytes()
    except FileNotFoundError:
        return ProcessObservation(process_id=process_id, exists=False)
    except PermissionError as exc:
        exists = _can_signal_zero(process_id)
        return ProcessObservation(process_id=process_id, exists=exists, error=str(exc))
    except OSError as exc:
        exists = _can_signal_zero(process_id)
        return ProcessObservation(process_id=process_id, exists=exists, error=str(exc))
    command_line = tuple(
        part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
    )
    return ProcessObservation(
        process_id=process_id,
        exists=True,
        command_line=command_line,
    )


def is_tcp_port_reachable(*, host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    return not is_tcp_port_available(
        host=host,
        port=port,
        timeout_seconds=timeout_seconds,
    )


def _validate_options(
    options: ForegroundProductionShutdownOptions,
) -> ForegroundProductionShutdownOptions:
    workdir = _require_non_empty(str(options.workdir), name="workdir")
    pid_file = _require_non_empty(options.pid_file, name="pid_file")
    log_file = _require_non_empty(options.log_file, name="log_file")
    host = _require_non_empty(options.host, name="host")
    port = _validate_positive_int(options.port, name="port")
    signal_name = _require_non_empty(options.signal_name, name="signal_name")
    if options.port_check_timeout_seconds <= 0:
        raise ValueError("port_check_timeout_seconds must be greater than zero")
    expected_markers = tuple(
        marker.strip() for marker in options.expected_command_markers if marker.strip()
    )
    return ForegroundProductionShutdownOptions(
        workdir=Path(workdir),
        pid_file=pid_file,
        log_file=log_file,
        host=host,
        port=port,
        require_pid_file=options.require_pid_file,
        check_port_reachable=options.check_port_reachable,
        port_check_timeout_seconds=options.port_check_timeout_seconds,
        expected_command_markers=expected_markers,
        signal_name=signal_name,
    )


def _pid_file_check(
    *,
    pid_file_path: Path,
    display_path: str,
    required: bool,
) -> tuple[int | None, ForegroundProductionShutdownCheck]:
    if not pid_file_path.exists():
        status = SHUTDOWN_CHECK_FAILED if required else SHUTDOWN_CHECK_WARNING
        detail = "PID file is missing." if required else "PID file is missing; dry-run only."
        return None, ForegroundProductionShutdownCheck(
            code="pid_file",
            status=status,
            detail=detail,
            metadata={"pid_file": display_path},
        )
    try:
        process_id = int(pid_file_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None, ForegroundProductionShutdownCheck(
            code="pid_file",
            status=SHUTDOWN_CHECK_FAILED,
            detail="PID file does not contain a valid integer process ID.",
            metadata={"pid_file": display_path},
        )
    if process_id <= 0:
        return None, ForegroundProductionShutdownCheck(
            code="pid_file",
            status=SHUTDOWN_CHECK_FAILED,
            detail="PID file contains a non-positive process ID.",
            metadata={"pid_file": display_path},
        )
    return process_id, ForegroundProductionShutdownCheck(
        code="pid_file",
        status=SHUTDOWN_CHECK_PASSED,
        detail="PID file contains a valid process ID.",
        metadata={"pid_file": display_path, "process_id": process_id},
    )


def _workdir_check(root: Path) -> ForegroundProductionShutdownCheck:
    if root.exists() and root.is_dir():
        return ForegroundProductionShutdownCheck(
            code="workdir",
            status=SHUTDOWN_CHECK_PASSED,
            detail="Working directory exists.",
            metadata={"workdir": str(root)},
        )
    return ForegroundProductionShutdownCheck(
        code="workdir",
        status=SHUTDOWN_CHECK_FAILED,
        detail="Working directory is missing or is not a directory.",
        metadata={"workdir": str(root)},
    )


def _process_check(
    *,
    observation: ProcessObservation | None,
    required: bool,
) -> ForegroundProductionShutdownCheck:
    if observation is None:
        status = SHUTDOWN_CHECK_FAILED if required else SHUTDOWN_CHECK_WARNING
        return ForegroundProductionShutdownCheck(
            code="process_exists",
            status=status,
            detail="No process ID is available for shutdown.",
        )
    if observation.exists:
        return ForegroundProductionShutdownCheck(
            code="process_exists",
            status=SHUTDOWN_CHECK_PASSED,
            detail="PID currently resolves to a running process.",
            metadata={"process_id": observation.process_id},
        )
    return ForegroundProductionShutdownCheck(
        code="process_exists",
        status=SHUTDOWN_CHECK_WARNING,
        detail="PID file is stale; process is not running.",
        metadata={"process_id": observation.process_id},
    )


def _command_guard_check(
    *,
    observation: ProcessObservation | None,
    expected_markers: tuple[str, ...],
) -> ForegroundProductionShutdownCheck:
    if not expected_markers:
        return ForegroundProductionShutdownCheck(
            code="command_guard",
            status=SHUTDOWN_CHECK_WARNING,
            detail="Command guard is disabled.",
        )
    if observation is None:
        return ForegroundProductionShutdownCheck(
            code="command_guard",
            status=SHUTDOWN_CHECK_WARNING,
            detail="Command guard skipped because no process ID is available.",
        )
    if not observation.exists:
        return ForegroundProductionShutdownCheck(
            code="command_guard",
            status=SHUTDOWN_CHECK_WARNING,
            detail="Command guard skipped because the process is not running.",
        )
    command_text = observation.shell_command
    if all(marker in command_text for marker in expected_markers):
        return ForegroundProductionShutdownCheck(
            code="command_guard",
            status=SHUTDOWN_CHECK_PASSED,
            detail="Observed command matches expected NeX-PCX foreground markers.",
            metadata={"expected_markers": list(expected_markers)},
        )
    return ForegroundProductionShutdownCheck(
        code="command_guard",
        status=SHUTDOWN_CHECK_FAILED,
        detail="Observed command does not match expected NeX-PCX foreground markers.",
        metadata={
            "expected_markers": list(expected_markers),
            "observed_command": command_text,
            "error": observation.error,
        },
    )


def _port_reachable_check(
    *,
    reachable: bool | None,
    enabled: bool,
    host: str,
    port: int,
) -> ForegroundProductionShutdownCheck:
    if not enabled:
        return ForegroundProductionShutdownCheck(
            code="port_reachable",
            status=SHUTDOWN_CHECK_WARNING,
            detail="Port reachability check was skipped.",
            metadata={"host": host, "port": port},
        )
    if reachable:
        return ForegroundProductionShutdownCheck(
            code="port_reachable",
            status=SHUTDOWN_CHECK_PASSED,
            detail="Target web port is reachable before stop.",
            metadata={"host": host, "port": port},
        )
    return ForegroundProductionShutdownCheck(
        code="port_reachable",
        status=SHUTDOWN_CHECK_WARNING,
        detail="Target web port is not reachable before stop.",
        metadata={"host": host, "port": port},
    )


def _log_path_check(*, pid_file: str, log_file: str) -> ForegroundProductionShutdownCheck:
    if Path(log_file) == Path(pid_file):
        return ForegroundProductionShutdownCheck(
            code="log_path",
            status=SHUTDOWN_CHECK_FAILED,
            detail="Shutdown log file must differ from the PID file.",
        )
    return ForegroundProductionShutdownCheck(
        code="log_path",
        status=SHUTDOWN_CHECK_PASSED,
        detail="Shutdown log path is configured.",
        metadata={"log_file": log_file},
    )


def _overall_plan_status(checks: tuple[ForegroundProductionShutdownCheck, ...]) -> str:
    if any(check.status == SHUTDOWN_CHECK_FAILED for check in checks):
        return SHUTDOWN_PLAN_BLOCKED
    if any(check.status == SHUTDOWN_CHECK_WARNING for check in checks):
        return SHUTDOWN_PLAN_WARNING
    return SHUTDOWN_PLAN_READY


def _process_observation_payload(
    observation: ProcessObservation | None,
) -> dict[str, object] | None:
    if observation is None:
        return None
    return {
        "process_id": observation.process_id,
        "exists": observation.exists,
        "command_line": list(observation.command_line),
        "shell_command": observation.shell_command,
        "error": observation.error,
    }


def _can_signal_zero(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
