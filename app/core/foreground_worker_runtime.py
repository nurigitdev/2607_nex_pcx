"""Foreground worker runtime visibility from supervisor evidence and PID files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.foreground_app_worker_supervisor import (
    DEFAULT_SUPERVISED_WEB_PID_FILE,
    DEFAULT_SUPERVISOR_PID_FILE,
    SUPERVISOR_STATUS_BLOCKED,
    SUPERVISOR_STATUS_FAILED,
    SUPERVISOR_STATUS_RUNNING,
)
from app.core.foreground_production_launch import resolve_launch_path
from app.core.foreground_production_shutdown import ProcessObservation, inspect_process
from app.core.foreground_worker_runner import (
    WORKER_RUN_STATUS_BLOCKED,
    WORKER_RUN_STATUS_FAILED,
    WORKER_RUN_STATUS_PARTIAL,
)

FOREGROUND_WORKER_RUNTIME_VERSION = 1

FOREGROUND_WORKER_RUNTIME_READY = "ready"
FOREGROUND_WORKER_RUNTIME_WARNING = "warning"
FOREGROUND_WORKER_RUNTIME_BLOCKED = "blocked"

DEFAULT_SUPERVISOR_EVIDENCE_PATH = "artifacts/foreground_app_worker_supervisor.json"
DEFAULT_WORKER_RUNNER_EVIDENCE_PATH = "artifacts/foreground_worker_runner.json"

BLOCKING_SUPERVISOR_STATUSES = (SUPERVISOR_STATUS_BLOCKED, SUPERVISOR_STATUS_FAILED)
BLOCKING_WORKER_RUNNER_STATUSES = (
    WORKER_RUN_STATUS_BLOCKED,
    WORKER_RUN_STATUS_FAILED,
    WORKER_RUN_STATUS_PARTIAL,
)


@dataclass(frozen=True)
class RuntimeEvidenceSnapshot:
    code: str
    path: str
    exists: bool
    status: str | None = None
    generated_at: datetime | None = None
    age_seconds: int | None = None
    error: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class RuntimeProcessSnapshot:
    code: str
    pid_file: str
    process_id: int | None
    exists: bool | None
    command_line: tuple[str, ...] = ()
    error: str | None = None

    @property
    def shell_command(self) -> str:
        if not self.command_line:
            return ""
        return " ".join(self.command_line)


@dataclass(frozen=True)
class ForegroundWorkerRuntimeReport:
    status: str
    generated_at: datetime
    workdir: str
    supervisor_evidence: RuntimeEvidenceSnapshot
    worker_runner_evidence: RuntimeEvidenceSnapshot
    supervisor_process: RuntimeProcessSnapshot
    web_process: RuntimeProcessSnapshot
    summary: dict[str, object]


def build_foreground_worker_runtime_report(
    workdir: str | Path = ".",
    *,
    supervisor_evidence_path: str = DEFAULT_SUPERVISOR_EVIDENCE_PATH,
    worker_runner_evidence_path: str = DEFAULT_WORKER_RUNNER_EVIDENCE_PATH,
    supervisor_pid_file: str = DEFAULT_SUPERVISOR_PID_FILE,
    web_pid_file: str = DEFAULT_SUPERVISED_WEB_PID_FILE,
    now: datetime | None = None,
) -> ForegroundWorkerRuntimeReport:
    root = Path(workdir)
    generated_at = now or datetime.now(UTC)
    supervisor_evidence = _read_evidence_snapshot(
        root=root,
        code="foreground_app_worker_supervisor",
        path=supervisor_evidence_path,
        now=generated_at,
    )
    worker_runner_evidence = _read_evidence_snapshot(
        root=root,
        code="foreground_worker_runner",
        path=worker_runner_evidence_path,
        now=generated_at,
    )
    supervisor_process = _read_process_snapshot(
        root=root,
        code="supervisor",
        pid_file=supervisor_pid_file,
    )
    web_process = _read_process_snapshot(root=root, code="web", pid_file=web_pid_file)
    status = _runtime_status(
        supervisor_evidence=supervisor_evidence,
        worker_runner_evidence=worker_runner_evidence,
        supervisor_process=supervisor_process,
        web_process=web_process,
    )
    return ForegroundWorkerRuntimeReport(
        status=status,
        generated_at=generated_at,
        workdir=str(root),
        supervisor_evidence=supervisor_evidence,
        worker_runner_evidence=worker_runner_evidence,
        supervisor_process=supervisor_process,
        web_process=web_process,
        summary=_runtime_summary(
            status=status,
            supervisor_evidence=supervisor_evidence,
            worker_runner_evidence=worker_runner_evidence,
            supervisor_process=supervisor_process,
            web_process=web_process,
        ),
    )


def foreground_worker_runtime_report_payload(
    report: ForegroundWorkerRuntimeReport,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_WORKER_RUNTIME_VERSION,
        "status": report.status,
        "generated_at": report.generated_at.isoformat(),
        "generated_at_label": report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": report.workdir,
        "summary": dict(report.summary),
        "supervisor_evidence": runtime_evidence_snapshot_payload(report.supervisor_evidence),
        "worker_runner_evidence": runtime_evidence_snapshot_payload(report.worker_runner_evidence),
        "supervisor_process": runtime_process_snapshot_payload(report.supervisor_process),
        "web_process": runtime_process_snapshot_payload(report.web_process),
    }


def runtime_evidence_snapshot_payload(snapshot: RuntimeEvidenceSnapshot) -> dict[str, object]:
    return {
        "code": snapshot.code,
        "path": snapshot.path,
        "exists": snapshot.exists,
        "status": snapshot.status,
        "generated_at": snapshot.generated_at.isoformat() if snapshot.generated_at else None,
        "generated_at_label": (
            snapshot.generated_at.strftime("%Y-%m-%d %H:%M:%S") if snapshot.generated_at else None
        ),
        "age_seconds": snapshot.age_seconds,
        "error": snapshot.error,
        "metadata": dict(snapshot.metadata or {}),
    }


def runtime_process_snapshot_payload(snapshot: RuntimeProcessSnapshot) -> dict[str, object]:
    return {
        "code": snapshot.code,
        "pid_file": snapshot.pid_file,
        "process_id": snapshot.process_id,
        "exists": snapshot.exists,
        "command_line": list(snapshot.command_line),
        "shell_command": snapshot.shell_command,
        "error": snapshot.error,
    }


def _read_evidence_snapshot(
    *,
    root: Path,
    code: str,
    path: str,
    now: datetime,
) -> RuntimeEvidenceSnapshot:
    evidence_path = resolve_launch_path(root, path)
    if not evidence_path.exists():
        return RuntimeEvidenceSnapshot(
            code=code,
            path=path,
            exists=False,
            error="Evidence file is missing.",
        )
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RuntimeEvidenceSnapshot(
            code=code,
            path=path,
            exists=True,
            error=f"Evidence file could not be read: {exc}",
        )
    if not isinstance(payload, dict):
        return RuntimeEvidenceSnapshot(
            code=code,
            path=path,
            exists=True,
            error="Evidence payload is not a JSON object.",
        )
    generated_at = _parse_datetime(payload.get("generated_at"))
    return RuntimeEvidenceSnapshot(
        code=code,
        path=path,
        exists=True,
        status=_string_or_none(payload.get("status")),
        generated_at=generated_at,
        age_seconds=_age_seconds(now=now, value=generated_at),
        metadata={
            "dry_run": payload.get("dry_run"),
            "message": payload.get("message"),
            "worker_cycle_count": payload.get("worker_cycle_count"),
            "failed_worker_cycle_count": payload.get("failed_worker_cycle_count"),
            "command_count": payload.get("command_count"),
            "failed_command_count": payload.get("failed_command_count"),
        },
    )


def _read_process_snapshot(
    *,
    root: Path,
    code: str,
    pid_file: str,
) -> RuntimeProcessSnapshot:
    pid_path = resolve_launch_path(root, pid_file)
    if not pid_path.exists():
        return RuntimeProcessSnapshot(
            code=code,
            pid_file=pid_file,
            process_id=None,
            exists=None,
            error="PID file is missing.",
        )
    try:
        process_id = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return RuntimeProcessSnapshot(
            code=code,
            pid_file=pid_file,
            process_id=None,
            exists=None,
            error="PID file does not contain a valid integer.",
        )
    observation = inspect_process(process_id)
    if observation is None:
        return RuntimeProcessSnapshot(
            code=code,
            pid_file=pid_file,
            process_id=process_id,
            exists=None,
            error="Process observation is unavailable.",
        )
    return _process_snapshot_from_observation(
        code=code,
        pid_file=pid_file,
        observation=observation,
    )


def _process_snapshot_from_observation(
    *,
    code: str,
    pid_file: str,
    observation: ProcessObservation,
) -> RuntimeProcessSnapshot:
    return RuntimeProcessSnapshot(
        code=code,
        pid_file=pid_file,
        process_id=observation.process_id,
        exists=observation.exists,
        command_line=observation.command_line,
        error=observation.error,
    )


def _runtime_status(
    *,
    supervisor_evidence: RuntimeEvidenceSnapshot,
    worker_runner_evidence: RuntimeEvidenceSnapshot,
    supervisor_process: RuntimeProcessSnapshot,
    web_process: RuntimeProcessSnapshot,
) -> str:
    if _evidence_is_blocking(supervisor_evidence, BLOCKING_SUPERVISOR_STATUSES):
        return FOREGROUND_WORKER_RUNTIME_BLOCKED
    if _evidence_is_blocking(worker_runner_evidence, BLOCKING_WORKER_RUNNER_STATUSES):
        return FOREGROUND_WORKER_RUNTIME_BLOCKED
    if supervisor_evidence.status == SUPERVISOR_STATUS_RUNNING:
        if supervisor_process.exists is True and web_process.exists is True:
            return FOREGROUND_WORKER_RUNTIME_READY
        return FOREGROUND_WORKER_RUNTIME_BLOCKED
    return FOREGROUND_WORKER_RUNTIME_WARNING


def _runtime_summary(
    *,
    status: str,
    supervisor_evidence: RuntimeEvidenceSnapshot,
    worker_runner_evidence: RuntimeEvidenceSnapshot,
    supervisor_process: RuntimeProcessSnapshot,
    web_process: RuntimeProcessSnapshot,
) -> dict[str, object]:
    return {
        "status": status,
        "supervisor_status": supervisor_evidence.status,
        "worker_runner_status": worker_runner_evidence.status,
        "supervisor_alive": supervisor_process.exists is True,
        "web_alive": web_process.exists is True,
        "supervisor_evidence_age_seconds": supervisor_evidence.age_seconds,
        "worker_runner_evidence_age_seconds": worker_runner_evidence.age_seconds,
        "failed_worker_cycle_count": (supervisor_evidence.metadata or {}).get(
            "failed_worker_cycle_count"
        ),
        "failed_worker_command_count": (worker_runner_evidence.metadata or {}).get(
            "failed_command_count"
        ),
    }


def _evidence_is_blocking(
    evidence: RuntimeEvidenceSnapshot,
    blocking_statuses: tuple[str, ...],
) -> bool:
    if evidence.error:
        return False
    if evidence.status in blocking_statuses:
        return True
    metadata = evidence.metadata or {}
    failed_cycles = int(metadata.get("failed_worker_cycle_count") or 0)
    failed_commands = int(metadata.get("failed_command_count") or 0)
    return failed_cycles > 0 or failed_commands > 0


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _age_seconds(*, now: datetime, value: datetime | None) -> int | None:
    if value is None:
        return None
    selected_now = now if now.tzinfo else now.replace(tzinfo=UTC)
    return max(0, int((selected_now.astimezone(UTC) - value).total_seconds()))


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
