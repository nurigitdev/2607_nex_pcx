"""Foreground web app plus worker-cycle supervisor planning and evidence."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.foreground_production_launch import (
    DEFAULT_DATABASE_URL_ENV,
    DEFAULT_PID_FILE,
    DEFAULT_PYTHON_BIN,
    LAUNCH_PLAN_BLOCKED,
    LAUNCH_PLAN_WARNING,
    ForegroundProductionLaunchOptions,
    ForegroundProductionLaunchPlan,
    build_foreground_production_launch_plan,
    foreground_production_launch_plan_payload,
)
from app.core.foreground_worker_runner import (
    DEFAULT_EMBEDDING_LIMIT_PER_PROFILE,
    DEFAULT_HEALTH_TIMEOUT_SECONDS,
    DEFAULT_MAX_HEALTH_ELAPSED_MS,
    DEFAULT_PIPELINE_LIMIT,
)
from app.core.service_startup_templates import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT

FOREGROUND_APP_WORKER_SUPERVISOR_VERSION = 1

SUPERVISOR_CHECK_PASSED = "passed"
SUPERVISOR_CHECK_WARNING = "warning"
SUPERVISOR_CHECK_FAILED = "failed"

SUPERVISOR_PLAN_READY = "ready"
SUPERVISOR_PLAN_WARNING = "warning"
SUPERVISOR_PLAN_BLOCKED = "blocked"

SUPERVISOR_STATUS_PLANNED = "planned"
SUPERVISOR_STATUS_RUNNING = "running"
SUPERVISOR_STATUS_EXITED = "exited"
SUPERVISOR_STATUS_FAILED = "failed"
SUPERVISOR_STATUS_BLOCKED = "blocked"

DEFAULT_SUPERVISOR_PID_FILE = "artifacts/foreground_app_worker_supervisor.pid"
DEFAULT_SUPERVISOR_LOG_FILE = "artifacts/foreground_app_worker_supervisor.log"
DEFAULT_SUPERVISED_WEB_PID_FILE = DEFAULT_PID_FILE
DEFAULT_WORKER_CYCLE_INTERVAL_SECONDS = 5.0
DEFAULT_WORKER_FAILURE_TOLERANCE = 3
DEFAULT_WORKER_RUNNER_JSON_OUTPUT = "artifacts/foreground_worker_runner.json"
DEFAULT_WORKER_RUNNER_MARKDOWN_OUTPUT = "artifacts/foreground_worker_runner.md"

ENV_FOREGROUND_WORKDIR = "NEX_PCX_FOREGROUND_WORKDIR"
ENV_FOREGROUND_PYTHON_BIN = "NEX_PCX_FOREGROUND_PYTHON_BIN"
ENV_FOREGROUND_HOST = "NEX_PCX_FOREGROUND_HOST"
ENV_FOREGROUND_PORT = "NEX_PCX_FOREGROUND_PORT"
ENV_FOREGROUND_DATABASE_URL_ENV = "NEX_PCX_FOREGROUND_DATABASE_URL_ENV"
ENV_FOREGROUND_REQUIRE_DATABASE_URL = "NEX_PCX_FOREGROUND_REQUIRE_DATABASE_URL"
ENV_FOREGROUND_SUPERVISOR_PID_FILE = "NEX_PCX_FOREGROUND_SUPERVISOR_PID_FILE"
ENV_FOREGROUND_WEB_PID_FILE = "NEX_PCX_FOREGROUND_WEB_PID_FILE"
ENV_FOREGROUND_SUPERVISOR_LOG_FILE = "NEX_PCX_FOREGROUND_SUPERVISOR_LOG_FILE"
ENV_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS = "NEX_PCX_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS"
ENV_FOREGROUND_WORKER_FAILURE_TOLERANCE = "NEX_PCX_FOREGROUND_WORKER_FAILURE_TOLERANCE"
ENV_FOREGROUND_PIPELINE_LIMIT = "NEX_PCX_FOREGROUND_PIPELINE_LIMIT"
ENV_FOREGROUND_EMBEDDING_LIMIT_PER_PROFILE = "NEX_PCX_FOREGROUND_EMBEDDING_LIMIT_PER_PROFILE"
ENV_FOREGROUND_GUARD_HEALTH_TIMEOUT_SECONDS = "NEX_PCX_FOREGROUND_GUARD_HEALTH_TIMEOUT_SECONDS"
ENV_FOREGROUND_MAX_PROVIDER_HEALTH_ELAPSED_MS = "NEX_PCX_FOREGROUND_MAX_PROVIDER_HEALTH_ELAPSED_MS"
ENV_FOREGROUND_WORKER_JSON_OUTPUT = "NEX_PCX_FOREGROUND_WORKER_JSON_OUTPUT"
ENV_FOREGROUND_WORKER_MARKDOWN_OUTPUT = "NEX_PCX_FOREGROUND_WORKER_MARKDOWN_OUTPUT"
ENV_FOREGROUND_NO_DEFAULT_QWEN_TOKEN_GUARD = "NEX_PCX_FOREGROUND_NO_DEFAULT_QWEN_TOKEN_GUARD"
ENV_FOREGROUND_CHECK_PORT_AVAILABLE = "NEX_PCX_FOREGROUND_CHECK_PORT_AVAILABLE"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ForegroundAppWorkerSupervisorOptions:
    workdir: str | Path = "."
    python_bin: str = DEFAULT_PYTHON_BIN
    host: str = DEFAULT_WEB_HOST
    port: int = DEFAULT_WEB_PORT
    database_url_env: str = DEFAULT_DATABASE_URL_ENV
    require_database_url: bool = True
    supervisor_pid_file: str = DEFAULT_SUPERVISOR_PID_FILE
    web_pid_file: str = DEFAULT_SUPERVISED_WEB_PID_FILE
    log_file: str = DEFAULT_SUPERVISOR_LOG_FILE
    worker_cycle_interval_seconds: float = DEFAULT_WORKER_CYCLE_INTERVAL_SECONDS
    worker_failure_tolerance: int = DEFAULT_WORKER_FAILURE_TOLERANCE
    pipeline_limit: int = DEFAULT_PIPELINE_LIMIT
    embedding_limit_per_profile: int = DEFAULT_EMBEDDING_LIMIT_PER_PROFILE
    guard_health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS
    max_provider_health_elapsed_ms: int = DEFAULT_MAX_HEALTH_ELAPSED_MS
    worker_json_output: str = DEFAULT_WORKER_RUNNER_JSON_OUTPUT
    worker_markdown_output: str = DEFAULT_WORKER_RUNNER_MARKDOWN_OUTPUT
    no_default_qwen_token_guard: bool = False
    check_port_available: bool = True


def foreground_app_worker_supervisor_options_from_environ(
    environ: Mapping[str, str],
    *,
    defaults: ForegroundAppWorkerSupervisorOptions | None = None,
) -> ForegroundAppWorkerSupervisorOptions:
    base = defaults or ForegroundAppWorkerSupervisorOptions()
    return _validate_options(
        ForegroundAppWorkerSupervisorOptions(
            workdir=_env_str(environ, ENV_FOREGROUND_WORKDIR, str(base.workdir)),
            python_bin=_env_str(environ, ENV_FOREGROUND_PYTHON_BIN, base.python_bin),
            host=_env_str(environ, ENV_FOREGROUND_HOST, base.host),
            port=_env_int(environ, ENV_FOREGROUND_PORT, base.port),
            database_url_env=_env_str(
                environ,
                ENV_FOREGROUND_DATABASE_URL_ENV,
                base.database_url_env,
            ),
            require_database_url=_env_bool(
                environ,
                ENV_FOREGROUND_REQUIRE_DATABASE_URL,
                base.require_database_url,
            ),
            supervisor_pid_file=_env_str(
                environ,
                ENV_FOREGROUND_SUPERVISOR_PID_FILE,
                base.supervisor_pid_file,
            ),
            web_pid_file=_env_str(environ, ENV_FOREGROUND_WEB_PID_FILE, base.web_pid_file),
            log_file=_env_str(
                environ,
                ENV_FOREGROUND_SUPERVISOR_LOG_FILE,
                base.log_file,
            ),
            worker_cycle_interval_seconds=_env_float(
                environ,
                ENV_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS,
                base.worker_cycle_interval_seconds,
            ),
            worker_failure_tolerance=_env_int(
                environ,
                ENV_FOREGROUND_WORKER_FAILURE_TOLERANCE,
                base.worker_failure_tolerance,
            ),
            pipeline_limit=_env_int(
                environ,
                ENV_FOREGROUND_PIPELINE_LIMIT,
                base.pipeline_limit,
            ),
            embedding_limit_per_profile=_env_int(
                environ,
                ENV_FOREGROUND_EMBEDDING_LIMIT_PER_PROFILE,
                base.embedding_limit_per_profile,
            ),
            guard_health_timeout_seconds=_env_float(
                environ,
                ENV_FOREGROUND_GUARD_HEALTH_TIMEOUT_SECONDS,
                base.guard_health_timeout_seconds,
            ),
            max_provider_health_elapsed_ms=_env_int(
                environ,
                ENV_FOREGROUND_MAX_PROVIDER_HEALTH_ELAPSED_MS,
                base.max_provider_health_elapsed_ms,
            ),
            worker_json_output=_env_str(
                environ,
                ENV_FOREGROUND_WORKER_JSON_OUTPUT,
                base.worker_json_output,
            ),
            worker_markdown_output=_env_str(
                environ,
                ENV_FOREGROUND_WORKER_MARKDOWN_OUTPUT,
                base.worker_markdown_output,
            ),
            no_default_qwen_token_guard=_env_bool(
                environ,
                ENV_FOREGROUND_NO_DEFAULT_QWEN_TOKEN_GUARD,
                base.no_default_qwen_token_guard,
            ),
            check_port_available=_env_bool(
                environ,
                ENV_FOREGROUND_CHECK_PORT_AVAILABLE,
                base.check_port_available,
            ),
        )
    )


@dataclass(frozen=True)
class ForegroundAppWorkerSupervisorCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ForegroundAppWorkerSupervisorPlan:
    status: str
    generated_at: datetime
    workdir: str
    python_bin: str
    launch_plan: ForegroundProductionLaunchPlan
    worker_command: tuple[str, ...]
    worker_cycle_interval_seconds: float
    worker_failure_tolerance: int
    supervisor_pid_file: str
    web_pid_file: str
    log_file: str
    checks: tuple[ForegroundAppWorkerSupervisorCheck, ...]

    @property
    def worker_shell_command(self) -> str:
        return _quote_command(self.worker_command)

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(SUPERVISOR_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(SUPERVISOR_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(SUPERVISOR_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


@dataclass(frozen=True)
class ForegroundWorkerCycleObservation:
    index: int
    command: tuple[str, ...]
    exit_code: int
    elapsed_ms: int
    status: str
    message: str = ""

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class ForegroundAppWorkerSupervisorEvidence:
    status: str
    dry_run: bool
    generated_at: datetime
    plan: ForegroundAppWorkerSupervisorPlan
    supervisor_process_id: int | None = None
    web_process_id: int | None = None
    returncode: int | None = None
    started_at: datetime | None = None
    exited_at: datetime | None = None
    worker_cycles: tuple[ForegroundWorkerCycleObservation, ...] = ()
    message: str = ""
    metadata: dict[str, object] | None = None

    @property
    def worker_cycle_count(self) -> int:
        return len(self.worker_cycles)

    @property
    def failed_worker_cycle_count(self) -> int:
        return sum(1 for cycle in self.worker_cycles if not cycle.succeeded)


def build_foreground_app_worker_supervisor_plan(
    options: ForegroundAppWorkerSupervisorOptions | None = None,
    *,
    generated_at: datetime | None = None,
    environ: Mapping[str, str] | None = None,
    port_available: bool | None = None,
) -> ForegroundAppWorkerSupervisorPlan:
    selected_options = _validate_options(options or ForegroundAppWorkerSupervisorOptions())
    launch_plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=selected_options.workdir,
            python_bin=selected_options.python_bin,
            host=selected_options.host,
            port=selected_options.port,
            database_url_env=selected_options.database_url_env,
            require_database_url=selected_options.require_database_url,
            pid_file=selected_options.web_pid_file,
            log_file=selected_options.log_file,
            check_port_available=selected_options.check_port_available,
        ),
        generated_at=generated_at,
        environ=environ,
        port_available=port_available,
    )
    checks = (
        _launch_plan_check(launch_plan),
        _supervisor_path_check(
            supervisor_pid_file=selected_options.supervisor_pid_file,
            web_pid_file=selected_options.web_pid_file,
            log_file=selected_options.log_file,
        ),
        _worker_cycle_check(
            interval_seconds=selected_options.worker_cycle_interval_seconds,
            worker_failure_tolerance=selected_options.worker_failure_tolerance,
            pipeline_limit=selected_options.pipeline_limit,
            embedding_limit_per_profile=selected_options.embedding_limit_per_profile,
            guard_health_timeout_seconds=selected_options.guard_health_timeout_seconds,
            max_provider_health_elapsed_ms=selected_options.max_provider_health_elapsed_ms,
        ),
    )
    worker_command = _worker_command(selected_options)
    return ForegroundAppWorkerSupervisorPlan(
        status=_overall_plan_status(checks),
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(Path(selected_options.workdir)),
        python_bin=selected_options.python_bin,
        launch_plan=launch_plan,
        worker_command=worker_command,
        worker_cycle_interval_seconds=selected_options.worker_cycle_interval_seconds,
        worker_failure_tolerance=selected_options.worker_failure_tolerance,
        supervisor_pid_file=selected_options.supervisor_pid_file,
        web_pid_file=selected_options.web_pid_file,
        log_file=selected_options.log_file,
        checks=checks,
    )


def build_foreground_app_worker_supervisor_evidence(
    plan: ForegroundAppWorkerSupervisorPlan,
    *,
    status: str,
    dry_run: bool,
    supervisor_process_id: int | None = None,
    web_process_id: int | None = None,
    returncode: int | None = None,
    started_at: datetime | None = None,
    exited_at: datetime | None = None,
    worker_cycles: tuple[ForegroundWorkerCycleObservation, ...] = (),
    message: str = "",
    metadata: dict[str, object] | None = None,
    generated_at: datetime | None = None,
) -> ForegroundAppWorkerSupervisorEvidence:
    selected_status = status.strip()
    if selected_status not in {
        SUPERVISOR_STATUS_PLANNED,
        SUPERVISOR_STATUS_RUNNING,
        SUPERVISOR_STATUS_EXITED,
        SUPERVISOR_STATUS_FAILED,
        SUPERVISOR_STATUS_BLOCKED,
    }:
        raise ValueError("unsupported foreground app worker supervisor status")
    return ForegroundAppWorkerSupervisorEvidence(
        status=selected_status,
        dry_run=dry_run,
        generated_at=generated_at or datetime.now(UTC),
        plan=plan,
        supervisor_process_id=supervisor_process_id,
        web_process_id=web_process_id,
        returncode=returncode,
        started_at=started_at,
        exited_at=exited_at,
        worker_cycles=tuple(worker_cycles),
        message=message,
        metadata=dict(metadata or {}),
    )


def foreground_app_worker_supervisor_evidence_payload(
    evidence: ForegroundAppWorkerSupervisorEvidence,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_APP_WORKER_SUPERVISOR_VERSION,
        "status": evidence.status,
        "dry_run": evidence.dry_run,
        "generated_at": evidence.generated_at.isoformat(),
        "generated_at_label": evidence.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "supervisor_process_id": evidence.supervisor_process_id,
        "web_process_id": evidence.web_process_id,
        "returncode": evidence.returncode,
        "started_at": _iso_or_none(evidence.started_at),
        "started_at_label": _label_or_none(evidence.started_at),
        "exited_at": _iso_or_none(evidence.exited_at),
        "exited_at_label": _label_or_none(evidence.exited_at),
        "worker_cycle_count": evidence.worker_cycle_count,
        "failed_worker_cycle_count": evidence.failed_worker_cycle_count,
        "message": evidence.message,
        "metadata": dict(evidence.metadata or {}),
        "plan": foreground_app_worker_supervisor_plan_payload(evidence.plan),
        "worker_cycles": [
            foreground_worker_cycle_observation_payload(cycle) for cycle in evidence.worker_cycles
        ],
    }


def foreground_app_worker_supervisor_plan_payload(
    plan: ForegroundAppWorkerSupervisorPlan,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_APP_WORKER_SUPERVISOR_VERSION,
        "status": plan.status,
        "generated_at": plan.generated_at.isoformat(),
        "generated_at_label": plan.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": plan.workdir,
        "python_bin": plan.python_bin,
        "worker_command": list(plan.worker_command),
        "worker_shell_command": plan.worker_shell_command,
        "worker_cycle_interval_seconds": plan.worker_cycle_interval_seconds,
        "worker_failure_tolerance": plan.worker_failure_tolerance,
        "supervisor_pid_file": plan.supervisor_pid_file,
        "web_pid_file": plan.web_pid_file,
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
        "launch_plan": foreground_production_launch_plan_payload(plan.launch_plan),
    }


def foreground_worker_cycle_observation_payload(
    cycle: ForegroundWorkerCycleObservation,
) -> dict[str, object]:
    return {
        "index": cycle.index,
        "command": list(cycle.command),
        "shell_command": cycle.shell_command,
        "exit_code": cycle.exit_code,
        "elapsed_ms": cycle.elapsed_ms,
        "status": cycle.status,
        "succeeded": cycle.succeeded,
        "message": cycle.message,
    }


def render_foreground_app_worker_supervisor_markdown(payload: dict[str, object]) -> str:
    plan = _dict(payload.get("plan"))
    launch_plan = _dict(plan.get("launch_plan"))
    lines = [
        "# Foreground App Worker Supervisor Evidence",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Dry Run: {_text(payload.get('dry_run'))}",
        f"- Supervisor PID: {_text(payload.get('supervisor_process_id'))}",
        f"- Web PID: {_text(payload.get('web_process_id'))}",
        f"- Worker Cycles: {_text(payload.get('worker_cycle_count'))}",
        f"- Failed Worker Cycles: {_text(payload.get('failed_worker_cycle_count'))}",
        f"- Message: {_text(payload.get('message'))}",
        "",
        "## Supervisor Plan",
        "",
        f"- Plan Status: `{_text(plan.get('status'))}`",
        f"- Workdir: `{_text(plan.get('workdir'))}`",
        f"- Host: `{_text(launch_plan.get('host'))}`",
        f"- Port: {_text(launch_plan.get('port'))}",
        f"- Health URL: `{_text(launch_plan.get('health_url'))}`",
        f"- Supervisor PID File: `{_text(plan.get('supervisor_pid_file'))}`",
        f"- Web PID File: `{_text(plan.get('web_pid_file'))}`",
        f"- Log File: `{_text(plan.get('log_file'))}`",
        f"- Worker Interval Seconds: {_text(plan.get('worker_cycle_interval_seconds'))}",
        f"- Worker Failure Tolerance: {_text(plan.get('worker_failure_tolerance'))}",
        f"- Worker Command: `{_text(plan.get('worker_shell_command'))}`",
        "",
        "## Checks",
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
            "## Worker Cycles",
            "",
            "| Index | Status | Exit | Elapsed Ms | Message |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for cycle in payload.get("worker_cycles", []):
        cycle_payload = _dict(cycle)
        lines.append(
            "| "
            f"{_md_cell(cycle_payload.get('index'))} | "
            f"{_md_cell(cycle_payload.get('status'))} | "
            f"{_md_cell(cycle_payload.get('exit_code'))} | "
            f"{_md_cell(cycle_payload.get('elapsed_ms'))} | "
            f"{_md_cell(cycle_payload.get('message'))} |"
        )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- This supervisor keeps the web app running while worker cycles poll the queue.",
            "- Worker cycles call `run_foreground_workers.py`; database URLs are passed by env.",
            "- Use dry-run planning before starting foreground operation.",
            "- Keep DGX resource pressure visible before disabling Qwen token guards.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _validate_options(
    options: ForegroundAppWorkerSupervisorOptions,
) -> ForegroundAppWorkerSupervisorOptions:
    workdir = _require_non_empty(str(options.workdir), name="workdir")
    python_bin = _require_non_empty(options.python_bin, name="python_bin")
    host = _require_non_empty(options.host, name="host")
    port = _validate_positive_int(options.port, name="port")
    database_url_env = _require_non_empty(options.database_url_env, name="database_url_env")
    supervisor_pid_file = _require_non_empty(
        options.supervisor_pid_file,
        name="supervisor_pid_file",
    )
    web_pid_file = _require_non_empty(options.web_pid_file, name="web_pid_file")
    log_file = _require_non_empty(options.log_file, name="log_file")
    worker_json_output = _require_non_empty(
        options.worker_json_output,
        name="worker_json_output",
    )
    worker_markdown_output = _require_non_empty(
        options.worker_markdown_output,
        name="worker_markdown_output",
    )
    return ForegroundAppWorkerSupervisorOptions(
        workdir=Path(workdir),
        python_bin=python_bin,
        host=host,
        port=port,
        database_url_env=database_url_env,
        require_database_url=options.require_database_url,
        supervisor_pid_file=supervisor_pid_file,
        web_pid_file=web_pid_file,
        log_file=log_file,
        worker_cycle_interval_seconds=options.worker_cycle_interval_seconds,
        worker_failure_tolerance=options.worker_failure_tolerance,
        pipeline_limit=options.pipeline_limit,
        embedding_limit_per_profile=options.embedding_limit_per_profile,
        guard_health_timeout_seconds=options.guard_health_timeout_seconds,
        max_provider_health_elapsed_ms=options.max_provider_health_elapsed_ms,
        worker_json_output=worker_json_output,
        worker_markdown_output=worker_markdown_output,
        no_default_qwen_token_guard=options.no_default_qwen_token_guard,
        check_port_available=options.check_port_available,
    )


def _worker_command(options: ForegroundAppWorkerSupervisorOptions) -> tuple[str, ...]:
    command = [
        options.python_bin,
        "scripts/run_foreground_workers.py",
        "--pipeline-limit",
        str(options.pipeline_limit),
        "--embedding-limit-per-profile",
        str(options.embedding_limit_per_profile),
        "--guard-health-timeout-seconds",
        _number_text(options.guard_health_timeout_seconds),
        "--max-provider-health-elapsed-ms",
        str(options.max_provider_health_elapsed_ms),
        "--continue-on-command-failure",
        "--json-output",
        options.worker_json_output,
        "--markdown-output",
        options.worker_markdown_output,
        "--pretty",
    ]
    if options.no_default_qwen_token_guard:
        command.append("--no-default-qwen-token-guard")
    return tuple(command)


def _launch_plan_check(
    launch_plan: ForegroundProductionLaunchPlan,
) -> ForegroundAppWorkerSupervisorCheck:
    if launch_plan.status == LAUNCH_PLAN_BLOCKED:
        return ForegroundAppWorkerSupervisorCheck(
            code="launch_plan",
            status=SUPERVISOR_CHECK_FAILED,
            detail="Foreground web launch plan is blocked.",
            metadata={"launch_failed_count": launch_plan.failed_count},
        )
    if launch_plan.status == LAUNCH_PLAN_WARNING:
        return ForegroundAppWorkerSupervisorCheck(
            code="launch_plan",
            status=SUPERVISOR_CHECK_WARNING,
            detail="Foreground web launch plan has warnings.",
            metadata={"launch_warning_count": launch_plan.warning_count},
        )
    return ForegroundAppWorkerSupervisorCheck(
        code="launch_plan",
        status=SUPERVISOR_CHECK_PASSED,
        detail="Foreground web launch plan is ready.",
    )


def _supervisor_path_check(
    *,
    supervisor_pid_file: str,
    web_pid_file: str,
    log_file: str,
) -> ForegroundAppWorkerSupervisorCheck:
    paths = {
        "supervisor_pid_file": Path(supervisor_pid_file),
        "web_pid_file": Path(web_pid_file),
        "log_file": Path(log_file),
    }
    if paths["supervisor_pid_file"] == paths["web_pid_file"]:
        return ForegroundAppWorkerSupervisorCheck(
            code="supervisor_paths",
            status=SUPERVISOR_CHECK_FAILED,
            detail="Supervisor PID file must differ from the web PID file.",
            metadata={key: str(value) for key, value in paths.items()},
        )
    if paths["log_file"] in {paths["supervisor_pid_file"], paths["web_pid_file"]}:
        return ForegroundAppWorkerSupervisorCheck(
            code="supervisor_paths",
            status=SUPERVISOR_CHECK_FAILED,
            detail="Supervisor log file must differ from PID files.",
            metadata={key: str(value) for key, value in paths.items()},
        )
    return ForegroundAppWorkerSupervisorCheck(
        code="supervisor_paths",
        status=SUPERVISOR_CHECK_PASSED,
        detail="Supervisor PID/log paths are separated.",
        metadata={key: str(value) for key, value in paths.items()},
    )


def _worker_cycle_check(
    *,
    interval_seconds: float,
    worker_failure_tolerance: int,
    pipeline_limit: int,
    embedding_limit_per_profile: int,
    guard_health_timeout_seconds: float,
    max_provider_health_elapsed_ms: int,
) -> ForegroundAppWorkerSupervisorCheck:
    invalid = []
    if interval_seconds <= 0:
        invalid.append("worker_cycle_interval_seconds")
    if worker_failure_tolerance < 0:
        invalid.append("worker_failure_tolerance")
    if pipeline_limit < 0:
        invalid.append("pipeline_limit")
    if embedding_limit_per_profile <= 0:
        invalid.append("embedding_limit_per_profile")
    if guard_health_timeout_seconds <= 0:
        invalid.append("guard_health_timeout_seconds")
    if max_provider_health_elapsed_ms <= 0:
        invalid.append("max_provider_health_elapsed_ms")
    if invalid:
        return ForegroundAppWorkerSupervisorCheck(
            code="worker_cycle",
            status=SUPERVISOR_CHECK_FAILED,
            detail="Worker cycle settings contain invalid values.",
            metadata={"invalid_fields": invalid},
        )
    if pipeline_limit == 0:
        return ForegroundAppWorkerSupervisorCheck(
            code="worker_cycle",
            status=SUPERVISOR_CHECK_WARNING,
            detail="Pipeline worker cycle is disabled; only embedding work can run.",
            metadata={
                "pipeline_limit": pipeline_limit,
                "embedding_limit_per_profile": embedding_limit_per_profile,
                "worker_failure_tolerance": worker_failure_tolerance,
            },
        )
    return ForegroundAppWorkerSupervisorCheck(
        code="worker_cycle",
        status=SUPERVISOR_CHECK_PASSED,
        detail="Worker cycle settings are bounded.",
        metadata={
            "interval_seconds": interval_seconds,
            "worker_failure_tolerance": worker_failure_tolerance,
            "pipeline_limit": pipeline_limit,
            "embedding_limit_per_profile": embedding_limit_per_profile,
            "guard_health_timeout_seconds": guard_health_timeout_seconds,
            "max_provider_health_elapsed_ms": max_provider_health_elapsed_ms,
        },
    )


def _overall_plan_status(checks: tuple[ForegroundAppWorkerSupervisorCheck, ...]) -> str:
    if any(check.status == SUPERVISOR_CHECK_FAILED for check in checks):
        return SUPERVISOR_PLAN_BLOCKED
    if any(check.status == SUPERVISOR_CHECK_WARNING for check in checks):
        return SUPERVISOR_PLAN_WARNING
    return SUPERVISOR_PLAN_READY


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


def _number_text(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _env_str(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    if value is None:
        return default
    selected_value = value.strip()
    return selected_value if selected_value else default


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None or not value.strip():
        return default
    selected_value = value.strip().lower()
    if selected_value in _TRUE_VALUES:
        return True
    if selected_value in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of: 1, 0, true, false, yes, no, on, off")


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
