"""DGX provider and vLLM snapshot collection runner planning/evidence."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.core.provider_resource_probe import DEFAULT_DGX_PROVIDER_RESOURCE_HOST

DGX_SNAPSHOT_COLLECTION_VERSION = 1

COLLECTION_CHECK_PASSED = "passed"
COLLECTION_CHECK_WARNING = "warning"
COLLECTION_CHECK_FAILED = "failed"

COLLECTION_PLAN_READY = "ready"
COLLECTION_PLAN_BLOCKED = "blocked"

COLLECTION_STATUS_PLANNED = "planned"
COLLECTION_STATUS_COMPLETED = "completed"
COLLECTION_STATUS_ATTENTION = "attention"
COLLECTION_STATUS_PARTIAL = "partial"
COLLECTION_STATUS_FAILED = "failed"
COLLECTION_STATUS_BLOCKED = "blocked"

COMMAND_STATUS_COLLECTED = "collected"
COMMAND_STATUS_ATTENTION = "attention"
COMMAND_STATUS_FAILED = "failed"

COMPONENT_VLLM_RUNTIME = "vllm_runtime"
COMPONENT_PROVIDER_RESOURCES = "provider_resources"

DEFAULT_DATABASE_URL_ENV = "NEX_PCX_DATABASE_URL"
DEFAULT_PROVIDER_SSH_USER = "nexpcx"
DEFAULT_REMOTE_WORKDIR = "/home/nexpcx/2607_nex_pcx"
DEFAULT_REMOTE_PYTHON_BIN = "/home/nexpcx/2607_nex_pcx/.venv/bin/python"
DEFAULT_DGX_VLLM_BASE_URL = "http://192.168.20.243:12000"
DEFAULT_DGX_VLLM_PROVIDER_NAME = "dgx_vllm_qwen36_27b_nvfp4"
DEFAULT_DGX_VLLM_MODEL_ID = "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4"
DEFAULT_VLLM_TIMEOUT_SECONDS = 5.0
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 15
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_COLLECTION_INTERVAL_SECONDS = 60.0
DEFAULT_COLLECTION_MAX_CYCLES = 1
DEFAULT_JSON_OUTPUT = "artifacts/dgx_snapshot_collection.json"
DEFAULT_MARKDOWN_OUTPUT = "artifacts/dgx_snapshot_collection.md"

ENV_COLLECTION_WORKDIR = "NEX_PCX_COLLECTION_WORKDIR"
ENV_COLLECTION_PYTHON_BIN = "NEX_PCX_COLLECTION_PYTHON_BIN"
ENV_COLLECTION_DATABASE_URL_ENV = "NEX_PCX_COLLECTION_DATABASE_URL_ENV"
ENV_COLLECTION_HOST = "NEX_PCX_COLLECTION_HOST"
ENV_COLLECTION_SSH_USER = "NEX_PCX_COLLECTION_SSH_USER"
ENV_COLLECTION_REMOTE_WORKDIR = "NEX_PCX_COLLECTION_REMOTE_WORKDIR"
ENV_COLLECTION_REMOTE_PYTHON_BIN = "NEX_PCX_COLLECTION_REMOTE_PYTHON_BIN"
ENV_COLLECTION_VLLM_BASE_URL = "NEX_PCX_COLLECTION_VLLM_BASE_URL"
ENV_COLLECTION_VLLM_PROVIDER_NAME = "NEX_PCX_COLLECTION_VLLM_PROVIDER_NAME"
ENV_COLLECTION_VLLM_MODEL_ID = "NEX_PCX_COLLECTION_VLLM_MODEL_ID"
ENV_COLLECTION_PROVIDER_SELECTORS = "NEX_PCX_COLLECTION_PROVIDER_SELECTORS"
ENV_COLLECTION_VLLM_TIMEOUT_SECONDS = "NEX_PCX_COLLECTION_VLLM_TIMEOUT_SECONDS"
ENV_COLLECTION_PROVIDER_TIMEOUT_SECONDS = "NEX_PCX_COLLECTION_PROVIDER_TIMEOUT_SECONDS"
ENV_COLLECTION_COMMAND_TIMEOUT_SECONDS = "NEX_PCX_COLLECTION_COMMAND_TIMEOUT_SECONDS"
ENV_COLLECTION_INTERVAL_SECONDS = "NEX_PCX_COLLECTION_INTERVAL_SECONDS"
ENV_COLLECTION_MAX_CYCLES = "NEX_PCX_COLLECTION_MAX_CYCLES"
ENV_COLLECTION_PROVIDER_LOCAL_ONLY = "NEX_PCX_COLLECTION_PROVIDER_LOCAL_ONLY"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class DgxSnapshotCollectionOptions:
    workdir: str | Path = "."
    python_bin: str = "./.venv/bin/python"
    database_url_env: str = DEFAULT_DATABASE_URL_ENV
    require_database_url: bool = True
    components: tuple[str, ...] = ("all",)
    host: str = DEFAULT_DGX_PROVIDER_RESOURCE_HOST
    provider_ssh_user: str | None = DEFAULT_PROVIDER_SSH_USER
    provider_remote_workdir: str = DEFAULT_REMOTE_WORKDIR
    provider_remote_python_bin: str = DEFAULT_REMOTE_PYTHON_BIN
    provider_selectors: tuple[str, ...] = ("all",)
    provider_local_only: bool = False
    vllm_base_url: str = DEFAULT_DGX_VLLM_BASE_URL
    vllm_provider_name: str = DEFAULT_DGX_VLLM_PROVIDER_NAME
    vllm_model_id: str = DEFAULT_DGX_VLLM_MODEL_ID
    vllm_timeout_seconds: float = DEFAULT_VLLM_TIMEOUT_SECONDS
    provider_timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    interval_seconds: float = DEFAULT_COLLECTION_INTERVAL_SECONDS
    max_cycles: int = DEFAULT_COLLECTION_MAX_CYCLES


@dataclass(frozen=True)
class DgxSnapshotCollectionCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DgxSnapshotCollectionCommandPlan:
    code: str
    component: str
    purpose: str
    command: tuple[str, ...]
    expected_payload_key: str
    bounded: bool
    risk_note: str

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command)


@dataclass(frozen=True)
class DgxSnapshotCollectionPlan:
    status: str
    generated_at: datetime
    workdir: str
    database_url_env: str
    host: str
    components: tuple[str, ...]
    provider_selectors: tuple[str, ...]
    provider_local_only: bool
    interval_seconds: float
    max_cycles: int
    command_timeout_seconds: float
    commands: tuple[DgxSnapshotCollectionCommandPlan, ...]
    checks: tuple[DgxSnapshotCollectionCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if check.status == COLLECTION_CHECK_FAILED)

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == COLLECTION_CHECK_WARNING)


@dataclass(frozen=True)
class DgxSnapshotCollectionCommandResult:
    code: str
    component: str
    command: tuple[str, ...]
    exit_code: int
    elapsed_ms: int
    status: str
    payload: dict[str, object] | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_message: str | None = None

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command)

    @property
    def collected(self) -> bool:
        return self.status in {COMMAND_STATUS_COLLECTED, COMMAND_STATUS_ATTENTION}


@dataclass(frozen=True)
class DgxSnapshotCollectionCycle:
    index: int
    started_at: datetime
    finished_at: datetime
    status: str
    results: tuple[DgxSnapshotCollectionCommandResult, ...]

    @property
    def result_count(self) -> int:
        return len(self.results)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.status == COMMAND_STATUS_FAILED)

    @property
    def attention_count(self) -> int:
        return sum(1 for result in self.results if result.status == COMMAND_STATUS_ATTENTION)

    @property
    def collected_count(self) -> int:
        return sum(1 for result in self.results if result.collected)


@dataclass(frozen=True)
class DgxSnapshotCollectionEvidence:
    status: str
    dry_run: bool
    generated_at: datetime
    plan: DgxSnapshotCollectionPlan
    cycles: tuple[DgxSnapshotCollectionCycle, ...] = ()
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def cycle_count(self) -> int:
        return len(self.cycles)

    @property
    def command_result_count(self) -> int:
        return sum(cycle.result_count for cycle in self.cycles)

    @property
    def failed_command_count(self) -> int:
        return sum(cycle.failed_count for cycle in self.cycles)

    @property
    def attention_command_count(self) -> int:
        return sum(cycle.attention_count for cycle in self.cycles)


def dgx_snapshot_collection_options_from_environ(
    environ: Mapping[str, str],
    *,
    defaults: DgxSnapshotCollectionOptions | None = None,
) -> DgxSnapshotCollectionOptions:
    base = defaults or DgxSnapshotCollectionOptions()
    return DgxSnapshotCollectionOptions(
        workdir=_env_str(environ, ENV_COLLECTION_WORKDIR, str(base.workdir)),
        python_bin=_env_str(environ, ENV_COLLECTION_PYTHON_BIN, base.python_bin),
        database_url_env=_env_str(
            environ,
            ENV_COLLECTION_DATABASE_URL_ENV,
            base.database_url_env,
        ),
        require_database_url=base.require_database_url,
        components=base.components,
        host=_env_str(environ, ENV_COLLECTION_HOST, base.host),
        provider_ssh_user=_env_optional_str(
            environ,
            ENV_COLLECTION_SSH_USER,
            base.provider_ssh_user,
        ),
        provider_remote_workdir=_env_str(
            environ,
            ENV_COLLECTION_REMOTE_WORKDIR,
            base.provider_remote_workdir,
        ),
        provider_remote_python_bin=_env_str(
            environ,
            ENV_COLLECTION_REMOTE_PYTHON_BIN,
            base.provider_remote_python_bin,
        ),
        provider_selectors=_env_csv(
            environ,
            ENV_COLLECTION_PROVIDER_SELECTORS,
            base.provider_selectors,
        ),
        provider_local_only=_env_bool(
            environ,
            ENV_COLLECTION_PROVIDER_LOCAL_ONLY,
            base.provider_local_only,
        ),
        vllm_base_url=_env_str(environ, ENV_COLLECTION_VLLM_BASE_URL, base.vllm_base_url),
        vllm_provider_name=_env_str(
            environ,
            ENV_COLLECTION_VLLM_PROVIDER_NAME,
            base.vllm_provider_name,
        ),
        vllm_model_id=_env_str(environ, ENV_COLLECTION_VLLM_MODEL_ID, base.vllm_model_id),
        vllm_timeout_seconds=_env_float(
            environ,
            ENV_COLLECTION_VLLM_TIMEOUT_SECONDS,
            base.vllm_timeout_seconds,
        ),
        provider_timeout_seconds=_env_int(
            environ,
            ENV_COLLECTION_PROVIDER_TIMEOUT_SECONDS,
            base.provider_timeout_seconds,
        ),
        command_timeout_seconds=_env_float(
            environ,
            ENV_COLLECTION_COMMAND_TIMEOUT_SECONDS,
            base.command_timeout_seconds,
        ),
        interval_seconds=_env_float(
            environ,
            ENV_COLLECTION_INTERVAL_SECONDS,
            base.interval_seconds,
        ),
        max_cycles=_env_int(environ, ENV_COLLECTION_MAX_CYCLES, base.max_cycles),
    )


def build_dgx_snapshot_collection_plan(
    options: DgxSnapshotCollectionOptions,
    *,
    environ: Mapping[str, str] | None = None,
    generated_at: datetime | None = None,
) -> DgxSnapshotCollectionPlan:
    selected_workdir = _require_non_empty(str(options.workdir), name="workdir")
    selected_python_bin = _require_non_empty(options.python_bin, name="python_bin")
    selected_database_url_env = _require_non_empty(
        options.database_url_env,
        name="database_url_env",
    )
    selected_host = _require_non_empty(options.host, name="host")
    selected_components = _normalize_components(options.components)
    selected_provider_selectors = _normalize_provider_selectors(options.provider_selectors)
    selected_interval = _validate_non_negative_float(
        options.interval_seconds,
        name="interval_seconds",
    )
    selected_max_cycles = _validate_positive_int(options.max_cycles, name="max_cycles")
    selected_vllm_timeout = _validate_positive_float(
        options.vllm_timeout_seconds,
        name="vllm_timeout_seconds",
    )
    selected_provider_timeout = _validate_positive_int(
        options.provider_timeout_seconds,
        name="provider_timeout_seconds",
    )
    selected_command_timeout = _validate_positive_float(
        options.command_timeout_seconds,
        name="command_timeout_seconds",
    )
    env = environ or {}
    database_url_present = bool(env.get(selected_database_url_env, "").strip())
    checks = [
        DgxSnapshotCollectionCheck(
            code="database_url",
            status=(
                COLLECTION_CHECK_PASSED
                if database_url_present or not options.require_database_url
                else COLLECTION_CHECK_FAILED
            ),
            detail=_database_url_check_detail(
                database_url_env=selected_database_url_env,
                database_url_present=database_url_present,
                require_database_url=options.require_database_url,
            ),
        ),
        DgxSnapshotCollectionCheck(
            code="components",
            status=COLLECTION_CHECK_PASSED,
            detail=f"Selected components: {', '.join(selected_components)}.",
        ),
    ]
    if COMPONENT_PROVIDER_RESOURCES in selected_components:
        provider_ssh_user = (
            options.provider_ssh_user.strip()
            if isinstance(options.provider_ssh_user, str) and options.provider_ssh_user.strip()
            else None
        )
        checks.append(
            DgxSnapshotCollectionCheck(
                code="provider_probe_mode",
                status=(
                    COLLECTION_CHECK_PASSED
                    if options.provider_local_only or provider_ssh_user
                    else COLLECTION_CHECK_FAILED
                ),
                detail=(
                    "Provider resource probe will run locally on the current host."
                    if options.provider_local_only
                    else (
                        f"Provider resource probe will be delegated through SSH user "
                        f"{provider_ssh_user}."
                        if provider_ssh_user
                        else "Provider resource probe requires --ssh-user or --provider-local-only."
                    )
                ),
                metadata={"provider_selectors": list(selected_provider_selectors)},
            )
        )

    commands = tuple(
        _build_command_plans(
            selected_components,
            python_bin=selected_python_bin,
            host=selected_host,
            provider_ssh_user=options.provider_ssh_user,
            provider_remote_workdir=options.provider_remote_workdir,
            provider_remote_python_bin=options.provider_remote_python_bin,
            provider_selectors=selected_provider_selectors,
            provider_local_only=options.provider_local_only,
            vllm_base_url=options.vllm_base_url,
            vllm_provider_name=options.vllm_provider_name,
            vllm_model_id=options.vllm_model_id,
            vllm_timeout_seconds=selected_vllm_timeout,
            provider_timeout_seconds=selected_provider_timeout,
        )
    )
    failed_count = sum(1 for check in checks if check.status == COLLECTION_CHECK_FAILED)
    return DgxSnapshotCollectionPlan(
        status=COLLECTION_PLAN_BLOCKED if failed_count else COLLECTION_PLAN_READY,
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(Path(selected_workdir)),
        database_url_env=selected_database_url_env,
        host=selected_host,
        components=selected_components,
        provider_selectors=selected_provider_selectors,
        provider_local_only=options.provider_local_only,
        interval_seconds=selected_interval,
        max_cycles=selected_max_cycles,
        command_timeout_seconds=selected_command_timeout,
        commands=commands,
        checks=tuple(checks),
    )


def classify_dgx_snapshot_command_result(
    *,
    code: str,
    exit_code: int,
    payload: dict[str, object] | None,
    error_message: str | None = None,
) -> str:
    if error_message or not payload:
        return COMMAND_STATUS_FAILED
    if code == COMPONENT_VLLM_RUNTIME:
        if not isinstance(payload.get("snapshot_record"), dict):
            return COMMAND_STATUS_FAILED
        return COMMAND_STATUS_COLLECTED if exit_code == 0 else COMMAND_STATUS_ATTENTION
    if code == COMPONENT_PROVIDER_RESOURCES:
        snapshot_records = payload.get("snapshot_records")
        if not isinstance(snapshot_records, list) or not snapshot_records:
            return COMMAND_STATUS_FAILED
        observed_status = str(payload.get("status") or "").strip().lower()
        if exit_code != 0 or observed_status in {"warning", "critical", "unknown"}:
            return COMMAND_STATUS_ATTENTION
        return COMMAND_STATUS_COLLECTED
    return COMMAND_STATUS_COLLECTED if exit_code == 0 else COMMAND_STATUS_FAILED


def build_dgx_snapshot_collection_cycle(
    *,
    index: int,
    results: Sequence[DgxSnapshotCollectionCommandResult],
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> DgxSnapshotCollectionCycle:
    selected_started_at = started_at or datetime.now(UTC)
    selected_finished_at = finished_at or datetime.now(UTC)
    return DgxSnapshotCollectionCycle(
        index=_validate_positive_int(index, name="index"),
        started_at=selected_started_at,
        finished_at=selected_finished_at,
        status=_cycle_status(results),
        results=tuple(results),
    )


def build_dgx_snapshot_collection_evidence(
    plan: DgxSnapshotCollectionPlan,
    *,
    status: str,
    dry_run: bool,
    cycles: Sequence[DgxSnapshotCollectionCycle] = (),
    message: str = "",
    metadata: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> DgxSnapshotCollectionEvidence:
    selected_status = status.strip()
    if selected_status not in {
        COLLECTION_STATUS_PLANNED,
        COLLECTION_STATUS_COMPLETED,
        COLLECTION_STATUS_ATTENTION,
        COLLECTION_STATUS_PARTIAL,
        COLLECTION_STATUS_FAILED,
        COLLECTION_STATUS_BLOCKED,
    }:
        raise ValueError("unsupported DGX snapshot collection status")
    return DgxSnapshotCollectionEvidence(
        status=selected_status,
        dry_run=dry_run,
        generated_at=generated_at or datetime.now(UTC),
        plan=plan,
        cycles=tuple(cycles),
        message=message,
        metadata=dict(metadata or {}),
    )


def dgx_snapshot_collection_status_from_cycles(
    cycles: Sequence[DgxSnapshotCollectionCycle],
) -> str:
    if not cycles:
        return COLLECTION_STATUS_COMPLETED
    if any(cycle.status == COLLECTION_STATUS_FAILED for cycle in cycles):
        return (
            COLLECTION_STATUS_PARTIAL
            if any(cycle.collected_count for cycle in cycles)
            else COLLECTION_STATUS_FAILED
        )
    if any(cycle.status == COLLECTION_STATUS_PARTIAL for cycle in cycles):
        return COLLECTION_STATUS_PARTIAL
    if any(cycle.status == COLLECTION_STATUS_ATTENTION for cycle in cycles):
        return COLLECTION_STATUS_ATTENTION
    return COLLECTION_STATUS_COMPLETED


def dgx_snapshot_collection_evidence_payload(
    evidence: DgxSnapshotCollectionEvidence,
) -> dict[str, object]:
    return {
        "version": DGX_SNAPSHOT_COLLECTION_VERSION,
        "status": evidence.status,
        "dry_run": evidence.dry_run,
        "generated_at": evidence.generated_at.isoformat(),
        "generated_at_label": evidence.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "message": evidence.message,
        "cycle_count": evidence.cycle_count,
        "command_result_count": evidence.command_result_count,
        "failed_command_count": evidence.failed_command_count,
        "attention_command_count": evidence.attention_command_count,
        "metadata": dict(evidence.metadata),
        "plan": dgx_snapshot_collection_plan_payload(evidence.plan),
        "cycles": [dgx_snapshot_collection_cycle_payload(cycle) for cycle in evidence.cycles],
    }


def dgx_snapshot_collection_plan_payload(
    plan: DgxSnapshotCollectionPlan,
) -> dict[str, object]:
    return {
        "version": DGX_SNAPSHOT_COLLECTION_VERSION,
        "status": plan.status,
        "generated_at": plan.generated_at.isoformat(),
        "generated_at_label": plan.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": plan.workdir,
        "database_url_env": plan.database_url_env,
        "host": plan.host,
        "components": list(plan.components),
        "provider_selectors": list(plan.provider_selectors),
        "provider_local_only": plan.provider_local_only,
        "interval_seconds": plan.interval_seconds,
        "max_cycles": plan.max_cycles,
        "command_timeout_seconds": plan.command_timeout_seconds,
        "check_count": plan.check_count,
        "failed_count": plan.failed_count,
        "warning_count": plan.warning_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "metadata": dict(check.metadata),
            }
            for check in plan.checks
        ],
        "commands": [
            {
                "code": command.code,
                "component": command.component,
                "purpose": command.purpose,
                "command": list(command.command),
                "shell_command": command.shell_command,
                "expected_payload_key": command.expected_payload_key,
                "bounded": command.bounded,
                "risk_note": command.risk_note,
            }
            for command in plan.commands
        ],
    }


def dgx_snapshot_collection_cycle_payload(
    cycle: DgxSnapshotCollectionCycle,
) -> dict[str, object]:
    return {
        "index": cycle.index,
        "started_at": cycle.started_at.isoformat(),
        "started_at_label": cycle.started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": cycle.finished_at.isoformat(),
        "finished_at_label": cycle.finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        "status": cycle.status,
        "result_count": cycle.result_count,
        "failed_count": cycle.failed_count,
        "attention_count": cycle.attention_count,
        "collected_count": cycle.collected_count,
        "results": [
            dgx_snapshot_collection_command_result_payload(result) for result in cycle.results
        ],
    }


def dgx_snapshot_collection_command_result_payload(
    result: DgxSnapshotCollectionCommandResult,
) -> dict[str, object]:
    return {
        "code": result.code,
        "component": result.component,
        "command": list(result.command),
        "shell_command": result.shell_command,
        "exit_code": result.exit_code,
        "elapsed_ms": result.elapsed_ms,
        "status": result.status,
        "collected": result.collected,
        "payload": result.payload,
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
        "error_message": result.error_message,
    }


def render_dgx_snapshot_collection_markdown(payload: dict[str, object]) -> str:
    plan = _dict(payload.get("plan"))
    provider_selectors = ", ".join(_strings(plan.get("provider_selectors"))) or "-"
    lines = [
        "# DGX Snapshot Collection Evidence",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Dry Run: {_text(payload.get('dry_run'))}",
        f"- Message: {_text(payload.get('message'))}",
        f"- Cycles: {_text(payload.get('cycle_count'))}",
        f"- Commands: {_text(payload.get('command_result_count'))}",
        f"- Attention Commands: {_text(payload.get('attention_command_count'))}",
        f"- Failed Commands: {_text(payload.get('failed_command_count'))}",
        "",
        "## Plan",
        "",
        f"- Plan Status: `{_text(plan.get('status'))}`",
        f"- Workdir: `{_text(plan.get('workdir'))}`",
        f"- Database URL Env: `{_text(plan.get('database_url_env'))}`",
        f"- Host: `{_text(plan.get('host'))}`",
        f"- Components: {_text(', '.join(_strings(plan.get('components'))) or '-')}",
        f"- Provider Selectors: {_text(provider_selectors)}",
        "",
        "## Checks",
        "",
        "| Code | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in _list(plan.get("checks")):
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
            "## Commands",
            "",
            "| Code | Component | Bounded | Purpose | Command |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for command in _list(plan.get("commands")):
        command_payload = _dict(command)
        lines.append(
            "| "
            f"{_md_cell(command_payload.get('code'))} | "
            f"{_md_cell(command_payload.get('component'))} | "
            f"{_md_cell(command_payload.get('bounded'))} | "
            f"{_md_cell(command_payload.get('purpose'))} | "
            f"`{_md_cell(command_payload.get('shell_command'))}` |"
        )
    lines.extend(
        [
            "",
            "## Cycle Results",
            "",
            "| Cycle | Code | Status | Exit | Elapsed Ms | Error |",
            "| ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for cycle in _list(payload.get("cycles")):
        cycle_payload = _dict(cycle)
        for result in _list(cycle_payload.get("results")):
            result_payload = _dict(result)
            lines.append(
                "| "
                f"{_md_cell(cycle_payload.get('index'))} | "
                f"{_md_cell(result_payload.get('code'))} | "
                f"{_md_cell(result_payload.get('status'))} | "
                f"{_md_cell(result_payload.get('exit_code'))} | "
                f"{_md_cell(result_payload.get('elapsed_ms'))} | "
                f"{_md_cell(result_payload.get('error_message'))} |"
            )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            (
                "- This runner fills the vLLM Runtime and Provider Resource menus with "
                "persisted snapshots."
            ),
            (
                "- vLLM metrics are collected through the OpenAI-compatible runtime "
                "`/metrics` endpoint."
            ),
            (
                "- Provider resources are collected from the DGX host process table "
                "through SSH delegation or local DGX execution."
            ),
            (
                "- Observed provider critical status is reported as attention when "
                "persistence succeeds."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _build_command_plans(
    components: tuple[str, ...],
    *,
    python_bin: str,
    host: str,
    provider_ssh_user: str | None,
    provider_remote_workdir: str,
    provider_remote_python_bin: str,
    provider_selectors: tuple[str, ...],
    provider_local_only: bool,
    vllm_base_url: str,
    vllm_provider_name: str,
    vllm_model_id: str | None,
    vllm_timeout_seconds: float,
    provider_timeout_seconds: int,
) -> list[DgxSnapshotCollectionCommandPlan]:
    commands: list[DgxSnapshotCollectionCommandPlan] = []
    if COMPONENT_VLLM_RUNTIME in components:
        command = [
            python_bin,
            "scripts/scrape_vllm_runtime_metrics.py",
            "--base-url",
            _require_non_empty(vllm_base_url, name="vllm_base_url"),
            "--provider-name",
            _require_non_empty(vllm_provider_name, name="vllm_provider_name"),
            "--timeout-seconds",
            _float_text(vllm_timeout_seconds),
            "--persist",
        ]
        if vllm_model_id is not None and vllm_model_id.strip():
            command.extend(["--model-id", vllm_model_id.strip()])
        commands.append(
            DgxSnapshotCollectionCommandPlan(
                code=COMPONENT_VLLM_RUNTIME,
                component=COMPONENT_VLLM_RUNTIME,
                purpose="Scrape vLLM /metrics and persist a runtime metric snapshot.",
                command=tuple(command),
                expected_payload_key="snapshot_record",
                bounded=True,
                risk_note="Read-only HTTP scrape followed by one DB insert.",
            )
        )
    if COMPONENT_PROVIDER_RESOURCES in components:
        command = [
            python_bin,
            "scripts/probe_provider_resources.py",
            "--host",
            host,
            "--timeout-seconds",
            str(provider_timeout_seconds),
        ]
        if provider_local_only:
            command.append("--local-only")
        else:
            command.extend(
                [
                    "--ssh-user",
                    _require_non_empty(provider_ssh_user or "", name="provider_ssh_user"),
                    "--remote-workdir",
                    _require_non_empty(provider_remote_workdir, name="provider_remote_workdir"),
                    "--remote-python-bin",
                    _require_non_empty(
                        provider_remote_python_bin,
                        name="provider_remote_python_bin",
                    ),
                ]
            )
        for provider_selector in provider_selectors:
            command.extend(["--provider", provider_selector])
        command.append("--persist")
        commands.append(
            DgxSnapshotCollectionCommandPlan(
                code=COMPONENT_PROVIDER_RESOURCES,
                component=COMPONENT_PROVIDER_RESOURCES,
                purpose="Probe DGX provider process resources and persist provider snapshots.",
                command=tuple(command),
                expected_payload_key="snapshot_records",
                bounded=True,
                risk_note="Read-only process probe followed by provider snapshot DB inserts.",
            )
        )
    return commands


def _normalize_components(components: Sequence[str]) -> tuple[str, ...]:
    aliases = {
        "all": (COMPONENT_VLLM_RUNTIME, COMPONENT_PROVIDER_RESOURCES),
        "vllm": (COMPONENT_VLLM_RUNTIME,),
        "vllm-runtime": (COMPONENT_VLLM_RUNTIME,),
        "vllm_runtime": (COMPONENT_VLLM_RUNTIME,),
        "provider": (COMPONENT_PROVIDER_RESOURCES,),
        "provider-resource": (COMPONENT_PROVIDER_RESOURCES,),
        "provider-resources": (COMPONENT_PROVIDER_RESOURCES,),
        "provider_resource": (COMPONENT_PROVIDER_RESOURCES,),
        "provider_resources": (COMPONENT_PROVIDER_RESOURCES,),
        "resources": (COMPONENT_PROVIDER_RESOURCES,),
    }
    selected: list[str] = []
    raw_components = tuple(
        component.strip().lower() for component in components if component.strip()
    )
    for component in raw_components or ("all",):
        if component not in aliases:
            raise ValueError(f"unsupported snapshot collection component: {component}")
        for normalized in aliases[component]:
            if normalized not in selected:
                selected.append(normalized)
    if not selected:
        raise ValueError("at least one snapshot collection component is required")
    return tuple(selected)


def _normalize_provider_selectors(provider_selectors: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(selector.strip() for selector in provider_selectors if selector.strip())
    return selected or ("all",)


def _cycle_status(results: Sequence[DgxSnapshotCollectionCommandResult]) -> str:
    if not results:
        return COLLECTION_STATUS_COMPLETED
    if any(result.status == COMMAND_STATUS_FAILED for result in results):
        return (
            COLLECTION_STATUS_PARTIAL
            if any(result.collected for result in results)
            else COLLECTION_STATUS_FAILED
        )
    if any(result.status == COMMAND_STATUS_ATTENTION for result in results):
        return COLLECTION_STATUS_ATTENTION
    return COLLECTION_STATUS_COMPLETED


def _env_str(environ: Mapping[str, str], key: str, default: str) -> str:
    value = environ.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _database_url_check_detail(
    *,
    database_url_env: str,
    database_url_present: bool,
    require_database_url: bool,
) -> str:
    if database_url_present:
        return f"{database_url_env} is configured."
    if require_database_url:
        return f"{database_url_env} is required for snapshot persistence."
    return f"{database_url_env} is not configured; dry-run planning is allowed."


def _env_optional_str(
    environ: Mapping[str, str],
    key: str,
    default: str | None,
) -> str | None:
    value = environ.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return default


def _env_csv(environ: Mapping[str, str], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = environ.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip()) or default


def _env_int(environ: Mapping[str, str], key: str, default: int) -> int:
    value = environ.get(key)
    return int(value) if isinstance(value, str) and value.strip() else default


def _env_float(environ: Mapping[str, str], key: str, default: float) -> float:
    value = environ.get(key)
    return float(value) if isinstance(value, str) and value.strip() else default


def _env_bool(environ: Mapping[str, str], key: str, default: bool) -> bool:
    value = environ.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{key} must be a boolean value")


def _require_non_empty(value: str, *, name: str) -> str:
    selected_value = value.strip()
    if not selected_value:
        raise ValueError(f"{name} is required")
    return selected_value


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(value)


def _validate_positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return float(value)


def _validate_non_negative_float(value: float, *, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return float(value)


def _float_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _quote_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    return [str(item) for item in _list(value)]


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
