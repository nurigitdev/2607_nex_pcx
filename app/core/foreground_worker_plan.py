"""Foreground worker command plan for supervised operations."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.service_startup_templates import DEFAULT_CHUNK_POLICY_NAMES

FOREGROUND_WORKER_PLAN_VERSION = 1
DEFAULT_PIPELINE_WORKER_NAME = "foreground-pipeline-worker"
DEFAULT_EMBEDDING_WORKER_NAME = "foreground-embedding-worker"
DEFAULT_LEASE_SECONDS = 300
DEFAULT_EMBEDDING_LIMIT = 5


@dataclass(frozen=True)
class ForegroundWorkerCommand:
    code: str
    purpose: str
    command: tuple[str, ...]
    bounded: bool
    risk_note: str

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command)


@dataclass(frozen=True)
class ForegroundWorkerPlan:
    generated_at: datetime
    workdir: str
    database_url_source: str
    lease_seconds: int
    embedding_limit: int
    chunk_policy_names: tuple[str, ...]
    commands: tuple[ForegroundWorkerCommand, ...]


def build_foreground_worker_plan(
    *,
    workdir: str | Path = ".",
    python_bin: str = "./.venv/bin/python",
    database_url_source: str = "${NEX_PCX_DATABASE_URL}",
    pipeline_worker_name: str = DEFAULT_PIPELINE_WORKER_NAME,
    embedding_worker_name: str = DEFAULT_EMBEDDING_WORKER_NAME,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    embedding_limit: int = DEFAULT_EMBEDDING_LIMIT,
    chunk_policy_names: tuple[str, ...] = DEFAULT_CHUNK_POLICY_NAMES,
    generated_at: datetime | None = None,
) -> ForegroundWorkerPlan:
    selected_workdir = _require_non_empty(str(workdir), name="workdir")
    selected_python_bin = _require_non_empty(python_bin, name="python_bin")
    selected_database_url_source = _require_non_empty(
        database_url_source,
        name="database_url_source",
    )
    selected_pipeline_worker_name = _require_non_empty(
        pipeline_worker_name,
        name="pipeline_worker_name",
    )
    selected_embedding_worker_name = _require_non_empty(
        embedding_worker_name,
        name="embedding_worker_name",
    )
    selected_lease_seconds = _validate_positive_int(lease_seconds, name="lease_seconds")
    selected_embedding_limit = _validate_positive_int(embedding_limit, name="embedding_limit")
    selected_chunk_policy_names = _validate_chunk_policy_names(chunk_policy_names)
    commands = (
        ForegroundWorkerCommand(
            code="pipeline_worker_help",
            purpose="Validate pipeline worker import and CLI arguments without claiming work.",
            command=(selected_python_bin, "scripts/process_pipeline_job.py", "--help"),
            bounded=True,
            risk_note="Read-only CLI check.",
        ),
        ForegroundWorkerCommand(
            code="embedding_worker_help",
            purpose="Validate embedding worker import and CLI arguments without claiming work.",
            command=(selected_python_bin, "scripts/process_embedding_job.py", "--help"),
            bounded=True,
            risk_note="Read-only CLI check.",
        ),
        ForegroundWorkerCommand(
            code="pipeline_worker_once",
            purpose="Process at most one queued pipeline job using the configured chunk policies.",
            command=(
                selected_python_bin,
                "scripts/process_pipeline_job.py",
                "--database-url",
                selected_database_url_source,
                "--worker-name",
                selected_pipeline_worker_name,
                "--lease-seconds",
                str(selected_lease_seconds),
                "--chunk-policy-names",
                *selected_chunk_policy_names,
            ),
            bounded=True,
            risk_note="Claims at most one pipeline job.",
        ),
        ForegroundWorkerCommand(
            code="embedding_worker_batch",
            purpose=(
                "Process a small bounded batch of pending embedding jobs through "
                "provider routes."
            ),
            command=(
                selected_python_bin,
                "scripts/process_embedding_job.py",
                "--database-url",
                selected_database_url_source,
                "--worker-name",
                selected_embedding_worker_name,
                "--provider-source",
                "route",
                "--require-route-readiness",
                "--limit",
                str(selected_embedding_limit),
                "--lease-seconds",
                str(selected_lease_seconds),
            ),
            bounded=True,
            risk_note="Claims at most the configured embedding job limit.",
        ),
    )
    return ForegroundWorkerPlan(
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(Path(selected_workdir)),
        database_url_source=selected_database_url_source,
        lease_seconds=selected_lease_seconds,
        embedding_limit=selected_embedding_limit,
        chunk_policy_names=selected_chunk_policy_names,
        commands=commands,
    )


def foreground_worker_plan_payload(plan: ForegroundWorkerPlan) -> dict[str, object]:
    return {
        "version": FOREGROUND_WORKER_PLAN_VERSION,
        "generated_at": plan.generated_at.isoformat(),
        "generated_at_label": plan.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": plan.workdir,
        "database_url_source": plan.database_url_source,
        "lease_seconds": plan.lease_seconds,
        "embedding_limit": plan.embedding_limit,
        "chunk_policy_names": list(plan.chunk_policy_names),
        "commands": [
            {
                **asdict(command),
                "shell_command": command.shell_command,
            }
            for command in plan.commands
        ],
    }


def render_foreground_worker_plan_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Foreground Worker Command Plan",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Workdir: `{_text(payload.get('workdir'))}`",
        f"- Database URL Source: `{_text(payload.get('database_url_source'))}`",
        f"- Lease Seconds: {_text(payload.get('lease_seconds'))}",
        f"- Embedding Limit: {_text(payload.get('embedding_limit'))}",
        "",
        "## Commands",
        "",
        "| Code | Bounded | Purpose | Risk Note | Command |",
        "| --- | --- | --- | --- | --- |",
    ]
    for command in payload.get("commands", []):
        command_payload = _dict(command)
        lines.append(
            "| "
            f"{_md_cell(command_payload.get('code'))} | "
            f"{_md_cell(command_payload.get('bounded'))} | "
            f"{_md_cell(command_payload.get('purpose'))} | "
            f"{_md_cell(command_payload.get('risk_note'))} | "
            f"`{_md_cell(command_payload.get('shell_command'))}` |"
        )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- Run help checks before foreground worker commands.",
            "- Run worker commands from the project root in a supervised terminal.",
            "- Keep `--limit` and lease settings bounded for pre-CX operation.",
            "- Stop and inspect logs if any command returns a failed job or provider route error.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _validate_chunk_policy_names(chunk_policy_names: tuple[str, ...]) -> tuple[str, ...]:
    selected_names = tuple(name.strip() for name in chunk_policy_names if name.strip())
    if not selected_names:
        raise ValueError("at least one chunk policy name is required")
    return selected_names


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


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
