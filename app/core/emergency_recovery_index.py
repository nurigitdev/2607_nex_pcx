"""Emergency recovery command index for NeX_PCX operations."""

import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

EMERGENCY_RECOVERY_INDEX_VERSION = 1


@dataclass(frozen=True)
class RecoveryCommand:
    code: str
    title: str
    description: str
    command: tuple[str, ...]
    destructive: bool = False
    requires_review: bool = False

    @property
    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


@dataclass(frozen=True)
class RecoveryScenario:
    code: str
    title: str
    severity: str
    first_check: str
    command_codes: tuple[str, ...]
    checklist: tuple[str, ...]
    stop_condition: str


@dataclass(frozen=True)
class EmergencyRecoveryIndex:
    generated_at: datetime
    workdir: str
    app_url: str
    provider_host: str
    commands: tuple[RecoveryCommand, ...]
    scenarios: tuple[RecoveryScenario, ...]

    @property
    def command_count(self) -> int:
        return len(self.commands)

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)


def build_emergency_recovery_index(
    *,
    workdir: Path | str,
    app_url: str = "http://127.0.0.1:8000",
    provider_host: str = "192.168.20.243",
    artifacts_dir: Path | str = "artifacts",
    generated_at: datetime | None = None,
) -> EmergencyRecoveryIndex:
    selected_workdir = str(workdir)
    selected_artifacts_dir = Path(artifacts_dir)
    commands = _build_commands(app_url=app_url, provider_host=provider_host)
    scenarios = _build_scenarios()
    _validate_scenarios(commands, scenarios)
    return EmergencyRecoveryIndex(
        generated_at=generated_at or datetime.now(UTC),
        workdir=selected_workdir,
        app_url=app_url.rstrip("/"),
        provider_host=provider_host,
        commands=tuple(_with_artifact_dir(command, selected_artifacts_dir) for command in commands),
        scenarios=scenarios,
    )


def emergency_recovery_index_payload(index: EmergencyRecoveryIndex) -> dict[str, object]:
    command_payloads = {command.code: _command_payload(command) for command in index.commands}
    return {
        "version": EMERGENCY_RECOVERY_INDEX_VERSION,
        "generated_at": index.generated_at.isoformat(),
        "generated_at_label": index.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": index.workdir,
        "app_url": index.app_url,
        "provider_host": index.provider_host,
        "command_count": index.command_count,
        "scenario_count": index.scenario_count,
        "commands": list(command_payloads.values()),
        "scenarios": [
            {
                "code": scenario.code,
                "title": scenario.title,
                "severity": scenario.severity,
                "first_check": scenario.first_check,
                "command_codes": list(scenario.command_codes),
                "commands": [
                    command_payloads[command_code] for command_code in scenario.command_codes
                ],
                "checklist": list(scenario.checklist),
                "stop_condition": scenario.stop_condition,
            }
            for scenario in index.scenarios
        ],
    }


def render_emergency_recovery_index_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# NeX_PCX Emergency Recovery Command Index",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Workdir: `{_text(payload.get('workdir'))}`",
        f"- App URL: `{_text(payload.get('app_url'))}`",
        f"- Provider Host: `{_text(payload.get('provider_host'))}`",
        "",
        "## Command Index",
        "",
        "| Code | Review | Description |",
        "| --- | --- | --- |",
    ]
    for command in payload.get("commands", []):
        command_payload = _dict(command)
        review = "yes" if command_payload.get("requires_review") else "no"
        lines.append(
            "| "
            f"{_md_cell(command_payload.get('code'))} | "
            f"{_md_cell(review)} | "
            f"{_md_cell(command_payload.get('description'))} |"
        )
        lines.extend(
            [
                "",
                "```bash",
                _text(command_payload.get("shell_command")),
                "```",
                "",
            ]
        )

    lines.extend(["## Recovery Scenarios", ""])
    for scenario in payload.get("scenarios", []):
        scenario_payload = _dict(scenario)
        lines.extend(
            [
                f"### {_text(scenario_payload.get('title'))}",
                "",
                f"- Severity: `{_text(scenario_payload.get('severity'))}`",
                f"- First Check: {_text(scenario_payload.get('first_check'))}",
                f"- Stop Condition: {_text(scenario_payload.get('stop_condition'))}",
                "",
                "Checklist:",
                "",
            ]
        )
        for item in scenario_payload.get("checklist", []):
            lines.append(f"- {_text(item)}")
        lines.extend(["", "Commands:", ""])
        for command in scenario_payload.get("commands", []):
            command_payload = _dict(command)
            lines.extend(
                [
                    f"- `{_text(command_payload.get('code'))}`",
                    "",
                    "```bash",
                    _text(command_payload.get("shell_command")),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def _build_commands(*, app_url: str, provider_host: str) -> tuple[RecoveryCommand, ...]:
    normalized_app_url = app_url.rstrip("/")
    return (
        RecoveryCommand(
            code="app_healthz",
            title="Application health check",
            description="Confirm the main FastAPI process responds.",
            command=("curl", "-fsS", f"{normalized_app_url}/healthz"),
        ),
        RecoveryCommand(
            code="startup_validation",
            title="Startup validation",
            description="Run the startup validation gate with app health.",
            command=(
                "./.venv/bin/python",
                "scripts/validate_operations_startup.py",
                "--app-url",
                normalized_app_url,
                "--pretty",
            ),
        ),
        RecoveryCommand(
            code="shutdown_drain",
            title="Shutdown drain check",
            description="Inspect running, stale, retryable, and exhausted queues.",
            command=(
                "./.venv/bin/python",
                "scripts/check_shutdown_drain.py",
                "--json-output",
                "{artifacts_dir}/shutdown_drain_check.json",
                "--markdown-output",
                "{artifacts_dir}/shutdown_drain_check.md",
                "--pretty",
            ),
        ),
        RecoveryCommand(
            code="runtime_audit",
            title="Runtime configuration audit",
            description="Review environment and filesystem configuration drift.",
            command=(
                "./.venv/bin/python",
                "scripts/audit_runtime_config.py",
                "--json-output",
                "{artifacts_dir}/runtime_config_audit.json",
                "--markdown-output",
                "{artifacts_dir}/runtime_config_audit.md",
                "--pretty",
            ),
        ),
        RecoveryCommand(
            code="provider_health_kure",
            title="KURE provider health",
            description="Confirm KURE remote provider health.",
            command=("curl", "-fsS", f"http://{provider_host}:9101/healthz"),
        ),
        RecoveryCommand(
            code="provider_health_bge",
            title="BGE provider health",
            description="Confirm BGE remote provider health.",
            command=("curl", "-fsS", f"http://{provider_host}:9102/healthz"),
        ),
        RecoveryCommand(
            code="provider_health_qwen",
            title="Qwen provider health",
            description="Confirm Qwen remote provider health.",
            command=("curl", "-fsS", f"http://{provider_host}:9103/healthz"),
        ),
        RecoveryCommand(
            code="provider_preflight",
            title="Provider route preflight",
            description="Persist route health and contract snapshots.",
            command=("./.venv/bin/python", "scripts/preflight_provider_routes.py"),
        ),
        RecoveryCommand(
            code="scheduled_preflight",
            title="Scheduled provider preflight",
            description="Run due provider preflight schedules.",
            command=(
                "./.venv/bin/python",
                "scripts/run_scheduled_provider_preflight.py",
                "--limit",
                "20",
            ),
        ),
        RecoveryCommand(
            code="pipeline_worker_once",
            title="Pipeline worker single pass",
            description="Process one queued pipeline job for manual recovery.",
            command=(
                "./.venv/bin/python",
                "scripts/process_pipeline_job.py",
                "--chunk-policy-names",
                "heading_512_64",
                "heading_1000_200",
                "heading_1500_200",
            ),
        ),
        RecoveryCommand(
            code="embedding_worker_batch",
            title="Embedding worker batch",
            description="Process a small route-aware embedding batch.",
            command=(
                "./.venv/bin/python",
                "scripts/process_embedding_job.py",
                "--provider-source",
                "route",
                "--require-route-readiness",
                "--limit",
                "20",
            ),
        ),
        RecoveryCommand(
            code="release_stale_embedding_lease",
            title="Release stale embedding lease",
            description="Release one reclaimable stale embedding job lease.",
            command=(
                "curl",
                "-fsS",
                "-X",
                "POST",
                f"{normalized_app_url}/api/admin/embedding-jobs/{{job_id}}/release-stale-lease",
            ),
            requires_review=True,
        ),
        RecoveryCommand(
            code="retry_failed_embedding_jobs",
            title="Retry failed embedding jobs",
            description="Retry failed embedding jobs after provider readiness is green.",
            command=(
                "curl",
                "-fsS",
                "-X",
                "POST",
                f"{normalized_app_url}/api/admin/embedding-jobs/retry-failed",
            ),
            requires_review=True,
        ),
        RecoveryCommand(
            code="retry_pipeline_job",
            title="Retry pipeline job",
            description="Retry a single retryable pipeline job.",
            command=(
                "curl",
                "-fsS",
                "-X",
                "POST",
                f"{normalized_app_url}/api/pipeline/jobs/{{job_id}}/retry",
            ),
            requires_review=True,
        ),
        RecoveryCommand(
            code="go_live_smoke",
            title="Go-live HTTP smoke",
            description="Run read-only go-live endpoint smoke checks.",
            command=(
                "./.venv/bin/python",
                "scripts/run_go_live_smoke.py",
                "--app-url",
                normalized_app_url,
                "--json-output",
                "{artifacts_dir}/go_live_smoke.json",
                "--markdown-output",
                "{artifacts_dir}/go_live_smoke.md",
                "--pretty",
            ),
        ),
        RecoveryCommand(
            code="retention_verification",
            title="Retention verification",
            description="Review retention settings and cleanup dry-run counts.",
            command=(
                "./.venv/bin/python",
                "scripts/verify_operational_retention.py",
                "--json-output",
                "{artifacts_dir}/operational_retention_verification.json",
                "--markdown-output",
                "{artifacts_dir}/operational_retention_verification.md",
                "--pretty",
            ),
        ),
        RecoveryCommand(
            code="search_runtime_failure_retry",
            title="Retry search runtime failures",
            description="Retry recorded search runtime failures after route recovery.",
            command=(
                "curl",
                "-fsS",
                "-X",
                "POST",
                f"{normalized_app_url}/api/search/logs/runtime-failures/retry",
            ),
            requires_review=True,
        ),
        RecoveryCommand(
            code="migration_upgrade",
            title="Apply migrations",
            description="Apply Alembic migrations to head after database verification.",
            command=("bash", "scripts/migrate.sh", "upgrade", "head"),
            requires_review=True,
        ),
    )


def _build_scenarios() -> tuple[RecoveryScenario, ...]:
    return (
        RecoveryScenario(
            code="app_unhealthy",
            title="Application Unhealthy",
            severity="critical",
            first_check="/healthz fails or the operations UI is unavailable.",
            command_codes=(
                "app_healthz",
                "startup_validation",
                "runtime_audit",
                "go_live_smoke",
            ),
            checklist=(
                "Confirm the process manager has a running app process.",
                "Check database connectivity before restarting workers.",
                "Capture the startup validation JSON before applying fixes.",
            ),
            stop_condition=(
                "Stop when startup validation is blocked by database or migration state."
            ),
        ),
        RecoveryScenario(
            code="provider_route_blocked",
            title="Provider Route Blocked",
            severity="critical",
            first_check="/admin/embedding-provider-routes shows blocked readiness.",
            command_codes=(
                "provider_health_kure",
                "provider_health_bge",
                "provider_health_qwen",
                "provider_preflight",
                "scheduled_preflight",
            ),
            checklist=(
                "Confirm every expected DGX provider responds from the app host.",
                "Run provider preflight before retrying embedding jobs.",
                "Record failed route IDs and snapshot IDs in the incident note.",
            ),
            stop_condition="Stop before retrying jobs if provider preflight still fails.",
        ),
        RecoveryScenario(
            code="pipeline_queue_stalled",
            title="Pipeline Queue Stalled",
            severity="high",
            first_check="/admin/jobs shows stale or retryable pipeline jobs.",
            command_codes=("shutdown_drain", "pipeline_worker_once", "retry_pipeline_job"),
            checklist=(
                "Identify stale, failed, and retryable pipeline job IDs.",
                "Run one worker pass to verify the extractor/chunker path.",
                "Retry only one job first when the root cause is uncertain.",
            ),
            stop_condition="Stop if the same job fails twice with the same error code.",
        ),
        RecoveryScenario(
            code="embedding_queue_stalled",
            title="Embedding Queue Stalled",
            severity="high",
            first_check="/admin/embedding-jobs shows stale leases or retryable failures.",
            command_codes=(
                "shutdown_drain",
                "provider_preflight",
                "release_stale_embedding_lease",
                "retry_failed_embedding_jobs",
                "embedding_worker_batch",
            ),
            checklist=(
                "Release only reclaimable stale leases.",
                "Retry failed embedding jobs only after route readiness is green.",
                "Use a small worker batch before restarting continuous workers.",
            ),
            stop_condition="Stop if stale leases immediately reappear after a worker batch.",
        ),
        RecoveryScenario(
            code="search_runtime_failures",
            title="Search Runtime Failures",
            severity="medium",
            first_check="/search or search history shows profile runtime failures.",
            command_codes=(
                "provider_preflight",
                "go_live_smoke",
                "search_runtime_failure_retry",
            ),
            checklist=(
                "Confirm failed profiles match provider route alerts.",
                "Retry search failures after provider route recovery.",
                "Compare recovered results against golden questions when available.",
            ),
            stop_condition="Stop if repeated query embedding failures occur for the same profile.",
        ),
        RecoveryScenario(
            code="operational_storage_growth",
            title="Operational Storage Growth",
            severity="medium",
            first_check="Logs, snapshots, or artifacts are growing faster than expected.",
            command_codes=("retention_verification",),
            checklist=(
                "Review dry-run counts before any destructive cleanup.",
                "Keep the verification report with incident evidence.",
                "Prefer UI cleanup actions so deleted counts are visible to operators.",
            ),
            stop_condition="Stop if dry-run counts are unexpectedly high for recent data.",
        ),
        RecoveryScenario(
            code="schema_or_migration_mismatch",
            title="Schema Or Migration Mismatch",
            severity="critical",
            first_check="Startup validation reports Alembic revision mismatch.",
            command_codes=("startup_validation", "migration_upgrade", "startup_validation"),
            checklist=(
                "Confirm the target database is not a restore smoke database.",
                "Back up the database before applying migrations in production.",
                "Run startup validation again after migrations finish.",
            ),
            stop_condition="Stop if migration head cannot be resolved from the deployed source.",
        ),
    )


def _with_artifact_dir(command: RecoveryCommand, artifacts_dir: Path) -> RecoveryCommand:
    command_parts = tuple(
        part.replace("{artifacts_dir}", str(artifacts_dir)) for part in command.command
    )
    return RecoveryCommand(
        code=command.code,
        title=command.title,
        description=command.description,
        command=command_parts,
        destructive=command.destructive,
        requires_review=command.requires_review,
    )


def _validate_scenarios(
    commands: tuple[RecoveryCommand, ...],
    scenarios: tuple[RecoveryScenario, ...],
) -> None:
    known_command_codes = {command.code for command in commands}
    for scenario in scenarios:
        missing_codes = [
            command_code
            for command_code in scenario.command_codes
            if command_code not in known_command_codes
        ]
        if missing_codes:
            msg = f"Scenario {scenario.code} references unknown commands: {missing_codes}"
            raise ValueError(msg)


def _command_payload(command: RecoveryCommand) -> dict[str, object]:
    return {
        "code": command.code,
        "title": command.title,
        "description": command.description,
        "command": list(command.command),
        "shell_command": command.shell_command,
        "destructive": command.destructive,
        "requires_review": command.requires_review,
    }


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("|", "\\|").replace("\n", " ")


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
