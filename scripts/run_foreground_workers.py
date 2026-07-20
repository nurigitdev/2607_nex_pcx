"""Run bounded foreground workers with provider resource guard evidence."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.foreground_worker_runner import (  # noqa: E402
    DEFAULT_EMBEDDING_LIMIT_PER_PROFILE,
    DEFAULT_HEALTH_TIMEOUT_SECONDS,
    DEFAULT_MAX_HEALTH_ELAPSED_MS,
    DEFAULT_PIPELINE_LIMIT,
    DEFAULT_WORKER_NAME_PREFIX,
    WORKER_RUN_STATUS_COMPLETED,
    WORKER_RUN_STATUS_FAILED,
    WORKER_RUN_STATUS_GUARDED,
    WORKER_RUN_STATUS_PARTIAL,
    WORKER_RUN_STATUS_PLANNED,
    WorkerCommandResult,
    build_foreground_worker_runner_evidence,
    build_foreground_worker_runner_plan,
    foreground_worker_runner_evidence_payload,
    merge_profile_token_limits,
    payload_to_json,
    render_foreground_worker_runner_markdown,
)
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded foreground pipeline/embedding workers with resource guards.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--python-bin", default="./.venv/bin/python")
    parser.add_argument("--worker-name-prefix", default=DEFAULT_WORKER_NAME_PREFIX)
    parser.add_argument("--pipeline-limit", type=int, default=DEFAULT_PIPELINE_LIMIT)
    parser.add_argument(
        "--embedding-limit-per-profile",
        type=int,
        default=DEFAULT_EMBEDDING_LIMIT_PER_PROFILE,
    )
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument(
        "--guard-health-timeout-seconds",
        type=float,
        default=DEFAULT_HEALTH_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-provider-health-elapsed-ms",
        type=int,
        default=DEFAULT_MAX_HEALTH_ELAPSED_MS,
    )
    parser.add_argument(
        "--profile-token-limit",
        action="append",
        default=[],
        help=(
            "Profile token guard in PROFILE=LIMIT format. LIMIT <= 0 removes a default "
            "limit for that profile."
        ),
    )
    parser.add_argument(
        "--no-default-qwen-token-guard",
        action="store_true",
        help="Disable conservative default token guards for Qwen profiles.",
    )
    parser.add_argument("--exclude-profile", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-command-failure", action="store_true")
    parser.add_argument("--json-output", default="artifacts/foreground_worker_runner.json")
    parser.add_argument("--markdown-output", default="artifacts/foreground_worker_runner.md")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")
    try:
        profile_token_limits = merge_profile_token_limits(
            args.profile_token_limit,
            include_defaults=not args.no_default_qwen_token_guard,
        )
    except ValueError as exc:
        parser.error(str(exc))

    pipeline_results: list[WorkerCommandResult] = []
    embedding_results: list[WorkerCommandResult] = []
    try:
        if not args.dry_run:
            pipeline_results = _run_pipeline_workers(
                database_url=database_url,
                workdir=Path(args.workdir),
                python_bin=args.python_bin,
                worker_name_prefix=args.worker_name_prefix,
                pipeline_limit=args.pipeline_limit,
                lease_seconds=args.lease_seconds,
                stop_on_failure=not args.continue_on_command_failure,
            )
        plan = build_foreground_worker_runner_plan(
            database_url,
            workdir=args.workdir,
            pipeline_limit=args.pipeline_limit,
            embedding_limit_per_profile=args.embedding_limit_per_profile,
            lease_seconds=args.lease_seconds,
            worker_name_prefix=args.worker_name_prefix,
            health_timeout_seconds=args.guard_health_timeout_seconds,
            max_health_elapsed_ms=args.max_provider_health_elapsed_ms,
            profile_token_limits=profile_token_limits,
            excluded_profiles=args.exclude_profile,
        )
        if not args.dry_run:
            embedding_results = _run_embedding_workers(
                plan=plan,
                database_url=database_url,
                workdir=Path(args.workdir),
                python_bin=args.python_bin,
                stop_on_failure=not args.continue_on_command_failure,
            )
    except ValueError as exc:
        parser.error(str(exc))

    status = _evidence_status(
        dry_run=args.dry_run,
        plan=plan,
        pipeline_results=pipeline_results,
        embedding_results=embedding_results,
    )
    evidence = build_foreground_worker_runner_evidence(
        plan,
        status=status,
        dry_run=args.dry_run,
        pipeline_results=pipeline_results,
        embedding_results=embedding_results,
        message=_evidence_message(status, args.dry_run),
        metadata={
            "script": "run_foreground_workers.py",
            "pipeline_result_count": len(pipeline_results),
            "embedding_result_count": len(embedding_results),
        },
    )
    payload = foreground_worker_runner_evidence_payload(evidence)
    _write_outputs(
        payload,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        pretty=args.pretty,
    )
    print(payload_to_json(payload, pretty=args.pretty))
    return 1 if status in {WORKER_RUN_STATUS_FAILED, WORKER_RUN_STATUS_PARTIAL} else 0


def _run_pipeline_workers(
    *,
    database_url: str,
    workdir: Path,
    python_bin: str,
    worker_name_prefix: str,
    pipeline_limit: int,
    lease_seconds: int,
    stop_on_failure: bool,
) -> list[WorkerCommandResult]:
    results = []
    for index in range(max(pipeline_limit, 0)):
        command = (
            python_bin,
            "scripts/process_pipeline_job.py",
            "--worker-name",
            f"{worker_name_prefix}-pipeline-{index + 1}",
            "--lease-seconds",
            str(lease_seconds),
        )
        result = _run_json_command(
            code=f"pipeline_worker_{index + 1}",
            command=command,
            workdir=workdir,
            database_url=database_url,
        )
        results.append(result)
        if not result.succeeded and stop_on_failure:
            break
        payload = result.payload or {}
        if payload.get("processed") is False:
            break
    return results


def _run_embedding_workers(
    *,
    plan,
    database_url: str,
    workdir: Path,
    python_bin: str,
    stop_on_failure: bool,
) -> list[WorkerCommandResult]:
    results = []
    for profile_name in plan.allowed_profiles:
        command = (
            python_bin,
            "scripts/process_embedding_job.py",
            "--worker-name",
            f"{plan.worker_name_prefix}-embedding-{profile_name}",
            "--profile-name",
            profile_name,
            "--provider-source",
            "route",
            "--skip-route-readiness",
            "--limit",
            str(plan.embedding_limit_per_profile),
            "--lease-seconds",
            str(plan.lease_seconds),
        )
        result = _run_json_command(
            code=f"embedding_worker_{profile_name}",
            command=command,
            workdir=workdir,
            database_url=database_url,
        )
        results.append(result)
        if not result.succeeded and stop_on_failure:
            break
        payload = result.payload or {}
        if int(payload.get("failed_count") or 0) > 0 and stop_on_failure:
            break
        if payload.get("status") == "failed" and stop_on_failure:
            break
    return results


def _run_json_command(
    *,
    code: str,
    command: tuple[str, ...],
    workdir: Path,
    database_url: str | None = None,
) -> WorkerCommandResult:
    started_at = perf_counter()
    env = {**os.environ}
    if database_url:
        env["NEX_PCX_DATABASE_URL"] = database_url
    completed = subprocess.run(
        command,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    payload, error_message = _parse_command_payload(completed.stdout)
    stdout = _redact_sensitive_text(completed.stdout, database_url)
    stderr = _redact_sensitive_text(completed.stderr, database_url)
    return WorkerCommandResult(
        code=code,
        command=command,
        exit_code=completed.returncode,
        elapsed_ms=elapsed_ms,
        payload=payload,
        stdout=stdout,
        stderr=stderr,
        error_message=_redact_sensitive_text(error_message or "", database_url) or None,
    )


def _parse_command_payload(stdout: str) -> tuple[dict[str, object] | None, str | None]:
    stripped_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not stripped_lines:
        return None, "Command did not write a JSON payload."
    try:
        payload = json.loads(stripped_lines[-1])
    except json.JSONDecodeError as exc:
        return None, f"Command payload could not be parsed: {exc}"
    if not isinstance(payload, dict):
        return None, "Command payload was not a JSON object."
    return payload, None


def _evidence_status(
    *,
    dry_run: bool,
    plan,
    pipeline_results: list[WorkerCommandResult],
    embedding_results: list[WorkerCommandResult],
) -> str:
    if dry_run:
        return WORKER_RUN_STATUS_PLANNED
    all_results = [*pipeline_results, *embedding_results]
    if any(not result.succeeded for result in all_results):
        return (
            WORKER_RUN_STATUS_PARTIAL
            if any(result.succeeded for result in all_results)
            else WORKER_RUN_STATUS_FAILED
        )
    if any(_command_payload_failed(result.payload) for result in all_results):
        return WORKER_RUN_STATUS_PARTIAL
    if plan.skipped_profiles:
        return WORKER_RUN_STATUS_GUARDED
    return WORKER_RUN_STATUS_COMPLETED


def _command_payload_failed(payload: dict[str, object] | None) -> bool:
    if not payload:
        return False
    if payload.get("status") == "failed":
        return True
    return int(payload.get("failed_count") or 0) > 0


def _evidence_message(status: str, dry_run: bool) -> str:
    if dry_run:
        return "Dry-run foreground worker plan was generated; no jobs were claimed."
    if status == WORKER_RUN_STATUS_GUARDED:
        return "Foreground workers completed allowed profiles; guarded profiles were skipped."
    if status == WORKER_RUN_STATUS_COMPLETED:
        return "Foreground workers completed bounded work successfully."
    if status == WORKER_RUN_STATUS_PARTIAL:
        return "Foreground workers completed partially; inspect failed command or job payloads."
    if status == WORKER_RUN_STATUS_FAILED:
        return "Foreground worker runner failed before completing bounded work."
    return "Foreground worker runner completed."


def _write_outputs(
    payload: dict[str, object],
    *,
    json_output: str | None,
    markdown_output: str | None,
    pretty: bool,
) -> None:
    if json_output:
        _write_text(Path(json_output), payload_to_json(payload, pretty=pretty) + "\n")
    if markdown_output:
        _write_text(Path(markdown_output), render_foreground_worker_runner_markdown(payload))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _redact_sensitive_text(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "<redacted-database-url>")


if __name__ == "__main__":
    raise SystemExit(main())
