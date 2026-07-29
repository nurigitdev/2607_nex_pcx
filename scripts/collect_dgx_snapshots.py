"""Collect and persist DGX vLLM/provider resource snapshots."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.dgx_snapshot_collection import (  # noqa: E402
    COLLECTION_PLAN_BLOCKED,
    COLLECTION_STATUS_ATTENTION,
    COLLECTION_STATUS_BLOCKED,
    COLLECTION_STATUS_COMPLETED,
    COLLECTION_STATUS_FAILED,
    COLLECTION_STATUS_PARTIAL,
    COLLECTION_STATUS_PLANNED,
    DEFAULT_DATABASE_URL_ENV,
    DEFAULT_JSON_OUTPUT,
    DEFAULT_MARKDOWN_OUTPUT,
    DgxSnapshotCollectionCommandResult,
    DgxSnapshotCollectionCycle,
    DgxSnapshotCollectionOptions,
    build_dgx_snapshot_collection_cycle,
    build_dgx_snapshot_collection_evidence,
    build_dgx_snapshot_collection_plan,
    classify_dgx_snapshot_command_result,
    dgx_snapshot_collection_evidence_payload,
    dgx_snapshot_collection_options_from_environ,
    dgx_snapshot_collection_status_from_cycles,
    payload_to_json,
    render_dgx_snapshot_collection_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect vLLM runtime and DGX provider resource snapshots.",
    )
    try:
        runtime_defaults = dgx_snapshot_collection_options_from_environ(
            os.environ,
            defaults=DgxSnapshotCollectionOptions(workdir=PROJECT_ROOT),
        )
    except ValueError as exc:
        parser.error(str(exc))

    parser.add_argument("--database-url", default=os.getenv(DEFAULT_DATABASE_URL_ENV))
    parser.add_argument("--database-url-env", default=runtime_defaults.database_url_env)
    parser.add_argument("--workdir", default=str(runtime_defaults.workdir))
    parser.add_argument("--python-bin", default=runtime_defaults.python_bin)
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help="Component to collect: all, vllm, or provider-resource. Repeatable.",
    )
    parser.add_argument("--host", default=runtime_defaults.host)
    parser.add_argument("--ssh-user", default=runtime_defaults.provider_ssh_user)
    parser.add_argument("--remote-workdir", default=runtime_defaults.provider_remote_workdir)
    parser.add_argument(
        "--remote-python-bin",
        default=runtime_defaults.provider_remote_python_bin,
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider resource selector forwarded to probe_provider_resources.py.",
    )
    parser.add_argument(
        "--provider-local-only",
        action="store_true",
        default=runtime_defaults.provider_local_only,
        help="Run provider resource probe on the current host instead of SSH delegation.",
    )
    parser.add_argument("--vllm-base-url", default=runtime_defaults.vllm_base_url)
    parser.add_argument("--vllm-provider-name", default=runtime_defaults.vllm_provider_name)
    parser.add_argument("--vllm-model-id", default=runtime_defaults.vllm_model_id)
    parser.add_argument(
        "--vllm-timeout-seconds",
        type=float,
        default=runtime_defaults.vllm_timeout_seconds,
    )
    parser.add_argument(
        "--provider-timeout-seconds",
        type=int,
        default=runtime_defaults.provider_timeout_seconds,
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=runtime_defaults.command_timeout_seconds,
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=runtime_defaults.interval_seconds,
    )
    parser.add_argument("--max-cycles", type=int, default=runtime_defaults.max_cycles)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-database-url",
        action="store_true",
        help="Allow plan generation without a database URL. Live collection still requires one.",
    )
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if args.allow_missing_database_url and not args.dry_run:
        parser.error("--allow-missing-database-url can only be used with --dry-run")
    effective_environ = {**os.environ}
    if args.database_url:
        effective_environ[args.database_url_env] = args.database_url

    options = DgxSnapshotCollectionOptions(
        workdir=args.workdir,
        python_bin=args.python_bin,
        database_url_env=args.database_url_env,
        require_database_url=not args.allow_missing_database_url,
        components=tuple(args.component) or ("all",),
        host=args.host,
        provider_ssh_user=args.ssh_user,
        provider_remote_workdir=args.remote_workdir,
        provider_remote_python_bin=args.remote_python_bin,
        provider_selectors=tuple(args.provider) or runtime_defaults.provider_selectors,
        provider_local_only=args.provider_local_only,
        vllm_base_url=args.vllm_base_url,
        vllm_provider_name=args.vllm_provider_name,
        vllm_model_id=args.vllm_model_id,
        vllm_timeout_seconds=args.vllm_timeout_seconds,
        provider_timeout_seconds=args.provider_timeout_seconds,
        command_timeout_seconds=args.command_timeout_seconds,
        interval_seconds=args.interval_seconds,
        max_cycles=args.max_cycles,
    )
    try:
        plan = build_dgx_snapshot_collection_plan(options, environ=effective_environ)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run or plan.status == COLLECTION_PLAN_BLOCKED:
        status = (
            COLLECTION_STATUS_BLOCKED
            if plan.status == COLLECTION_PLAN_BLOCKED
            else COLLECTION_STATUS_PLANNED
        )
        evidence = build_dgx_snapshot_collection_evidence(
            plan,
            status=status,
            dry_run=args.dry_run,
            message=_planned_message(plan.status, args.dry_run),
            metadata={"script": "collect_dgx_snapshots.py"},
        )
        payload = dgx_snapshot_collection_evidence_payload(evidence)
        _write_outputs(
            payload,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            pretty=args.pretty,
        )
        print(payload_to_json(payload, pretty=args.pretty))
        return 1 if status == COLLECTION_STATUS_BLOCKED else 0

    database_url = effective_environ.get(plan.database_url_env, "").strip()
    if not database_url:
        parser.error(f"--database-url or {plan.database_url_env} is required")

    cycles = []
    for index in range(1, plan.max_cycles + 1):
        cycles.append(_run_collection_cycle(plan, index=index, database_url=database_url))
        if index < plan.max_cycles:
            time.sleep(plan.interval_seconds)

    status = dgx_snapshot_collection_status_from_cycles(cycles)
    evidence = build_dgx_snapshot_collection_evidence(
        plan,
        status=status,
        dry_run=False,
        cycles=cycles,
        message=_completion_message(status),
        metadata={"script": "collect_dgx_snapshots.py"},
    )
    payload = dgx_snapshot_collection_evidence_payload(evidence)
    _write_outputs(
        payload,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        pretty=args.pretty,
    )
    print(payload_to_json(payload, pretty=args.pretty))
    return 1 if status in {COLLECTION_STATUS_FAILED, COLLECTION_STATUS_PARTIAL} else 0


def _run_collection_cycle(
    plan,
    *,
    index: int,
    database_url: str,
) -> DgxSnapshotCollectionCycle:
    started_at = datetime.now(UTC)
    results = [
        _run_collection_command(
            command_plan,
            workdir=Path(plan.workdir),
            database_url=database_url,
            database_url_env=plan.database_url_env,
            timeout_seconds=plan.command_timeout_seconds,
        )
        for command_plan in plan.commands
    ]
    return build_dgx_snapshot_collection_cycle(
        index=index,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        results=results,
    )


def _run_collection_command(
    command_plan,
    *,
    workdir: Path,
    database_url: str,
    database_url_env: str,
    timeout_seconds: float,
) -> DgxSnapshotCollectionCommandResult:
    started_at = perf_counter()
    env = {**os.environ, database_url_env: database_url}
    try:
        completed = subprocess.run(
            command_plan.command,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        payload, error_message = _parse_command_payload(stdout)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _string_or_empty(exc.stdout)
        stderr = _string_or_empty(exc.stderr)
        payload = None
        error_message = f"Command timed out after {timeout_seconds:g}s."
        exit_code = 124
    except OSError as exc:
        stdout = ""
        stderr = str(exc)
        payload = None
        error_message = str(exc)
        exit_code = 127
    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    redacted_stdout = _redact_sensitive_text(stdout, database_url)
    redacted_stderr = _redact_sensitive_text(stderr, database_url)
    redacted_error = _redact_sensitive_text(error_message or "", database_url) or None
    status = classify_dgx_snapshot_command_result(
        code=command_plan.code,
        exit_code=exit_code,
        payload=payload,
        error_message=redacted_error,
    )
    return DgxSnapshotCollectionCommandResult(
        code=command_plan.code,
        component=command_plan.component,
        command=command_plan.command,
        exit_code=exit_code,
        elapsed_ms=elapsed_ms,
        status=status,
        payload=payload,
        stdout_tail=_tail(redacted_stdout),
        stderr_tail=_tail(redacted_stderr),
        error_message=redacted_error,
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


def _planned_message(plan_status: str, dry_run: bool) -> str:
    if plan_status == COLLECTION_PLAN_BLOCKED:
        return "DGX snapshot collection plan is blocked; inspect failed checks."
    if dry_run:
        return "Dry-run DGX snapshot collection plan was generated; no snapshots were collected."
    return "DGX snapshot collection plan was generated."


def _completion_message(status: str) -> str:
    if status == COLLECTION_STATUS_COMPLETED:
        return "DGX snapshot collection completed successfully."
    if status == COLLECTION_STATUS_ATTENTION:
        return "DGX snapshots were collected, but observed provider status needs attention."
    if status == COLLECTION_STATUS_PARTIAL:
        return "DGX snapshot collection completed partially; inspect failed commands."
    if status == COLLECTION_STATUS_FAILED:
        return "DGX snapshot collection failed before persisting expected snapshots."
    return "DGX snapshot collection finished."


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
        _write_text(Path(markdown_output), render_dgx_snapshot_collection_markdown(payload))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tail(text: str, *, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _redact_sensitive_text(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "<redacted-database-url>")


def _string_or_empty(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
