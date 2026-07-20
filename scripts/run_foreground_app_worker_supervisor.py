"""Run the foreground web app with automatic worker-cycle supervision."""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.foreground_app_worker_supervisor import (  # noqa: E402
    SUPERVISOR_PLAN_BLOCKED,
    SUPERVISOR_STATUS_BLOCKED,
    SUPERVISOR_STATUS_EXITED,
    SUPERVISOR_STATUS_FAILED,
    SUPERVISOR_STATUS_PLANNED,
    SUPERVISOR_STATUS_RUNNING,
    ForegroundAppWorkerSupervisorOptions,
    ForegroundWorkerCycleObservation,
    build_foreground_app_worker_supervisor_evidence,
    build_foreground_app_worker_supervisor_plan,
    foreground_app_worker_supervisor_evidence_payload,
    foreground_app_worker_supervisor_options_from_environ,
    payload_to_json,
    render_foreground_app_worker_supervisor_markdown,
)
from app.core.foreground_production_launch import (  # noqa: E402
    resolve_launch_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NeX-PCX foreground web app with automatic worker cycles.",
    )
    try:
        runtime_defaults = foreground_app_worker_supervisor_options_from_environ(
            os.environ,
            defaults=ForegroundAppWorkerSupervisorOptions(workdir=PROJECT_ROOT),
        )
    except ValueError as exc:
        parser.error(str(exc))

    parser.add_argument("--workdir", default=str(runtime_defaults.workdir))
    parser.add_argument("--python-bin", default=runtime_defaults.python_bin)
    parser.add_argument("--host", default=runtime_defaults.host)
    parser.add_argument("--port", type=int, default=runtime_defaults.port)
    parser.add_argument("--database-url-env", default=runtime_defaults.database_url_env)
    parser.add_argument("--supervisor-pid-file", default=runtime_defaults.supervisor_pid_file)
    parser.add_argument("--web-pid-file", default=runtime_defaults.web_pid_file)
    parser.add_argument("--log-file", default=runtime_defaults.log_file)
    parser.add_argument(
        "--worker-cycle-interval-seconds",
        type=float,
        default=runtime_defaults.worker_cycle_interval_seconds,
    )
    parser.add_argument("--pipeline-limit", type=int, default=runtime_defaults.pipeline_limit)
    parser.add_argument(
        "--embedding-limit-per-profile",
        type=int,
        default=runtime_defaults.embedding_limit_per_profile,
    )
    parser.add_argument(
        "--guard-health-timeout-seconds",
        type=float,
        default=runtime_defaults.guard_health_timeout_seconds,
    )
    parser.add_argument(
        "--max-provider-health-elapsed-ms",
        type=int,
        default=runtime_defaults.max_provider_health_elapsed_ms,
    )
    parser.add_argument("--worker-json-output", default=runtime_defaults.worker_json_output)
    parser.add_argument(
        "--worker-markdown-output",
        default=runtime_defaults.worker_markdown_output,
    )
    qwen_guard_group = parser.add_mutually_exclusive_group()
    qwen_guard_group.set_defaults(
        no_default_qwen_token_guard=runtime_defaults.no_default_qwen_token_guard
    )
    qwen_guard_group.add_argument(
        "--no-default-qwen-token-guard",
        dest="no_default_qwen_token_guard",
        action="store_true",
    )
    qwen_guard_group.add_argument(
        "--default-qwen-token-guard",
        dest="no_default_qwen_token_guard",
        action="store_false",
    )
    parser.add_argument(
        "--json-output",
        default="artifacts/foreground_app_worker_supervisor.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="artifacts/foreground_app_worker_supervisor.md",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-worker-cycles", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    port_check_group = parser.add_mutually_exclusive_group()
    port_check_group.set_defaults(check_port_available=runtime_defaults.check_port_available)
    port_check_group.add_argument(
        "--check-port",
        dest="check_port_available",
        action="store_true",
    )
    port_check_group.add_argument(
        "--skip-port-check",
        dest="check_port_available",
        action="store_false",
    )
    database_group = parser.add_mutually_exclusive_group()
    database_group.set_defaults(require_database_url=runtime_defaults.require_database_url)
    database_group.add_argument(
        "--require-database-url",
        dest="require_database_url",
        action="store_true",
    )
    database_group.add_argument(
        "--allow-missing-database-url",
        dest="require_database_url",
        action="store_false",
        help="Allow dry-run planning when NEX_PCX_DATABASE_URL is not configured.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    if not args.require_database_url and not args.dry_run:
        parser.error("missing database URL can only be allowed with --dry-run")

    plan = build_foreground_app_worker_supervisor_plan(
        ForegroundAppWorkerSupervisorOptions(
            workdir=args.workdir,
            python_bin=args.python_bin,
            host=args.host,
            port=args.port,
            database_url_env=args.database_url_env,
            require_database_url=args.require_database_url,
            supervisor_pid_file=args.supervisor_pid_file,
            web_pid_file=args.web_pid_file,
            log_file=args.log_file,
            worker_cycle_interval_seconds=args.worker_cycle_interval_seconds,
            pipeline_limit=args.pipeline_limit,
            embedding_limit_per_profile=args.embedding_limit_per_profile,
            guard_health_timeout_seconds=args.guard_health_timeout_seconds,
            max_provider_health_elapsed_ms=args.max_provider_health_elapsed_ms,
            worker_json_output=args.worker_json_output,
            worker_markdown_output=args.worker_markdown_output,
            no_default_qwen_token_guard=args.no_default_qwen_token_guard,
            check_port_available=args.check_port_available,
        ),
        environ=os.environ,
    )
    if args.dry_run or plan.status == SUPERVISOR_PLAN_BLOCKED:
        status = (
            SUPERVISOR_STATUS_BLOCKED
            if plan.status == SUPERVISOR_PLAN_BLOCKED
            else SUPERVISOR_STATUS_PLANNED
        )
        evidence = build_foreground_app_worker_supervisor_evidence(
            plan,
            status=status,
            dry_run=args.dry_run,
            message=_dry_run_message(plan.status, args.dry_run),
        )
        _write_outputs(
            evidence=evidence,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            pretty=args.pretty,
        )
        return 1 if plan.status == SUPERVISOR_PLAN_BLOCKED else 0

    return _run_supervisor(
        plan=plan,
        startup_timeout_seconds=args.startup_timeout_seconds,
        max_worker_cycles=args.max_worker_cycles,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        pretty=args.pretty,
    )


def _run_supervisor(
    *,
    plan,
    startup_timeout_seconds: float,
    max_worker_cycles: int | None,
    json_output: str | None,
    markdown_output: str | None,
    pretty: bool,
) -> int:
    workdir = Path(plan.workdir)
    supervisor_pid_file = resolve_launch_path(workdir, plan.supervisor_pid_file)
    web_pid_file = resolve_launch_path(workdir, plan.web_pid_file)
    log_file = resolve_launch_path(workdir, plan.log_file)
    for path in (supervisor_pid_file, web_pid_file, log_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    supervisor_pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    worker_cycles: list[ForegroundWorkerCycleObservation] = []
    stop_requested = False

    def request_stop(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started_at = datetime.now(UTC)
    web_process = None
    returncode = 0
    final_status = SUPERVISOR_STATUS_EXITED
    message = "Supervisor exited cleanly."
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        with log_file.open("a", encoding="utf-8", buffering=1) as log_handle:
            _write_log_line(
                log_handle, f"Starting supervised web: {plan.launch_plan.shell_command}"
            )
            web_process = subprocess.Popen(
                plan.launch_plan.command,
                cwd=workdir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            web_pid_file.write_text(f"{web_process.pid}\n", encoding="utf-8")
            startup_metadata = _wait_for_health(
                health_url=plan.launch_plan.health_url,
                process=web_process,
                timeout_seconds=startup_timeout_seconds,
            )
            running_evidence = build_foreground_app_worker_supervisor_evidence(
                plan,
                status=SUPERVISOR_STATUS_RUNNING,
                dry_run=False,
                supervisor_process_id=os.getpid(),
                web_process_id=web_process.pid,
                started_at=started_at,
                message="Foreground web app started; worker supervisor is polling the queue.",
                metadata=startup_metadata,
            )
            _write_outputs(
                evidence=running_evidence,
                json_output=json_output,
                markdown_output=markdown_output,
                pretty=pretty,
            )
            cycle_index = 0
            while not stop_requested:
                if web_process.poll() is not None:
                    returncode = web_process.returncode or 0
                    final_status = (
                        SUPERVISOR_STATUS_EXITED if returncode == 0 else SUPERVISOR_STATUS_FAILED
                    )
                    message = f"Supervised web process exited with return code {returncode}."
                    break
                if max_worker_cycles is not None and cycle_index >= max_worker_cycles:
                    message = f"Supervisor reached max worker cycles: {max_worker_cycles}."
                    break
                cycle_index += 1
                worker_cycles.append(
                    _run_worker_cycle(
                        index=cycle_index,
                        command=plan.worker_command,
                        workdir=workdir,
                        env=env,
                        log_handle=log_handle,
                        should_stop=lambda: stop_requested,
                    )
                )
                if worker_cycles[-1].exit_code != 0:
                    final_status = SUPERVISOR_STATUS_FAILED
                    returncode = worker_cycles[-1].exit_code
                    message = "Worker cycle failed; supervisor is stopping."
                    break
                _sleep_with_stop(
                    seconds=plan.worker_cycle_interval_seconds,
                    should_stop=lambda: stop_requested,
                )
            if stop_requested:
                message = "Supervisor received stop signal."
    except KeyboardInterrupt:
        message = "Supervisor was interrupted."
        returncode = 130
        final_status = SUPERVISOR_STATUS_EXITED
    finally:
        if web_process is not None and web_process.poll() is None:
            _terminate_process(web_process)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
    exited_at = datetime.now(UTC)
    evidence = build_foreground_app_worker_supervisor_evidence(
        plan,
        status=final_status,
        dry_run=False,
        supervisor_process_id=os.getpid(),
        web_process_id=web_process.pid if web_process else None,
        returncode=returncode,
        started_at=started_at,
        exited_at=exited_at,
        worker_cycles=tuple(worker_cycles),
        message=message,
    )
    _write_outputs(
        evidence=evidence,
        json_output=json_output,
        markdown_output=markdown_output,
        pretty=pretty,
    )
    return returncode


def _run_worker_cycle(
    *,
    index: int,
    command: tuple[str, ...],
    workdir: Path,
    env: dict[str, str],
    log_handle,
    should_stop,
) -> ForegroundWorkerCycleObservation:
    started_at = perf_counter()
    _write_log_line(log_handle, f"Starting worker cycle {index}: {_quote_command(command)}")
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output_lines: list[str] = []
    while process.poll() is None:
        if should_stop():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break
        line = process.stdout.readline() if process.stdout else ""
        if line:
            output_lines.append(line)
            log_handle.write(line)
        else:
            time.sleep(0.1)
    if process.stdout:
        for line in process.stdout:
            output_lines.append(line)
            log_handle.write(line)
    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    exit_code = process.returncode if process.returncode is not None else 1
    status = "succeeded" if exit_code == 0 else "failed"
    message = _worker_cycle_message(output_lines)
    _write_log_line(log_handle, f"Worker cycle {index} {status} with exit code {exit_code}.")
    return ForegroundWorkerCycleObservation(
        index=index,
        command=command,
        exit_code=exit_code,
        elapsed_ms=elapsed_ms,
        status=status,
        message=message,
    )


def _wait_for_health(*, health_url: str, process, timeout_seconds: float) -> dict[str, object]:
    if timeout_seconds <= 0:
        return {"startup_health_status": "skipped", "startup_health_url": health_url}
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return {
                "startup_health_status": "failed",
                "startup_health_url": health_url,
                "startup_health_error": "process exited before health check passed",
            }
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                return {
                    "startup_health_status": "passed",
                    "startup_health_url": health_url,
                    "startup_health_code": response.status,
                }
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    return {
        "startup_health_status": "warning",
        "startup_health_url": health_url,
        "startup_health_error": last_error or "startup health check timed out",
    }


def _sleep_with_stop(*, seconds: float, should_stop) -> None:
    deadline = time.monotonic() + max(seconds, 0)
    while time.monotonic() < deadline:
        if should_stop():
            return
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def _terminate_process(process) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _worker_cycle_message(output_lines: list[str]) -> str:
    for line in reversed(output_lines):
        stripped = line.strip()
        if stripped:
            return stripped[:300]
    return "Worker cycle produced no output."


def _write_outputs(
    *,
    evidence,
    json_output: str | None,
    markdown_output: str | None,
    pretty: bool,
) -> None:
    payload = foreground_app_worker_supervisor_evidence_payload(evidence)
    json_text = payload_to_json(payload, pretty=pretty)
    markdown_text = render_foreground_app_worker_supervisor_markdown(payload)
    if json_output:
        _write_text(Path(json_output), json_text + "\n")
    else:
        print(json_text)
    if markdown_output:
        _write_text(Path(markdown_output), markdown_text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_log_line(log_handle, message: str) -> None:
    log_handle.write(f"[{datetime.now(UTC).isoformat()}] {message}\n")


def _dry_run_message(plan_status: str, dry_run: bool) -> str:
    if plan_status == SUPERVISOR_PLAN_BLOCKED:
        return "Supervisor plan is blocked; web app and worker cycles were not started."
    if dry_run:
        return "Dry-run supervisor plan was generated; no process was started."
    return "Supervisor process was not started."


def _quote_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
