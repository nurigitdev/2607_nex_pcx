"""Run NeX-PCX in foreground mode and write PID/log launch evidence."""

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.foreground_production_launch import (  # noqa: E402
    DEFAULT_DATABASE_URL_ENV,
    DEFAULT_LOG_FILE,
    DEFAULT_PID_FILE,
    DEFAULT_PYTHON_BIN,
    LAUNCH_PLAN_BLOCKED,
    LAUNCH_STATUS_BLOCKED,
    LAUNCH_STATUS_EXITED,
    LAUNCH_STATUS_FAILED,
    LAUNCH_STATUS_PLANNED,
    LAUNCH_STATUS_RUNNING,
    ForegroundProductionLaunchOptions,
    build_foreground_production_launch_evidence,
    build_foreground_production_launch_plan,
    foreground_production_launch_evidence_payload,
    payload_to_json,
    render_foreground_production_launch_markdown,
    resolve_launch_path,
)
from app.core.service_startup_templates import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NeX-PCX FastAPI app in foreground mode with PID/log evidence.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--database-url-env", default=DEFAULT_DATABASE_URL_ENV)
    parser.add_argument("--pid-file", default=DEFAULT_PID_FILE)
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    parser.add_argument("--json-output", default="artifacts/foreground_production_launch.json")
    parser.add_argument("--markdown-output", default="artifacts/foreground_production_launch.md")
    parser.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-port-check", action="store_true")
    parser.add_argument(
        "--allow-missing-database-url",
        action="store_true",
        help="Allow dry-run planning when NEX_PCX_DATABASE_URL is not configured.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    if args.allow_missing_database_url and not args.dry_run:
        parser.error("--allow-missing-database-url can only be used with --dry-run")

    plan = build_foreground_production_launch_plan(
        ForegroundProductionLaunchOptions(
            workdir=args.workdir,
            python_bin=args.python_bin,
            host=args.host,
            port=args.port,
            database_url_env=args.database_url_env,
            require_database_url=not args.allow_missing_database_url,
            pid_file=args.pid_file,
            log_file=args.log_file,
            check_port_available=not args.skip_port_check,
        ),
        environ=os.environ,
    )
    if args.dry_run or plan.status == LAUNCH_PLAN_BLOCKED:
        status = (
            LAUNCH_STATUS_BLOCKED if plan.status == LAUNCH_PLAN_BLOCKED else LAUNCH_STATUS_PLANNED
        )
        evidence = build_foreground_production_launch_evidence(
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
        return 1 if plan.status == LAUNCH_PLAN_BLOCKED else 0
    return _run_foreground_process(
        plan=plan,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        startup_timeout_seconds=args.startup_timeout_seconds,
        pretty=args.pretty,
    )


def _run_foreground_process(
    *,
    plan,
    json_output: str | None,
    markdown_output: str | None,
    startup_timeout_seconds: float,
    pretty: bool,
) -> int:
    workdir = Path(plan.workdir)
    pid_file = resolve_launch_path(workdir, plan.pid_file)
    log_file = resolve_launch_path(workdir, plan.log_file)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    started_at = datetime.now(UTC)
    with log_file.open("a", encoding="utf-8", buffering=1) as log_handle:
        _write_log_line(log_handle, f"Starting NeX-PCX foreground app: {plan.shell_command}")
        process = subprocess.Popen(
            plan.command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        startup_metadata = _wait_for_health(
            health_url=plan.health_url,
            process=process,
            timeout_seconds=startup_timeout_seconds,
        )
        running_evidence = build_foreground_production_launch_evidence(
            plan,
            status=LAUNCH_STATUS_RUNNING,
            dry_run=False,
            process_id=process.pid,
            started_at=started_at,
            message="Foreground process started. Keep this terminal open.",
            metadata=startup_metadata,
        )
        _write_outputs(
            evidence=running_evidence,
            json_output=json_output,
            markdown_output=markdown_output,
            pretty=pretty,
        )
        try:
            _stream_process_output(process, log_handle)
            returncode = process.wait()
            final_status = LAUNCH_STATUS_EXITED if returncode == 0 else LAUNCH_STATUS_FAILED
            message = f"Foreground process exited with return code {returncode}."
            exit_code = returncode
        except KeyboardInterrupt:
            process.terminate()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
            final_status = LAUNCH_STATUS_EXITED
            message = "Foreground process was interrupted and terminated."
            exit_code = 130
        exited_at = datetime.now(UTC)
        final_evidence = build_foreground_production_launch_evidence(
            plan,
            status=final_status,
            dry_run=False,
            process_id=process.pid,
            returncode=returncode,
            started_at=started_at,
            exited_at=exited_at,
            message=message,
            metadata=startup_metadata,
        )
        _write_outputs(
            evidence=final_evidence,
            json_output=json_output,
            markdown_output=markdown_output,
            pretty=pretty,
        )
        _write_log_line(log_handle, message)
        return exit_code


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


def _stream_process_output(process, log_handle) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)


def _write_outputs(
    *,
    evidence,
    json_output: str | None,
    markdown_output: str | None,
    pretty: bool,
) -> None:
    payload = foreground_production_launch_evidence_payload(evidence)
    json_text = payload_to_json(payload, pretty=pretty)
    markdown_text = render_foreground_production_launch_markdown(payload)
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
    if plan_status == LAUNCH_PLAN_BLOCKED:
        return "Launch plan is blocked; foreground process was not started."
    if dry_run:
        return "Dry-run launch plan was generated; foreground process was not started."
    return "Foreground process was not started."


if __name__ == "__main__":
    raise SystemExit(main())
