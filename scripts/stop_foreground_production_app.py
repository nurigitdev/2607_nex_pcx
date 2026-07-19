"""Stop a foreground NeX-PCX app process and write shutdown evidence."""

import argparse
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.foreground_production_launch import (  # noqa: E402
    DEFAULT_PID_FILE,
    resolve_launch_path,
)
from app.core.foreground_production_shutdown import (  # noqa: E402
    DEFAULT_SHUTDOWN_LOG_FILE,
    SHUTDOWN_PLAN_BLOCKED,
    SHUTDOWN_STATUS_BLOCKED,
    SHUTDOWN_STATUS_FAILED,
    SHUTDOWN_STATUS_NO_PROCESS,
    SHUTDOWN_STATUS_PLANNED,
    SHUTDOWN_STATUS_STOPPED,
    ForegroundProductionShutdownOptions,
    build_foreground_production_shutdown_evidence,
    build_foreground_production_shutdown_plan,
    foreground_production_shutdown_evidence_payload,
    inspect_process,
    is_tcp_port_reachable,
    payload_to_json,
    render_foreground_production_shutdown_markdown,
)
from app.core.service_startup_templates import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stop a NeX-PCX foreground process with PID/log evidence.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--pid-file", default=DEFAULT_PID_FILE)
    parser.add_argument("--log-file", default=DEFAULT_SHUTDOWN_LOG_FILE)
    parser.add_argument("--host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--json-output", default="artifacts/foreground_production_shutdown.json")
    parser.add_argument("--markdown-output", default="artifacts/foreground_production_shutdown.md")
    parser.add_argument("--wait-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-port-check", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    plan = build_foreground_production_shutdown_plan(
        ForegroundProductionShutdownOptions(
            workdir=args.workdir,
            pid_file=args.pid_file,
            log_file=args.log_file,
            host=args.host,
            port=args.port,
            require_pid_file=not args.dry_run,
            check_port_reachable=not args.skip_port_check,
        )
    )
    if args.dry_run or plan.status == SHUTDOWN_PLAN_BLOCKED:
        status = (
            SHUTDOWN_STATUS_BLOCKED
            if plan.status == SHUTDOWN_PLAN_BLOCKED
            else SHUTDOWN_STATUS_PLANNED
        )
        evidence = build_foreground_production_shutdown_evidence(
            plan,
            status=status,
            dry_run=args.dry_run,
            process_id=plan.process_id,
            signal_name=plan.signal_name,
            message=_dry_run_message(plan.status, args.dry_run),
        )
        _write_outputs(
            evidence=evidence,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            pretty=args.pretty,
        )
        return 1 if plan.status == SHUTDOWN_PLAN_BLOCKED else 0
    return _stop_foreground_process(
        plan=plan,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        wait_timeout_seconds=args.wait_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        pretty=args.pretty,
    )


def _stop_foreground_process(
    *,
    plan,
    json_output: str | None,
    markdown_output: str | None,
    wait_timeout_seconds: float,
    poll_interval_seconds: float,
    pretty: bool,
) -> int:
    if plan.process_id is None or not (
        plan.process_observation and plan.process_observation.exists
    ):
        evidence = build_foreground_production_shutdown_evidence(
            plan,
            status=SHUTDOWN_STATUS_NO_PROCESS,
            dry_run=False,
            process_id=plan.process_id,
            signal_name=plan.signal_name,
            message="No running foreground process was found for the PID file.",
        )
        _write_outputs(
            evidence=evidence,
            json_output=json_output,
            markdown_output=markdown_output,
            pretty=pretty,
        )
        return 0
    workdir = Path(plan.workdir)
    log_file = resolve_launch_path(workdir, plan.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    with log_file.open("a", encoding="utf-8", buffering=1) as log_handle:
        _write_log_line(log_handle, f"Sending {plan.signal_name} to PID {plan.process_id}.")
        try:
            os.kill(plan.process_id, signal.SIGTERM)
        except ProcessLookupError:
            message = f"Foreground process {plan.process_id} already exited before SIGTERM."
            _write_log_line(log_handle, message)
            evidence = build_foreground_production_shutdown_evidence(
                plan,
                status=SHUTDOWN_STATUS_NO_PROCESS,
                dry_run=False,
                process_id=plan.process_id,
                signal_name=plan.signal_name,
                started_at=started_at,
                stopped_at=datetime.now(UTC),
                port_released_after_stop=not is_tcp_port_reachable(
                    host=plan.host,
                    port=plan.port,
                ),
                message=message,
            )
            _write_outputs(
                evidence=evidence,
                json_output=json_output,
                markdown_output=markdown_output,
                pretty=pretty,
            )
            return 0
        except PermissionError as exc:
            message = f"Permission denied while sending {plan.signal_name}: {exc}"
            _write_log_line(log_handle, message)
            evidence = build_foreground_production_shutdown_evidence(
                plan,
                status=SHUTDOWN_STATUS_FAILED,
                dry_run=False,
                process_id=plan.process_id,
                signal_name=plan.signal_name,
                started_at=started_at,
                stopped_at=datetime.now(UTC),
                message=message,
                metadata={"error": str(exc)},
            )
            _write_outputs(
                evidence=evidence,
                json_output=json_output,
                markdown_output=markdown_output,
                pretty=pretty,
            )
            return 1
        stopped = _wait_for_process_exit(
            process_id=plan.process_id,
            timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        stopped_at = datetime.now(UTC)
        port_released = not is_tcp_port_reachable(host=plan.host, port=plan.port)
        if stopped:
            status = SHUTDOWN_STATUS_STOPPED
            message = f"Foreground process {plan.process_id} stopped after {plan.signal_name}."
            exit_code = 0
        else:
            status = SHUTDOWN_STATUS_FAILED
            message = (
                f"Foreground process {plan.process_id} did not stop within "
                f"{wait_timeout_seconds:.1f} seconds."
            )
            exit_code = 1
        _write_log_line(log_handle, message)
    evidence = build_foreground_production_shutdown_evidence(
        plan,
        status=status,
        dry_run=False,
        process_id=plan.process_id,
        signal_name=plan.signal_name,
        started_at=started_at,
        stopped_at=stopped_at,
        port_released_after_stop=port_released,
        message=message,
        metadata={
            "wait_timeout_seconds": wait_timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
        },
    )
    _write_outputs(
        evidence=evidence,
        json_output=json_output,
        markdown_output=markdown_output,
        pretty=pretty,
    )
    return exit_code


def _wait_for_process_exit(
    *,
    process_id: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> bool:
    deadline = time.monotonic() + max(timeout_seconds, 0)
    interval = max(poll_interval_seconds, 0.1)
    while time.monotonic() <= deadline:
        observation = inspect_process(process_id)
        if observation is None or not observation.exists:
            return True
        time.sleep(interval)
    return False


def _write_outputs(
    *,
    evidence,
    json_output: str | None,
    markdown_output: str | None,
    pretty: bool,
) -> None:
    payload = foreground_production_shutdown_evidence_payload(evidence)
    json_text = payload_to_json(payload, pretty=pretty)
    markdown_text = render_foreground_production_shutdown_markdown(payload)
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
    if plan_status == SHUTDOWN_PLAN_BLOCKED:
        return "Shutdown plan is blocked; no signal was sent."
    if dry_run:
        return "Dry-run shutdown plan was generated; no signal was sent."
    return "No signal was sent."


if __name__ == "__main__":
    raise SystemExit(main())
