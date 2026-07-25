"""Manage the DGX remote reranker provider as a background process."""

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rerankers import RemoteRerankerProviderClient  # noqa: E402
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_GPU_HOST,
    DEFAULT_GPU_USER,
    DEFAULT_GPU_WORKDIR,
)
from scripts.plan_remote_reranker_foreground_smoke import (  # noqa: E402
    build_reranker_foreground_smoke_plan,
)
from scripts.run_remote_reranker_foreground_smoke import (  # noqa: E402
    HealthObservation,
    _health_mismatches,
    _probe_health_once,
)
from scripts.run_remote_reranker_request_smoke import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS as DEFAULT_REQUEST_TIMEOUT_SECONDS,
    RemoteRerankerRequestSmokeReport,
    build_reranker_request_smoke_plan,
    run_reranker_request_smoke,
)
from scripts.run_reranker_provider import (  # noqa: E402
    DEFAULT_RERANKER_PROVIDER_NAME,
    DEFAULT_RERANKER_PROVIDER_PORT,
)

LifecycleAction = Literal["start", "status", "stop", "smoke"]
DEFAULT_HEALTH_TIMEOUT_SECONDS = 5.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_STOP_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class RemoteRerankerBackgroundLifecyclePlan:
    action: LifecycleAction
    provider_name: str
    backend: str
    reranker_profile_name: str
    provider_model_id: str
    device: str
    ssh_target: str
    workdir: str
    models_dir: str
    model_dir_name: str
    base_url: str
    health_url: str
    host: str
    port: int
    route_host: str
    pid_file: str
    log_file: str
    process_pattern: str
    remote_status_command: str
    remote_start_command: str
    remote_stop_command: str


@dataclass(frozen=True)
class RemoteCommandObservation:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    values: dict[str, str]


@dataclass(frozen=True)
class RemoteRerankerBackgroundLifecycleReport:
    plan: RemoteRerankerBackgroundLifecyclePlan
    action: LifecycleAction
    status: str
    pid: str | None
    command_observation: RemoteCommandObservation
    health_checked: bool
    health_ok: bool
    health_status_code: int | None
    health_payload: dict[str, Any] | None
    health_error: str | None
    health_mismatches: tuple[str, ...]
    request_smoke_checked: bool
    request_smoke_passed: bool | None
    request_smoke_summary: dict[str, Any] | None
    elapsed_ms: int
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error is not None or not self.command_observation.ok:
            return False
        if self.action == "stop":
            return self.status in {"stopped", "not_running"} and not self.health_ok
        expected_statuses = {"running", "already_running", "started"}
        if self.action == "smoke":
            expected_statuses = {"running", "already_running", "started"}
        request_ok = not self.request_smoke_checked or self.request_smoke_passed is True
        return (
            self.status in expected_statuses
            and self.health_ok
            and not self.health_mismatches
            and request_ok
        )


def build_background_lifecycle_plan(
    *,
    action: LifecycleAction,
    host: str = DEFAULT_GPU_HOST,
    ssh_user: str = DEFAULT_GPU_USER,
    workdir: str = DEFAULT_GPU_WORKDIR,
    route_host: str | None = None,
    provider_name: str = DEFAULT_RERANKER_PROVIDER_NAME,
    port: int = DEFAULT_RERANKER_PROVIDER_PORT,
    pid_file: str | None = None,
    log_file: str | None = None,
) -> RemoteRerankerBackgroundLifecyclePlan:
    foreground_plan = build_reranker_foreground_smoke_plan(
        host=host,
        ssh_user=ssh_user,
        workdir=workdir,
        route_host=route_host,
        provider_name=provider_name,
        port=port,
    )
    selected_pid_file = _validate_nonblank(
        pid_file or f"run/remote_reranker_provider_{foreground_plan.port}.pid",
        "pid_file",
    )
    selected_log_file = _validate_nonblank(
        log_file or f"logs/remote_reranker_provider_{foreground_plan.port}.log",
        "log_file",
    )
    process_pattern = (
        f"[u]vicorn app.reranker_provider_service:app --host "
        f"{foreground_plan.host} --port {foreground_plan.port}"
    )
    plan = RemoteRerankerBackgroundLifecyclePlan(
        action=action,
        provider_name=foreground_plan.provider_name,
        backend=foreground_plan.backend,
        reranker_profile_name=foreground_plan.reranker_profile_name,
        provider_model_id=foreground_plan.provider_model_id,
        device=foreground_plan.device,
        ssh_target=foreground_plan.ssh_target,
        workdir=foreground_plan.workdir,
        models_dir=foreground_plan.models_dir,
        model_dir_name=foreground_plan.model_dir_name,
        base_url=foreground_plan.base_url,
        health_url=foreground_plan.health_url,
        host=foreground_plan.host,
        port=foreground_plan.port,
        route_host=foreground_plan.route_host,
        pid_file=selected_pid_file,
        log_file=selected_log_file,
        process_pattern=process_pattern,
        remote_status_command="",
        remote_start_command="",
        remote_stop_command="",
    )
    return RemoteRerankerBackgroundLifecyclePlan(
        **{
            **asdict(plan),
            "remote_status_command": _remote_status_command(plan),
            "remote_start_command": _remote_start_command(
                plan, foreground_plan.launch_plan.shell_command
            ),
            "remote_stop_command": _remote_stop_command(plan),
        }
    )


def run_background_lifecycle(
    plan: RemoteRerankerBackgroundLifecyclePlan,
    *,
    request_smoke: bool = False,
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> RemoteRerankerBackgroundLifecycleReport:
    started_at = time.perf_counter()
    try:
        if plan.action == "start":
            command = plan.remote_start_command
        elif plan.action == "stop":
            command = plan.remote_stop_command
        else:
            command = plan.remote_status_command
        command_observation = _run_remote_command(plan, command)
    except OSError as exc:
        command_observation = RemoteCommandObservation(
            ok=False,
            exit_code=1,
            stdout="",
            stderr="",
            values={},
        )
        return _report(
            plan,
            command_observation=command_observation,
            health_observation=HealthObservation(
                ok=False, status_code=None, payload=None, error=None
            ),
            health_mismatches=(),
            request_report=None,
            request_smoke_checked=request_smoke or plan.action == "smoke",
            elapsed_ms=_elapsed_ms(started_at),
            error=str(exc),
        )

    health_observation = _health_for_action(
        plan,
        health_timeout_seconds=health_timeout_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    health_mismatches = (
        _health_mismatches(health_observation.payload, plan=plan)
        if health_observation.ok and plan.action != "stop"
        else ()
    )

    should_smoke = request_smoke or plan.action == "smoke"
    request_report = None
    if should_smoke and health_observation.ok and not health_mismatches:
        request_report = _run_request_smoke(plan, request_timeout_seconds=request_timeout_seconds)

    return _report(
        plan,
        command_observation=command_observation,
        health_observation=health_observation,
        health_mismatches=health_mismatches,
        request_report=request_report,
        request_smoke_checked=should_smoke,
        elapsed_ms=_elapsed_ms(started_at),
        error=None,
    )


def write_json_report(report: RemoteRerankerBackgroundLifecycleReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_report_payload(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_markdown_report(
    report: RemoteRerankerBackgroundLifecycleReport, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def _run_remote_command(
    plan: RemoteRerankerBackgroundLifecyclePlan,
    remote_command: str,
) -> RemoteCommandObservation:
    completed = subprocess.run(
        ("ssh", plan.ssh_target, remote_command),
        capture_output=True,
        check=False,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return RemoteCommandObservation(
        ok=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        values=_parse_key_values(stdout),
    )


def _remote_status_command(plan: RemoteRerankerBackgroundLifecyclePlan) -> str:
    return _remote_prelude(plan) + (
        f"pid=$(pgrep -f {shlex.quote(plan.process_pattern)} | head -n 1 || true); "
        f"file_pid=''; if [ -f {shlex.quote(plan.pid_file)} ]; then "
        f"file_pid=$(cat {shlex.quote(plan.pid_file)} 2>/dev/null || true); fi; "
        'if [ -n "$pid" ]; then '
        f"printf '%s\\n' \"$pid\" > {shlex.quote(plan.pid_file)}; "
        "printf 'status=running\\npid=%s\\n' \"$pid\"; "
        "else "
        "printf 'status=stopped\\npid=\\n'; "
        "fi; "
        "printf 'file_pid=%s\\n' \"$file_pid\"; "
        f"printf 'pid_file=%s\\n' {shlex.quote(plan.pid_file)}; "
        f"printf 'log_file=%s\\n' {shlex.quote(plan.log_file)}"
    )


def _remote_start_command(
    plan: RemoteRerankerBackgroundLifecyclePlan,
    launch_shell_command: str,
) -> str:
    return _remote_prelude(plan) + (
        f"pid=$(pgrep -f {shlex.quote(plan.process_pattern)} | head -n 1 || true); "
        'if [ -n "$pid" ]; then '
        f"printf '%s\\n' \"$pid\" > {shlex.quote(plan.pid_file)}; "
        "printf 'status=already_running\\npid=%s\\n' \"$pid\"; "
        f"printf 'pid_file=%s\\n' {shlex.quote(plan.pid_file)}; "
        f"printf 'log_file=%s\\n' {shlex.quote(plan.log_file)}; "
        "exit 0; "
        "fi; "
        f"nohup sh -c {shlex.quote(launch_shell_command)} "
        f"> {shlex.quote(plan.log_file)} 2>&1 < /dev/null & "
        "pid=$!; "
        f"printf '%s\\n' \"$pid\" > {shlex.quote(plan.pid_file)}; "
        "printf 'status=started\\npid=%s\\n' \"$pid\"; "
        f"printf 'pid_file=%s\\n' {shlex.quote(plan.pid_file)}; "
        f"printf 'log_file=%s\\n' {shlex.quote(plan.log_file)}"
    )


def _remote_stop_command(plan: RemoteRerankerBackgroundLifecyclePlan) -> str:
    return _remote_prelude(plan) + (
        f"pid=$(pgrep -f {shlex.quote(plan.process_pattern)} | head -n 1 || true); "
        f'if [ -z "$pid" ] && [ -f {shlex.quote(plan.pid_file)} ]; then '
        f"file_pid=$(cat {shlex.quote(plan.pid_file)} 2>/dev/null || true); "
        'if [ -n "$file_pid" ] && kill -0 "$file_pid" 2>/dev/null; then '
        'pid="$file_pid"; fi; fi; '
        'if [ -z "$pid" ]; then '
        f"rm -f {shlex.quote(plan.pid_file)}; "
        "printf 'status=not_running\\npid=\\n'; "
        f"printf 'pid_file=%s\\n' {shlex.quote(plan.pid_file)}; "
        f"printf 'log_file=%s\\n' {shlex.quote(plan.log_file)}; "
        "exit 0; "
        "fi; "
        'kill -TERM "$pid" 2>/dev/null || true; '
        f"for _ in $(seq 1 {DEFAULT_STOP_TIMEOUT_SECONDS}); do "
        'if kill -0 "$pid" 2>/dev/null; then sleep 1; else break; fi; '
        "done; "
        'if kill -0 "$pid" 2>/dev/null; then '
        'kill -KILL "$pid" 2>/dev/null || true; status=force_stopped; '
        "else status=stopped; fi; "
        f"rm -f {shlex.quote(plan.pid_file)}; "
        'printf \'status=%s\\npid=%s\\n\' "$status" "$pid"; '
        f"printf 'pid_file=%s\\n' {shlex.quote(plan.pid_file)}; "
        f"printf 'log_file=%s\\n' {shlex.quote(plan.log_file)}"
    )


def _remote_prelude(plan: RemoteRerankerBackgroundLifecyclePlan) -> str:
    return (
        f"cd {shlex.quote(plan.workdir)} && "
        f"mkdir -p {shlex.quote(str(Path(plan.pid_file).parent))} "
        f"{shlex.quote(str(Path(plan.log_file).parent))} && "
    )


def _health_for_action(
    plan: RemoteRerankerBackgroundLifecyclePlan,
    *,
    health_timeout_seconds: float,
    startup_timeout_seconds: float,
    poll_interval_seconds: float,
) -> HealthObservation:
    if plan.action == "start":
        return _wait_for_health(
            plan.health_url,
            startup_timeout_seconds=startup_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            health_timeout_seconds=health_timeout_seconds,
        )
    observation = _probe_health_once(plan.health_url, timeout_seconds=health_timeout_seconds)
    if plan.action == "stop" and observation.ok:
        return observation
    return observation


def _wait_for_health(
    url: str,
    *,
    startup_timeout_seconds: float,
    poll_interval_seconds: float,
    health_timeout_seconds: float,
) -> HealthObservation:
    deadline = time.monotonic() + startup_timeout_seconds
    last_observation = HealthObservation(ok=False, status_code=None, payload=None, error=None)
    while time.monotonic() < deadline:
        last_observation = _probe_health_once(url, timeout_seconds=health_timeout_seconds)
        if last_observation.ok:
            return last_observation
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
    return last_observation


def _run_request_smoke(
    plan: RemoteRerankerBackgroundLifecyclePlan,
    *,
    request_timeout_seconds: float,
) -> RemoteRerankerRequestSmokeReport:
    request_plan = build_reranker_request_smoke_plan(
        base_url=plan.base_url,
        provider_name=plan.provider_name,
        provider_model_id=plan.provider_model_id,
        reranker_profile_name=plan.reranker_profile_name,
        expected_backend=plan.backend,
        expected_device=plan.device,
        timeout_seconds=request_timeout_seconds,
    )
    provider = RemoteRerankerProviderClient(
        request_plan.base_url,
        timeout_seconds=request_plan.timeout_seconds,
    )
    try:
        return run_reranker_request_smoke(provider, plan=request_plan)
    finally:
        provider.close()


def _report(
    plan: RemoteRerankerBackgroundLifecyclePlan,
    *,
    command_observation: RemoteCommandObservation,
    health_observation: HealthObservation,
    health_mismatches: tuple[str, ...],
    request_report: RemoteRerankerRequestSmokeReport | None,
    request_smoke_checked: bool,
    elapsed_ms: int,
    error: str | None,
) -> RemoteRerankerBackgroundLifecycleReport:
    values = command_observation.values
    status = values.get("status") or ("failed" if not command_observation.ok else "unknown")
    pid = values.get("pid") or None
    request_summary = _request_smoke_summary(request_report) if request_report is not None else None
    return RemoteRerankerBackgroundLifecycleReport(
        plan=plan,
        action=plan.action,
        status=status,
        pid=pid,
        command_observation=command_observation,
        health_checked=True,
        health_ok=health_observation.ok,
        health_status_code=health_observation.status_code,
        health_payload=health_observation.payload,
        health_error=health_observation.error,
        health_mismatches=health_mismatches,
        request_smoke_checked=request_smoke_checked,
        request_smoke_passed=request_report.passed if request_report is not None else None,
        request_smoke_summary=request_summary,
        elapsed_ms=elapsed_ms,
        error=error,
    )


def _request_smoke_summary(report: RemoteRerankerRequestSmokeReport) -> dict[str, Any]:
    observation = report.observation
    return {
        "passed": report.passed,
        "request_elapsed_ms": observation.request_elapsed_ms,
        "provider_elapsed_ms": observation.provider_elapsed_ms,
        "candidate_count": observation.candidate_count,
        "returned_count": observation.returned_count,
        "result_previews": [asdict(preview) for preview in observation.result_previews],
        "runtime_metadata": observation.runtime_metadata,
        "mismatches": list(observation.mismatches),
        "error": observation.error,
    }


def _parse_key_values(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key.strip()] = value.strip()
    return values


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _report_payload(report: RemoteRerankerBackgroundLifecycleReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "action": report.action,
        "status": report.status,
        "pid": report.pid,
        "plan": asdict(report.plan),
        "command_observation": asdict(report.command_observation),
        "health_checked": report.health_checked,
        "health_ok": report.health_ok,
        "health_status_code": report.health_status_code,
        "health_payload": report.health_payload,
        "health_error": report.health_error,
        "health_mismatches": list(report.health_mismatches),
        "request_smoke_checked": report.request_smoke_checked,
        "request_smoke_passed": report.request_smoke_passed,
        "request_smoke_summary": report.request_smoke_summary,
        "elapsed_ms": report.elapsed_ms,
        "error": report.error,
    }


def _print_human_report(report: RemoteRerankerBackgroundLifecycleReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"Remote reranker background lifecycle {report.action}: {status}")
    print(f"- provider_name: {report.plan.provider_name}")
    print(f"- base_url: {report.plan.base_url}")
    print(f"- status: {report.status}")
    print(f"- pid: {report.pid}")
    print(f"- pid_file: {report.plan.pid_file}")
    print(f"- log_file: {report.plan.log_file}")
    print(f"- health_ok: {report.health_ok}")
    print(f"- request_smoke_passed: {report.request_smoke_passed}")
    if report.error:
        print(f"- error: {report.error}")
    if report.health_mismatches:
        print("- health_mismatches:")
        for mismatch in report.health_mismatches:
            print(f"  - {mismatch}")
    if report.command_observation.stderr:
        print(f"- stderr: {report.command_observation.stderr}")


def _markdown_report(report: RemoteRerankerBackgroundLifecycleReport) -> str:
    lines = [
        "# Remote Reranker Background Lifecycle Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `action`: `{report.action}`",
        f"- `status`: `{report.status}`",
        f"- `pid`: `{report.pid}`",
        f"- `base_url`: `{report.plan.base_url}`",
        f"- `health_ok`: `{str(report.health_ok).lower()}`",
        f"- `request_smoke_passed`: `{report.request_smoke_passed}`",
        f"- `elapsed_ms`: `{report.elapsed_ms}`",
        "",
        "## Files",
        "",
        f"- `pid_file`: `{report.plan.workdir}/{report.plan.pid_file}`",
        f"- `log_file`: `{report.plan.workdir}/{report.plan.log_file}`",
        "",
        "## Health Payload",
        "",
        "```json",
        json.dumps(report.health_payload or {}, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if report.request_smoke_summary is not None:
        lines.extend(
            [
                "## Request Smoke Summary",
                "",
                "```json",
                json.dumps(report.request_smoke_summary, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    if report.health_mismatches:
        lines.extend(["## Health Mismatches", ""])
        lines.extend(f"- `{mismatch}`" for mismatch in report.health_mismatches)
        lines.append("")
    if report.error:
        lines.extend(["## Error", "", f"`{report.error}`", ""])
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the DGX remote reranker provider background lifecycle.",
    )
    parser.add_argument(
        "action",
        choices=("start", "status", "stop", "smoke"),
        help="Lifecycle action to run.",
    )
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--ssh-user", default=DEFAULT_GPU_USER)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--provider-name", default=DEFAULT_RERANKER_PROVIDER_NAME)
    parser.add_argument("--port", type=int, default=DEFAULT_RERANKER_PROVIDER_PORT)
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--request-smoke", action="store_true")
    parser.add_argument(
        "--health-timeout-seconds", type=float, default=DEFAULT_HEALTH_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--startup-timeout-seconds", type=float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_background_lifecycle_plan(
            action=args.action,
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            route_host=args.route_host,
            provider_name=args.provider_name,
            port=args.port,
            pid_file=args.pid_file,
            log_file=args.log_file,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": asdict(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    report = run_background_lifecycle(
        plan,
        request_smoke=args.request_smoke,
        health_timeout_seconds=args.health_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    if args.json_output:
        write_json_report(report, Path(args.json_output))
    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
