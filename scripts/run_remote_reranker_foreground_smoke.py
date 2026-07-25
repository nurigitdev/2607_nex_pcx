"""Run a foreground launch health smoke for the remote reranker provider."""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rerankers import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    REMOTE_RERANKER_PROVIDER_MODE,
)
from app.reranker_provider_service import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_DIR_NAME,
    RERANKER_PROVIDER_BACKEND_QWEN,
)
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_GPU_HOST,
    DEFAULT_GPU_USER,
    DEFAULT_GPU_WORKDIR,
)
from scripts.plan_remote_reranker_foreground_smoke import (  # noqa: E402
    RemoteRerankerForegroundSmokePlan,
    build_reranker_foreground_smoke_plan,
)
from scripts.run_reranker_provider import (  # noqa: E402
    DEFAULT_RERANKER_PROVIDER_NAME,
    DEFAULT_RERANKER_PROVIDER_PORT,
)

DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
DEFAULT_HEALTH_TIMEOUT_SECONDS = 5.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 15.0
OUTPUT_TAIL_CHARACTERS = 4000


@dataclass(frozen=True)
class HealthObservation:
    ok: bool
    status_code: int | None
    payload: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class RemoteRerankerForegroundSmokeResult:
    provider_name: str
    base_url: str
    health_url: str
    launch_command: tuple[str, ...]
    pre_launch_health_reachable: bool
    startup_timeout_seconds: float
    health_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_timeout_seconds: float
    launched: bool
    health_checked: bool
    health_ok: bool
    health_attempts: int
    health_status_code: int | None
    health_payload: dict[str, Any] | None
    health_error: str | None
    health_mismatches: tuple[str, ...]
    process_exit_code_before_stop: int | None
    process_exit_code_after_stop: int | None
    stopped: bool
    stop_confirmed: bool
    remote_stop_attempted: bool
    remote_stop_exit_code: int | None
    remote_stop_stdout: str
    remote_stop_stderr: str
    post_stop_health_reachable: bool
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.launched
            and self.health_ok
            and not self.health_mismatches
            and self.stopped
            and self.stop_confirmed
            and (not self.remote_stop_attempted or self.remote_stop_exit_code == 0)
            and not self.post_stop_health_reachable
            and self.error is None
        )


def _automated_ssh_launch_command(
    plan: RemoteRerankerForegroundSmokePlan,
) -> tuple[str, ...]:
    return ("ssh", plan.ssh_target, plan.remote_launch_command)


def _remote_stop_command(plan: RemoteRerankerForegroundSmokePlan) -> tuple[str, ...]:
    pattern = f"[u]vicorn app.reranker_provider_service:app --host {plan.host} --port {plan.port}"
    remote_command = (
        f"pkill -TERM -f {shlex.quote(pattern)} || true; "
        "sleep 1; "
        f"pkill -KILL -f {shlex.quote(pattern)} || true"
    )
    return ("ssh", plan.ssh_target, remote_command)


def _validate_positive(value: float, *, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _probe_health_once(url: str, *, timeout_seconds: float) -> HealthObservation:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return HealthObservation(
            ok=False,
            status_code=exc.code,
            payload=None,
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return HealthObservation(ok=False, status_code=None, payload=None, error=str(exc))

    try:
        payload = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError as exc:
        return HealthObservation(
            ok=False,
            status_code=status_code,
            payload=None,
            error=f"Invalid JSON health response: {exc}",
        )
    if not isinstance(payload, dict):
        return HealthObservation(
            ok=False,
            status_code=status_code,
            payload=None,
            error="Health response must be a JSON object.",
        )
    return HealthObservation(
        ok=200 <= status_code < 300,
        status_code=status_code,
        payload=payload,
        error=None if 200 <= status_code < 300 else f"HTTP {status_code}",
    )


def _health_mismatches(
    payload: dict[str, Any] | None,
    *,
    plan: RemoteRerankerForegroundSmokePlan,
) -> tuple[str, ...]:
    if payload is None:
        return ("health response payload is missing",)

    mismatches: list[str] = []
    expected_scalars = {
        "ready": True,
        "provider_type": REMOTE_RERANKER_PROVIDER_MODE,
        "provider_model_id": plan.provider_model_id,
        "reranker_profile_name": plan.reranker_profile_name,
        "device": plan.device,
    }
    for key, expected in expected_scalars.items():
        actual = payload.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")

    runtime_metadata = payload.get("runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        mismatches.append("runtime_metadata: expected JSON object")
        return tuple(mismatches)

    expected_runtime_metadata = {
        "service": "nex_pcx_reranker_provider_service",
        "backend": plan.backend,
        "model_dir": f"{plan.models_dir}/{plan.model_dir_name}",
        "model_dir_exists": True,
    }
    for key, expected in expected_runtime_metadata.items():
        actual = runtime_metadata.get(key)
        if actual != expected:
            mismatches.append(f"runtime_metadata.{key}: expected {expected!r}, got {actual!r}")

    return tuple(mismatches)


def _read_tail(file_obj: TextIO, *, limit: int = OUTPUT_TAIL_CHARACTERS) -> str:
    file_obj.flush()
    file_obj.seek(0)
    content = file_obj.read()
    return content[-limit:]


def _stop_process(
    process: subprocess.Popen[str],
    *,
    shutdown_timeout_seconds: float,
) -> tuple[bool, int | None]:
    exit_code = process.poll()
    if exit_code is not None:
        return True, exit_code

    process.terminate()
    try:
        return True, process.wait(timeout=shutdown_timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            return False, process.wait(timeout=shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            return False, process.poll()


def _wait_until_health_unreachable(
    url: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    health_timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        observation = _probe_health_once(url, timeout_seconds=health_timeout_seconds)
        if not observation.ok:
            return True
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
    return False


def run_foreground_smoke(
    plan: RemoteRerankerForegroundSmokePlan,
    *,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
) -> RemoteRerankerForegroundSmokeResult:
    startup_timeout_seconds = _validate_positive(
        startup_timeout_seconds,
        name="startup_timeout_seconds",
    )
    health_timeout_seconds = _validate_positive(
        health_timeout_seconds,
        name="health_timeout_seconds",
    )
    poll_interval_seconds = _validate_positive(
        poll_interval_seconds,
        name="poll_interval_seconds",
    )
    shutdown_timeout_seconds = _validate_positive(
        shutdown_timeout_seconds,
        name="shutdown_timeout_seconds",
    )
    launch_command = _automated_ssh_launch_command(plan)
    started_at = time.monotonic()
    health_checked = False
    health_attempts = 0
    health_observation = HealthObservation(
        ok=False,
        status_code=None,
        payload=None,
        error="Health check was not attempted.",
    )
    health_mismatches: tuple[str, ...] = ()
    exit_code_before_stop: int | None = None
    exit_code_after_stop: int | None = None
    stopped = False
    stop_confirmed = False
    remote_stop_attempted = False
    remote_stop_exit_code: int | None = None
    remote_stop_stdout = ""
    remote_stop_stderr = ""
    post_stop_health_reachable = False
    launched = False
    error: str | None = None
    pre_launch_observation = _probe_health_once(
        plan.health_url,
        timeout_seconds=min(health_timeout_seconds, 1.0),
    )
    pre_launch_health_reachable = pre_launch_observation.ok

    if pre_launch_health_reachable:
        return _result(
            plan=plan,
            launch_command=launch_command,
            pre_launch_health_reachable=True,
            startup_timeout_seconds=startup_timeout_seconds,
            health_timeout_seconds=health_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
            launched=False,
            health_checked=False,
            health_ok=False,
            health_attempts=0,
            health_status_code=pre_launch_observation.status_code,
            health_payload=pre_launch_observation.payload,
            health_error=pre_launch_observation.error,
            health_mismatches=(),
            process_exit_code_before_stop=None,
            process_exit_code_after_stop=None,
            stopped=False,
            stop_confirmed=False,
            remote_stop_attempted=False,
            remote_stop_exit_code=None,
            remote_stop_stdout="",
            remote_stop_stderr="",
            post_stop_health_reachable=True,
            elapsed_seconds=max(0.0, time.monotonic() - started_at),
            stdout_tail="",
            stderr_tail="",
            error=(
                "Health URL is already reachable before launch; "
                "stop the existing reranker provider or use another port."
            ),
        )

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stdout:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stderr:
            try:
                process = subprocess.Popen(
                    launch_command,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
                launched = True
            except OSError as exc:
                return _result(
                    plan=plan,
                    launch_command=launch_command,
                    pre_launch_health_reachable=pre_launch_health_reachable,
                    startup_timeout_seconds=startup_timeout_seconds,
                    health_timeout_seconds=health_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    shutdown_timeout_seconds=shutdown_timeout_seconds,
                    launched=False,
                    health_checked=False,
                    health_ok=False,
                    health_attempts=0,
                    health_status_code=None,
                    health_payload=None,
                    health_error=None,
                    health_mismatches=(),
                    process_exit_code_before_stop=None,
                    process_exit_code_after_stop=None,
                    stopped=False,
                    stop_confirmed=False,
                    remote_stop_attempted=False,
                    remote_stop_exit_code=None,
                    remote_stop_stdout="",
                    remote_stop_stderr="",
                    post_stop_health_reachable=False,
                    elapsed_seconds=max(0.0, time.monotonic() - started_at),
                    stdout_tail=_read_tail(stdout),
                    stderr_tail=_read_tail(stderr),
                    error=str(exc),
                )

            deadline = time.monotonic() + startup_timeout_seconds
            while time.monotonic() < deadline:
                exit_code_before_stop = process.poll()
                if exit_code_before_stop is not None:
                    error = (
                        f"Remote reranker provider process exited early with code "
                        f"{exit_code_before_stop}."
                    )
                    break

                health_checked = True
                health_attempts += 1
                health_observation = _probe_health_once(
                    plan.health_url,
                    timeout_seconds=health_timeout_seconds,
                )
                if health_observation.ok:
                    health_mismatches = _health_mismatches(
                        health_observation.payload,
                        plan=plan,
                    )
                    break

                time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
            else:
                error = f"Health check did not pass within {startup_timeout_seconds:.1f} seconds."

            if health_observation.ok and health_mismatches:
                error = "Health response did not match the expected reranker provider plan."

            exit_code_before_stop = process.poll()
            stopped, exit_code_after_stop = _stop_process(
                process,
                shutdown_timeout_seconds=shutdown_timeout_seconds,
            )
            stop_confirmed = exit_code_after_stop is not None
            if health_observation.ok:
                health_unreachable = _wait_until_health_unreachable(
                    plan.health_url,
                    timeout_seconds=shutdown_timeout_seconds,
                    poll_interval_seconds=min(poll_interval_seconds, 0.5),
                    health_timeout_seconds=min(health_timeout_seconds, 1.0),
                )
                post_stop_health_reachable = not health_unreachable
                if post_stop_health_reachable:
                    remote_stop_attempted = True
                    try:
                        remote_stop = subprocess.run(
                            _remote_stop_command(plan),
                            capture_output=True,
                            text=True,
                            timeout=shutdown_timeout_seconds,
                            check=False,
                        )
                        remote_stop_exit_code = remote_stop.returncode
                        remote_stop_stdout = remote_stop.stdout[-OUTPUT_TAIL_CHARACTERS:]
                        remote_stop_stderr = remote_stop.stderr[-OUTPUT_TAIL_CHARACTERS:]
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        remote_stop_exit_code = None
                        remote_stop_stderr = str(exc)
                        if error is None:
                            error = f"Remote stop fallback failed: {exc}"
                    health_unreachable = _wait_until_health_unreachable(
                        plan.health_url,
                        timeout_seconds=shutdown_timeout_seconds,
                        poll_interval_seconds=min(poll_interval_seconds, 0.5),
                        health_timeout_seconds=min(health_timeout_seconds, 1.0),
                    )
                    post_stop_health_reachable = not health_unreachable
                    if post_stop_health_reachable and error is None:
                        error = "Health URL was still reachable after remote stop fallback."

            return _result(
                plan=plan,
                launch_command=launch_command,
                pre_launch_health_reachable=pre_launch_health_reachable,
                startup_timeout_seconds=startup_timeout_seconds,
                health_timeout_seconds=health_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                shutdown_timeout_seconds=shutdown_timeout_seconds,
                launched=launched,
                health_checked=health_checked,
                health_ok=health_observation.ok and not health_mismatches,
                health_attempts=health_attempts,
                health_status_code=health_observation.status_code,
                health_payload=health_observation.payload,
                health_error=health_observation.error,
                health_mismatches=health_mismatches,
                process_exit_code_before_stop=exit_code_before_stop,
                process_exit_code_after_stop=exit_code_after_stop,
                stopped=stopped,
                stop_confirmed=stop_confirmed,
                remote_stop_attempted=remote_stop_attempted,
                remote_stop_exit_code=remote_stop_exit_code,
                remote_stop_stdout=remote_stop_stdout,
                remote_stop_stderr=remote_stop_stderr,
                post_stop_health_reachable=post_stop_health_reachable,
                elapsed_seconds=max(0.0, time.monotonic() - started_at),
                stdout_tail=_read_tail(stdout),
                stderr_tail=_read_tail(stderr),
                error=error,
            )


def _result(
    *,
    plan: RemoteRerankerForegroundSmokePlan,
    launch_command: tuple[str, ...],
    pre_launch_health_reachable: bool,
    startup_timeout_seconds: float,
    health_timeout_seconds: float,
    poll_interval_seconds: float,
    shutdown_timeout_seconds: float,
    launched: bool,
    health_checked: bool,
    health_ok: bool,
    health_attempts: int,
    health_status_code: int | None,
    health_payload: dict[str, Any] | None,
    health_error: str | None,
    health_mismatches: tuple[str, ...],
    process_exit_code_before_stop: int | None,
    process_exit_code_after_stop: int | None,
    stopped: bool,
    stop_confirmed: bool,
    remote_stop_attempted: bool,
    remote_stop_exit_code: int | None,
    remote_stop_stdout: str,
    remote_stop_stderr: str,
    post_stop_health_reachable: bool,
    elapsed_seconds: float,
    stdout_tail: str,
    stderr_tail: str,
    error: str | None,
) -> RemoteRerankerForegroundSmokeResult:
    return RemoteRerankerForegroundSmokeResult(
        provider_name=plan.provider_name,
        base_url=plan.base_url,
        health_url=plan.health_url,
        launch_command=launch_command,
        pre_launch_health_reachable=pre_launch_health_reachable,
        startup_timeout_seconds=startup_timeout_seconds,
        health_timeout_seconds=health_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        launched=launched,
        health_checked=health_checked,
        health_ok=health_ok,
        health_attempts=health_attempts,
        health_status_code=health_status_code,
        health_payload=health_payload,
        health_error=health_error,
        health_mismatches=health_mismatches,
        process_exit_code_before_stop=process_exit_code_before_stop,
        process_exit_code_after_stop=process_exit_code_after_stop,
        stopped=stopped,
        stop_confirmed=stop_confirmed,
        remote_stop_attempted=remote_stop_attempted,
        remote_stop_exit_code=remote_stop_exit_code,
        remote_stop_stdout=remote_stop_stdout,
        remote_stop_stderr=remote_stop_stderr,
        post_stop_health_reachable=post_stop_health_reachable,
        elapsed_seconds=elapsed_seconds,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=error,
    )


def _result_payload(result: RemoteRerankerForegroundSmokeResult) -> dict[str, Any]:
    return {**asdict(result), "passed": result.passed}


def _print_human_result(result: RemoteRerankerForegroundSmokeResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"Remote reranker foreground smoke: {status}")
    print(f"- provider_name: {result.provider_name}")
    print(f"- health_url: {result.health_url}")
    print(f"- pre_launch_health_reachable: {result.pre_launch_health_reachable}")
    print(f"- launched: {result.launched}")
    print(f"- health_ok: {result.health_ok}")
    print(f"- health_attempts: {result.health_attempts}")
    print(f"- stopped: {result.stopped}")
    print(f"- stop_confirmed: {result.stop_confirmed}")
    print(f"- remote_stop_attempted: {result.remote_stop_attempted}")
    if result.remote_stop_attempted:
        print(f"- remote_stop_exit_code: {result.remote_stop_exit_code}")
    print(f"- post_stop_health_reachable: {result.post_stop_health_reachable}")
    print(f"- elapsed_seconds: {result.elapsed_seconds:.2f}")
    if result.health_mismatches:
        print("- health_mismatches:")
        for mismatch in result.health_mismatches:
            print(f"  - {mismatch}")
    if result.error:
        print(f"- error: {result.error}")
    if result.stderr_tail:
        print("- stderr_tail:")
        print(result.stderr_tail)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a foreground launch health smoke for the remote reranker provider.",
    )
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--ssh-user", default=DEFAULT_GPU_USER)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--model-dir-name", default=DEFAULT_RERANKER_MODEL_DIR_NAME)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--provider-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_RERANKER_PROVIDER_PORT)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--provider-name", default=DEFAULT_RERANKER_PROVIDER_NAME)
    parser.add_argument("--provider-model-id", default=DEFAULT_RERANKER_MODEL_ID)
    parser.add_argument("--profile-name", default=DEFAULT_RERANKER_PROFILE_NAME)
    parser.add_argument("--backend", default=RERANKER_PROVIDER_BACKEND_QWEN)
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=float,
        default=DEFAULT_HEALTH_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--shutdown-timeout-seconds",
        type=float,
        default=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_reranker_foreground_smoke_plan(
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            models_dir=args.models_dir,
            model_dir_name=args.model_dir_name,
            python_bin=args.python_bin,
            provider_host=args.provider_host,
            port=args.port,
            route_host=args.route_host,
            device=args.device,
            provider_name=args.provider_name,
            provider_model_id=args.provider_model_id,
            reranker_profile_name=args.profile_name,
            backend=args.backend,
        )
        result = run_foreground_smoke(
            plan,
            startup_timeout_seconds=args.startup_timeout_seconds,
            health_timeout_seconds=args.health_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(_result_payload(result), ensure_ascii=False))
    else:
        _print_human_result(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
