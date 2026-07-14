"""Run a sequential remote embedding request smoke suite."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.embedding_provider_presets import (  # noqa: E402
    InvalidEmbeddingProviderPresetError,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from app.core.embedding_providers import RemoteEmbeddingProviderClient  # noqa: E402
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_GPU_HOST,
    DEFAULT_GPU_USER,
    DEFAULT_GPU_WORKDIR,
    RemoteProviderForegroundSmokePlan,
    build_foreground_smoke_plan,
)
from scripts.run_remote_provider_embedding_smoke import (  # noqa: E402
    DEFAULT_SMOKE_TEXT,
    RemoteProviderEmbeddingSmokePlan,
    RemoteProviderEmbeddingSmokeReport,
    build_embedding_smoke_plan,
    run_embedding_request_smoke,
)
from scripts.run_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_HEALTH_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    HealthObservation,
    _automated_ssh_launch_command,
    _health_mismatches,
    _probe_health_once,
    _read_tail,
    _remote_stop_command,
    _stop_process,
    _validate_positive,
    _wait_until_health_unreachable,
)

DEFAULT_PROVIDER_ORDER = ("kure", "bge", "qwen")
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS = 300.0
OUTPUT_TAIL_CHARACTERS = 4000


@dataclass(frozen=True)
class RemoteProviderEmbeddingSuiteProviderPlan:
    provider: str
    foreground_plan: RemoteProviderForegroundSmokePlan
    embedding_plan: RemoteProviderEmbeddingSmokePlan
    startup_timeout_seconds: float
    request_timeout_seconds: float


@dataclass(frozen=True)
class RemoteProviderEmbeddingSuitePlan:
    providers: tuple[RemoteProviderEmbeddingSuiteProviderPlan, ...]
    health_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_timeout_seconds: float
    fail_fast: bool


@dataclass(frozen=True)
class RemoteProviderEmbeddingSuiteProviderResult:
    provider: str
    provider_name: str
    base_url: str
    health_url: str
    launch_command: tuple[str, ...]
    startup_timeout_seconds: float
    request_timeout_seconds: float
    health_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_timeout_seconds: float
    pre_launch_health_reachable: bool
    launched: bool
    health_checked: bool
    health_ok: bool
    health_attempts: int
    health_status_code: int | None
    health_payload: dict[str, Any] | None
    health_error: str | None
    health_mismatches: tuple[str, ...]
    embedding_report: RemoteProviderEmbeddingSmokeReport | None
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
            and self.embedding_report is not None
            and self.embedding_report.passed
            and self.stopped
            and self.stop_confirmed
            and (not self.remote_stop_attempted or self.remote_stop_exit_code == 0)
            and not self.post_stop_health_reachable
            and self.error is None
        )


@dataclass(frozen=True)
class RemoteProviderEmbeddingSuiteReport:
    plan: RemoteProviderEmbeddingSuitePlan
    results: tuple[RemoteProviderEmbeddingSuiteProviderResult, ...]
    total_elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)


def build_embedding_smoke_suite_plan(
    provider_names: tuple[str, ...] | None = None,
    *,
    host: str = DEFAULT_GPU_HOST,
    ssh_user: str = DEFAULT_GPU_USER,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    python_bin: str | None = None,
    provider_host: str = "0.0.0.0",
    route_host: str | None = None,
    device: str = DEFAULT_DEVICE,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    request_timeout_seconds: float | None = None,
    qwen_request_timeout_seconds: float = DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS,
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    input_type: str = "document",
    texts: tuple[str, ...] = (DEFAULT_SMOKE_TEXT,),
    fail_fast: bool = False,
) -> RemoteProviderEmbeddingSuitePlan:
    startup_timeout_seconds = _validate_positive(
        startup_timeout_seconds,
        name="startup_timeout_seconds",
    )
    selected_request_timeout_seconds = (
        None
        if request_timeout_seconds is None
        else _validate_positive(request_timeout_seconds, name="request_timeout_seconds")
    )
    qwen_request_timeout_seconds = _validate_positive(
        qwen_request_timeout_seconds,
        name="qwen_request_timeout_seconds",
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

    selected_provider_names = _normalize_provider_names(provider_names)
    provider_plans = []
    for provider_name in selected_provider_names:
        preset = get_embedding_provider_preset(provider_name)
        foreground_plan = build_foreground_smoke_plan(
            preset,
            host=host,
            ssh_user=ssh_user,
            workdir=workdir,
            models_dir=models_dir,
            python_bin=python_bin,
            provider_host=provider_host,
            route_host=route_host,
            device=device,
        )
        provider_request_timeout_seconds = (
            selected_request_timeout_seconds
            if selected_request_timeout_seconds is not None
            else _default_request_timeout_seconds(provider_name)
        )
        if provider_name == "qwen" and selected_request_timeout_seconds is None:
            provider_request_timeout_seconds = qwen_request_timeout_seconds
        embedding_plan = build_embedding_smoke_plan(
            preset,
            base_url=foreground_plan.base_url,
            input_type=input_type,
            texts=texts,
            timeout_seconds=provider_request_timeout_seconds,
        )
        provider_plans.append(
            RemoteProviderEmbeddingSuiteProviderPlan(
                provider=provider_name,
                foreground_plan=foreground_plan,
                embedding_plan=embedding_plan,
                startup_timeout_seconds=startup_timeout_seconds,
                request_timeout_seconds=provider_request_timeout_seconds,
            )
        )

    return RemoteProviderEmbeddingSuitePlan(
        providers=tuple(provider_plans),
        health_timeout_seconds=health_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        fail_fast=fail_fast,
    )


def run_embedding_smoke_suite(
    plan: RemoteProviderEmbeddingSuitePlan,
) -> RemoteProviderEmbeddingSuiteReport:
    started_at = time.monotonic()
    results: list[RemoteProviderEmbeddingSuiteProviderResult] = []
    for provider_plan in plan.providers:
        result = run_provider_embedding_smoke_session(
            provider_plan,
            health_timeout_seconds=plan.health_timeout_seconds,
            poll_interval_seconds=plan.poll_interval_seconds,
            shutdown_timeout_seconds=plan.shutdown_timeout_seconds,
        )
        results.append(result)
        if plan.fail_fast and not result.passed:
            break

    return RemoteProviderEmbeddingSuiteReport(
        plan=plan,
        results=tuple(results),
        total_elapsed_seconds=max(0.0, time.monotonic() - started_at),
    )


def run_provider_embedding_smoke_session(
    provider_plan: RemoteProviderEmbeddingSuiteProviderPlan,
    *,
    health_timeout_seconds: float,
    poll_interval_seconds: float,
    shutdown_timeout_seconds: float,
) -> RemoteProviderEmbeddingSuiteProviderResult:
    foreground_plan = provider_plan.foreground_plan
    launch_command = _automated_ssh_launch_command(foreground_plan)
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
    embedding_report: RemoteProviderEmbeddingSmokeReport | None = None
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
        foreground_plan.health_url,
        timeout_seconds=min(health_timeout_seconds, 1.0),
    )
    pre_launch_health_reachable = pre_launch_observation.ok
    if pre_launch_health_reachable:
        return _provider_result(
            provider_plan,
            launch_command=launch_command,
            health_timeout_seconds=health_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
            pre_launch_health_reachable=True,
            launched=False,
            health_checked=False,
            health_ok=False,
            health_attempts=0,
            health_status_code=pre_launch_observation.status_code,
            health_payload=pre_launch_observation.payload,
            health_error=pre_launch_observation.error,
            health_mismatches=(),
            embedding_report=None,
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
                "stop the existing provider or use another port."
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
                return _provider_result(
                    provider_plan,
                    launch_command=launch_command,
                    health_timeout_seconds=health_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    shutdown_timeout_seconds=shutdown_timeout_seconds,
                    pre_launch_health_reachable=pre_launch_health_reachable,
                    launched=False,
                    health_checked=False,
                    health_ok=False,
                    health_attempts=0,
                    health_status_code=None,
                    health_payload=None,
                    health_error=None,
                    health_mismatches=(),
                    embedding_report=None,
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

            deadline = time.monotonic() + provider_plan.startup_timeout_seconds
            while time.monotonic() < deadline:
                exit_code_before_stop = process.poll()
                if exit_code_before_stop is not None:
                    error = (
                        f"Remote provider process exited early with code {exit_code_before_stop}."
                    )
                    break

                health_checked = True
                health_attempts += 1
                health_observation = _probe_health_once(
                    foreground_plan.health_url,
                    timeout_seconds=health_timeout_seconds,
                )
                if health_observation.ok:
                    health_mismatches = _health_mismatches(
                        health_observation.payload,
                        plan=foreground_plan,
                    )
                    break
                time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
            else:
                error = (
                    "Health check did not pass within "
                    f"{provider_plan.startup_timeout_seconds:.1f} seconds."
                )

            if health_observation.ok and health_mismatches:
                error = "Health response did not match the expected provider plan."

            if health_observation.ok and not health_mismatches:
                embedding_report = _run_request_smoke(provider_plan)
                if not embedding_report.passed and error is None:
                    error = "Embedding request smoke failed."

            exit_code_before_stop = process.poll()
            stopped, exit_code_after_stop = _stop_process(
                process,
                shutdown_timeout_seconds=shutdown_timeout_seconds,
            )
            stop_confirmed = exit_code_after_stop is not None
            if health_observation.ok:
                (
                    post_stop_health_reachable,
                    remote_stop_attempted,
                    remote_stop_exit_code,
                    remote_stop_stdout,
                    remote_stop_stderr,
                    stop_error,
                ) = _confirm_remote_stop(
                    foreground_plan,
                    health_timeout_seconds=health_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    shutdown_timeout_seconds=shutdown_timeout_seconds,
                )
                if stop_error and error is None:
                    error = stop_error

            return _provider_result(
                provider_plan,
                launch_command=launch_command,
                health_timeout_seconds=health_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                shutdown_timeout_seconds=shutdown_timeout_seconds,
                pre_launch_health_reachable=pre_launch_health_reachable,
                launched=launched,
                health_checked=health_checked,
                health_ok=health_observation.ok and not health_mismatches,
                health_attempts=health_attempts,
                health_status_code=health_observation.status_code,
                health_payload=health_observation.payload,
                health_error=health_observation.error,
                health_mismatches=health_mismatches,
                embedding_report=embedding_report,
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


def _run_request_smoke(
    provider_plan: RemoteProviderEmbeddingSuiteProviderPlan,
    *,
    provider_factory: Callable[..., RemoteEmbeddingProviderClient] = RemoteEmbeddingProviderClient,
) -> RemoteProviderEmbeddingSmokeReport:
    provider = provider_factory(
        provider_plan.embedding_plan.base_url,
        timeout_seconds=provider_plan.embedding_plan.timeout_seconds,
    )
    try:
        return run_embedding_request_smoke(provider, plan=provider_plan.embedding_plan)
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _confirm_remote_stop(
    foreground_plan: RemoteProviderForegroundSmokePlan,
    *,
    health_timeout_seconds: float,
    poll_interval_seconds: float,
    shutdown_timeout_seconds: float,
) -> tuple[bool, bool, int | None, str, str, str | None]:
    health_unreachable = _wait_until_health_unreachable(
        foreground_plan.health_url,
        timeout_seconds=shutdown_timeout_seconds,
        poll_interval_seconds=min(poll_interval_seconds, 0.5),
        health_timeout_seconds=min(health_timeout_seconds, 1.0),
    )
    post_stop_health_reachable = not health_unreachable
    remote_stop_attempted = False
    remote_stop_exit_code: int | None = None
    remote_stop_stdout = ""
    remote_stop_stderr = ""
    error: str | None = None
    if post_stop_health_reachable:
        remote_stop_attempted = True
        try:
            remote_stop = subprocess.run(
                _remote_stop_command(foreground_plan),
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
            error = f"Remote stop fallback failed: {exc}"
        health_unreachable = _wait_until_health_unreachable(
            foreground_plan.health_url,
            timeout_seconds=shutdown_timeout_seconds,
            poll_interval_seconds=min(poll_interval_seconds, 0.5),
            health_timeout_seconds=min(health_timeout_seconds, 1.0),
        )
        post_stop_health_reachable = not health_unreachable
        if post_stop_health_reachable and error is None:
            error = "Health URL was still reachable after remote stop fallback."
    return (
        post_stop_health_reachable,
        remote_stop_attempted,
        remote_stop_exit_code,
        remote_stop_stdout,
        remote_stop_stderr,
        error,
    )


def _provider_result(
    provider_plan: RemoteProviderEmbeddingSuiteProviderPlan,
    *,
    launch_command: tuple[str, ...],
    health_timeout_seconds: float,
    poll_interval_seconds: float,
    shutdown_timeout_seconds: float,
    pre_launch_health_reachable: bool,
    launched: bool,
    health_checked: bool,
    health_ok: bool,
    health_attempts: int,
    health_status_code: int | None,
    health_payload: dict[str, Any] | None,
    health_error: str | None,
    health_mismatches: tuple[str, ...],
    embedding_report: RemoteProviderEmbeddingSmokeReport | None,
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
) -> RemoteProviderEmbeddingSuiteProviderResult:
    foreground_plan = provider_plan.foreground_plan
    return RemoteProviderEmbeddingSuiteProviderResult(
        provider=provider_plan.provider,
        provider_name=foreground_plan.provider_name,
        base_url=foreground_plan.base_url,
        health_url=foreground_plan.health_url,
        launch_command=launch_command,
        startup_timeout_seconds=provider_plan.startup_timeout_seconds,
        request_timeout_seconds=provider_plan.request_timeout_seconds,
        health_timeout_seconds=health_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        pre_launch_health_reachable=pre_launch_health_reachable,
        launched=launched,
        health_checked=health_checked,
        health_ok=health_ok,
        health_attempts=health_attempts,
        health_status_code=health_status_code,
        health_payload=health_payload,
        health_error=health_error,
        health_mismatches=health_mismatches,
        embedding_report=embedding_report,
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


def _normalize_provider_names(provider_names: tuple[str, ...] | None) -> tuple[str, ...]:
    selected_names = provider_names or DEFAULT_PROVIDER_ORDER
    normalized_names = tuple(name.strip().lower() for name in selected_names if name.strip())
    if not normalized_names:
        raise InvalidEmbeddingProviderPresetError("At least one provider is required")
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in normalized_names:
        get_embedding_provider_preset(name)
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    return tuple(unique_names)


def _default_request_timeout_seconds(provider_name: str) -> float:
    return (
        DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS
        if provider_name == "qwen"
        else DEFAULT_REQUEST_TIMEOUT_SECONDS
    )


def _load_texts(text_args: list[str], texts_file: str | None) -> tuple[str, ...]:
    texts = [text.strip() for text in text_args if text.strip()]
    if texts_file:
        file_texts = [
            line.strip()
            for line in Path(texts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        texts.extend(file_texts)
    return tuple(texts or (DEFAULT_SMOKE_TEXT,))


def _suite_plan_payload(plan: RemoteProviderEmbeddingSuitePlan) -> dict[str, Any]:
    return {
        "providers": [_provider_plan_payload(provider) for provider in plan.providers],
        "health_timeout_seconds": plan.health_timeout_seconds,
        "poll_interval_seconds": plan.poll_interval_seconds,
        "shutdown_timeout_seconds": plan.shutdown_timeout_seconds,
        "fail_fast": plan.fail_fast,
    }


def _provider_plan_payload(
    provider_plan: RemoteProviderEmbeddingSuiteProviderPlan,
) -> dict[str, Any]:
    return {
        "provider": provider_plan.provider,
        "foreground_plan": asdict(provider_plan.foreground_plan),
        "embedding_plan": _embedding_plan_payload(provider_plan.embedding_plan),
        "startup_timeout_seconds": provider_plan.startup_timeout_seconds,
        "request_timeout_seconds": provider_plan.request_timeout_seconds,
    }


def _embedding_plan_payload(plan: RemoteProviderEmbeddingSmokePlan) -> dict[str, Any]:
    return {
        **asdict(plan),
        "texts": [f"<text:{index + 1}>" for index in range(len(plan.texts))],
    }


def _embedding_report_payload(report: RemoteProviderEmbeddingSmokeReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _embedding_plan_payload(report.plan),
        "observations": [asdict(observation) for observation in report.observations],
        "total_elapsed_ms": report.total_elapsed_ms,
    }


def _result_payload(result: RemoteProviderEmbeddingSuiteProviderResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["passed"] = result.passed
    payload["embedding_report"] = (
        _embedding_report_payload(result.embedding_report)
        if result.embedding_report is not None
        else None
    )
    return payload


def _suite_report_payload(report: RemoteProviderEmbeddingSuiteReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _suite_plan_payload(report.plan),
        "results": [_result_payload(result) for result in report.results],
        "total_elapsed_seconds": report.total_elapsed_seconds,
    }


def _print_human_report(report: RemoteProviderEmbeddingSuiteReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"Remote embedding request smoke suite: {status}")
    print(f"- providers: {len(report.results)}/{len(report.plan.providers)} executed")
    print(f"- total_elapsed_seconds: {report.total_elapsed_seconds:.2f}")
    for result in report.results:
        print(
            f"- {result.provider}: "
            f"passed={result.passed} "
            f"health_ok={result.health_ok} "
            f"post_stop_health_reachable={result.post_stop_health_reachable}"
        )
        if result.embedding_report is not None:
            for observation in result.embedding_report.observations:
                print(
                    f"  - {observation.case.profile_name}: "
                    f"dimension={observation.dimension} "
                    f"request_elapsed_ms={observation.request_elapsed_ms} "
                    f"preview={list(observation.embedding_preview)}"
                )
        if result.error:
            print(f"  - error: {result.error}")


def write_markdown_report(
    report: RemoteProviderEmbeddingSuiteReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: RemoteProviderEmbeddingSuiteReport) -> str:
    lines = [
        "# Remote Embedding Request Smoke Suite Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `total_elapsed_seconds`: `{report.total_elapsed_seconds:.2f}`",
        f"- providers executed: `{len(report.results)}`",
        "",
        "## Provider Results",
        "",
        "| Provider | Passed | Base URL | Health | Request Cases | Error |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in report.results:
        case_count = (
            len(result.embedding_report.observations) if result.embedding_report is not None else 0
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result.provider}`",
                    f"`{str(result.passed).lower()}`",
                    f"`{result.base_url}`",
                    f"`{str(result.health_ok).lower()}`",
                    f"`{case_count}`",
                    f"`{result.error or ''}`",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Embedding Vector Previews")
    lines.append("")
    lines.append("| Provider | Profile | Dimension | First 3 values from first vector |")
    lines.append("| --- | --- | ---: | --- |")
    for result in report.results:
        if result.embedding_report is None:
            continue
        for observation in result.embedding_report.observations:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{result.provider}`",
                        f"`{observation.case.profile_name}`",
                        f"`{observation.dimension}`",
                        f"`{list(observation.embedding_preview)}`",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description="Run a sequential request smoke suite against remote embedding providers.",
    )
    parser.add_argument("--provider", choices=preset_names, action="append", default=[])
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--ssh-user", default=DEFAULT_GPU_USER)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--provider-host", default="0.0.0.0")
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--qwen-request-timeout-seconds",
        type=float,
        default=DEFAULT_QWEN_REQUEST_TIMEOUT_SECONDS,
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
    parser.add_argument("--input-type", choices=("query", "document"), default="document")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--texts-file", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_embedding_smoke_suite_plan(
            provider_names=tuple(args.provider) or None,
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            models_dir=args.models_dir,
            python_bin=args.python_bin,
            provider_host=args.provider_host,
            route_host=args.route_host,
            device=args.device,
            startup_timeout_seconds=args.startup_timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            qwen_request_timeout_seconds=args.qwen_request_timeout_seconds,
            health_timeout_seconds=args.health_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
            input_type=args.input_type,
            texts=_load_texts(args.text, args.texts_file),
            fail_fast=args.fail_fast,
        )
    except (InvalidEmbeddingProviderPresetError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _suite_plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    report = run_embedding_smoke_suite(plan)
    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))

    if args.json:
        print(json.dumps(_suite_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
