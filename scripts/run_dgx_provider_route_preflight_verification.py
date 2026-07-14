"""Run DGX provider route preflight checks with managed remote launches."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_provider_presets import (  # noqa: E402
    InvalidEmbeddingProviderPresetError,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_GPU_HOST,
    DEFAULT_GPU_USER,
    DEFAULT_GPU_WORKDIR,
    RemoteProviderForegroundSmokePlan,
    build_foreground_smoke_plan,
)
from scripts.preflight_provider_routes import run_preflight  # noqa: E402
from scripts.run_remote_provider_embedding_smoke_suite import (  # noqa: E402
    _confirm_remote_stop,
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
    _stop_process,
    _validate_positive,
)

DEFAULT_PROVIDER_ORDER = ("kure", "bge", "qwen")
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class DgxProviderRoutePreflightProviderPlan:
    provider: str
    foreground_plan: RemoteProviderForegroundSmokePlan
    profile_names: tuple[str, ...]
    startup_timeout_seconds: float


@dataclass(frozen=True)
class DgxProviderRoutePreflightPlan:
    database_url: str
    providers: tuple[DgxProviderRoutePreflightProviderPlan, ...]
    active_only: bool
    health_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_timeout_seconds: float
    fail_fast: bool


@dataclass(frozen=True)
class DgxProviderRoutePreflightProfileResult:
    profile_name: str
    route_count: int
    passed_count: int
    failed_count: int
    preflight_payload: dict[str, Any] | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.route_count > 0
            and self.failed_count == 0
            and self.passed_count == self.route_count
            and self.error is None
        )


@dataclass(frozen=True)
class DgxProviderRoutePreflightProviderResult:
    provider: str
    provider_name: str
    base_url: str
    health_url: str
    launch_command: tuple[str, ...]
    profile_names: tuple[str, ...]
    startup_timeout_seconds: float
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
    profile_results: tuple[DgxProviderRoutePreflightProfileResult, ...]
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
            and bool(self.profile_results)
            and all(result.passed for result in self.profile_results)
            and self.stopped
            and self.stop_confirmed
            and (not self.remote_stop_attempted or self.remote_stop_exit_code == 0)
            and not self.post_stop_health_reachable
            and self.error is None
        )


@dataclass(frozen=True)
class DgxProviderRoutePreflightReport:
    plan: DgxProviderRoutePreflightPlan
    results: tuple[DgxProviderRoutePreflightProviderResult, ...]
    total_elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)


def build_dgx_provider_route_preflight_plan(
    provider_names: tuple[str, ...] | None = None,
    *,
    database_url: str,
    host: str = DEFAULT_GPU_HOST,
    ssh_user: str = DEFAULT_GPU_USER,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    python_bin: str | None = None,
    provider_host: str = "0.0.0.0",
    route_host: str | None = None,
    device: str = DEFAULT_DEVICE,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    active_only: bool = True,
    fail_fast: bool = False,
) -> DgxProviderRoutePreflightPlan:
    if not database_url.strip():
        raise ValueError("database_url is required")
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

    provider_plans: list[DgxProviderRoutePreflightProviderPlan] = []
    for provider_name in _normalize_provider_names(provider_names):
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
        provider_plans.append(
            DgxProviderRoutePreflightProviderPlan(
                provider=provider_name,
                foreground_plan=foreground_plan,
                profile_names=tuple(foreground_plan.profile_names),
                startup_timeout_seconds=startup_timeout_seconds,
            )
        )

    return DgxProviderRoutePreflightPlan(
        database_url=database_url,
        providers=tuple(provider_plans),
        active_only=active_only,
        health_timeout_seconds=health_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        fail_fast=fail_fast,
    )


def run_dgx_provider_route_preflight_suite(
    plan: DgxProviderRoutePreflightPlan,
) -> DgxProviderRoutePreflightReport:
    started_at = time.monotonic()
    results: list[DgxProviderRoutePreflightProviderResult] = []
    for provider_plan in plan.providers:
        result = run_provider_route_preflight_session(
            provider_plan,
            database_url=plan.database_url,
            active_only=plan.active_only,
            health_timeout_seconds=plan.health_timeout_seconds,
            poll_interval_seconds=plan.poll_interval_seconds,
            shutdown_timeout_seconds=plan.shutdown_timeout_seconds,
            fail_fast=plan.fail_fast,
        )
        results.append(result)
        if plan.fail_fast and not result.passed:
            break

    return DgxProviderRoutePreflightReport(
        plan=plan,
        results=tuple(results),
        total_elapsed_seconds=max(0.0, time.monotonic() - started_at),
    )


def run_provider_route_preflight_session(
    provider_plan: DgxProviderRoutePreflightProviderPlan,
    *,
    database_url: str,
    active_only: bool,
    health_timeout_seconds: float,
    poll_interval_seconds: float,
    shutdown_timeout_seconds: float,
    fail_fast: bool,
) -> DgxProviderRoutePreflightProviderResult:
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
    profile_results: list[DgxProviderRoutePreflightProfileResult] = []
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
            profile_results=(),
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
                    profile_results=(),
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
                for profile_name in provider_plan.profile_names:
                    profile_result = _run_profile_preflight(
                        database_url,
                        profile_name=profile_name,
                        active_only=active_only,
                    )
                    profile_results.append(profile_result)
                    if fail_fast and not profile_result.passed:
                        break
                if any(not result.passed for result in profile_results) and error is None:
                    error = "One or more route preflight checks failed."

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
                profile_results=tuple(profile_results),
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


def _run_profile_preflight(
    database_url: str,
    *,
    profile_name: str,
    active_only: bool,
) -> DgxProviderRoutePreflightProfileResult:
    try:
        payload = run_preflight(
            database_url,
            profile_name=profile_name,
            active_only=active_only,
        )
    except Exception as exc:
        return DgxProviderRoutePreflightProfileResult(
            profile_name=profile_name,
            route_count=0,
            passed_count=0,
            failed_count=1,
            preflight_payload=None,
            error=str(exc),
        )

    route_count = _int_payload_value(payload, "route_count")
    passed_count = _int_payload_value(payload, "passed_count")
    failed_count = _int_payload_value(payload, "failed_count")
    error = None
    if route_count == 0:
        scope = "active " if active_only else ""
        error = f"No {scope}provider routes matched profile {profile_name!r}."
    elif failed_count > 0:
        error = "One or more provider route contracts failed."

    return DgxProviderRoutePreflightProfileResult(
        profile_name=profile_name,
        route_count=route_count,
        passed_count=passed_count,
        failed_count=failed_count,
        preflight_payload=dict(payload),
        error=error,
    )


def _int_payload_value(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    return int(value) if isinstance(value, int | float | str) and str(value).strip() else 0


def _provider_result(
    provider_plan: DgxProviderRoutePreflightProviderPlan,
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
    profile_results: tuple[DgxProviderRoutePreflightProfileResult, ...],
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
) -> DgxProviderRoutePreflightProviderResult:
    foreground_plan = provider_plan.foreground_plan
    return DgxProviderRoutePreflightProviderResult(
        provider=provider_plan.provider,
        provider_name=foreground_plan.provider_name,
        base_url=foreground_plan.base_url,
        health_url=foreground_plan.health_url,
        launch_command=launch_command,
        profile_names=provider_plan.profile_names,
        startup_timeout_seconds=provider_plan.startup_timeout_seconds,
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
        profile_results=profile_results,
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


def _redact_database_url(database_url: str) -> str:
    try:
        parts = urlsplit(database_url)
    except ValueError:
        return "<configured>"
    if not parts.scheme or not parts.netloc:
        return "<configured>"
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, hostinfo = netloc.rsplit("@", 1)
        if ":" in userinfo:
            username, _password = userinfo.split(":", 1)
            userinfo = f"{username}:***"
        else:
            userinfo = "***"
        netloc = f"{userinfo}@{hostinfo}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _plan_payload(plan: DgxProviderRoutePreflightPlan) -> dict[str, Any]:
    return {
        "database_url": _redact_database_url(plan.database_url),
        "providers": [_provider_plan_payload(provider) for provider in plan.providers],
        "active_only": plan.active_only,
        "health_timeout_seconds": plan.health_timeout_seconds,
        "poll_interval_seconds": plan.poll_interval_seconds,
        "shutdown_timeout_seconds": plan.shutdown_timeout_seconds,
        "fail_fast": plan.fail_fast,
    }


def _provider_plan_payload(
    provider_plan: DgxProviderRoutePreflightProviderPlan,
) -> dict[str, Any]:
    return {
        "provider": provider_plan.provider,
        "foreground_plan": asdict(provider_plan.foreground_plan),
        "profile_names": list(provider_plan.profile_names),
        "startup_timeout_seconds": provider_plan.startup_timeout_seconds,
    }


def _profile_result_payload(result: DgxProviderRoutePreflightProfileResult) -> dict[str, Any]:
    return {**asdict(result), "passed": result.passed}


def _result_payload(result: DgxProviderRoutePreflightProviderResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["passed"] = result.passed
    payload["profile_results"] = [
        _profile_result_payload(profile_result) for profile_result in result.profile_results
    ]
    return payload


def _report_payload(report: DgxProviderRoutePreflightReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "results": [_result_payload(result) for result in report.results],
        "total_elapsed_seconds": report.total_elapsed_seconds,
    }


def _print_human_report(report: DgxProviderRoutePreflightReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"DGX provider route preflight verification: {status}")
    print(f"- providers: {len(report.results)}/{len(report.plan.providers)} executed")
    print(f"- total_elapsed_seconds: {report.total_elapsed_seconds:.2f}")
    for result in report.results:
        print(
            f"- {result.provider}: "
            f"passed={result.passed} "
            f"health_ok={result.health_ok} "
            f"profiles={len(result.profile_results)}/{len(result.profile_names)} "
            f"post_stop_health_reachable={result.post_stop_health_reachable}"
        )
        for profile_result in result.profile_results:
            print(
                f"  - {profile_result.profile_name}: "
                f"passed={profile_result.passed} "
                f"routes={profile_result.route_count} "
                f"failed={profile_result.failed_count}"
            )
        if result.error:
            print(f"  - error: {result.error}")


def write_markdown_report(
    report: DgxProviderRoutePreflightReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: DgxProviderRoutePreflightReport) -> str:
    lines = [
        "# DGX Provider Route Preflight Verification Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `database_url`: `{_redact_database_url(report.plan.database_url)}`",
        f"- `active_only`: `{str(report.plan.active_only).lower()}`",
        f"- `total_elapsed_seconds`: `{report.total_elapsed_seconds:.2f}`",
        f"- providers executed: `{len(report.results)}`",
        "",
        "## Provider Results",
        "",
        "| Provider | Passed | Base URL | Health | Profiles | Error |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in report.results:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result.provider}`",
                    f"`{str(result.passed).lower()}`",
                    f"`{result.base_url}`",
                    f"`{str(result.health_ok).lower()}`",
                    f"`{len(result.profile_results)}/{len(result.profile_names)}`",
                    f"`{result.error or ''}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Profile Preflight Results",
            "",
            (
                "| Provider | Profile | Passed | Routes | Passed | Failed | "
                "Contract Snapshots | Health Snapshots | Error |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for result in report.results:
        for profile_result in result.profile_results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{result.provider}`",
                        f"`{profile_result.profile_name}`",
                        f"`{str(profile_result.passed).lower()}`",
                        f"`{profile_result.route_count}`",
                        f"`{profile_result.passed_count}`",
                        f"`{profile_result.failed_count}`",
                        f"`{_snapshot_ids(profile_result, 'contract_snapshot_id')}`",
                        f"`{_snapshot_ids(profile_result, 'health_snapshot_id')}`",
                        f"`{profile_result.error or ''}`",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _snapshot_ids(
    profile_result: DgxProviderRoutePreflightProfileResult,
    key: str,
) -> str:
    payload = profile_result.preflight_payload or {}
    results = payload.get("results")
    if not isinstance(results, list):
        return ""
    values = [
        str(row.get(key)) for row in results if isinstance(row, dict) and row.get(key) is not None
    ]
    return ", ".join(values)


def _build_arg_parser() -> argparse.ArgumentParser:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description=(
            "Launch DGX remote embedding providers sequentially and run profile-scoped "
            "provider route preflight checks against the NeX-PCX database."
        ),
    )
    parser.add_argument("--provider", choices=preset_names, action="append", default=[])
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--include-inactive", action="store_true")
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
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    try:
        plan = build_dgx_provider_route_preflight_plan(
            provider_names=tuple(args.provider) or None,
            database_url=database_url,
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            models_dir=args.models_dir,
            python_bin=args.python_bin,
            provider_host=args.provider_host,
            route_host=args.route_host,
            device=args.device,
            startup_timeout_seconds=args.startup_timeout_seconds,
            health_timeout_seconds=args.health_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
            active_only=not args.include_inactive,
            fail_fast=args.fail_fast,
        )
    except (InvalidEmbeddingProviderPresetError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    report = run_dgx_provider_route_preflight_suite(plan)
    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
