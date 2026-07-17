"""App-host managed service restart validation."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

APP_HOST_RESTART_CHECK_PASSED = "passed"
APP_HOST_RESTART_CHECK_WARNING = "warning"
APP_HOST_RESTART_CHECK_FAILED = "failed"
APP_HOST_RESTART_CHECK_SKIPPED = "skipped"

APP_HOST_RESTART_STATUS_READY = "ready"
APP_HOST_RESTART_STATUS_WARNING = "warning"
APP_HOST_RESTART_STATUS_BLOCKED = "blocked"

DEFAULT_WEB_SERVICE_NAME = "nex-pcx-web"
DEFAULT_PIPELINE_WORKER_SERVICE_NAME = "nex-pcx-pipeline-worker"
DEFAULT_EMBEDDING_WORKER_SERVICE_NAME = "nex-pcx-embedding-worker"
DEFAULT_SERVICE_NAMES = (
    DEFAULT_WEB_SERVICE_NAME,
    DEFAULT_PIPELINE_WORKER_SERVICE_NAME,
    DEFAULT_EMBEDDING_WORKER_SERVICE_NAME,
)
DEFAULT_EXPECTED_APP_NAME = "NeX_PCX"
VALID_SYSTEMD_SCOPES = ("user", "system")


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class HttpJsonResult:
    status_code: int
    payload: dict[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True)
class AppHostServiceRestartValidationOptions:
    scope: str = "user"
    service_names: tuple[str, ...] = DEFAULT_SERVICE_NAMES
    web_service_name: str = DEFAULT_WEB_SERVICE_NAME
    app_base_url: str | None = None
    expected_app_name: str = DEFAULT_EXPECTED_APP_NAME
    restart_web: bool = False
    health_timeout_seconds: float = 5.0
    systemctl_path: str = "systemctl"


@dataclass(frozen=True)
class AppHostServiceRestartValidationCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class AppHostServiceRestartValidationReport:
    status: str
    checked_at: datetime
    scope: str
    checks: tuple[AppHostServiceRestartValidationCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(APP_HOST_RESTART_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(APP_HOST_RESTART_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(APP_HOST_RESTART_CHECK_FAILED)

    @property
    def skipped_count(self) -> int:
        return self._count_status(APP_HOST_RESTART_CHECK_SKIPPED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


CommandRunner = Callable[[tuple[str, ...]], CommandResult]
HttpJsonGetter = Callable[[str, float], HttpJsonResult]


def build_app_host_service_restart_validation_report(
    options: AppHostServiceRestartValidationOptions | None = None,
    *,
    checked_at: datetime | None = None,
    command_runner: CommandRunner | None = None,
    http_json_getter: HttpJsonGetter | None = None,
) -> AppHostServiceRestartValidationReport:
    selected_options = _validate_options(options or AppHostServiceRestartValidationOptions())
    selected_command_runner = command_runner or _run_command
    selected_http_json_getter = http_json_getter or _default_http_json_getter
    checks = [
        _systemctl_version_check(
            selected_options,
            command_runner=selected_command_runner,
        ),
        _systemd_scope_check(
            selected_options,
            command_runner=selected_command_runner,
        ),
    ]

    for service_name in selected_options.service_names:
        checks.append(
            _service_status_check(
                selected_options,
                service_name=service_name,
                command_runner=selected_command_runner,
            )
        )

    if selected_options.restart_web:
        checks.append(
            _web_restart_check(
                selected_options,
                command_runner=selected_command_runner,
            )
        )

    if selected_options.app_base_url:
        checks.append(
            _app_health_check(
                selected_options,
                http_json_getter=selected_http_json_getter,
            )
        )
        checks.append(
            _app_identity_check(
                selected_options,
                http_json_getter=selected_http_json_getter,
            )
        )
    elif selected_options.restart_web:
        checks.append(
            AppHostServiceRestartValidationCheck(
                code="app_url_after_restart",
                status=APP_HOST_RESTART_CHECK_FAILED,
                detail="--app-url is required when --restart-web is used.",
                metadata={"app_url_configured": False},
            )
        )

    report_checks = tuple(checks)
    return AppHostServiceRestartValidationReport(
        status=_overall_status(report_checks),
        checked_at=checked_at or datetime.now(UTC),
        scope=selected_options.scope,
        checks=report_checks,
    )


def app_host_service_restart_validation_report_payload(
    report: AppHostServiceRestartValidationReport,
) -> dict[str, object]:
    return {
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": report.scope,
        "check_count": report.check_count,
        "passed_count": report.passed_count,
        "warning_count": report.warning_count,
        "failed_count": report.failed_count,
        "skipped_count": report.skipped_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "metadata": dict(check.metadata or {}),
            }
            for check in report.checks
        ],
    }


def render_app_host_service_restart_validation_markdown(
    payload: dict[str, object],
) -> str:
    lines = [
        "# App Host Service Restart Validation",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Systemd Scope: `{_text(payload.get('scope'))}`",
        f"- Checks: {_text(payload.get('check_count'))}",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
        f"- Skipped: {_text(payload.get('skipped_count'))}",
        "",
        "## Checks",
        "",
        "| Code | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} |"
        )

    lines.extend(["", "## Operator Notes", ""])
    if payload.get("status") == APP_HOST_RESTART_STATUS_BLOCKED:
        lines.append(
            "- Resolve failed checks before relying on automatic restart after process failure, "
            "logout, or reboot."
        )
    else:
        lines.append("- Keep this report with the go-live handoff evidence.")
    lines.append("- Use `--restart-web` only during a controlled maintenance window.")
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _validate_options(
    options: AppHostServiceRestartValidationOptions,
) -> AppHostServiceRestartValidationOptions:
    selected_scope = options.scope.strip()
    if selected_scope not in VALID_SYSTEMD_SCOPES:
        raise ValueError("scope must be one of: user, system")
    service_names = tuple(name.strip() for name in options.service_names if name.strip())
    if not service_names:
        raise ValueError("at least one service name is required")
    web_service_name = options.web_service_name.strip()
    if not web_service_name:
        raise ValueError("web_service_name is required")
    expected_app_name = options.expected_app_name.strip()
    if not expected_app_name:
        raise ValueError("expected_app_name is required")
    systemctl_path = options.systemctl_path.strip()
    if not systemctl_path:
        raise ValueError("systemctl_path is required")
    if options.health_timeout_seconds <= 0:
        raise ValueError("health_timeout_seconds must be greater than zero")
    return AppHostServiceRestartValidationOptions(
        scope=selected_scope,
        service_names=service_names,
        web_service_name=web_service_name,
        app_base_url=options.app_base_url.rstrip("/") if options.app_base_url else None,
        expected_app_name=expected_app_name,
        restart_web=options.restart_web,
        health_timeout_seconds=options.health_timeout_seconds,
        systemctl_path=systemctl_path,
    )


def _systemctl_version_check(
    options: AppHostServiceRestartValidationOptions,
    *,
    command_runner: CommandRunner,
) -> AppHostServiceRestartValidationCheck:
    command = (options.systemctl_path, "--version")
    result = command_runner(command)
    if result.returncode != 0:
        return _failed_command_check(
            code="systemctl_available",
            detail_prefix="systemctl is not available",
            result=result,
        )
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else "systemctl"
    return AppHostServiceRestartValidationCheck(
        code="systemctl_available",
        status=APP_HOST_RESTART_CHECK_PASSED,
        detail=f"{first_line} is available.",
        metadata=_command_metadata(result),
    )


def _systemd_scope_check(
    options: AppHostServiceRestartValidationOptions,
    *,
    command_runner: CommandRunner,
) -> AppHostServiceRestartValidationCheck:
    result = command_runner(
        _systemctl_command(
            options,
            "list-unit-files",
            f"{options.web_service_name}.service",
            "--no-pager",
        )
    )
    if result.returncode != 0:
        return _failed_command_check(
            code="systemd_scope",
            detail_prefix=f"{options.scope} systemd scope is not reachable",
            result=result,
        )
    return AppHostServiceRestartValidationCheck(
        code="systemd_scope",
        status=APP_HOST_RESTART_CHECK_PASSED,
        detail=f"{options.scope} systemd scope is reachable.",
        metadata=_command_metadata(result),
    )


def _service_status_check(
    options: AppHostServiceRestartValidationOptions,
    *,
    service_name: str,
    command_runner: CommandRunner,
) -> AppHostServiceRestartValidationCheck:
    unit_name = _unit_name(service_name)
    result = command_runner(
        _systemctl_command(
            options,
            "show",
            unit_name,
            "--property=LoadState,ActiveState,SubState,UnitFileState,FragmentPath,Restart,NRestarts",
            "--no-pager",
        )
    )
    if result.returncode != 0:
        return _failed_command_check(
            code=f"service_status:{service_name}",
            detail_prefix=f"{unit_name} status command failed",
            result=result,
        )

    properties = _parse_systemctl_properties(result.stdout)
    metadata = {**_command_metadata(result), "properties": properties}
    if properties.get("LoadState") != "loaded":
        return AppHostServiceRestartValidationCheck(
            code=f"service_status:{service_name}",
            status=APP_HOST_RESTART_CHECK_FAILED,
            detail=f"{unit_name} is not loaded.",
            metadata=metadata,
        )
    if properties.get("ActiveState") != "active":
        active_state = properties.get("ActiveState") or "unknown"
        sub_state = properties.get("SubState") or "unknown"
        return AppHostServiceRestartValidationCheck(
            code=f"service_status:{service_name}",
            status=APP_HOST_RESTART_CHECK_FAILED,
            detail=f"{unit_name} is {active_state}/{sub_state}.",
            metadata=metadata,
        )

    restart_policy = properties.get("Restart") or ""
    if restart_policy in {"", "no"}:
        return AppHostServiceRestartValidationCheck(
            code=f"service_status:{service_name}",
            status=APP_HOST_RESTART_CHECK_WARNING,
            detail=f"{unit_name} is active but has no restart policy.",
            metadata=metadata,
        )

    return AppHostServiceRestartValidationCheck(
        code=f"service_status:{service_name}",
        status=APP_HOST_RESTART_CHECK_PASSED,
        detail=f"{unit_name} is active with Restart={restart_policy}.",
        metadata=metadata,
    )


def _web_restart_check(
    options: AppHostServiceRestartValidationOptions,
    *,
    command_runner: CommandRunner,
) -> AppHostServiceRestartValidationCheck:
    unit_name = _unit_name(options.web_service_name)
    result = command_runner(_systemctl_command(options, "restart", unit_name))
    if result.returncode != 0:
        return _failed_command_check(
            code="web_service_restart",
            detail_prefix=f"{unit_name} restart failed",
            result=result,
        )
    return AppHostServiceRestartValidationCheck(
        code="web_service_restart",
        status=APP_HOST_RESTART_CHECK_PASSED,
        detail=f"{unit_name} restart command completed.",
        metadata=_command_metadata(result),
    )


def _app_health_check(
    options: AppHostServiceRestartValidationOptions,
    *,
    http_json_getter: HttpJsonGetter,
) -> AppHostServiceRestartValidationCheck:
    url = f"{options.app_base_url}/healthz"
    result = http_json_getter(url, options.health_timeout_seconds)
    if result.error:
        return AppHostServiceRestartValidationCheck(
            code="app_health_after_restart",
            status=APP_HOST_RESTART_CHECK_FAILED,
            detail=f"App health request failed: {result.error}",
            metadata={"url": url, "status_code": result.status_code},
        )
    payload = result.payload or {}
    if result.status_code != 200 or payload.get("status") != "ok":
        return AppHostServiceRestartValidationCheck(
            code="app_health_after_restart",
            status=APP_HOST_RESTART_CHECK_FAILED,
            detail="App health endpoint did not return status=ok.",
            metadata={"url": url, "status_code": result.status_code, "payload": payload},
        )
    return AppHostServiceRestartValidationCheck(
        code="app_health_after_restart",
        status=APP_HOST_RESTART_CHECK_PASSED,
        detail="App health endpoint returned status=ok.",
        metadata={"url": url, "status_code": result.status_code},
    )


def _app_identity_check(
    options: AppHostServiceRestartValidationOptions,
    *,
    http_json_getter: HttpJsonGetter,
) -> AppHostServiceRestartValidationCheck:
    url = f"{options.app_base_url}/openapi.json"
    result = http_json_getter(url, options.health_timeout_seconds)
    if result.error:
        return AppHostServiceRestartValidationCheck(
            code="app_identity_after_restart",
            status=APP_HOST_RESTART_CHECK_FAILED,
            detail=f"App identity request failed: {result.error}",
            metadata={"url": url, "status_code": result.status_code},
        )
    payload = result.payload or {}
    title = _dict(payload.get("info")).get("title")
    if title != options.expected_app_name:
        return AppHostServiceRestartValidationCheck(
            code="app_identity_after_restart",
            status=APP_HOST_RESTART_CHECK_FAILED,
            detail=(
                f"OpenAPI title is {_text(title)!r}; expected " f"{options.expected_app_name!r}."
            ),
            metadata={"url": url, "status_code": result.status_code, "title": title},
        )
    return AppHostServiceRestartValidationCheck(
        code="app_identity_after_restart",
        status=APP_HOST_RESTART_CHECK_PASSED,
        detail=f"OpenAPI title matches {options.expected_app_name}.",
        metadata={"url": url, "status_code": result.status_code, "title": title},
    )


def _systemctl_command(
    options: AppHostServiceRestartValidationOptions,
    *parts: str,
) -> tuple[str, ...]:
    command = [options.systemctl_path]
    if options.scope == "user":
        command.append("--user")
    command.extend(parts)
    return tuple(command)


def _unit_name(service_name: str) -> str:
    return service_name if service_name.endswith(".service") else f"{service_name}.service"


def _parse_systemctl_properties(stdout: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def _failed_command_check(
    *,
    code: str,
    detail_prefix: str,
    result: CommandResult,
) -> AppHostServiceRestartValidationCheck:
    error = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
    return AppHostServiceRestartValidationCheck(
        code=code,
        status=APP_HOST_RESTART_CHECK_FAILED,
        detail=f"{detail_prefix}: {error}",
        metadata=_command_metadata(result),
    )


def _command_metadata(result: CommandResult) -> dict[str, object]:
    return {
        "command": list(result.command),
        "returncode": result.returncode,
        "stdout": _truncate(result.stdout.strip()),
        "stderr": _truncate(result.stderr.strip()),
    }


def _run_command(command: tuple[str, ...]) -> CommandResult:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return CommandResult(command=command, returncode=127, stderr=str(exc))
    return CommandResult(
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _default_http_json_getter(url: str, timeout_seconds: float) -> HttpJsonResult:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)
            return HttpJsonResult(
                status_code=int(response.status),
                payload=payload if isinstance(payload, dict) else {},
            )
    except urllib.error.HTTPError as exc:
        return HttpJsonResult(status_code=int(exc.code), error=str(exc))
    except Exception as exc:
        return HttpJsonResult(status_code=0, error=str(exc))


def _overall_status(checks: tuple[AppHostServiceRestartValidationCheck, ...]) -> str:
    if any(check.status == APP_HOST_RESTART_CHECK_FAILED for check in checks):
        return APP_HOST_RESTART_STATUS_BLOCKED
    if any(check.status == APP_HOST_RESTART_CHECK_WARNING for check in checks):
        return APP_HOST_RESTART_STATUS_WARNING
    return APP_HOST_RESTART_STATUS_READY


def _truncate(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
