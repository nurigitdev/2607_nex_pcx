"""Foreground operation validation for app-host go-live checks."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

FOREGROUND_CHECK_PASSED = "passed"
FOREGROUND_CHECK_WARNING = "warning"
FOREGROUND_CHECK_FAILED = "failed"

FOREGROUND_STATUS_READY = "ready"
FOREGROUND_STATUS_WARNING = "warning"
FOREGROUND_STATUS_BLOCKED = "blocked"

DEFAULT_EXPECTED_APP_NAME = "NeX_PCX"
DEFAULT_WEB_COMMAND = (
    "./.venv/bin/python",
    "-m",
    "uvicorn",
    "app.main:create_app",
    "--factory",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
)
DEFAULT_PIPELINE_WORKER_COMMAND = (
    "./.venv/bin/python",
    "scripts/process_pipeline_job.py",
    "--help",
)
DEFAULT_EMBEDDING_WORKER_COMMAND = (
    "./.venv/bin/python",
    "scripts/process_embedding_job.py",
    "--help",
)


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
class ForegroundOperationsValidationOptions:
    app_base_url: str = "http://127.0.0.1:8000"
    expected_app_name: str = DEFAULT_EXPECTED_APP_NAME
    acknowledge_no_auto_restart: bool = False
    health_timeout_seconds: float = 5.0
    pipeline_worker_command: tuple[str, ...] = DEFAULT_PIPELINE_WORKER_COMMAND
    embedding_worker_command: tuple[str, ...] = DEFAULT_EMBEDDING_WORKER_COMMAND
    web_command: tuple[str, ...] = DEFAULT_WEB_COMMAND


@dataclass(frozen=True)
class ForegroundOperationsValidationCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ForegroundOperationsValidationReport:
    status: str
    checked_at: datetime
    app_base_url: str
    checks: tuple[ForegroundOperationsValidationCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(FOREGROUND_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(FOREGROUND_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(FOREGROUND_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


CommandRunner = Callable[[tuple[str, ...]], CommandResult]
HttpJsonGetter = Callable[[str, float], HttpJsonResult]


def build_foreground_operations_validation_report(
    options: ForegroundOperationsValidationOptions | None = None,
    *,
    checked_at: datetime | None = None,
    command_runner: CommandRunner | None = None,
    http_json_getter: HttpJsonGetter | None = None,
) -> ForegroundOperationsValidationReport:
    selected_options = _validate_options(options or ForegroundOperationsValidationOptions())
    selected_command_runner = command_runner or _run_command
    selected_http_json_getter = http_json_getter or _default_http_json_getter
    checks = (
        _foreground_mode_acknowledgement_check(selected_options),
        _foreground_web_command_check(selected_options),
        _app_health_check(selected_options, http_json_getter=selected_http_json_getter),
        _app_identity_check(selected_options, http_json_getter=selected_http_json_getter),
        _worker_cli_check(
            code="pipeline_worker_cli",
            command=selected_options.pipeline_worker_command,
            command_runner=selected_command_runner,
        ),
        _worker_cli_check(
            code="embedding_worker_cli",
            command=selected_options.embedding_worker_command,
            command_runner=selected_command_runner,
        ),
    )
    return ForegroundOperationsValidationReport(
        status=_overall_status(checks),
        checked_at=checked_at or datetime.now(UTC),
        app_base_url=selected_options.app_base_url,
        checks=checks,
    )


def foreground_operations_validation_report_payload(
    report: ForegroundOperationsValidationReport,
) -> dict[str, object]:
    return {
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "app_base_url": report.app_base_url,
        "check_count": report.check_count,
        "passed_count": report.passed_count,
        "warning_count": report.warning_count,
        "failed_count": report.failed_count,
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


def render_foreground_operations_validation_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Foreground Operations Validation",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- App URL: `{_text(payload.get('app_base_url'))}`",
        f"- Checks: {_text(payload.get('check_count'))}",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
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
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- Foreground operation is acceptable for a controlled pre-CX run when an operator "
            "keeps the terminal/session alive.",
            "- This mode does not provide automatic restart after process failure, logout, "
            "or reboot.",
            "- Promote to systemd or another supervisor when continuous unattended operation "
            "is required.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _validate_options(
    options: ForegroundOperationsValidationOptions,
) -> ForegroundOperationsValidationOptions:
    app_base_url = options.app_base_url.rstrip("/")
    if not app_base_url.startswith(("http://", "https://")):
        raise ValueError("app_base_url must be an absolute HTTP URL")
    expected_app_name = options.expected_app_name.strip()
    if not expected_app_name:
        raise ValueError("expected_app_name is required")
    if options.health_timeout_seconds <= 0:
        raise ValueError("health_timeout_seconds must be greater than zero")
    pipeline_worker_command = _validate_command(
        options.pipeline_worker_command,
        name="pipeline_worker_command",
    )
    embedding_worker_command = _validate_command(
        options.embedding_worker_command,
        name="embedding_worker_command",
    )
    web_command = _validate_command(options.web_command, name="web_command")
    return ForegroundOperationsValidationOptions(
        app_base_url=app_base_url,
        expected_app_name=expected_app_name,
        acknowledge_no_auto_restart=options.acknowledge_no_auto_restart,
        health_timeout_seconds=options.health_timeout_seconds,
        pipeline_worker_command=pipeline_worker_command,
        embedding_worker_command=embedding_worker_command,
        web_command=web_command,
    )


def _validate_command(command: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    selected_command = tuple(part.strip() for part in command if part.strip())
    if not selected_command:
        raise ValueError(f"{name} is required")
    return selected_command


def _foreground_mode_acknowledgement_check(
    options: ForegroundOperationsValidationOptions,
) -> ForegroundOperationsValidationCheck:
    if not options.acknowledge_no_auto_restart:
        return ForegroundOperationsValidationCheck(
            code="foreground_no_auto_restart_ack",
            status=FOREGROUND_CHECK_FAILED,
            detail=(
                "Foreground operation was not acknowledged. Pass "
                "--acknowledge-no-auto-restart to accept manual process supervision."
            ),
            metadata={"acknowledged": False},
        )
    return ForegroundOperationsValidationCheck(
        code="foreground_no_auto_restart_ack",
        status=FOREGROUND_CHECK_WARNING,
        detail="Foreground mode accepted with no automatic restart guarantee.",
        metadata={"acknowledged": True},
    )


def _foreground_web_command_check(
    options: ForegroundOperationsValidationOptions,
) -> ForegroundOperationsValidationCheck:
    return ForegroundOperationsValidationCheck(
        code="foreground_web_command",
        status=FOREGROUND_CHECK_PASSED,
        detail="Foreground web launch command is documented for operator use.",
        metadata={"command": list(options.web_command)},
    )


def _app_health_check(
    options: ForegroundOperationsValidationOptions,
    *,
    http_json_getter: HttpJsonGetter,
) -> ForegroundOperationsValidationCheck:
    url = f"{options.app_base_url}/healthz"
    result = http_json_getter(url, options.health_timeout_seconds)
    if result.error:
        return ForegroundOperationsValidationCheck(
            code="app_healthz",
            status=FOREGROUND_CHECK_FAILED,
            detail=f"App health request failed: {result.error}",
            metadata={"url": url, "status_code": result.status_code},
        )
    payload = result.payload or {}
    if result.status_code != 200 or payload.get("status") != "ok":
        return ForegroundOperationsValidationCheck(
            code="app_healthz",
            status=FOREGROUND_CHECK_FAILED,
            detail="App health endpoint did not return status=ok.",
            metadata={"url": url, "status_code": result.status_code, "payload": payload},
        )
    return ForegroundOperationsValidationCheck(
        code="app_healthz",
        status=FOREGROUND_CHECK_PASSED,
        detail="App health endpoint returned status=ok.",
        metadata={"url": url, "status_code": result.status_code},
    )


def _app_identity_check(
    options: ForegroundOperationsValidationOptions,
    *,
    http_json_getter: HttpJsonGetter,
) -> ForegroundOperationsValidationCheck:
    url = f"{options.app_base_url}/openapi.json"
    result = http_json_getter(url, options.health_timeout_seconds)
    if result.error:
        return ForegroundOperationsValidationCheck(
            code="app_identity",
            status=FOREGROUND_CHECK_FAILED,
            detail=f"App identity request failed: {result.error}",
            metadata={"url": url, "status_code": result.status_code},
        )
    payload = result.payload or {}
    title = _dict(payload.get("info")).get("title")
    if title != options.expected_app_name:
        return ForegroundOperationsValidationCheck(
            code="app_identity",
            status=FOREGROUND_CHECK_FAILED,
            detail=f"OpenAPI title is {_text(title)!r}; expected {options.expected_app_name!r}.",
            metadata={"url": url, "status_code": result.status_code, "title": title},
        )
    return ForegroundOperationsValidationCheck(
        code="app_identity",
        status=FOREGROUND_CHECK_PASSED,
        detail=f"OpenAPI title matches {options.expected_app_name}.",
        metadata={"url": url, "status_code": result.status_code, "title": title},
    )


def _worker_cli_check(
    *,
    code: str,
    command: tuple[str, ...],
    command_runner: CommandRunner,
) -> ForegroundOperationsValidationCheck:
    result = command_runner(command)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return ForegroundOperationsValidationCheck(
            code=code,
            status=FOREGROUND_CHECK_FAILED,
            detail=f"Worker CLI check failed: {error}",
            metadata=_command_metadata(result),
        )
    return ForegroundOperationsValidationCheck(
        code=code,
        status=FOREGROUND_CHECK_PASSED,
        detail="Worker CLI import and argument parsing succeeded.",
        metadata=_command_metadata(result),
    )


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


def _overall_status(checks: tuple[ForegroundOperationsValidationCheck, ...]) -> str:
    if any(check.status == FOREGROUND_CHECK_FAILED for check in checks):
        return FOREGROUND_STATUS_BLOCKED
    if any(check.status == FOREGROUND_CHECK_WARNING for check in checks):
        return FOREGROUND_STATUS_WARNING
    return FOREGROUND_STATUS_READY


def _command_metadata(result: CommandResult) -> dict[str, object]:
    return {
        "command": list(result.command),
        "returncode": result.returncode,
        "stdout": _truncate(result.stdout.strip()),
        "stderr": _truncate(result.stderr.strip()),
    }


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
