from datetime import UTC, datetime

import pytest

from app.core.app_host_service_restart_validation import (
    APP_HOST_RESTART_STATUS_BLOCKED,
    APP_HOST_RESTART_STATUS_READY,
    APP_HOST_RESTART_STATUS_WARNING,
    AppHostServiceRestartValidationOptions,
    CommandResult,
    HttpJsonResult,
    app_host_service_restart_validation_report_payload,
    build_app_host_service_restart_validation_report,
    render_app_host_service_restart_validation_markdown,
)


def test_app_host_service_restart_validation_ready_with_app_identity() -> None:
    command_runner = make_command_runner(
        {
            ("systemctl", "--version"): CommandResult(
                command=("systemctl", "--version"),
                returncode=0,
                stdout="systemd 255\n",
            ),
            (
                "systemctl",
                "--user",
                "list-unit-files",
                "nex-pcx-web.service",
                "--no-pager",
            ): CommandResult(
                command=(),
                returncode=0,
                stdout="nex-pcx-web.service enabled\n",
            ),
            service_show_command("nex-pcx-web.service"): service_show_result(restart="on-failure"),
            service_show_command("nex-pcx-pipeline-worker.service"): service_show_result(
                restart="always"
            ),
            service_show_command("nex-pcx-embedding-worker.service"): service_show_result(
                restart="always"
            ),
        }
    )

    report = build_app_host_service_restart_validation_report(
        AppHostServiceRestartValidationOptions(app_base_url="http://127.0.0.1:8000/"),
        checked_at=datetime(2026, 7, 17, 1, 2, 3, tzinfo=UTC),
        command_runner=command_runner,
        http_json_getter=healthy_app_getter,
    )
    payload = app_host_service_restart_validation_report_payload(report)

    assert report.status == APP_HOST_RESTART_STATUS_READY
    assert payload["checked_at_label"] == "2026-07-17 01:02:03"
    assert payload["passed_count"] == 7
    assert payload["failed_count"] == 0


def test_app_host_service_restart_validation_blocks_when_user_bus_unavailable() -> None:
    command_runner = make_command_runner(
        {
            ("systemctl", "--version"): CommandResult(
                command=("systemctl", "--version"),
                returncode=0,
                stdout="systemd 255\n",
            )
        },
        default=CommandResult(
            command=(),
            returncode=1,
            stderr="Failed to connect to bus: No data available",
        ),
    )

    report = build_app_host_service_restart_validation_report(
        AppHostServiceRestartValidationOptions(service_names=("nex-pcx-web",)),
        command_runner=command_runner,
    )
    payload = app_host_service_restart_validation_report_payload(report)
    markdown = render_app_host_service_restart_validation_markdown(payload)

    assert report.status == APP_HOST_RESTART_STATUS_BLOCKED
    assert payload["failed_count"] == 2
    assert "Failed to connect to bus" in payload["checks"][1]["detail"]
    assert "Resolve failed checks" in markdown


def test_app_host_service_restart_validation_can_restart_web_service() -> None:
    captured_commands: list[tuple[str, ...]] = []

    def command_runner(command: tuple[str, ...]) -> CommandResult:
        captured_commands.append(command)
        if command == service_show_command("nex-pcx-web.service"):
            return service_show_result(restart="on-failure")
        return CommandResult(command=command, returncode=0, stdout="ok\n")

    report = build_app_host_service_restart_validation_report(
        AppHostServiceRestartValidationOptions(
            service_names=("nex-pcx-web",),
            app_base_url="http://127.0.0.1:8000",
            restart_web=True,
        ),
        command_runner=command_runner,
        http_json_getter=healthy_app_getter,
    )

    assert report.status == APP_HOST_RESTART_STATUS_READY
    assert ("systemctl", "--user", "restart", "nex-pcx-web.service") in captured_commands


def test_app_host_service_restart_validation_warns_without_restart_policy() -> None:
    command_runner = make_command_runner(
        {
            ("systemctl", "--version"): CommandResult(command=(), returncode=0),
            (
                "systemctl",
                "--user",
                "list-unit-files",
                "nex-pcx-web.service",
                "--no-pager",
            ): CommandResult(command=(), returncode=0),
            service_show_command("nex-pcx-web.service"): service_show_result(restart="no"),
        }
    )

    report = build_app_host_service_restart_validation_report(
        AppHostServiceRestartValidationOptions(service_names=("nex-pcx-web",)),
        command_runner=command_runner,
    )

    assert report.status == APP_HOST_RESTART_STATUS_WARNING
    assert report.warning_count == 1


def test_app_host_service_restart_validation_fails_wrong_app_identity() -> None:
    def wrong_identity_getter(url: str, timeout_seconds: float) -> HttpJsonResult:
        if url.endswith("/healthz"):
            return HttpJsonResult(status_code=200, payload={"status": "ok"})
        return HttpJsonResult(status_code=200, payload={"info": {"title": "Other"}})

    command_runner = make_command_runner(
        {
            ("systemctl", "--version"): CommandResult(command=(), returncode=0),
            (
                "systemctl",
                "--user",
                "list-unit-files",
                "nex-pcx-web.service",
                "--no-pager",
            ): CommandResult(command=(), returncode=0),
            service_show_command("nex-pcx-web.service"): service_show_result(restart="on-failure"),
        }
    )

    report = build_app_host_service_restart_validation_report(
        AppHostServiceRestartValidationOptions(
            service_names=("nex-pcx-web",),
            app_base_url="http://127.0.0.1:8000",
        ),
        command_runner=command_runner,
        http_json_getter=wrong_identity_getter,
    )

    assert report.status == APP_HOST_RESTART_STATUS_BLOCKED
    assert report.checks[-1].code == "app_identity_after_restart"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (AppHostServiceRestartValidationOptions(scope="bad"), "scope must be one of"),
        (
            AppHostServiceRestartValidationOptions(service_names=()),
            "at least one service name",
        ),
        (
            AppHostServiceRestartValidationOptions(health_timeout_seconds=0),
            "health_timeout_seconds",
        ),
    ],
)
def test_app_host_service_restart_validation_rejects_invalid_options(
    options: AppHostServiceRestartValidationOptions,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_app_host_service_restart_validation_report(options)


def make_command_runner(
    responses: dict[tuple[str, ...], CommandResult],
    *,
    default: CommandResult | None = None,
):
    def command_runner(command: tuple[str, ...]) -> CommandResult:
        if command in responses:
            result = responses[command]
            return CommandResult(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        fallback = default or CommandResult(command=command, returncode=0)
        return CommandResult(
            command=command,
            returncode=fallback.returncode,
            stdout=fallback.stdout,
            stderr=fallback.stderr,
        )

    return command_runner


def service_show_command(service_name: str) -> tuple[str, ...]:
    return (
        "systemctl",
        "--user",
        "show",
        service_name,
        "--property=LoadState,ActiveState,SubState,UnitFileState,FragmentPath,Restart,NRestarts",
        "--no-pager",
    )


def service_show_result(*, restart: str) -> CommandResult:
    return CommandResult(
        command=(),
        returncode=0,
        stdout=(
            "LoadState=loaded\n"
            "ActiveState=active\n"
            "SubState=running\n"
            "UnitFileState=enabled\n"
            "FragmentPath=/home/tprover/.config/systemd/user/nex-pcx-web.service\n"
            f"Restart={restart}\n"
            "NRestarts=1\n"
        ),
    )


def healthy_app_getter(url: str, timeout_seconds: float) -> HttpJsonResult:
    if url.endswith("/healthz"):
        return HttpJsonResult(status_code=200, payload={"status": "ok"})
    return HttpJsonResult(status_code=200, payload={"info": {"title": "NeX_PCX"}})
