from datetime import UTC, datetime

import pytest

from app.core.foreground_operations_validation import (
    FOREGROUND_STATUS_BLOCKED,
    FOREGROUND_STATUS_WARNING,
    CommandResult,
    ForegroundOperationsValidationOptions,
    HttpJsonResult,
    build_foreground_operations_validation_report,
    foreground_operations_validation_report_payload,
    render_foreground_operations_validation_markdown,
)


def test_foreground_operations_validation_warns_when_acknowledged() -> None:
    report = build_foreground_operations_validation_report(
        ForegroundOperationsValidationOptions(
            app_base_url="http://127.0.0.1:8000/",
            acknowledge_no_auto_restart=True,
        ),
        checked_at=datetime(2026, 7, 17, 9, 10, 11, tzinfo=UTC),
        command_runner=successful_command_runner,
        http_json_getter=healthy_app_getter,
    )
    payload = foreground_operations_validation_report_payload(report)
    markdown = render_foreground_operations_validation_markdown(payload)

    assert report.status == FOREGROUND_STATUS_WARNING
    assert payload["checked_at_label"] == "2026-07-17 09:10:11"
    assert payload["passed_count"] == 5
    assert payload["warning_count"] == 1
    assert "no automatic restart" in markdown


def test_foreground_operations_validation_blocks_without_acknowledgement() -> None:
    report = build_foreground_operations_validation_report(
        ForegroundOperationsValidationOptions(),
        command_runner=successful_command_runner,
        http_json_getter=healthy_app_getter,
    )

    assert report.status == FOREGROUND_STATUS_BLOCKED
    assert report.checks[0].code == "foreground_no_auto_restart_ack"
    assert report.checks[0].status == "failed"


def test_foreground_operations_validation_blocks_wrong_app_identity() -> None:
    def wrong_identity_getter(url: str, timeout_seconds: float) -> HttpJsonResult:
        if url.endswith("/healthz"):
            return HttpJsonResult(status_code=200, payload={"status": "ok"})
        return HttpJsonResult(status_code=200, payload={"info": {"title": "Other"}})

    report = build_foreground_operations_validation_report(
        ForegroundOperationsValidationOptions(acknowledge_no_auto_restart=True),
        command_runner=successful_command_runner,
        http_json_getter=wrong_identity_getter,
    )

    assert report.status == FOREGROUND_STATUS_BLOCKED
    assert report.checks[3].code == "app_identity"


def test_foreground_operations_validation_blocks_failed_worker_cli() -> None:
    def failed_command_runner(command: tuple[str, ...]) -> CommandResult:
        if any(part.endswith("process_embedding_job.py") for part in command):
            return CommandResult(command=command, returncode=2, stderr="bad import")
        return CommandResult(command=command, returncode=0, stdout="usage\n")

    report = build_foreground_operations_validation_report(
        ForegroundOperationsValidationOptions(acknowledge_no_auto_restart=True),
        command_runner=failed_command_runner,
        http_json_getter=healthy_app_getter,
    )

    assert report.status == FOREGROUND_STATUS_BLOCKED
    assert report.checks[-1].code == "embedding_worker_cli"
    assert "bad import" in report.checks[-1].detail


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            ForegroundOperationsValidationOptions(app_base_url="127.0.0.1:8000"),
            "absolute HTTP URL",
        ),
        (
            ForegroundOperationsValidationOptions(expected_app_name=""),
            "expected_app_name",
        ),
        (
            ForegroundOperationsValidationOptions(health_timeout_seconds=0),
            "health_timeout_seconds",
        ),
        (
            ForegroundOperationsValidationOptions(pipeline_worker_command=()),
            "pipeline_worker_command",
        ),
    ],
)
def test_foreground_operations_validation_rejects_invalid_options(
    options: ForegroundOperationsValidationOptions,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_foreground_operations_validation_report(options)


def successful_command_runner(command: tuple[str, ...]) -> CommandResult:
    return CommandResult(command=command, returncode=0, stdout="usage\n")


def healthy_app_getter(url: str, timeout_seconds: float) -> HttpJsonResult:
    if url.endswith("/healthz"):
        return HttpJsonResult(status_code=200, payload={"status": "ok"})
    return HttpJsonResult(status_code=200, payload={"info": {"title": "NeX_PCX"}})
