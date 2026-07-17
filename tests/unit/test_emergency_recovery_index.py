from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core import emergency_recovery_index
from app.core.emergency_recovery_index import (
    RecoveryCommand,
    RecoveryScenario,
    build_emergency_recovery_index,
    emergency_recovery_index_payload,
    payload_to_json,
    render_emergency_recovery_index_markdown,
)


def test_emergency_recovery_index_payload_and_markdown(tmp_path: Path) -> None:
    generated_at = datetime(2026, 7, 17, 11, 12, 13, tzinfo=UTC)

    index = build_emergency_recovery_index(
        workdir=tmp_path,
        app_url="http://127.0.0.1:8000/",
        provider_host="192.168.20.243",
        artifacts_dir="ops-artifacts",
        generated_at=generated_at,
    )
    payload = emergency_recovery_index_payload(index)
    markdown = render_emergency_recovery_index_markdown(payload)
    json_text = payload_to_json(payload, pretty=True)

    assert payload["generated_at_label"] == "2026-07-17 11:12:13"
    assert payload["app_url"] == "http://127.0.0.1:8000"
    assert payload["command_count"] >= 10
    assert payload["scenario_count"] >= 5
    assert "Provider Route Blocked" in markdown
    assert "ops-artifacts/go_live_smoke.json" in markdown
    assert "release-stale-lease" in markdown
    assert '"scenario_count"' in json_text


def test_emergency_recovery_index_marks_review_commands(tmp_path: Path) -> None:
    index = build_emergency_recovery_index(workdir=tmp_path)
    payload = emergency_recovery_index_payload(index)
    commands = {command["code"]: command for command in payload["commands"]}

    assert commands["retry_failed_embedding_jobs"]["requires_review"] is True
    assert commands["retry_pipeline_job"]["requires_review"] is True
    assert commands["app_healthz"]["requires_review"] is False
    assert "curl -fsS" in commands["app_healthz"]["shell_command"]


def test_emergency_recovery_index_validates_scenario_command_references() -> None:
    commands = (
        RecoveryCommand(
            code="known",
            title="Known",
            description="Known command",
            command=("echo", "known"),
        ),
    )
    scenarios = (
        RecoveryScenario(
            code="broken",
            title="Broken",
            severity="critical",
            first_check="test",
            command_codes=("missing",),
            checklist=("check",),
            stop_condition="stop",
        ),
    )

    with pytest.raises(ValueError, match="unknown commands"):
        emergency_recovery_index._validate_scenarios(commands, scenarios)
