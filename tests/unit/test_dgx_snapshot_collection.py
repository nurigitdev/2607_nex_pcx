import json
from datetime import UTC, datetime

import pytest

from app.core.dgx_snapshot_collection import (
    COLLECTION_PLAN_BLOCKED,
    COLLECTION_PLAN_READY,
    COLLECTION_STATUS_ATTENTION,
    COLLECTION_STATUS_PARTIAL,
    COMMAND_STATUS_ATTENTION,
    COMMAND_STATUS_COLLECTED,
    COMMAND_STATUS_FAILED,
    COMPONENT_PROVIDER_RESOURCES,
    COMPONENT_VLLM_RUNTIME,
    DgxSnapshotCollectionCommandResult,
    DgxSnapshotCollectionOptions,
    build_dgx_snapshot_collection_cycle,
    build_dgx_snapshot_collection_evidence,
    build_dgx_snapshot_collection_plan,
    classify_dgx_snapshot_command_result,
    dgx_snapshot_collection_evidence_payload,
    dgx_snapshot_collection_options_from_environ,
    dgx_snapshot_collection_status_from_cycles,
    render_dgx_snapshot_collection_markdown,
)

NOW = datetime(2026, 7, 29, 15, 0, tzinfo=UTC)


def test_build_dgx_snapshot_collection_plan_defaults_to_both_snapshot_commands() -> None:
    plan = build_dgx_snapshot_collection_plan(
        DgxSnapshotCollectionOptions(workdir="/repo", python_bin="python"),
        environ={"NEX_PCX_DATABASE_URL": "postgresql://user:secret@db/app"},
        generated_at=NOW,
    )
    payload = dgx_snapshot_collection_evidence_payload(
        build_dgx_snapshot_collection_evidence(
            plan,
            status="planned",
            dry_run=True,
            generated_at=NOW,
        )
    )

    assert plan.status == COLLECTION_PLAN_READY
    assert [command.code for command in plan.commands] == [
        COMPONENT_VLLM_RUNTIME,
        COMPONENT_PROVIDER_RESOURCES,
    ]
    assert "scripts/scrape_vllm_runtime_metrics.py" in plan.commands[0].shell_command
    assert "scripts/probe_provider_resources.py" in plan.commands[1].shell_command
    assert "--ssh-user nexpcx" in plan.commands[1].shell_command
    assert "--persist" in plan.commands[0].command
    assert "--persist" in plan.commands[1].command
    assert "secret" not in json.dumps(payload)


def test_build_dgx_snapshot_collection_plan_supports_local_provider_probe() -> None:
    plan = build_dgx_snapshot_collection_plan(
        DgxSnapshotCollectionOptions(
            components=("provider-resource",),
            provider_local_only=True,
            provider_selectors=("vllm", "reranker"),
        ),
        environ={"NEX_PCX_DATABASE_URL": "postgresql://db"},
    )

    command = plan.commands[0]
    assert plan.components == (COMPONENT_PROVIDER_RESOURCES,)
    assert "--local-only" in command.command
    assert "--ssh-user" not in command.command
    assert command.command.count("--provider") == 2
    assert "vllm" in command.command
    assert "reranker" in command.command


def test_dgx_snapshot_collection_plan_reports_blockers_and_validation_errors() -> None:
    missing_db_plan = build_dgx_snapshot_collection_plan(
        DgxSnapshotCollectionOptions(),
        environ={},
    )
    dry_run_plan = build_dgx_snapshot_collection_plan(
        DgxSnapshotCollectionOptions(require_database_url=False),
        environ={},
    )

    assert missing_db_plan.status == COLLECTION_PLAN_BLOCKED
    assert missing_db_plan.failed_count == 1
    assert dry_run_plan.status == COLLECTION_PLAN_READY
    assert "dry-run planning is allowed" in dry_run_plan.checks[0].detail
    with pytest.raises(ValueError, match="unsupported snapshot collection component"):
        build_dgx_snapshot_collection_plan(
            DgxSnapshotCollectionOptions(components=("database",)),
            environ={"NEX_PCX_DATABASE_URL": "postgresql://db"},
        )
    with pytest.raises(ValueError, match="provider_ssh_user"):
        build_dgx_snapshot_collection_plan(
            DgxSnapshotCollectionOptions(
                components=("provider-resource",),
                provider_ssh_user=None,
            ),
            environ={"NEX_PCX_DATABASE_URL": "postgresql://db"},
        )


def test_dgx_snapshot_collection_options_from_environment() -> None:
    options = dgx_snapshot_collection_options_from_environ(
        {
            "NEX_PCX_COLLECTION_WORKDIR": "/ops",
            "NEX_PCX_COLLECTION_PROVIDER_SELECTORS": "vllm, reranker",
            "NEX_PCX_COLLECTION_PROVIDER_LOCAL_ONLY": "true",
            "NEX_PCX_COLLECTION_INTERVAL_SECONDS": "2.5",
            "NEX_PCX_COLLECTION_MAX_CYCLES": "3",
        }
    )

    assert str(options.workdir) == "/ops"
    assert options.provider_selectors == ("vllm", "reranker")
    assert options.provider_local_only is True
    assert options.interval_seconds == 2.5
    assert options.max_cycles == 3
    with pytest.raises(ValueError, match="boolean"):
        dgx_snapshot_collection_options_from_environ(
            {"NEX_PCX_COLLECTION_PROVIDER_LOCAL_ONLY": "sometimes"}
        )


def test_classify_dgx_snapshot_command_result_distinguishes_attention_from_failure() -> None:
    assert (
        classify_dgx_snapshot_command_result(
            code=COMPONENT_VLLM_RUNTIME,
            exit_code=0,
            payload={"snapshot_record": {"snapshot_id": 1}},
        )
        == COMMAND_STATUS_COLLECTED
    )
    assert (
        classify_dgx_snapshot_command_result(
            code=COMPONENT_PROVIDER_RESOURCES,
            exit_code=1,
            payload={"status": "critical", "snapshot_records": [{"snapshot_id": 2}]},
        )
        == COMMAND_STATUS_ATTENTION
    )
    assert (
        classify_dgx_snapshot_command_result(
            code=COMPONENT_PROVIDER_RESOURCES,
            exit_code=0,
            payload={"status": "ok", "snapshot_records": []},
        )
        == COMMAND_STATUS_FAILED
    )
    assert (
        classify_dgx_snapshot_command_result(
            code=COMPONENT_VLLM_RUNTIME,
            exit_code=0,
            payload={"provider_name": "missing-record"},
        )
        == COMMAND_STATUS_FAILED
    )
    assert (
        classify_dgx_snapshot_command_result(
            code="custom",
            exit_code=7,
            payload={"status": "failed"},
        )
        == COMMAND_STATUS_FAILED
    )
    assert (
        classify_dgx_snapshot_command_result(
            code=COMPONENT_VLLM_RUNTIME,
            exit_code=0,
            payload={"snapshot_record": {"snapshot_id": 1}},
            error_message="bad json",
        )
        == COMMAND_STATUS_FAILED
    )


def test_dgx_snapshot_collection_evidence_payload_and_markdown() -> None:
    plan = build_dgx_snapshot_collection_plan(
        DgxSnapshotCollectionOptions(components=("all",)),
        environ={"NEX_PCX_DATABASE_URL": "postgresql://db"},
        generated_at=NOW,
    )
    cycle = build_dgx_snapshot_collection_cycle(
        index=1,
        started_at=NOW,
        finished_at=NOW,
        results=(
            _result(COMPONENT_VLLM_RUNTIME, COMMAND_STATUS_COLLECTED),
            _result(COMPONENT_PROVIDER_RESOURCES, COMMAND_STATUS_ATTENTION, exit_code=1),
        ),
    )
    evidence_status = dgx_snapshot_collection_status_from_cycles([cycle])
    evidence = build_dgx_snapshot_collection_evidence(
        plan,
        status=evidence_status,
        dry_run=False,
        cycles=(cycle,),
        generated_at=NOW,
        message="collected with attention",
    )
    payload = dgx_snapshot_collection_evidence_payload(evidence)
    markdown = render_dgx_snapshot_collection_markdown(payload)

    assert cycle.status == COLLECTION_STATUS_ATTENTION
    assert payload["status"] == COLLECTION_STATUS_ATTENTION
    assert payload["attention_command_count"] == 1
    assert "DGX Snapshot Collection Evidence" in markdown
    assert "Provider resources are collected from the DGX host" in markdown
    assert json.loads(json.dumps(payload))["cycles"][0]["results"][0]["collected"] is True


def test_dgx_snapshot_collection_cycle_status_handles_partial_failure() -> None:
    failed_cycle = build_dgx_snapshot_collection_cycle(
        index=1,
        started_at=NOW,
        finished_at=NOW,
        results=(
            _result(COMPONENT_VLLM_RUNTIME, COMMAND_STATUS_COLLECTED),
            _result(COMPONENT_PROVIDER_RESOURCES, COMMAND_STATUS_FAILED, exit_code=127),
        ),
    )

    assert failed_cycle.status == COLLECTION_STATUS_PARTIAL
    assert dgx_snapshot_collection_status_from_cycles([failed_cycle]) == COLLECTION_STATUS_PARTIAL


def _result(
    code: str,
    status: str,
    *,
    exit_code: int = 0,
) -> DgxSnapshotCollectionCommandResult:
    return DgxSnapshotCollectionCommandResult(
        code=code,
        component=code,
        command=("python", "script.py"),
        exit_code=exit_code,
        elapsed_ms=12,
        status=status,
        payload={"snapshot_record": {"snapshot_id": 1}},
    )
