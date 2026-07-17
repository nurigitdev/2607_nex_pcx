from datetime import UTC, datetime
from pathlib import Path

from app.core import production_environment_validation as validation
from app.core.config import Settings
from app.core.go_live_readiness import GoLiveReadinessReport
from app.core.operations_startup_validation import OperationsStartupValidationReport
from app.core.production_environment_validation import (
    PRODUCTION_STATUS_BLOCKED,
    PRODUCTION_STATUS_READY,
    PRODUCTION_STATUS_WARNING,
    ProductionValidationOptions,
    build_production_environment_validation_report,
    payload_to_json,
    production_environment_validation_payload,
    render_production_environment_validation_markdown,
)
from app.core.runtime_config_audit import RuntimeConfigAuditReport


def _settings(**overrides) -> Settings:
    values = {
        "environment": "production",
        "database_url": "postgresql://nex_pcx_app:secret@127.0.0.1:5432/nex_pcx_app",
        "upload_storage_dir": Path("/srv/nex_pcx/uploads"),
        "embedding_models_dir": Path("/srv/nex_pcx/models"),
        "embedding_provider_mode": "remote",
        "embedding_require_route_readiness": True,
    }
    values.update(overrides)
    return Settings(**values)


def _patch_nested_reports(monkeypatch, *, status: str = "ready") -> None:
    monkeypatch.setattr(
        validation,
        "build_runtime_config_audit_report",
        lambda settings, *, checked_at, project_root: RuntimeConfigAuditReport(
            status=status,
            checked_at=checked_at,
            checks=(),
        ),
    )
    monkeypatch.setattr(
        validation,
        "build_operations_startup_validation_report",
        lambda settings, *, checked_at, options: OperationsStartupValidationReport(
            status=status,
            checked_at=checked_at,
            checks=(),
        ),
    )
    monkeypatch.setattr(
        validation,
        "build_go_live_readiness_report",
        lambda settings, *, checked_at: GoLiveReadinessReport(
            status=status,
            checked_at=checked_at,
            sections=(),
        ),
    )


def test_production_environment_validation_ready_payload_and_markdown(
    monkeypatch,
    tmp_path,
) -> None:
    checked_at = datetime(2026, 7, 17, 17, 18, 19, tzinfo=UTC)
    _patch_nested_reports(monkeypatch)

    report = build_production_environment_validation_report(
        _settings(),
        project_root=tmp_path,
        checked_at=checked_at,
        options=ProductionValidationOptions(
            app_url="http://127.0.0.1:8000",
            expected_database_name="nex_pcx_app",
        ),
    )
    payload = production_environment_validation_payload(report)
    markdown = render_production_environment_validation_markdown(payload)

    assert report.status == PRODUCTION_STATUS_READY
    assert payload["checked_at_label"] == "2026-07-17 17:18:19"
    assert payload["failed_guard_count"] == 0
    assert "runtime_config_audit" in markdown
    assert '"status": "ready"' in payload_to_json(payload, pretty=True)


def test_production_environment_validation_blocks_strict_guard_failures(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_nested_reports(monkeypatch)

    report = build_production_environment_validation_report(
        _settings(
            environment="local",
            database_url="postgresql://nex_pcx_test:secret@127.0.0.1:5432/nex_pcx_test",
            embedding_provider_mode="mock",
            embedding_require_route_readiness=False,
        ),
        project_root=tmp_path,
        options=ProductionValidationOptions(
            expected_database_name="nex_pcx_app",
        ),
    )
    payload = production_environment_validation_payload(report)
    failed_codes = {
        check["code"] for check in payload["guard_checks"] if check["status"] == "failed"
    }

    assert report.status == PRODUCTION_STATUS_BLOCKED
    assert {
        "environment",
        "database_target",
        "embedding_provider_mode",
        "embedding_route_readiness",
    } <= failed_codes


def test_production_environment_validation_warns_when_guards_are_relaxed(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_nested_reports(monkeypatch)

    report = build_production_environment_validation_report(
        _settings(
            environment="local",
            database_url="postgresql://nex_pcx_dev:secret@127.0.0.1:5432/nex_pcx_dev",
            embedding_provider_mode="mock",
            embedding_require_route_readiness=False,
        ),
        project_root=tmp_path,
        options=ProductionValidationOptions(
            require_production_env=False,
            require_remote_provider=False,
            require_route_readiness=False,
        ),
    )

    assert report.status == PRODUCTION_STATUS_WARNING
    assert report.warning_guard_count >= 3


def test_production_environment_validation_blocks_nested_sections(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_nested_reports(monkeypatch, status="blocked")

    report = build_production_environment_validation_report(
        _settings(),
        project_root=tmp_path,
        options=ProductionValidationOptions(app_url="http://127.0.0.1:8000"),
    )

    assert report.status == PRODUCTION_STATUS_BLOCKED
