from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.runtime_config_audit import (
    CONFIG_AUDIT_STATUS_BLOCKED,
    CONFIG_AUDIT_STATUS_READY,
    CONFIG_AUDIT_STATUS_WARNING,
    CONFIG_CHECK_FAILED,
    build_runtime_config_audit_report,
    mask_database_url,
    render_runtime_config_audit_markdown,
    runtime_config_audit_report_payload,
)


def _production_settings(tmp_path: Path, **overrides) -> Settings:
    upload_dir = tmp_path / "uploads"
    models_dir = tmp_path / "models"
    upload_dir.mkdir()
    models_dir.mkdir()
    values = {
        "environment": "production",
        "database_url": "postgresql://app:secret@127.0.0.1:5432/nex_pcx_app",
        "upload_storage_dir": upload_dir,
        "embedding_models_dir": models_dir,
        "embedding_provider_mode": "remote",
        "embedding_require_route_readiness": True,
        "embedding_route_readiness_failure_mode": "defer",
        "embedding_route_readiness_defer_seconds": 300,
    }
    values.update(overrides)
    return Settings(**values)


def test_audit_ready_for_production_remote_route_settings(tmp_path) -> None:
    report = build_runtime_config_audit_report(
        _production_settings(tmp_path),
        checked_at=datetime(2026, 7, 17, 5, 6, 7, tzinfo=UTC),
    )
    payload = runtime_config_audit_report_payload(report)
    markdown = render_runtime_config_audit_markdown(payload)

    assert report.status == CONFIG_AUDIT_STATUS_READY
    assert report.failed_count == 0
    assert payload["checked_at_label"] == "2026-07-17 05:06:07"
    assert "Runtime Configuration Audit" in markdown
    assert "postgresql://app:***@127.0.0.1:5432/nex_pcx_app" in markdown


def test_audit_blocks_missing_database_and_invalid_modes(tmp_path) -> None:
    report = build_runtime_config_audit_report(
        _production_settings(
            tmp_path,
            database_url=None,
            embedding_provider_mode="broken",
            embedding_route_readiness_failure_mode="skip",
            embedding_route_readiness_defer_seconds=0,
        )
    )
    payload = runtime_config_audit_report_payload(report)
    failed_codes = {
        check["code"] for check in payload["checks"] if check["status"] == CONFIG_CHECK_FAILED
    }

    assert report.status == CONFIG_AUDIT_STATUS_BLOCKED
    assert {
        "database_url",
        "embedding_provider_mode",
        "embedding_route_readiness_failure_mode",
        "embedding_route_readiness_defer_seconds",
    }.issubset(failed_codes)


def test_audit_warns_for_local_relative_paths_and_disabled_readiness(tmp_path) -> None:
    (tmp_path / "uploads").mkdir()
    report = build_runtime_config_audit_report(
        Settings(
            environment="local",
            database_url="postgresql://app:secret@example/db",
            upload_storage_dir=Path("uploads"),
            embedding_models_dir=tmp_path / "missing-models",
            embedding_provider_mode="mock",
            embedding_require_route_readiness=False,
        ),
        project_root=tmp_path,
    )
    payload = runtime_config_audit_report_payload(report)
    warning_codes = {
        check["code"] for check in payload["checks"] if check["status"] == "warning"
    }

    assert report.status == CONFIG_AUDIT_STATUS_WARNING
    assert "environment" in warning_codes
    assert "upload_storage_dir" in warning_codes
    assert "embedding_models_dir" in warning_codes
    assert "embedding_route_readiness" in warning_codes


def test_audit_warns_for_production_test_database_and_short_defer(tmp_path) -> None:
    report = build_runtime_config_audit_report(
        _production_settings(
            tmp_path,
            test_database_url="postgresql://test:secret@example/test",
            embedding_route_readiness_defer_seconds=30,
        )
    )
    payload = runtime_config_audit_report_payload(report)
    warning_codes = {
        check["code"] for check in payload["checks"] if check["status"] == "warning"
    }

    assert report.status == CONFIG_AUDIT_STATUS_WARNING
    assert "test_database_url" in warning_codes
    assert "embedding_route_readiness_defer_seconds" in warning_codes


def test_mask_database_url_handles_edge_cases() -> None:
    assert mask_database_url(None) is None
    assert mask_database_url("not-a-url") == "***"
    assert mask_database_url("postgresql://example/db") == "postgresql://***@example/db"
