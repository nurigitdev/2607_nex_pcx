import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.core import operational_retention_verification as retention
from app.core.admin_logging import LogSettings
from app.core.embedding_provider_route_retention import (
    ProviderRouteCleanupResult,
    ProviderRouteRetentionSettings,
)
from app.core.embedding_worker_batch_run_retention import (
    EmbeddingBatchRunCleanupResult,
    EmbeddingBatchRunRetentionSettings,
)
from app.core.operational_retention_verification import (
    RETENTION_CHECK_FAILED,
    RETENTION_CHECK_SKIPPED,
    RETENTION_STATUS_BLOCKED,
    RETENTION_STATUS_READY,
    RETENTION_STATUS_WARNING,
    build_operational_retention_verification_report,
    operational_retention_verification_report_payload,
    render_operational_retention_verification_markdown,
)
from app.core.search_logs import SearchLogCleanupResult, SearchLogRetentionSettings


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None) -> None:
        self.query = query
        self.params = params

    def fetchone(self):
        return {"ok": 1}


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return _FakeCursor()


def _patch_ready_dependencies(monkeypatch, checked_at: datetime) -> None:
    monkeypatch.setattr(retention, "connect", lambda database_url: _FakeConnection())
    monkeypatch.setattr(
        retention,
        "load_admin_log_retention_settings",
        lambda database_url: LogSettings(enabled=True, min_level="INFO", retention_days=7),
    )
    monkeypatch.setattr(
        retention,
        "load_search_log_retention_settings",
        lambda database_url: SearchLogRetentionSettings(
            enabled=True,
            retention_days=30,
            cleanup_batch_size=1000,
        ),
    )
    monkeypatch.setattr(
        retention,
        "cleanup_expired_search_logs",
        lambda database_url, *, dry_run: SearchLogCleanupResult(
            enabled=True,
            dry_run=dry_run,
            retention_days=30,
            cleanup_batch_size=1000,
            expired_count=2,
            deleted_count=0,
            cutoff_at=checked_at,
        ),
    )
    monkeypatch.setattr(
        retention,
        "load_provider_route_retention_settings",
        lambda database_url: ProviderRouteRetentionSettings(
            enabled=True,
            retention_days=30,
            cleanup_batch_size=1000,
        ),
    )
    monkeypatch.setattr(
        retention,
        "cleanup_expired_provider_route_records",
        lambda database_url, *, dry_run: ProviderRouteCleanupResult(
            enabled=True,
            dry_run=dry_run,
            retention_days=30,
            cleanup_batch_size=1000,
            expired_health_snapshot_count=1,
            expired_contract_snapshot_count=2,
            expired_preflight_run_count=3,
            deleted_health_snapshot_count=0,
            deleted_contract_snapshot_count=0,
            deleted_preflight_run_count=0,
            cutoff_at=checked_at,
        ),
    )
    monkeypatch.setattr(
        retention,
        "load_embedding_batch_run_retention_settings",
        lambda database_url: EmbeddingBatchRunRetentionSettings(
            enabled=True,
            retention_days=30,
            cleanup_batch_size=1000,
        ),
    )
    monkeypatch.setattr(
        retention,
        "cleanup_expired_embedding_batch_run_records",
        lambda database_url, *, dry_run: EmbeddingBatchRunCleanupResult(
            enabled=True,
            dry_run=dry_run,
            retention_days=30,
            cleanup_batch_size=1000,
            expired_batch_run_count=4,
            deleted_batch_run_count=0,
            cutoff_at=checked_at,
        ),
    )


def test_operational_retention_verification_ready_payload_and_markdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checked_at = datetime(2026, 7, 17, 9, 10, 11, tzinfo=UTC)
    (tmp_path / "artifacts").mkdir()
    _patch_ready_dependencies(monkeypatch, checked_at)

    report = build_operational_retention_verification_report(
        "postgresql://nex_pcx_dev:secret@127.0.0.1:5432/nex_pcx_dev",
        project_root=tmp_path,
        checked_at=checked_at,
    )
    payload = operational_retention_verification_report_payload(report)
    markdown = render_operational_retention_verification_markdown(payload)

    assert report.status == RETENTION_STATUS_READY
    assert payload["checked_at_label"] == "2026-07-17 09:10:11"
    assert payload["passed_count"] == payload["check_count"]
    assert "provider_route_cleanup_preview" in markdown
    provider_metadata = next(
        check["metadata"]
        for check in payload["checks"]
        if check["code"] == "provider_route_cleanup_preview"
    )
    assert provider_metadata["expired_count"] == 6


def test_operational_retention_verification_warns_for_policy_and_old_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checked_at = datetime(2026, 7, 17, 9, 10, 11, tzinfo=UTC)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    old_artifact = artifacts_dir / "old.md"
    old_artifact.write_text("old evidence", encoding="utf-8")
    old_timestamp = (checked_at - timedelta(days=31)).timestamp()
    os.utime(old_artifact, (old_timestamp, old_timestamp))
    _patch_ready_dependencies(monkeypatch, checked_at)
    monkeypatch.setattr(
        retention,
        "load_admin_log_retention_settings",
        lambda database_url: LogSettings(enabled=False, min_level="INFO", retention_days=7),
    )
    monkeypatch.setattr(
        retention,
        "load_search_log_retention_settings",
        lambda database_url: SearchLogRetentionSettings(
            enabled=True,
            retention_days=120,
            cleanup_batch_size=1000,
        ),
    )
    monkeypatch.setattr(
        retention,
        "cleanup_expired_search_logs",
        lambda database_url, *, dry_run: SearchLogCleanupResult(
            enabled=True,
            dry_run=dry_run,
            retention_days=120,
            cleanup_batch_size=1000,
            expired_count=0,
            deleted_count=0,
            cutoff_at=checked_at,
        ),
    )

    report = build_operational_retention_verification_report(
        "postgresql://example/db",
        project_root=tmp_path,
        max_retention_days=90,
        artifact_retention_days=30,
        checked_at=checked_at,
    )
    payload = operational_retention_verification_report_payload(report)
    warning_codes = {check["code"] for check in payload["checks"] if check["status"] == "warning"}

    assert report.status == RETENTION_STATUS_WARNING
    assert "admin_log_retention" in warning_codes
    assert "search_log_cleanup_preview" in warning_codes
    assert "artifacts_retention_review" in warning_codes


def test_operational_retention_verification_blocks_missing_database_and_bad_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "artifacts").write_text("not a directory", encoding="utf-8")

    report = build_operational_retention_verification_report(
        None,
        project_root=tmp_path,
    )
    payload = operational_retention_verification_report_payload(report)
    failed_codes = {
        check["code"] for check in payload["checks"] if check["status"] == RETENTION_CHECK_FAILED
    }
    skipped_codes = {
        check["code"] for check in payload["checks"] if check["status"] == RETENTION_CHECK_SKIPPED
    }

    assert report.status == RETENTION_STATUS_BLOCKED
    assert "database_url" in failed_codes
    assert "artifacts_retention_review" in failed_codes
    assert "search_log_cleanup_preview" in skipped_codes


def test_operational_retention_verification_blocks_dependency_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checked_at = datetime(2026, 7, 17, 9, 10, 11, tzinfo=UTC)
    (tmp_path / "artifacts").mkdir()
    _patch_ready_dependencies(monkeypatch, checked_at)
    monkeypatch.setattr(
        retention,
        "load_admin_log_retention_settings",
        lambda database_url: (_ for _ in ()).throw(RuntimeError("settings missing")),
    )

    report = build_operational_retention_verification_report(
        "postgresql://example/db",
        project_root=tmp_path,
    )
    payload = operational_retention_verification_report_payload(report)
    admin_check = next(
        check for check in payload["checks"] if check["code"] == "admin_log_retention"
    )

    assert report.status == RETENTION_STATUS_BLOCKED
    assert admin_check["status"] == RETENTION_CHECK_FAILED
    assert "settings missing" in admin_check["detail"]
