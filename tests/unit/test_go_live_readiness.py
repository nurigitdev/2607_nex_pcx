from types import SimpleNamespace

from app.core import go_live_readiness as go_live
from app.core.config import Settings


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
        return None

    def fetchone(self) -> dict[str, int]:
        return {"ok": 1}

    def fetchall(self) -> list[dict[str, object]]:
        return []


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor()


def _install_successful_database_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(go_live, "connect", lambda database_url: FakeConnection())
    monkeypatch.setattr(
        go_live,
        "audit_embedding_model_readiness",
        lambda models_dir: (
            SimpleNamespace(ready=True),
            SimpleNamespace(ready=True),
        ),
    )
    monkeypatch.setattr(
        go_live,
        "list_active_embedding_profiles",
        lambda database_url: [SimpleNamespace(profile_name="kure_v1_1024")],
    )
    monkeypatch.setattr(
        go_live,
        "get_embedding_provider_route_readiness_summary",
        lambda database_url, active_only=True: SimpleNamespace(
            route_count=2,
            ready_count=2,
            blocked_count=0,
            needs_preflight_count=0,
        ),
    )
    monkeypatch.setattr(
        go_live,
        "list_embedding_provider_preflight_schedules",
        lambda database_url: [SimpleNamespace(is_enabled=True)],
    )
    monkeypatch.setattr(
        go_live,
        "get_pipeline_queue_summary",
        lambda database_url: SimpleNamespace(
            queued_count=0,
            running_count=0,
            stale_running_count=0,
            retryable_failed_count=0,
            exhausted_failed_count=0,
            retryable_canceled_count=0,
            exhausted_canceled_count=0,
        ),
    )
    monkeypatch.setattr(
        go_live,
        "get_embedding_job_backlog_summary",
        lambda database_url: SimpleNamespace(
            pending_count=0,
            running_count=0,
            stale_running_count=0,
            retryable_failed_count=0,
            exhausted_failed_count=0,
        ),
    )
    monkeypatch.setattr(
        go_live,
        "get_dashboard_recent_failures",
        lambda database_url, limit=10: SimpleNamespace(
            total_count=0,
            provider_alert_count=0,
            app_error_count=0,
            parsing_failure_count=0,
        ),
    )
    monkeypatch.setattr(
        go_live,
        "load_log_settings",
        lambda connection: SimpleNamespace(enabled=True, retention_days=7),
    )
    monkeypatch.setattr(
        go_live,
        "load_provider_route_retention_settings",
        lambda database_url: SimpleNamespace(enabled=True, retention_days=30),
    )
    monkeypatch.setattr(
        go_live,
        "load_embedding_batch_run_retention_settings",
        lambda database_url: SimpleNamespace(enabled=True, retention_days=30),
    )
    monkeypatch.setattr(
        go_live,
        "load_search_log_retention_settings",
        lambda database_url: SimpleNamespace(enabled=True, retention_days=30),
    )


def _check_by_code(report, code: str):
    for section in report.sections:
        for check in section.checks:
            if check.code == code:
                return check
    raise AssertionError(f"Missing readiness check: {code}")


def test_go_live_readiness_blocks_without_database_url(tmp_path) -> None:
    report = go_live.build_go_live_readiness_report(
        Settings(
            database_url=None,
            upload_storage_dir=tmp_path / "uploads",
            embedding_models_dir=tmp_path / "models",
        )
    )

    assert report.status == "blocked"
    assert _check_by_code(report, "database_configured").status == "failed"
    assert _check_by_code(report, "database_connectivity").status == "skipped"
    assert _check_by_code(report, "database_backed_checks").status == "skipped"
    assert _check_by_code(report, "upload_storage_writable").status == "passed"
    assert _check_by_code(report, "embedding_model_bundles").status == "warning"


def test_go_live_readiness_blocks_when_storage_is_not_directory(tmp_path) -> None:
    storage_file = tmp_path / "uploads"
    storage_file.write_text("not a directory", encoding="utf-8")

    report = go_live.build_go_live_readiness_report(
        Settings(database_url=None, upload_storage_dir=storage_file)
    )

    assert report.status == "blocked"
    storage_check = _check_by_code(report, "upload_storage_writable")
    assert storage_check.status == "failed"
    assert storage_check.metadata["path"] == str(storage_file)


def test_go_live_readiness_passes_when_all_checks_are_ready(tmp_path, monkeypatch) -> None:
    _install_successful_database_dependencies(monkeypatch)

    report = go_live.build_go_live_readiness_report(
        Settings(
            database_url="postgresql://example/db",
            upload_storage_dir=tmp_path / "uploads",
            embedding_models_dir=tmp_path / "models",
        )
    )
    payload = go_live.go_live_readiness_report_payload(report)

    assert report.status == "ready"
    assert report.failed_count == 0
    assert report.warning_count == 0
    assert payload["status"] == "ready"
    assert payload["passed_count"] == report.check_count


def test_go_live_readiness_warns_for_partial_operational_readiness(
    tmp_path,
    monkeypatch,
) -> None:
    _install_successful_database_dependencies(monkeypatch)
    monkeypatch.setattr(
        go_live,
        "get_embedding_provider_route_readiness_summary",
        lambda database_url, active_only=True: SimpleNamespace(
            route_count=3,
            ready_count=2,
            blocked_count=1,
            needs_preflight_count=1,
        ),
    )
    monkeypatch.setattr(
        go_live,
        "list_embedding_provider_preflight_schedules",
        lambda database_url: [SimpleNamespace(is_enabled=False)],
    )
    monkeypatch.setattr(
        go_live,
        "get_dashboard_recent_failures",
        lambda database_url, limit=10: SimpleNamespace(
            total_count=1,
            provider_alert_count=1,
            app_error_count=0,
            parsing_failure_count=0,
        ),
    )

    report = go_live.build_go_live_readiness_report(
        Settings(
            database_url="postgresql://example/db",
            upload_storage_dir=tmp_path / "uploads",
        )
    )

    assert report.status == "warning"
    assert _check_by_code(report, "provider_route_readiness").status == "warning"
    assert _check_by_code(report, "provider_preflight_schedule").status == "warning"
    assert _check_by_code(report, "recent_operational_failures").status == "warning"


def test_go_live_readiness_blocks_when_database_backed_check_raises(
    tmp_path,
    monkeypatch,
) -> None:
    _install_successful_database_dependencies(monkeypatch)
    monkeypatch.setattr(
        go_live,
        "list_active_embedding_profiles",
        lambda database_url: (_ for _ in ()).throw(RuntimeError("schema missing")),
    )

    report = go_live.build_go_live_readiness_report(
        Settings(
            database_url="postgresql://example/db",
            upload_storage_dir=tmp_path / "uploads",
        )
    )

    assert report.status == "blocked"
    check = _check_by_code(report, "active_embedding_profiles")
    assert check.status == "failed"
    assert check.metadata["error"] == "schema missing"
