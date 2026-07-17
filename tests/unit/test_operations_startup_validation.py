from types import SimpleNamespace

from app.core import operations_startup_validation as startup
from app.core.config import Settings


class FakeCursor:
    def __init__(self, current_revisions: tuple[str, ...]) -> None:
        self.current_revisions = current_revisions
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
        self.query = query

    def fetchone(self) -> dict[str, int]:
        return {"ok": 1}

    def fetchall(self) -> list[dict[str, str]]:
        return [{"version_num": revision} for revision in self.current_revisions]


class FakeConnection:
    def __init__(self, current_revisions: tuple[str, ...]) -> None:
        self.current_revisions = current_revisions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.current_revisions)


def make_go_live_report(status: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        check_count=6,
        passed_count=5 if status == "ready" else 4,
        warning_count=1 if status == "warning" else 0,
        failed_count=1 if status == "blocked" else 0,
        skipped_count=0,
    )


def install_successful_dependencies(
    monkeypatch,
    *,
    current_revisions: tuple[str, ...] = ("head-one",),
    head_revisions: tuple[str, ...] = ("head-one",),
    go_live_status: str = "ready",
    app_payload: dict[str, object] | None = None,
    preflight_payload: dict[str, object] | None = None,
) -> None:
    monkeypatch.setattr(
        startup,
        "connect",
        lambda database_url: FakeConnection(current_revisions),
    )
    monkeypatch.setattr(startup, "_head_revisions", lambda database_url: head_revisions)
    monkeypatch.setattr(
        startup,
        "build_go_live_readiness_report",
        lambda settings: make_go_live_report(go_live_status),
    )
    monkeypatch.setattr(
        startup,
        "_fetch_json",
        lambda url, *, timeout_seconds: app_payload
        if app_payload is not None
        else {"status": "ok", "service": "NeX_PCX"},
    )
    monkeypatch.setattr(
        startup,
        "run_embedding_provider_route_preflight",
        lambda database_url, **kwargs: preflight_payload
        if preflight_payload is not None
        else {
            "route_count": 2,
            "passed_count": 2,
            "failed_count": 0,
            "profile_name": kwargs.get("profile_name"),
            "active_only": kwargs.get("active_only"),
        },
    )


def check_by_code(report: startup.OperationsStartupValidationReport) -> dict[str, str]:
    return {check.code: check.status for check in report.checks}


def test_startup_validation_blocks_when_database_url_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        startup,
        "build_go_live_readiness_report",
        lambda settings: make_go_live_report("blocked"),
    )

    report = startup.build_operations_startup_validation_report(Settings(database_url=None))
    checks = check_by_code(report)

    assert report.status == startup.STARTUP_STATUS_BLOCKED
    assert checks["database_url"] == startup.STARTUP_CHECK_FAILED
    assert checks["database_connectivity"] == startup.STARTUP_CHECK_SKIPPED
    assert checks["alembic_revision"] == startup.STARTUP_CHECK_SKIPPED
    assert checks["app_healthz"] == startup.STARTUP_CHECK_SKIPPED
    assert checks["provider_route_preflight"] == startup.STARTUP_CHECK_SKIPPED


def test_startup_validation_passes_full_startup_path(monkeypatch) -> None:
    install_successful_dependencies(monkeypatch)

    report = startup.build_operations_startup_validation_report(
        Settings(database_url="postgresql://example/db"),
        options=startup.OperationsStartupValidationOptions(
            app_base_url="http://127.0.0.1:8000/",
            run_provider_preflight=True,
            profile_name="kure_v1",
            include_inactive_routes=True,
            health_timeout_seconds=2.5,
        ),
    )
    payload = startup.operations_startup_validation_report_payload(report)
    checks = check_by_code(report)

    assert report.status == startup.STARTUP_STATUS_READY
    assert report.failed_count == 0
    assert checks["database_connectivity"] == startup.STARTUP_CHECK_PASSED
    assert checks["alembic_revision"] == startup.STARTUP_CHECK_PASSED
    assert checks["app_healthz"] == startup.STARTUP_CHECK_PASSED
    assert checks["go_live_readiness"] == startup.STARTUP_CHECK_PASSED
    assert checks["provider_route_preflight"] == startup.STARTUP_CHECK_PASSED
    assert payload["passed_count"] == report.passed_count
    preflight = next(
        check
        for check in payload["checks"]
        if check["code"] == "provider_route_preflight"
    )
    assert preflight["metadata"]["profile_name"] == "kure_v1"
    assert preflight["metadata"]["active_only"] is False


def test_startup_validation_warns_when_go_live_has_warnings(monkeypatch) -> None:
    install_successful_dependencies(monkeypatch, go_live_status="warning")

    report = startup.build_operations_startup_validation_report(
        Settings(database_url="postgresql://example/db"),
    )
    checks = check_by_code(report)

    assert report.status == startup.STARTUP_STATUS_WARNING
    assert checks["go_live_readiness"] == startup.STARTUP_CHECK_WARNING
    assert checks["app_healthz"] == startup.STARTUP_CHECK_SKIPPED
    assert checks["provider_route_preflight"] == startup.STARTUP_CHECK_SKIPPED


def test_startup_validation_blocks_for_revision_app_and_preflight_failures(monkeypatch) -> None:
    install_successful_dependencies(
        monkeypatch,
        current_revisions=("old-head",),
        head_revisions=("new-head",),
        app_payload={"status": "starting"},
        preflight_payload={
            "route_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "profile_name": None,
            "active_only": True,
        },
    )

    report = startup.build_operations_startup_validation_report(
        Settings(database_url="postgresql://example/db"),
        options=startup.OperationsStartupValidationOptions(
            app_base_url="http://127.0.0.1:8000",
            run_provider_preflight=True,
        ),
    )
    checks = check_by_code(report)

    assert report.status == startup.STARTUP_STATUS_BLOCKED
    assert checks["alembic_revision"] == startup.STARTUP_CHECK_FAILED
    assert checks["app_healthz"] == startup.STARTUP_CHECK_FAILED
    assert checks["provider_route_preflight"] == startup.STARTUP_CHECK_FAILED


def test_startup_validation_skips_database_dependent_checks_when_connectivity_fails(
    monkeypatch,
) -> None:
    def raising_connect(database_url: str):
        raise RuntimeError("database offline")

    monkeypatch.setattr(startup, "connect", raising_connect)
    monkeypatch.setattr(
        startup,
        "build_go_live_readiness_report",
        lambda settings: make_go_live_report("blocked"),
    )

    report = startup.build_operations_startup_validation_report(
        Settings(database_url="postgresql://example/db"),
        options=startup.OperationsStartupValidationOptions(run_provider_preflight=True),
    )
    checks = check_by_code(report)

    assert report.status == startup.STARTUP_STATUS_BLOCKED
    assert checks["database_connectivity"] == startup.STARTUP_CHECK_FAILED
    assert checks["alembic_revision"] == startup.STARTUP_CHECK_SKIPPED
    assert checks["provider_route_preflight"] == startup.STARTUP_CHECK_SKIPPED
