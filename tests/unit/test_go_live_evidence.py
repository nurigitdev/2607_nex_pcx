from datetime import UTC, datetime

from app.core import go_live_evidence
from app.core.config import Settings
from app.core.go_live_readiness import (
    GO_LIVE_CHECK_PASSED,
    GO_LIVE_CHECK_WARNING,
    GO_LIVE_STATUS_READY,
    GO_LIVE_STATUS_WARNING,
    GoLiveReadinessCheck,
    GoLiveReadinessReport,
    GoLiveReadinessSection,
)
from app.core.operations_startup_validation import (
    STARTUP_CHECK_PASSED,
    STARTUP_CHECK_WARNING,
    STARTUP_STATUS_READY,
    STARTUP_STATUS_WARNING,
    OperationsStartupValidationCheck,
    OperationsStartupValidationReport,
)

FIXED_TIME = datetime(2026, 7, 17, 4, 5, 6, tzinfo=UTC)


def make_startup_report(status: str = STARTUP_STATUS_READY) -> OperationsStartupValidationReport:
    check_status = (
        STARTUP_CHECK_WARNING if status == STARTUP_STATUS_WARNING else STARTUP_CHECK_PASSED
    )
    return OperationsStartupValidationReport(
        status=status,
        checked_at=FIXED_TIME,
        checks=(
            OperationsStartupValidationCheck(
                code="database_connectivity",
                status=STARTUP_CHECK_PASSED,
                detail="Database connection succeeded.",
            ),
            OperationsStartupValidationCheck(
                code="provider_route_preflight",
                status=check_status,
                detail="Provider route preflight detail | with pipe.",
            ),
        ),
    )


def make_go_live_report(status: str = GO_LIVE_STATUS_READY) -> GoLiveReadinessReport:
    check_status = (
        GO_LIVE_CHECK_WARNING if status == GO_LIVE_STATUS_WARNING else GO_LIVE_CHECK_PASSED
    )
    return GoLiveReadinessReport(
        status=status,
        checked_at=FIXED_TIME,
        sections=(
            GoLiveReadinessSection(
                code="providers",
                checks=(
                    GoLiveReadinessCheck(
                        code="provider_route_readiness",
                        status=check_status,
                        detail="2/2 active provider routes are ready.",
                        metadata={"ready_count": 2, "route_count": 2},
                    ),
                ),
            ),
        ),
    )


def install_report_builders(monkeypatch, *, startup_status: str, go_live_status: str) -> None:
    captured = {}

    def fake_startup_builder(settings, *, options, checked_at):
        captured["startup_options"] = options
        captured["startup_checked_at"] = checked_at
        return make_startup_report(startup_status)

    def fake_go_live_builder(settings, *, checked_at):
        captured["go_live_checked_at"] = checked_at
        return make_go_live_report(go_live_status)

    monkeypatch.setattr(
        go_live_evidence,
        "build_operations_startup_validation_report",
        fake_startup_builder,
    )
    monkeypatch.setattr(go_live_evidence, "build_go_live_readiness_report", fake_go_live_builder)
    monkeypatch.setattr(
        go_live_evidence,
        "_git_output",
        lambda project_root, *args: {
            ("rev-parse", "--short", "HEAD"): "abc1234",
            ("rev-parse", "--abbrev-ref", "HEAD"): "master",
            ("status", "--porcelain"): " M README.md",
        }.get(args),
    )


def test_go_live_evidence_snapshot_combines_readiness_and_masks_database_url(
    monkeypatch,
    tmp_path,
) -> None:
    install_report_builders(
        monkeypatch,
        startup_status=STARTUP_STATUS_READY,
        go_live_status=GO_LIVE_STATUS_READY,
    )
    settings = Settings(
        database_url="postgresql://nex_pcx_app:secret@127.0.0.1:5432/nex_pcx_app",
        upload_storage_dir=tmp_path / "uploads",
        embedding_models_dir=tmp_path / "models",
        environment="production",
    )

    snapshot = go_live_evidence.build_go_live_evidence_snapshot(
        settings,
        generated_at=FIXED_TIME,
        project_root=tmp_path,
    )

    assert snapshot["version"] == 1
    assert snapshot["status"] == go_live_evidence.GO_LIVE_EVIDENCE_STATUS_READY
    assert snapshot["runtime"]["database_url_masked"] == (
        "postgresql://nex_pcx_app:***@127.0.0.1:5432/nex_pcx_app"
    )
    assert "secret" not in str(snapshot)
    assert snapshot["provenance"]["git_commit"] == "abc1234"
    assert snapshot["provenance"]["git_branch"] == "master"
    assert snapshot["provenance"]["git_dirty"] is True
    assert snapshot["summary"]["startup_validation_status"] == "ready"
    assert snapshot["summary"]["go_live_readiness_status"] == "ready"


def test_go_live_evidence_snapshot_warns_when_any_signal_warns(monkeypatch, tmp_path) -> None:
    install_report_builders(
        monkeypatch,
        startup_status=STARTUP_STATUS_WARNING,
        go_live_status=GO_LIVE_STATUS_READY,
    )

    snapshot = go_live_evidence.build_go_live_evidence_snapshot(
        Settings(database_url="postgresql://user:secret@example/db"),
        generated_at=FIXED_TIME,
        project_root=tmp_path,
    )

    assert snapshot["status"] == go_live_evidence.GO_LIVE_EVIDENCE_STATUS_WARNING
    assert snapshot["startup_validation"]["warning_count"] == 1


def test_go_live_evidence_markdown_includes_tables_and_escapes_cells(
    monkeypatch,
    tmp_path,
) -> None:
    install_report_builders(
        monkeypatch,
        startup_status=STARTUP_STATUS_WARNING,
        go_live_status=GO_LIVE_STATUS_WARNING,
    )
    snapshot = go_live_evidence.build_go_live_evidence_snapshot(
        Settings(database_url="postgresql://user:secret@example/db"),
        generated_at=FIXED_TIME,
        project_root=tmp_path,
    )

    markdown = go_live_evidence.render_go_live_evidence_markdown(snapshot)

    assert "# NeX_PCX Go-Live Evidence Snapshot" in markdown
    assert "| Startup Validation | warning | 0 | 1 |" in markdown
    assert "provider_route_preflight" in markdown
    assert "detail \\| with pipe" in markdown
    assert "provider_route_readiness" in markdown
    assert "secret" not in markdown


def test_git_provenance_returns_none_when_git_is_unavailable(monkeypatch, tmp_path) -> None:
    def raising_run(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(go_live_evidence.subprocess, "run", raising_run)

    provenance = go_live_evidence._provenance_payload(tmp_path)

    assert provenance == {
        "project_root": str(tmp_path),
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }


def test_mask_database_url_handles_blank_and_invalid_values() -> None:
    assert go_live_evidence._mask_database_url(None) is None
    assert go_live_evidence._mask_database_url("not-a-url") == "***"
    assert go_live_evidence._mask_database_url("postgresql://example/db") == (
        "postgresql://***@example/db"
    )
