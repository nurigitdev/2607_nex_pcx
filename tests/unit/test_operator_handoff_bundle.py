from datetime import UTC, datetime
from pathlib import Path

from app.core.operator_handoff_bundle import (
    DEFAULT_HANDOFF_EVIDENCE_PATHS,
    export_operator_handoff_bundle,
    operator_handoff_bundle_payload,
    payload_to_json,
    render_operator_handoff_bundle_markdown,
)


def test_export_operator_handoff_bundle_copies_evidence_and_writes_manifest(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "work"
    bundle_dir = tmp_path / "bundle"
    evidence_path = workdir / "artifacts" / "go_live_evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"status":"ready"}\n', encoding="utf-8")

    bundle = export_operator_handoff_bundle(
        workdir=workdir,
        bundle_dir=bundle_dir,
        evidence_paths=("artifacts/go_live_evidence.json",),
        git_commit="abc123",
        generated_at=datetime(2026, 7, 17, 13, 14, 15, tzinfo=UTC),
    )
    payload = operator_handoff_bundle_payload(bundle)
    markdown = render_operator_handoff_bundle_markdown(payload)

    assert bundle.missing_required_count == 0
    assert bundle.included_count == 1
    assert payload["generated_at_label"] == "2026-07-17 13:14:15"
    assert payload["files"][0]["sha256"]
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "handoff.md").exists()
    assert (bundle_dir / "evidence" / "artifacts" / "go_live_evidence.json").exists()
    assert "abc123" in markdown
    assert '"included_count": 1' in payload_to_json(payload, pretty=True)


def test_export_operator_handoff_bundle_reports_missing_required_files(
    tmp_path: Path,
) -> None:
    bundle = export_operator_handoff_bundle(
        workdir=tmp_path / "work",
        bundle_dir=tmp_path / "bundle",
        evidence_paths=("artifacts/missing.json",),
        copy_files=False,
    )
    payload = operator_handoff_bundle_payload(bundle)
    markdown = render_operator_handoff_bundle_markdown(payload)

    assert bundle.missing_required_count == 1
    assert bundle.included_count == 0
    assert payload["files"][0]["exists"] is False
    assert "Missing Required Evidence" in markdown
    assert "artifacts/missing.json" in markdown


def test_export_operator_handoff_bundle_no_copy_keeps_checksum_without_inclusion(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "work"
    evidence_path = workdir / "artifacts" / "go_live_smoke.md"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("# smoke\n", encoding="utf-8")

    bundle = export_operator_handoff_bundle(
        workdir=workdir,
        bundle_dir=tmp_path / "bundle",
        evidence_paths=("artifacts/go_live_smoke.md",),
        copy_files=False,
    )
    payload = operator_handoff_bundle_payload(bundle)

    assert bundle.missing_required_count == 0
    assert bundle.included_count == 0
    assert payload["files"][0]["exists"] is True
    assert payload["files"][0]["included"] is False
    assert payload["files"][0]["copied_path"] is None


def test_default_handoff_evidence_paths_include_production_evidence_docs() -> None:
    assert "docs/production_database_revision_alignment.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "docs/production_provider_route_settings.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "docs/production_remote_provider_startup_evidence.md" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "docs/production_remote_provider_user_systemd_evidence.md" in (
        DEFAULT_HANDOFF_EVIDENCE_PATHS
    )
    assert "docs/production_app_host_startup_evidence.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "docs/production_operator_handoff_bundle_evidence.md" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "docs/production_app_identity_validation_evidence.md" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "docs/production_port_cutover_evidence.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "docs/production_app_host_service_restart_evidence.md" in (
        DEFAULT_HANDOFF_EVIDENCE_PATHS
    )
    assert "docs/production_foreground_operations_evidence.md" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "docs/production_foreground_go_live_summary_evidence.md" in (
        DEFAULT_HANDOFF_EVIDENCE_PATHS
    )
    assert "docs/production_foreground_worker_plan_evidence.md" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "artifacts/foreground_operations_validation.json" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "artifacts/foreground_operations_validation.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "artifacts/foreground_go_live_summary.json" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "artifacts/foreground_go_live_summary.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "artifacts/foreground_worker_plan.json" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "artifacts/foreground_worker_plan.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
    assert "artifacts/app_host_service_restart_validation.json" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "artifacts/app_host_service_restart_validation.md" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "artifacts/production_environment_validation.json" in (DEFAULT_HANDOFF_EVIDENCE_PATHS)
    assert "artifacts/production_environment_validation.md" in DEFAULT_HANDOFF_EVIDENCE_PATHS
