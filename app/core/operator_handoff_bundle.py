"""Operator handoff bundle export for NeX_PCX go-live evidence."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

OPERATOR_HANDOFF_BUNDLE_VERSION = 1

DEFAULT_HANDOFF_EVIDENCE_PATHS = (
    "docs/production_database_revision_alignment.md",
    "docs/production_provider_route_settings.md",
    "docs/production_remote_provider_startup_evidence.md",
    "docs/production_remote_provider_user_systemd_evidence.md",
    "docs/production_app_host_startup_evidence.md",
    "docs/production_operator_handoff_bundle_evidence.md",
    "docs/production_app_identity_validation_evidence.md",
    "docs/production_port_cutover_evidence.md",
    "docs/production_app_host_service_restart_evidence.md",
    "docs/production_foreground_operations_evidence.md",
    "docs/production_foreground_go_live_summary_evidence.md",
    "artifacts/foreground_operations_validation.json",
    "artifacts/foreground_operations_validation.md",
    "artifacts/foreground_go_live_summary.json",
    "artifacts/foreground_go_live_summary.md",
    "artifacts/app_host_service_restart_validation.json",
    "artifacts/app_host_service_restart_validation.md",
    "artifacts/production_environment_validation.json",
    "artifacts/production_environment_validation.md",
    "artifacts/go_live_evidence.json",
    "artifacts/go_live_evidence.md",
    "artifacts/shutdown_drain_check.json",
    "artifacts/shutdown_drain_check.md",
    "artifacts/runtime_config_audit.json",
    "artifacts/runtime_config_audit.md",
    "artifacts/backup_restore_smoke.json",
    "artifacts/backup_restore_smoke.md",
    "artifacts/go_live_smoke.json",
    "artifacts/go_live_smoke.md",
    "artifacts/emergency_recovery_index.json",
    "artifacts/emergency_recovery_index.md",
    "artifacts/operational_retention_verification.json",
    "artifacts/operational_retention_verification.md",
)


@dataclass(frozen=True)
class HandoffEvidenceFile:
    source_path: str
    exists: bool
    required: bool
    included: bool
    size_bytes: int | None = None
    sha256: str | None = None
    copied_path: str | None = None


@dataclass(frozen=True)
class OperatorHandoffBundle:
    generated_at: datetime
    bundle_dir: str
    workdir: str
    git_commit: str | None
    app_url: str
    provider_host: str
    files: tuple[HandoffEvidenceFile, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def included_count(self) -> int:
        return sum(1 for file in self.files if file.included)

    @property
    def missing_required_count(self) -> int:
        return sum(1 for file in self.files if file.required and not file.exists)


def export_operator_handoff_bundle(
    *,
    workdir: Path | str,
    bundle_dir: Path | str,
    evidence_paths: tuple[str, ...] = DEFAULT_HANDOFF_EVIDENCE_PATHS,
    git_commit: str | None = None,
    app_url: str = "http://127.0.0.1:8000",
    provider_host: str = "192.168.20.243",
    copy_files: bool = True,
    generated_at: datetime | None = None,
) -> OperatorHandoffBundle:
    root = Path(workdir)
    output_dir = Path(bundle_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = tuple(
        _collect_evidence_file(
            root=root,
            output_dir=output_dir,
            relative_path=relative_path,
            copy_file=copy_files,
        )
        for relative_path in evidence_paths
    )
    bundle = OperatorHandoffBundle(
        generated_at=generated_at or datetime.now(UTC),
        bundle_dir=str(output_dir),
        workdir=str(root),
        git_commit=git_commit,
        app_url=app_url.rstrip("/"),
        provider_host=provider_host,
        files=files,
    )
    payload = operator_handoff_bundle_payload(bundle)
    _write_text(output_dir / "manifest.json", payload_to_json(payload, pretty=True) + "\n")
    _write_text(output_dir / "handoff.md", render_operator_handoff_bundle_markdown(payload))
    return bundle


def operator_handoff_bundle_payload(bundle: OperatorHandoffBundle) -> dict[str, object]:
    return {
        "version": OPERATOR_HANDOFF_BUNDLE_VERSION,
        "generated_at": bundle.generated_at.isoformat(),
        "generated_at_label": bundle.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "bundle_dir": bundle.bundle_dir,
        "workdir": bundle.workdir,
        "git_commit": bundle.git_commit,
        "app_url": bundle.app_url,
        "provider_host": bundle.provider_host,
        "file_count": bundle.file_count,
        "included_count": bundle.included_count,
        "missing_required_count": bundle.missing_required_count,
        "files": [
            {
                "source_path": file.source_path,
                "exists": file.exists,
                "required": file.required,
                "included": file.included,
                "size_bytes": file.size_bytes,
                "sha256": file.sha256,
                "copied_path": file.copied_path,
            }
            for file in bundle.files
        ],
    }


def render_operator_handoff_bundle_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# NeX_PCX Operator Handoff Bundle",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Workdir: `{_text(payload.get('workdir'))}`",
        f"- Bundle Dir: `{_text(payload.get('bundle_dir'))}`",
        f"- Git Commit: `{_text(payload.get('git_commit'))}`",
        f"- App URL: `{_text(payload.get('app_url'))}`",
        f"- Provider Host: `{_text(payload.get('provider_host'))}`",
        f"- Included Files: {_text(payload.get('included_count'))}",
        f"- Missing Required Files: {_text(payload.get('missing_required_count'))}",
        "",
        "## Evidence Files",
        "",
        "| Source | Included | Size | SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    for file in payload.get("files", []):
        file_payload = _dict(file)
        included = "yes" if file_payload.get("included") else "no"
        lines.append(
            "| "
            f"{_md_cell(file_payload.get('source_path'))} | "
            f"{_md_cell(included)} | "
            f"{_md_cell(file_payload.get('size_bytes'))} | "
            f"{_md_cell(file_payload.get('sha256'))} |"
        )

    missing = [
        _dict(file).get("source_path")
        for file in payload.get("files", [])
        if _dict(file).get("required") and not _dict(file).get("exists")
    ]
    if missing:
        lines.extend(["", "## Missing Required Evidence", ""])
        for source_path in missing:
            lines.append(f"- `{_text(source_path)}`")

    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- Keep `manifest.json` with the release or incident record.",
            "- Review missing evidence before declaring go-live complete.",
            "- Do not add database passwords or private provider tokens to this bundle.",
        ]
    )
    return "\n".join(lines)


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _collect_evidence_file(
    *,
    root: Path,
    output_dir: Path,
    relative_path: str,
    copy_file: bool,
) -> HandoffEvidenceFile:
    source = root / relative_path
    if not source.exists() or not source.is_file():
        return HandoffEvidenceFile(
            source_path=relative_path,
            exists=False,
            required=True,
            included=False,
        )

    digest = _sha256_file(source)
    copied_path = None
    if copy_file:
        destination = output_dir / "evidence" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_path = str(destination)

    return HandoffEvidenceFile(
        source_path=relative_path,
        exists=True,
        required=True,
        included=copy_file,
        size_bytes=source.stat().st_size,
        sha256=digest,
        copied_path=copied_path,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("|", "\\|").replace("\n", " ")
