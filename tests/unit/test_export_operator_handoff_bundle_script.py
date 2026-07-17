import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.operator_handoff_bundle import HandoffEvidenceFile, OperatorHandoffBundle


def _load_export_operator_handoff_bundle_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "export_operator_handoff_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_operator_handoff_bundle_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export_operator_handoff_bundle_script = _load_export_operator_handoff_bundle_module()


def make_bundle(*, missing_required_count: int = 0) -> OperatorHandoffBundle:
    files = ()
    if missing_required_count:
        files = (
            HandoffEvidenceFile(
                source_path="artifacts/missing.json",
                exists=False,
                required=True,
                included=False,
            ),
        )
    return OperatorHandoffBundle(
        generated_at=datetime(2026, 7, 17, 14, 15, 16, tzinfo=UTC),
        bundle_dir="/tmp/bundle",
        workdir="/tmp/work",
        git_commit="abc123",
        app_url="http://app",
        provider_host="provider",
        files=files,
    )


def test_main_exports_bundle_and_prints_payload(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_export(
        *,
        workdir,
        bundle_dir,
        git_commit,
        app_url,
        provider_host,
        copy_files,
    ):
        captured["workdir"] = workdir
        captured["bundle_dir"] = bundle_dir
        captured["git_commit"] = git_commit
        captured["app_url"] = app_url
        captured["provider_host"] = provider_host
        captured["copy_files"] = copy_files
        return make_bundle()

    monkeypatch.setattr(
        export_operator_handoff_bundle_script,
        "_git_commit",
        lambda workdir: "detected-sha",
    )
    monkeypatch.setattr(
        export_operator_handoff_bundle_script,
        "export_operator_handoff_bundle",
        fake_export,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_operator_handoff_bundle.py",
            "--workdir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "handoff"),
            "--app-url",
            "http://app.local",
            "--provider-host",
            "provider.local",
            "--no-copy",
            "--pretty",
        ],
    )

    assert export_operator_handoff_bundle_script.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["git_commit"] == "abc123"
    assert captured["git_commit"] == "detected-sha"
    assert captured["copy_files"] is False
    assert captured["app_url"] == "http://app.local"
    assert captured["provider_host"] == "provider.local"


def test_main_returns_nonzero_when_required_evidence_is_missing(monkeypatch) -> None:
    def fake_export(
        *,
        workdir,
        bundle_dir,
        git_commit,
        app_url,
        provider_host,
        copy_files,
    ):
        return make_bundle(missing_required_count=1)

    monkeypatch.setattr(
        export_operator_handoff_bundle_script,
        "_git_commit",
        lambda workdir: None,
    )
    monkeypatch.setattr(
        export_operator_handoff_bundle_script,
        "export_operator_handoff_bundle",
        fake_export,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_operator_handoff_bundle.py", "--git-commit", "manual-sha"],
    )

    assert export_operator_handoff_bundle_script.main() == 1


def test_git_commit_returns_none_on_error(monkeypatch, tmp_path) -> None:
    def fake_run(*args, **kwargs):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(export_operator_handoff_bundle_script.subprocess, "run", fake_run)

    assert export_operator_handoff_bundle_script._git_commit(tmp_path) is None
