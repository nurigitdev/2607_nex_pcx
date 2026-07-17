import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.release_version_snapshot import (
    RELEASE_STATUS_READY,
    RELEASE_STATUS_WARNING,
    ReleaseGitState,
    ReleaseVersionSnapshot,
)


def _load_export_release_version_snapshot_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "export_release_version_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_release_version_snapshot_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export_release_version_snapshot = _load_export_release_version_snapshot_module()


def make_snapshot(status: str) -> ReleaseVersionSnapshot:
    return ReleaseVersionSnapshot(
        status=status,
        generated_at=datetime(2026, 7, 17, 16, 17, 18, tzinfo=UTC),
        workdir="/tmp/project",
        release_version="0.1.0",
        tag_name="nex-pcx-v0.1.0",
        app_version="0.1.0",
        pyproject_version="0.1.0",
        version_consistent=True,
        git_state=ReleaseGitState(
            branch="master",
            commit_sha="abc123",
            latest_tag=None,
            status_lines=(),
            recent_commits=("abc123 Slice 280",),
        ),
        commands=(),
    )


def test_main_writes_outputs_and_passes_options(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "release" / "snapshot.json"
    markdown_output = tmp_path / "release" / "snapshot.md"

    def fake_build_snapshot(*, workdir, release_version, tag_prefix):
        captured["workdir"] = workdir
        captured["release_version"] = release_version
        captured["tag_prefix"] = tag_prefix
        return make_snapshot(RELEASE_STATUS_READY)

    monkeypatch.setattr(
        export_release_version_snapshot,
        "build_release_version_snapshot",
        fake_build_snapshot,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_release_version_snapshot.py",
            "--workdir",
            str(tmp_path),
            "--release-version",
            "0.1.1",
            "--tag-prefix",
            "release-",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    assert export_release_version_snapshot.main() == 0
    assert captured == {
        "workdir": tmp_path,
        "release_version": "0.1.1",
        "tag_prefix": "release-",
    }
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "ready"
    assert "Release Version Snapshot" in markdown_output.read_text(encoding="utf-8")


def test_main_handles_warning_exit_codes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        export_release_version_snapshot,
        "build_release_version_snapshot",
        lambda *, workdir, release_version, tag_prefix: make_snapshot(
            RELEASE_STATUS_WARNING
        ),
    )
    monkeypatch.setattr(sys, "argv", ["export_release_version_snapshot.py"])

    assert export_release_version_snapshot.main() == 1
    assert '"status": "warning"' in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_release_version_snapshot.py", "--allow-warning"],
    )

    assert export_release_version_snapshot.main() == 0
