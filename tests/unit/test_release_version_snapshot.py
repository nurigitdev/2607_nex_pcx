from datetime import UTC, datetime
from pathlib import Path

from app.core.release_version_snapshot import (
    RELEASE_STATUS_READY,
    RELEASE_STATUS_WARNING,
    ReleaseGitState,
    build_release_version_snapshot,
    payload_to_json,
    read_pyproject_version,
    release_version_snapshot_payload,
    render_release_version_snapshot_markdown,
)


def _git_state(*, dirty: bool = False) -> ReleaseGitState:
    return ReleaseGitState(
        branch="master",
        commit_sha="abc123",
        latest_tag="nex-pcx-v0.0.9",
        status_lines=("?? scratch.txt",) if dirty else (),
        recent_commits=("abc123 Slice 280", "def456 Slice 279"),
    )


def test_release_version_snapshot_ready_payload_and_markdown(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'nex-pcx'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )

    snapshot = build_release_version_snapshot(
        workdir=tmp_path,
        git_state=_git_state(),
        generated_at=datetime(2026, 7, 17, 15, 16, 17, tzinfo=UTC),
        app_version="0.1.0",
    )
    payload = release_version_snapshot_payload(snapshot)
    markdown = render_release_version_snapshot_markdown(payload)
    json_text = payload_to_json(payload, pretty=True)

    assert snapshot.status == RELEASE_STATUS_READY
    assert payload["generated_at_label"] == "2026-07-17 15:16:17"
    assert payload["tag_name"] == "nex-pcx-v0.1.0"
    assert payload["version_consistent"] is True
    assert "git tag -a nex-pcx-v0.1.0" in markdown
    assert "Recent Commits" in markdown
    assert '"release_version": "0.1.0"' in json_text


def test_release_version_snapshot_warns_for_dirty_or_mismatched_versions(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'nex-pcx'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )

    dirty_snapshot = build_release_version_snapshot(
        workdir=tmp_path,
        git_state=_git_state(dirty=True),
        app_version="0.1.0",
    )
    mismatch_snapshot = build_release_version_snapshot(
        workdir=tmp_path,
        release_version="0.2.0",
        git_state=_git_state(),
        app_version="0.1.0",
    )

    assert dirty_snapshot.status == RELEASE_STATUS_WARNING
    assert dirty_snapshot.git_state.dirty is True
    assert mismatch_snapshot.status == RELEASE_STATUS_WARNING
    assert mismatch_snapshot.version_consistent is False


def test_read_pyproject_version_handles_missing_or_invalid_file(tmp_path: Path) -> None:
    assert read_pyproject_version(tmp_path) is None

    (tmp_path / "pyproject.toml").write_text("not toml = [", encoding="utf-8")

    assert read_pyproject_version(tmp_path) is None


def test_release_version_snapshot_handles_missing_git_state(tmp_path: Path) -> None:
    snapshot = build_release_version_snapshot(
        workdir=tmp_path,
        release_version="0.1.0",
        tag_prefix="release-",
        git_state=ReleaseGitState(
            branch=None,
            commit_sha=None,
            latest_tag=None,
            status_lines=(),
            recent_commits=(),
        ),
        app_version="0.1.0",
    )
    payload = release_version_snapshot_payload(snapshot)

    assert snapshot.status == RELEASE_STATUS_WARNING
    assert payload["tag_name"] == "release-0.1.0"
    assert payload["git"]["commit_sha"] is None
