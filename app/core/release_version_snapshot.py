"""Release version snapshot and tag command guide."""

import json
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app import __version__ as app_module_version

RELEASE_VERSION_SNAPSHOT_SCHEMA_VERSION = 1

RELEASE_STATUS_READY = "ready"
RELEASE_STATUS_WARNING = "warning"


@dataclass(frozen=True)
class ReleaseGitState:
    branch: str | None
    commit_sha: str | None
    latest_tag: str | None
    status_lines: tuple[str, ...]
    recent_commits: tuple[str, ...]

    @property
    def dirty(self) -> bool:
        return bool(self.status_lines)


@dataclass(frozen=True)
class ReleaseTagCommand:
    code: str
    description: str
    command: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


@dataclass(frozen=True)
class ReleaseVersionSnapshot:
    status: str
    generated_at: datetime
    workdir: str
    release_version: str
    tag_name: str
    app_version: str
    pyproject_version: str | None
    version_consistent: bool
    git_state: ReleaseGitState
    commands: tuple[ReleaseTagCommand, ...]


def build_release_version_snapshot(
    *,
    workdir: Path | str,
    release_version: str | None = None,
    tag_prefix: str = "nex-pcx-v",
    generated_at: datetime | None = None,
    git_state: ReleaseGitState | None = None,
    app_version: str = app_module_version,
) -> ReleaseVersionSnapshot:
    root = Path(workdir)
    pyproject_version = read_pyproject_version(root)
    selected_release_version = release_version or pyproject_version or app_version
    tag_name = f"{tag_prefix}{selected_release_version}"
    selected_git_state = git_state or collect_release_git_state(root)
    version_consistent = _version_consistent(
        app_version=app_version,
        pyproject_version=pyproject_version,
        release_version=selected_release_version,
    )
    commands = _release_tag_commands(tag_name)
    status = (
        RELEASE_STATUS_READY
        if version_consistent and not selected_git_state.dirty and selected_git_state.commit_sha
        else RELEASE_STATUS_WARNING
    )
    return ReleaseVersionSnapshot(
        status=status,
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(root),
        release_version=selected_release_version,
        tag_name=tag_name,
        app_version=app_version,
        pyproject_version=pyproject_version,
        version_consistent=version_consistent,
        git_state=selected_git_state,
        commands=commands,
    )


def release_version_snapshot_payload(
    snapshot: ReleaseVersionSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": RELEASE_VERSION_SNAPSHOT_SCHEMA_VERSION,
        "status": snapshot.status,
        "generated_at": snapshot.generated_at.isoformat(),
        "generated_at_label": snapshot.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": snapshot.workdir,
        "release_version": snapshot.release_version,
        "tag_name": snapshot.tag_name,
        "app_version": snapshot.app_version,
        "pyproject_version": snapshot.pyproject_version,
        "version_consistent": snapshot.version_consistent,
        "git": {
            "branch": snapshot.git_state.branch,
            "commit_sha": snapshot.git_state.commit_sha,
            "latest_tag": snapshot.git_state.latest_tag,
            "dirty": snapshot.git_state.dirty,
            "status_lines": list(snapshot.git_state.status_lines),
            "recent_commits": list(snapshot.git_state.recent_commits),
        },
        "commands": [
            {
                "code": command.code,
                "description": command.description,
                "command": list(command.command),
                "shell_command": command.shell_command,
            }
            for command in snapshot.commands
        ],
    }


def render_release_version_snapshot_markdown(payload: dict[str, object]) -> str:
    git_payload = _dict(payload.get("git"))
    lines = [
        "# NeX_PCX Release Version Snapshot",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Release Version: `{_text(payload.get('release_version'))}`",
        f"- Tag Name: `{_text(payload.get('tag_name'))}`",
        f"- App Version: `{_text(payload.get('app_version'))}`",
        f"- Pyproject Version: `{_text(payload.get('pyproject_version'))}`",
        f"- Version Consistent: `{_text(payload.get('version_consistent'))}`",
        f"- Branch: `{_text(git_payload.get('branch'))}`",
        f"- Commit: `{_text(git_payload.get('commit_sha'))}`",
        f"- Latest Tag: `{_text(git_payload.get('latest_tag'))}`",
        f"- Dirty Worktree: `{_text(git_payload.get('dirty'))}`",
        "",
        "## Tag Commands",
        "",
    ]
    for command in payload.get("commands", []):
        command_payload = _dict(command)
        lines.extend(
            [
                f"### {_text(command_payload.get('code'))}",
                "",
                _text(command_payload.get("description")),
                "",
                "```bash",
                _text(command_payload.get("shell_command")),
                "```",
                "",
            ]
        )

    status_lines = git_payload.get("status_lines") or []
    if status_lines:
        lines.extend(["## Worktree Status", "", "```text"])
        lines.extend(_text(line) for line in status_lines)
        lines.extend(["```", ""])

    recent_commits = git_payload.get("recent_commits") or []
    if recent_commits:
        lines.extend(["## Recent Commits", "", "```text"])
        lines.extend(_text(commit) for commit in recent_commits)
        lines.extend(["```", ""])

    lines.extend(
        [
            "## Release Notes Checklist",
            "",
            "- Confirm the operator handoff bundle has no missing required evidence.",
            "- Confirm the release commit matches the deployed source.",
            "- Create and push the annotated tag only after the worktree is clean.",
            "- Record the tag URL and deployment timestamp in the release note.",
        ]
    )
    return "\n".join(lines)


def collect_release_git_state(workdir: Path) -> ReleaseGitState:
    return ReleaseGitState(
        branch=_git_output(workdir, "rev-parse", "--abbrev-ref", "HEAD"),
        commit_sha=_git_output(workdir, "rev-parse", "HEAD"),
        latest_tag=_git_output(workdir, "describe", "--tags", "--abbrev=0"),
        status_lines=tuple(_git_lines(workdir, "status", "--short")),
        recent_commits=tuple(_git_lines(workdir, "log", "--oneline", "-5")),
    )


def read_pyproject_version(workdir: Path) -> str | None:
    pyproject_path = workdir / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project", {})
    version = project.get("version") if isinstance(project, dict) else None
    return str(version) if version else None


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _release_tag_commands(tag_name: str) -> tuple[ReleaseTagCommand, ...]:
    return (
        ReleaseTagCommand(
            code="create_annotated_tag",
            description="Create an annotated release tag at the current HEAD.",
            command=("git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"),
        ),
        ReleaseTagCommand(
            code="push_release_tag",
            description="Push the release tag to origin after local verification.",
            command=("git", "push", "origin", tag_name),
        ),
        ReleaseTagCommand(
            code="show_release_tag",
            description="Inspect the tag object and associated commit.",
            command=("git", "show", "--stat", tag_name),
        ),
    )


def _version_consistent(
    *,
    app_version: str,
    pyproject_version: str | None,
    release_version: str,
) -> bool:
    versions = [app_version, release_version]
    if pyproject_version is not None:
        versions.append(pyproject_version)
    return len(set(versions)) == 1


def _git_output(workdir: Path, *args: str) -> str | None:
    lines = _git_lines(workdir, *args)
    return lines[0] if lines else None


def _git_lines(workdir: Path, *args: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)
