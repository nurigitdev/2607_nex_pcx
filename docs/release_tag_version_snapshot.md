# NeX_PCX Release Tag + Version Snapshot

The release version snapshot records the exact source state that should be
tagged for go-live. It does not create a git tag automatically.

Run it after the operator handoff bundle has been exported:

```bash
./.venv/bin/python scripts/export_release_version_snapshot.py \
  --json-output artifacts/release_version_snapshot.json \
  --markdown-output artifacts/release_version_snapshot.md \
  --pretty
```

If the worktree contains expected untracked local files, use `--allow-warning`
to keep the export command from failing while still recording the dirty status.

## What It Records

- Application version from `app.__version__`
- Project version from `pyproject.toml`
- Selected release version and recommended tag name
- Current branch and commit SHA
- Latest existing tag, when one exists
- Worktree status lines
- Recent commit list
- Annotated tag, push, and show commands

## Operator Flow

1. Confirm `artifacts/operator_handoff/latest/manifest.json` has
   `missing_required_count = 0`.
2. Run the release version snapshot export.
3. Confirm the snapshot status is `ready`.
4. Execute the generated `create_annotated_tag` command.
5. Execute the generated `push_release_tag` command.
6. Execute the generated `show_release_tag` command and attach the output to the
   release note.

## Stop Rules

- Stop if `app.__version__`, `pyproject.toml`, and the selected release version
  do not match.
- Stop if the worktree is dirty and the dirty files are not explicitly expected.
- Stop if the commit SHA does not match the source deployed to the application
  host.
