# NeX_PCX Backup + Restore Smoke Guide

This guide defines the minimum backup and restore rehearsal before go-live.
The smoke runner does not restore into the source database. It checks guardrails
and emits a command manifest that an operator can review and execute.

## What Must Be Backed Up

| Area | Why |
| --- | --- |
| PostgreSQL database | File metadata, permissions, chunks, embeddings, jobs, logs, provider routes |
| Upload storage directory | Original source files uploaded by users |
| `artifacts/` directory | Go-live evidence, benchmark results, operator exports |
| Git commit/tag | Exact source code used for the environment |

## Generate Smoke Report

```bash
./.venv/bin/python scripts/run_backup_restore_smoke.py \
  --backup-dir artifacts/backups/latest \
  --json-output artifacts/backup_restore_smoke.json \
  --markdown-output artifacts/backup_restore_smoke.md \
  --pretty
```

When a restore target exists, include it so the runner can verify that the
restore database URL is distinct from the source:

```bash
./.venv/bin/python scripts/run_backup_restore_smoke.py \
  --restore-database-url postgresql://<restore-user>:<password>@127.0.0.1:5432/nex_pcx_restore \
  --backup-dir artifacts/backups/latest \
  --pretty
```

The generated command manifest uses `${NEX_PCX_DATABASE_URL}` and
`${NEX_PCX_RESTORE_DATABASE_URL}` placeholders so credentials do not land in
operator evidence files.

## Guardrails

- The source database URL must be configured.
- The restore database URL must not match the source database URL.
- `pg_dump`, `pg_restore`, and `psql` should be available on `PATH`.
- Upload storage and artifact paths are checked before the command manifest is trusted.
- Database URLs are masked in report metadata, but command manifests use environment
  placeholders where possible.

## Restore Rehearsal

1. Create an empty restore database.
2. Run the smoke runner with `--restore-database-url`.
3. Execute the generated `database_backup` and `dump_list_smoke` commands.
4. Restore into the empty restore database, never into the source database.
5. Start the app against the restore database only long enough to run startup validation.
