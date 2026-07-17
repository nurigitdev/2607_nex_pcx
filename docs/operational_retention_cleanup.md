# NeX_PCX Operational Retention + Cleanup Verification

NeX_PCX stores operational data for troubleshooting and experiment
reproducibility. Go-live operations need a repeatable way to confirm that those
records will not grow without review.

This verification runner reads retention settings and performs cleanup dry-run
previews only. It does not delete database rows or artifact files.

## Command

```bash
./.venv/bin/python scripts/verify_operational_retention.py \
  --json-output artifacts/operational_retention_verification.json \
  --markdown-output artifacts/operational_retention_verification.md \
  --pretty
```

Use `--strict` when warnings should fail the operator gate.

## What It Checks

| Area | Source | Expected Result |
| --- | --- | --- |
| Database URL | `NEX_PCX_DATABASE_URL` | Configured and connectable |
| Admin logs | `app_log_settings` | Enabled, with retention within the recommended window |
| Search logs | `search_logs` retention helpers | Dry-run expired row count is available |
| Provider route operations | route health, contract, and preflight retention helpers | Dry-run expired row counts are available |
| Embedding batch runs | worker batch run retention helpers | Dry-run expired row count is available |
| Artifacts | `artifacts/` under the project root | Old evidence files are visible for operator review |

## Status Meaning

| Status | Meaning |
| --- | --- |
| `ready` | Required settings are available and no warnings were raised |
| `warning` | Verification can run, but a cleanup setting or artifact review needs attention |
| `blocked` | Database access or a required retention query failed |

## Cleanup Boundary

The runner deliberately avoids destructive cleanup. Use the existing admin
cleanup APIs or UI actions when an operator has reviewed the dry-run counts:

- Search log cleanup: `/api/search/logs/cleanup`
- Provider route operations cleanup: `/api/admin/embedding-provider-routes/cleanup`
- Embedding batch run cleanup: `/api/admin/embedding-batch-runs/cleanup`

Keep the generated JSON and Markdown reports with go-live evidence so the
operator can prove which retention settings were active at release time.
