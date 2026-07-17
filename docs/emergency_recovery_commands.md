# NeX_PCX Emergency Recovery Commands

This document is the operator shortcut for urgent recovery. It complements
`docs/operations_runbook.md`; use the runbook for the full procedure and this
index when an incident needs fast triage.

Generate the current command index:

```bash
./.venv/bin/python scripts/render_emergency_recovery_index.py \
  --json-output artifacts/emergency_recovery_index.json \
  --markdown-output artifacts/emergency_recovery_index.md \
  --pretty
```

The generated Markdown groups commands by recovery scenario. Commands marked
with review should be executed only after the operator has recorded the target
job ID, route, or failure scope.

## Fast Triage Order

1. Check application health.
2. Run startup validation if the app or database looks unhealthy.
3. Run shutdown drain check when queues appear stuck.
4. Run provider health and preflight checks before retrying embedding jobs.
5. Retry a single job first when the root cause is uncertain.
6. Save generated JSON and Markdown evidence under `artifacts/`.

## Stop Rules

- Stop retry loops when the same error repeats for the same job or profile.
- Stop destructive cleanup until dry-run counts have been reviewed.
- Stop migration recovery if the active database might be a restore smoke target.
- Stop worker restarts when stale leases immediately reappear after one batch.

## Useful Screens

| Purpose | Screen |
| --- | --- |
| App readiness | `/admin/go-live-readiness` |
| Pipeline queue | `/admin/jobs` |
| Embedding queue | `/admin/embedding-jobs` |
| Provider routes | `/admin/embedding-provider-routes` |
| Logs | `/admin/logs` |
| Search failures | `/search/logs` |

## Evidence To Keep

```bash
artifacts/emergency_recovery_index.json
artifacts/emergency_recovery_index.md
artifacts/shutdown_drain_check.json
artifacts/runtime_config_audit.json
artifacts/go_live_smoke.json
artifacts/operational_retention_verification.json
```
