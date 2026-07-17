# Production Foreground Go-Live Summary Evidence

Date: 2026-07-17

## Scope

Slice 292 adds a foreground go-live evidence summary. It reads generated JSON
evidence and produces a single foreground-mode decision record.

Foreground mode treats these signals as required:

- `artifacts/foreground_operations_validation.json`
- `artifacts/production_environment_validation.json`
- `artifacts/go_live_evidence.json`
- `artifacts/go_live_smoke.json`

It treats app-host service restart validation as optional hardening while
systemd registration is intentionally deferred:

- `artifacts/app_host_service_restart_validation.json`

## Command

```bash
./.venv/bin/python scripts/summarize_foreground_go_live.py \
  --json-output artifacts/foreground_go_live_summary.json \
  --markdown-output artifacts/foreground_go_live_summary.md \
  --pretty
```

## Expected Result

The expected status is `warning` when:

- all required evidence files exist and have accepted statuses, and
- foreground operation is acknowledged, or
- service restart hardening is still blocked because systemd units are not
  installed.

The expected status is `blocked` when any required evidence file is missing,
unparseable, or reports an unaccepted status.

## Generated Evidence

- `artifacts/foreground_go_live_summary.json`
- `artifacts/foreground_go_live_summary.md`

## Operator Note

Use this summary as the final foreground-mode review before exporting the
operator handoff bundle. It does not replace the individual evidence files; it
ties them together so the operator can see why a supervised foreground run is
allowed even while service restart validation remains a hardening item.
