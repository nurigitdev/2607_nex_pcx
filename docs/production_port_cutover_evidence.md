# Production Port Cutover Evidence

Date: 2026-07-17

## Scope

Slice 289 verified that NeX-PCX can own the production application URL on port
`8000` after the previous process on that port was stopped.

## App URL

- App URL: `http://127.0.0.1:8000`
- Expected app identity: `NeX_PCX`
- `/openapi.json` title: `NeX_PCX`

## Startup Validation

Operations startup validation was run with provider preflight enabled:

- Overall status: `ready`
- Checks: `7`
- Passed: `7`
- Warning: `0`
- Failed: `0`
- Provider routes checked: `4`
- Provider route preflight passed: `4`

Key app checks:

- `app_healthz`: `passed`
- `app_identity`: `passed`

## Production Validation

Production environment validation was regenerated under:

- `artifacts/production_environment_validation.json`
- `artifacts/production_environment_validation.md`

Summary:

- Overall status: `ready`
- Failed guards: `0`
- Warning guards: `0`
- Runtime config audit: `ready`, `9/9` checks passed
- Operations startup validation: `ready`, `7/7` checks passed
- Go-live readiness: `ready`, `11/11` checks passed

## Go-Live Smoke

Go-live HTTP smoke evidence was regenerated under:

- `artifacts/go_live_smoke.json`
- `artifacts/go_live_smoke.md`

Summary:

- Overall status: `ready`
- Checks: `6`
- Passed: `6`
- Warning: `0`
- Failed: `0`

## Evidence Refresh

The following evidence artifacts were refreshed using
`http://127.0.0.1:8000`:

- `artifacts/production_environment_validation.json`
- `artifacts/production_environment_validation.md`
- `artifacts/go_live_evidence.json`
- `artifacts/go_live_evidence.md`
- `artifacts/go_live_smoke.json`
- `artifacts/go_live_smoke.md`
- `artifacts/emergency_recovery_index.json`
- `artifacts/emergency_recovery_index.md`

## Operator Note

Slice 289 used a foreground Uvicorn process to prove ownership of port `8000`.
Slice 290 should replace this with managed app-host services before declaring
the runtime fully operational.
