# Production Operator Handoff Bundle Evidence

Date: 2026-07-17

## Scope

Slice 287 extended the operator handoff bundle to include the production
evidence documents created while preparing the production database, provider
routes, DGX remote providers, and app-host startup path.

## Port Note

During verification, `http://127.0.0.1:8000` was not serving NeX-PCX. Its
OpenAPI title was `Hermes Agent Hub`, so NeX-PCX go-live HTTP smoke checks
correctly returned `404` for NeX-PCX admin APIs on that port.

The existing process was not stopped. NeX-PCX was launched separately on
`http://127.0.0.1:18080` for non-destructive validation.

## Generated Runtime Evidence

The following runtime evidence files were regenerated under `artifacts/` using
the production database and the DGX provider routes:

- `artifacts/runtime_config_audit.json`
- `artifacts/runtime_config_audit.md`
- `artifacts/shutdown_drain_check.json`
- `artifacts/shutdown_drain_check.md`
- `artifacts/backup_restore_smoke.json`
- `artifacts/backup_restore_smoke.md`
- `artifacts/go_live_evidence.json`
- `artifacts/go_live_evidence.md`
- `artifacts/go_live_smoke.json`
- `artifacts/go_live_smoke.md`
- `artifacts/emergency_recovery_index.json`
- `artifacts/emergency_recovery_index.md`
- `artifacts/operational_retention_verification.json`
- `artifacts/operational_retention_verification.md`

## Bundle Export

The bundle export command was run with the NeX-PCX validation URL:

```bash
./.venv/bin/python scripts/export_operator_handoff_bundle.py \
  --workdir /home/tprover/2607_nex_pcx \
  --output-dir artifacts/operator_handoff/latest \
  --app-url http://127.0.0.1:18080 \
  --provider-host 192.168.20.243 \
  --pretty
```

Result:

- Bundle directory: `artifacts/operator_handoff/latest`
- Source evidence files: `24`
- Included files: `24`
- Missing required files: `0`
- Provider host: `192.168.20.243`
- App URL used for smoke evidence: `http://127.0.0.1:18080`

## Included Production Evidence Docs

The default handoff evidence list now includes:

- `docs/production_database_revision_alignment.md`
- `docs/production_provider_route_settings.md`
- `docs/production_remote_provider_startup_evidence.md`
- `docs/production_remote_provider_user_systemd_evidence.md`
- `docs/production_app_host_startup_evidence.md`
- `docs/production_app_identity_validation_evidence.md`
- `docs/production_port_cutover_evidence.md`
- `artifacts/production_environment_validation.json`
- `artifacts/production_environment_validation.md`

## Operator Note

Before binding NeX-PCX to production port `8000`, stop or move the process
currently serving `Hermes Agent Hub` on that port. Re-run go-live smoke and the
handoff bundle export with the final production URL after the port is reassigned.
