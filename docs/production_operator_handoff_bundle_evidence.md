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
  --app-url http://127.0.0.1:8000 \
  --provider-host 192.168.20.243 \
  --pretty
```

Result:

- Bundle directory: `artifacts/operator_handoff/latest`
- Source evidence files: `33`
- Included files: `33`
- Missing required files: `0`
- Provider host: `192.168.20.243`
- App URL used for handoff evidence: `http://127.0.0.1:8000`

## Included Production Evidence Docs

The default handoff evidence list now includes:

- `docs/production_database_revision_alignment.md`
- `docs/production_provider_route_settings.md`
- `docs/production_remote_provider_startup_evidence.md`
- `docs/production_remote_provider_user_systemd_evidence.md`
- `docs/production_app_host_startup_evidence.md`
- `docs/production_app_identity_validation_evidence.md`
- `docs/production_port_cutover_evidence.md`
- `docs/production_app_host_service_restart_evidence.md`
- `docs/production_foreground_operations_evidence.md`
- `docs/production_foreground_go_live_summary_evidence.md`
- `artifacts/foreground_operations_validation.json`
- `artifacts/foreground_operations_validation.md`
- `artifacts/foreground_go_live_summary.json`
- `artifacts/foreground_go_live_summary.md`
- `artifacts/app_host_service_restart_validation.json`
- `artifacts/app_host_service_restart_validation.md`
- `artifacts/production_environment_validation.json`
- `artifacts/production_environment_validation.md`

## Operator Note

Port `8000` has been verified for NeX-PCX ownership. Slice 291 records
foreground operation as the accepted supervised pre-CX mode; Slice 290 service
restart validation remains a later hardening check until reviewed systemd units
are installed and rechecked.
