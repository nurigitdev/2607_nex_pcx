# NeX_PCX Operator Handoff Bundle

The operator handoff bundle collects go-live evidence into one directory and
writes a manifest with file size and SHA-256 checksums.

Run it after the startup, retention, backup, smoke, emergency recovery, and
shutdown evidence files have been generated:

```bash
./.venv/bin/python scripts/export_operator_handoff_bundle.py \
  --output-dir artifacts/operator_handoff/latest \
  --pretty
```

The command writes:

```bash
artifacts/operator_handoff/latest/manifest.json
artifacts/operator_handoff/latest/handoff.md
artifacts/operator_handoff/latest/evidence/
```

## Exit Code

| Exit Code | Meaning |
| --- | --- |
| `0` | All required evidence files were found |
| `1` | One or more required evidence files were missing |

## Default Evidence List

- `docs/production_database_revision_alignment.md`
- `docs/production_provider_route_settings.md`
- `docs/production_remote_provider_startup_evidence.md`
- `docs/production_remote_provider_user_systemd_evidence.md`
- `docs/production_remote_reranker_user_systemd_evidence.md`
- `docs/production_app_host_startup_evidence.md`
- `docs/production_operator_handoff_bundle_evidence.md`
- `docs/production_app_identity_validation_evidence.md`
- `docs/production_port_cutover_evidence.md`
- `docs/production_app_host_service_restart_evidence.md`
- `docs/production_foreground_launch_evidence.md`
- `docs/production_foreground_shutdown_evidence.md`
- `docs/production_foreground_operations_evidence.md`
- `docs/production_foreground_go_live_summary_evidence.md`
- `docs/production_foreground_worker_plan_evidence.md`
- `docs/production_foreground_worker_runner_evidence.md`
- `docs/production_foreground_app_worker_supervisor_evidence.md`
- `docs/production_foreground_worker_runtime_visibility.md`
- `docs/production_upload_queue_auto_processing_smoke.md`
- `docs/production_foreground_final_handoff_evidence.md`
- `artifacts/foreground_production_launch.json`
- `artifacts/foreground_production_launch.md`
- `artifacts/foreground_production_shutdown.json`
- `artifacts/foreground_production_shutdown.md`
- `artifacts/foreground_operations_validation.json`
- `artifacts/foreground_operations_validation.md`
- `artifacts/foreground_go_live_summary.json`
- `artifacts/foreground_go_live_summary.md`
- `artifacts/foreground_worker_plan.json`
- `artifacts/foreground_worker_plan.md`
- `artifacts/foreground_worker_runner.json`
- `artifacts/foreground_worker_runner.md`
- `artifacts/foreground_app_worker_supervisor.json`
- `artifacts/foreground_app_worker_supervisor.md`
- `artifacts/upload_queue_auto_processing_smoke.json`
- `artifacts/upload_queue_auto_processing_smoke.md`
- `artifacts/foreground_final_handoff.json`
- `artifacts/foreground_final_handoff.md`
- `artifacts/app_host_service_restart_validation.json`
- `artifacts/app_host_service_restart_validation.md`
- `artifacts/production_environment_validation.json`
- `artifacts/production_environment_validation.md`
- `artifacts/go_live_evidence.json`
- `artifacts/go_live_evidence.md`
- `artifacts/shutdown_drain_check.json`
- `artifacts/shutdown_drain_check.md`
- `artifacts/runtime_config_audit.json`
- `artifacts/runtime_config_audit.md`
- `artifacts/backup_restore_smoke.json`
- `artifacts/backup_restore_smoke.md`
- `artifacts/go_live_smoke.json`
- `artifacts/go_live_smoke.md`
- `artifacts/emergency_recovery_index.json`
- `artifacts/emergency_recovery_index.md`
- `artifacts/operational_retention_verification.json`
- `artifacts/operational_retention_verification.md`

## Operator Notes

- Review `missing_required_count` before declaring go-live complete.
- Keep the bundle directory with the release or incident record.
- Do not place raw database credentials or private provider tokens in evidence files.
