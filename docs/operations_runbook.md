# NeX_PCX Operations Runbook

This runbook is the operator-facing checklist for starting, validating, stopping,
and recovering a NeX_PCX environment. It complements the provider-specific
procedure in `docs/provider_operations_playbook.md`.
App-host service templates are documented in `docs/service_process_startup.md`.
Backup and restore rehearsal is documented in `docs/backup_restore_smoke.md`.
HTTP go-live smoke checks are documented in `docs/go_live_smoke.md`.
Production environment final validation is documented in
`docs/production_environment_validation.md`.
Operational retention verification is documented in
`docs/operational_retention_cleanup.md`.
Emergency recovery commands are indexed in
`docs/emergency_recovery_commands.md`.
Operator handoff bundles are documented in
`docs/operator_handoff_bundle.md`.
Release tag/version snapshots are documented in
`docs/release_tag_version_snapshot.md`.

## Scope

- Main FastAPI application and admin UI.
- PostgreSQL plus pgvector schema and migrations.
- Upload, extraction, pipeline, chunking, embedding, and search workers.
- Remote embedding providers for KURE, BGE, and Qwen profiles.
- Go-live readiness, provider route readiness, logs, and worker monitors.

## Required Runtime Values

Set these values in the process manager, shell profile, or deployment secret
store. Keep database passwords out of this document and out of committed files.

```bash
export NEX_PCX_DATABASE_URL="postgresql://<user>:<password>@<host>:5432/<database>"
export NEX_PCX_UPLOAD_STORAGE_DIR="/srv/nex_pcx/uploads"
export NEX_PCX_MODELS_DIR="/srv/nex_pcx/models"
export NEX_PCX_EMBEDDING_PROVIDER_MODE="remote"
export NEX_PCX_EMBEDDING_PROVIDER_SOURCE="route"
export NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS="true"
export NEX_PCX_EMBEDDING_ROUTE_READINESS_FAILURE_MODE="defer"
export NEX_PCX_EMBEDDING_ROUTE_READINESS_DEFER_SECONDS="300"
```

Recommended provider route base URLs for the current DGX development server:

| Profile | Base URL |
| --- | --- |
| `kure_v1` | `http://192.168.20.243:9101` |
| `bge_m3` | `http://192.168.20.243:9102` |
| `qwen3_4b_1000` | `http://192.168.20.243:9103` |
| `qwen3_4b_2560` | `http://192.168.20.243:9103` |

## Startup Checklist

1. Generate and review app-host service templates when bootstrapping a host.

   ```bash
   ./.venv/bin/python scripts/render_service_startup_templates.py \
     --workdir /home/tprover/2607_nex_pcx \
     --user nexpcx \
     --output-dir deployment \
     --write \
     --pretty
   ```

   Review `deployment/env/nex-pcx.env` and the generated systemd units before
   installing them. Keep environment files with real credentials out of git.
   Use `--user-systemd` when the app host must run units under
   `systemctl --user` instead of the system unit directory.

2. Move to the application directory and activate the virtual environment.

   ```bash
   cd /home/tprover/2607_nex_pcx
   source .venv/bin/activate
   ```

3. Confirm runtime values are present.

   ```bash
   test -n "${NEX_PCX_DATABASE_URL:-}"
   test -n "${NEX_PCX_UPLOAD_STORAGE_DIR:-}"
   test -n "${NEX_PCX_MODELS_DIR:-}"
   ```

4. Apply database migrations.

   ```bash
   bash scripts/migrate.sh upgrade head
   ```

5. Start the main web application with foreground app and worker supervisor
   evidence.

   ```bash
   ./.venv/bin/python scripts/run_foreground_app_worker_supervisor.py \
     --pipeline-limit 1 \
     --embedding-limit-per-profile 5 \
     --worker-cycle-interval-seconds 5 \
     --json-output artifacts/foreground_app_worker_supervisor.json \
     --markdown-output artifacts/foreground_app_worker_supervisor.md \
     --pretty
   ```

   Run this in a supervised terminal. The runner starts Uvicorn, writes
   supervisor and web PID files, and polls the queue through bounded worker
   cycles. Use `--dry-run` first when you only want to verify the command and
   evidence paths. Use the standalone `run_foreground_production_app.py` only
   for diagnostics where queue processing must intentionally remain disabled.

6. For a controlled pre-CX run, foreground operation is acceptable when an
   operator keeps the session alive and acknowledges that there is no automatic
   restart after failure, logout, or reboot. Validate that foreground mode is
   the active operating assumption:

   ```bash
   ./.venv/bin/python scripts/validate_foreground_operations.py \
     --app-url http://127.0.0.1:8000 \
     --acknowledge-no-auto-restart \
     --json-output artifacts/foreground_operations_validation.json \
     --markdown-output artifacts/foreground_operations_validation.md \
     --pretty
   ```

   A `warning` status is expected in foreground mode because the no-auto-restart
   limitation is explicitly recorded. Treat `blocked` as a stop signal.

7. Confirm application health.

   ```bash
   curl -fsS http://127.0.0.1:8000/healthz
   ```

8. Confirm foreground worker runtime visibility.

   ```bash
   curl -fsS http://127.0.0.1:8000/api/admin/foreground-worker-runtime
   ```

   `ready` means supervisor evidence is running and both supervisor/web
   processes are alive. `warning` is acceptable when only dry-run evidence is
   present. `blocked` means stale processes or worker failures need operator
   action.

9. Run the upload queue auto-processing smoke after supervisor startup.

   ```bash
   ./.venv/bin/python scripts/run_upload_queue_auto_processing_smoke.py \
     --app-url http://127.0.0.1:8000 \
     --poll-attempts 12 \
     --poll-interval-seconds 5 \
     --json-output artifacts/upload_queue_auto_processing_smoke.json \
     --markdown-output artifacts/upload_queue_auto_processing_smoke.md \
     --pretty
   ```

   Use `--dry-run` while DGX resources are under pressure. Live smoke uploads a
   small Markdown file and can trigger foreground worker processing.

10. Confirm remote provider health from the application host.

   ```bash
   curl -fsS http://192.168.20.243:9101/healthz
   curl -fsS http://192.168.20.243:9102/healthz
   curl -fsS http://192.168.20.243:9103/healthz
   ```

11. Run provider route preflight and store snapshots.

   ```bash
   ./.venv/bin/python scripts/preflight_provider_routes.py
   ```

12. Run the runtime configuration audit.

   ```bash
   ./.venv/bin/python scripts/audit_runtime_config.py \
     --project-root /home/tprover/2607_nex_pcx \
     --json-output artifacts/runtime_config_audit.json \
     --markdown-output artifacts/runtime_config_audit.md \
     --pretty
   ```

   Use `--strict` when warnings should fail the operator gate. Database URLs are
   masked in both JSON and Markdown output.

13. Run the production environment final validation.

   ```bash
   ./.venv/bin/python scripts/validate_production_environment.py \
     --expected-database-name nex_pcx_app \
     --app-url http://127.0.0.1:8000 \
     --json-output artifacts/production_environment_validation.json \
     --markdown-output artifacts/production_environment_validation.md \
     --pretty
   ```

   Add `--run-provider-preflight` only when this validation should persist
   provider health and contract snapshots.

14. Run the startup validation runner.

   ```bash
   ./.venv/bin/python scripts/validate_operations_startup.py \
     --app-url http://127.0.0.1:8000 \
     --run-provider-preflight \
     --pretty
   ```

   Use `--strict` when warnings should fail the startup gate. Omit
   `--run-provider-preflight` for a dry operator check that does not create
   provider health or contract snapshots. When `--app-url` is supplied, the
   runner checks both `/healthz` and `/openapi.json` so another FastAPI process
   cannot pass as NeX-PCX by exposing only a compatible health endpoint.

   After the app-host units are installed, verify managed-service restart
   readiness. The default mode is read-only and confirms that systemd can see
   the web and worker units, that they are active, and that restart policies are
   configured:

   ```bash
   ./.venv/bin/python scripts/validate_app_host_service_restart.py \
     --scope user \
     --app-url http://127.0.0.1:8000 \
     --json-output artifacts/app_host_service_restart_validation.json \
     --markdown-output artifacts/app_host_service_restart_validation.md \
     --pretty
   ```

   Use `--scope system` for system-level units. Use `--restart-web` only during
   a controlled maintenance window; it restarts `nex-pcx-web.service` and then
   verifies `/healthz` plus `/openapi.json` identity.

15. Export the go-live evidence snapshot.

   ```bash
   ./.venv/bin/python scripts/export_go_live_evidence.py \
     --app-url http://127.0.0.1:8000 \
     --run-provider-preflight \
     --json-output artifacts/go_live_evidence.json \
     --markdown-output artifacts/go_live_evidence.md \
     --pretty
   ```

   The JSON file is intended for automated review. The Markdown file is intended
   for an operator handoff note. Database credentials are masked in both files.

16. Verify operational retention and cleanup dry-run previews.

   ```bash
   ./.venv/bin/python scripts/verify_operational_retention.py \
     --json-output artifacts/operational_retention_verification.json \
     --markdown-output artifacts/operational_retention_verification.md \
     --pretty
   ```

   Use `--strict` when retention warnings should fail the startup gate. The
   runner does not delete database rows or artifact files.

17. Generate the backup and restore smoke report.

   ```bash
   ./.venv/bin/python scripts/run_backup_restore_smoke.py \
     --backup-dir artifacts/backups/latest \
     --json-output artifacts/backup_restore_smoke.json \
     --markdown-output artifacts/backup_restore_smoke.md \
     --pretty
   ```

   Add `--restore-database-url` when an empty restore target is prepared. The
   runner blocks when the restore URL matches the source database URL.

18. Run the HTTP go-live smoke runner.

   ```bash
   ./.venv/bin/python scripts/run_go_live_smoke.py \
     --app-url http://127.0.0.1:8000 \
     --json-output artifacts/go_live_smoke.json \
     --markdown-output artifacts/go_live_smoke.md \
     --pretty
   ```

19. Summarize foreground go-live evidence after the foreground validation and
   core evidence files have been generated.

   ```bash
   ./.venv/bin/python scripts/summarize_foreground_go_live.py \
     --json-output artifacts/foreground_go_live_summary.json \
     --markdown-output artifacts/foreground_go_live_summary.md \
     --pretty
   ```

   `warning` is acceptable when all required evidence passes and the only
   warning comes from foreground operation or deferred service hardening.

20. Render the foreground worker command plan before processing queue work.

   ```bash
   ./.venv/bin/python scripts/render_foreground_worker_plan.py \
     --json-output artifacts/foreground_worker_plan.json \
     --markdown-output artifacts/foreground_worker_plan.md \
     --pretty
   ```

   Review the generated bounded commands before running pipeline or embedding
   workers in a supervised terminal.

21. Run the guarded foreground worker runner when queue work must be processed
   during foreground operation.

   ```bash
   ./.venv/bin/python scripts/run_foreground_workers.py \
     --pipeline-limit 1 \
     --embedding-limit-per-profile 5 \
     --guard-health-timeout-seconds 5 \
     --max-provider-health-elapsed-ms 5000 \
     --json-output artifacts/foreground_worker_runner.json \
     --markdown-output artifacts/foreground_worker_runner.md \
     --pretty
   ```

   Run with `--dry-run` first. The default guard keeps large Qwen chunks queued
   when the pending token count exceeds the conservative foreground limit. After
   DGX memory and swap are stable, rerun with `--no-default-qwen-token-guard`,
   `--pipeline-limit 0`, and a small `--embedding-limit-per-profile` to process
   remaining Qwen work under supervision.

22. Render the final foreground handoff checklist.

   ```bash
   ./.venv/bin/python scripts/summarize_foreground_final_handoff.py \
     --json-output artifacts/foreground_final_handoff.json \
     --markdown-output artifacts/foreground_final_handoff.md \
     --pretty
   ```

   A `warning` status is acceptable for supervised foreground operation when
   failed checks are zero. A `blocked` status must stop handoff.

23. Generate the emergency recovery command index.

   ```bash
   ./.venv/bin/python scripts/render_emergency_recovery_index.py \
     --json-output artifacts/emergency_recovery_index.json \
     --markdown-output artifacts/emergency_recovery_index.md \
     --pretty
   ```

   Keep this index near the operator during go-live so incident recovery
   commands are available without searching through the full runbook.

24. Export the operator handoff bundle.

   ```bash
   ./.venv/bin/python scripts/export_operator_handoff_bundle.py \
     --output-dir artifacts/operator_handoff/latest \
     --pretty
   ```

   The command exits with `1` when required evidence is missing. Review
   `artifacts/operator_handoff/latest/handoff.md` before declaring go-live
   complete.

25. Export the release version snapshot.

   ```bash
   ./.venv/bin/python scripts/export_release_version_snapshot.py \
     --json-output artifacts/release_version_snapshot.json \
     --markdown-output artifacts/release_version_snapshot.md \
     --pretty
   ```

   Review the generated tag commands. Create and push the annotated release tag
   only after the worktree is clean and the deployed commit is confirmed.

19. Open the readiness screens.

   - `/admin/go-live-readiness`
   - `/admin/embedding-provider-routes`
   - `/admin/jobs`
   - `/admin/embedding-jobs`

20. Start one or more pipeline workers.

   ```bash
   ./.venv/bin/python scripts/process_pipeline_job.py \
     --chunk-policy-names heading_512_64 heading_1000_200 heading_1500_200
   ```

21. Start one or more embedding workers with route-aware provider selection.

    ```bash
    ./.venv/bin/python scripts/process_embedding_job.py --provider-source route --require-route-readiness --limit 20
    ```

22. Confirm queues drain and new search data is visible.

    - Check `/admin/jobs` for pipeline job state.
    - Check `/admin/embedding-jobs` for embedding job state.
    - Check `/admin/embedding-coverage` and `/admin/multi-policy-ingestion-coverage`.
    - Run a small search comparison from `/search`.

## Shutdown Checklist

1. Pause scheduled ingestion, scheduled preflight, and manual worker starts.
2. Run the shutdown drain check runner and save the evidence.

   ```bash
   ./.venv/bin/python scripts/check_shutdown_drain.py \
     --json-output artifacts/shutdown_drain_check.json \
     --markdown-output artifacts/shutdown_drain_check.md \
     --pretty
   ```

   - `ready`: queues are drained and shutdown can proceed.
   - `warning`: no running blockers exist, but queued/pending or retryable work remains.
   - `blocked`: running, stale running, or exhausted failure signals must be handled first.

3. Let running workers finish their current batch when possible.
4. Stop embedding workers before stopping remote embedding providers.
5. Stop pipeline workers after current extraction/chunking jobs finish.
6. Stop the main web application with foreground shutdown evidence.

   ```bash
   ./.venv/bin/python scripts/stop_foreground_production_app.py \
     --json-output artifacts/foreground_production_shutdown.json \
     --markdown-output artifacts/foreground_production_shutdown.md \
     --pid-file artifacts/foreground_production_launch.pid \
     --log-file artifacts/foreground_production_shutdown.log \
     --pretty
   ```

   Run with `--dry-run` first when verifying the PID file and command guard.
   The actual stop sends `SIGTERM` only when the PID belongs to the expected
   NeX-PCX foreground Uvicorn process.

7. Stop remote providers on the DGX host in this order:
   - Qwen provider on port `9103`
   - BGE provider on port `9102`
   - KURE provider on port `9101`
8. Capture final operator evidence:
   - `/admin/go-live-readiness`
   - `/admin/embedding-provider-routes`
   - `/admin/jobs`
   - `/admin/embedding-jobs`
   - `/admin/logs`
9. Record any unfinished job IDs before maintenance begins.

## Daily Operations Checklist

- Review `/admin/go-live-readiness` first.
- Review provider route operations summary and unacknowledged alerts.
- Run due provider route preflight schedules if the UI reports due schedules.

  ```bash
  ./.venv/bin/python scripts/run_scheduled_provider_preflight.py --limit 20
  ```

- Review recent operational failures on the dashboard.
- Review stale embedding leases and retry failed embedding jobs when the cause
  has been fixed.
- Run `scripts/verify_operational_retention.py` and review dry-run cleanup
  counts before executing any destructive cleanup.
- Confirm extraction artifact quality for newly uploaded file types.
- Run a small known query from `/search` against all active profiles.

## Restart After Failure

1. Check logs first: `/admin/logs`.
2. Confirm database connectivity and migration state.
3. Restart the main application and check `/healthz`.
4. Restart remote providers and run provider route preflight.
5. Reopen `/admin/go-live-readiness`.
6. Recover stale leases before retrying failed jobs.
7. Retry failed embedding jobs only after provider readiness is green.
8. Use `artifacts/emergency_recovery_index.md` for the exact command sequence
   and stop conditions for the active incident type.

## Evidence To Keep

- Go-live evidence JSON and Markdown:

  ```bash
  artifacts/go_live_evidence.json
  artifacts/go_live_evidence.md
  ```

- Shutdown drain check JSON and Markdown:

  ```bash
  artifacts/shutdown_drain_check.json
  artifacts/shutdown_drain_check.md
  ```

- Runtime configuration audit JSON and Markdown:

  ```bash
  artifacts/runtime_config_audit.json
  artifacts/runtime_config_audit.md
  ```

- Production environment validation JSON and Markdown:

  ```bash
  artifacts/production_environment_validation.json
  artifacts/production_environment_validation.md
  ```

- Backup and restore smoke JSON and Markdown:

  ```bash
  artifacts/backup_restore_smoke.json
  artifacts/backup_restore_smoke.md
  ```

- Go-live HTTP smoke JSON and Markdown:

  ```bash
  artifacts/go_live_smoke.json
  artifacts/go_live_smoke.md
  ```

- Emergency recovery index JSON and Markdown:

  ```bash
  artifacts/emergency_recovery_index.json
  artifacts/emergency_recovery_index.md
  ```

- Operational retention verification JSON and Markdown:

  ```bash
  artifacts/operational_retention_verification.json
  artifacts/operational_retention_verification.md
  ```

- Operator handoff bundle manifest and Markdown:

  ```bash
  artifacts/operator_handoff/latest/manifest.json
  artifacts/operator_handoff/latest/handoff.md
  ```

- Release version snapshot JSON and Markdown:

  ```bash
  artifacts/release_version_snapshot.json
  artifacts/release_version_snapshot.md
  ```

- Git commit SHA and deployment timestamp.
- Alembic head revision.
- Provider route IDs and active profile list.
- Latest provider preflight run IDs.
- Go-live readiness status and screenshot.
- Queue backlog counts before and after maintenance.
- Failure IDs or log IDs for every manual recovery action.

## Escalation Guide

| Symptom | First Check | Next Action |
| --- | --- | --- |
| Upload succeeds but no chunks appear | `/admin/jobs` | Restart pipeline worker or recover stale lease |
| Chunks exist but embeddings do not appear | `/admin/embedding-jobs` | Run provider preflight, then retry failed jobs |
| Search returns no results | `/admin/embedding-coverage` | Confirm profile and chunk policy coverage |
| Provider route blocked | `/admin/embedding-provider-routes` | Check health, contract snapshot, and alerts |
| Go-live readiness blocked | `/admin/go-live-readiness` | Resolve the blocking section before release |
