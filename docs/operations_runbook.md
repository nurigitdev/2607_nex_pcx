# NeX_PCX Operations Runbook

This runbook is the operator-facing checklist for starting, validating, stopping,
and recovering a NeX_PCX environment. It complements the provider-specific
procedure in `docs/provider_operations_playbook.md`.

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

1. Move to the application directory and activate the virtual environment.

   ```bash
   cd /home/tprover/2607_nex_pcx
   source .venv/bin/activate
   ```

2. Confirm runtime values are present.

   ```bash
   test -n "${NEX_PCX_DATABASE_URL:-}"
   test -n "${NEX_PCX_UPLOAD_STORAGE_DIR:-}"
   test -n "${NEX_PCX_MODELS_DIR:-}"
   ```

3. Apply database migrations.

   ```bash
   bash scripts/migrate.sh upgrade head
   ```

4. Start the main web application.

   ```bash
   ./.venv/bin/uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
   ```

5. Confirm application health.

   ```bash
   curl -fsS http://127.0.0.1:8000/healthz
   ```

6. Confirm remote provider health from the application host.

   ```bash
   curl -fsS http://192.168.20.243:9101/healthz
   curl -fsS http://192.168.20.243:9102/healthz
   curl -fsS http://192.168.20.243:9103/healthz
   ```

7. Run provider route preflight and store snapshots.

   ```bash
   ./.venv/bin/python scripts/preflight_provider_routes.py
   ```

8. Open the readiness screens.

   - `/admin/go-live-readiness`
   - `/admin/embedding-provider-routes`
   - `/admin/jobs`
   - `/admin/embedding-jobs`

9. Start one or more pipeline workers.

   ```bash
   ./.venv/bin/python scripts/process_pipeline_job.py --chunk-policy-names heading_512_64,heading_1000_200,heading_1500_200
   ```

10. Start one or more embedding workers with route-aware provider selection.

    ```bash
    ./.venv/bin/python scripts/process_embedding_job.py --provider-source route --require-route-readiness --limit 20
    ```

11. Confirm queues drain and new search data is visible.

    - Check `/admin/jobs` for pipeline job state.
    - Check `/admin/embedding-jobs` for embedding job state.
    - Check `/admin/embedding-coverage` and `/admin/multi-policy-ingestion-coverage`.
    - Run a small search comparison from `/search`.

## Shutdown Checklist

1. Pause scheduled ingestion, scheduled preflight, and manual worker starts.
2. Let running workers finish their current batch when possible.
3. Stop embedding workers before stopping remote embedding providers.
4. Stop pipeline workers after current extraction/chunking jobs finish.
5. Stop the main web application.
6. Stop remote providers on the DGX host in this order:
   - Qwen provider on port `9103`
   - BGE provider on port `9102`
   - KURE provider on port `9101`
7. Capture final operator evidence:
   - `/admin/go-live-readiness`
   - `/admin/embedding-provider-routes`
   - `/admin/jobs`
   - `/admin/embedding-jobs`
   - `/admin/logs`
8. Record any unfinished job IDs before maintenance begins.

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

## Evidence To Keep

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
