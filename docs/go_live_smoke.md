# NeX_PCX End-to-End Go-Live Smoke Guide

This smoke checks the running application over HTTP. It is intentionally
read-only: it verifies process health, readiness, queue APIs, provider readiness,
and search operations APIs without creating user data.

## Run Smoke

```bash
./.venv/bin/python scripts/run_go_live_smoke.py \
  --app-url http://127.0.0.1:8000 \
  --json-output artifacts/go_live_smoke.json \
  --markdown-output artifacts/go_live_smoke.md \
  --pretty
```

Use `--strict` when warnings should fail the operator gate.

## Checked Endpoints

| Check | Endpoint |
| --- | --- |
| Application health | `/healthz` |
| Go-live readiness | `/api/admin/go-live-readiness` |
| Pipeline queue | `/api/dashboard/pipeline-queue` |
| Embedding backlog | `/api/dashboard/embedding-backlog` |
| Provider readiness | `/api/admin/embedding-provider-routes/readiness` |
| Search operations | `/api/search/logs/operations-summary` |

## Interpretation

- `ready`: every endpoint responded and readiness is ready.
- `warning`: the application responded but go-live readiness reported warning.
- `blocked`: at least one endpoint failed, returned a non-200 response, or readiness
  reported blocked/unknown.
