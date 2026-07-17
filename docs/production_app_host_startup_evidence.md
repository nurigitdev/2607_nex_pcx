# Production App Host Startup Evidence

Date: 2026-07-17

## Scope

Slice 286 validated the application-host startup path after the production
database and DGX remote embedding providers were prepared. The slice also added
user-level systemd template support for the web, pipeline worker, and embedding
worker services.

## Template Generation

The app-host service template generator now supports user-level systemd units:

```bash
./.venv/bin/python scripts/render_service_startup_templates.py \
  --workdir /home/tprover/2607_nex_pcx \
  --user tprover \
  --output-dir /tmp/nex_pcx_slice286_app_host_deployment \
  --user-systemd \
  --pretty \
  --json-output /tmp/nex_pcx_slice286_service_templates.json
```

Generated user unit previews install to `default.target` and omit system-only
`User=`, `Group=`, `NoNewPrivileges=true`, and `PrivateTmp=true` directives.

| Unit | Restart | Target |
| --- | --- | --- |
| `nex-pcx-web.service` | `on-failure` | `default.target` |
| `nex-pcx-pipeline-worker.service` | `always` | `default.target` |
| `nex-pcx-embedding-worker.service` | `always` | `default.target` |

## Non-Destructive Worker Command Checks

Worker scripts were validated with `--help` to confirm CLI import and argument
parsing without claiming production queue work:

| Command | Result |
| --- | --- |
| `./.venv/bin/python scripts/process_pipeline_job.py --help` | passed |
| `./.venv/bin/python scripts/process_embedding_job.py --help` | passed |

## Runtime Health

Application health check:

```bash
curl -s http://127.0.0.1:8000/healthz
```

Result:

```json
{"status":"ok"}
```

## Startup Validation

Operations startup validation was run against the production database with DGX
provider route preflight enabled. The database URL used a production credential
and is intentionally not repeated here.

| Check | Status |
| --- | --- |
| `database_url` | `passed` |
| `database_connectivity` | `passed` |
| `alembic_revision` | `passed` |
| `app_healthz` | `passed` |
| `go_live_readiness` | `passed` |
| `provider_route_preflight` | `passed` |

Summary:

- Overall status: `ready`
- Passed checks: `6`
- Warning checks: `0`
- Failed checks: `0`
- Provider routes checked: `4`
- Provider route preflight passed: `4`

## Production Validation

Final production validation was run with absolute storage and model paths:

- `NEX_PCX_UPLOAD_STORAGE_DIR=/home/tprover/2607_nex_pcx/storage/uploads`
- `NEX_PCX_MODELS_DIR=/home/tprover/2607_nex_pcx/models`

Summary:

- Overall status: `ready`
- Failed guards: `0`
- Warning guards: `0`
- Runtime config audit: `ready`, `9/9` checks passed
- Operations startup validation: `ready`, `6/6` checks passed
- Go-live readiness: `ready`, `11/11` checks passed

## Operator Note

System-level app-host units remain the preferred production hardening path.
When `--user-systemd` is used, an administrator should enable lingering for the
service account or confirm another session manager keeps the user systemd
manager alive after logout and reboot.
