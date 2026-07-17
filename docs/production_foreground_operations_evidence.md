# Production Foreground Operations Evidence

Date: 2026-07-17

## Scope

Slice 291 records foreground operation as an accepted pre-CX operating mode.
Systemd registration remains a later hardening path, but it is not required to
continue controlled experiments when an operator keeps the app-host terminal
session alive.

## Foreground Web Launch

Run the web app from the project directory:

```bash
NEX_PCX_ENV=production \
NEX_PCX_DATABASE_URL='<production database url>' \
NEX_PCX_UPLOAD_STORAGE_DIR=/home/tprover/2607_nex_pcx/storage/uploads \
NEX_PCX_MODELS_DIR=/home/tprover/2607_nex_pcx/models \
NEX_PCX_EMBEDDING_PROVIDER_MODE=remote \
NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS=true \
./.venv/bin/uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Foreground Validation

Run this in a second terminal after the app starts:

```bash
./.venv/bin/python scripts/validate_foreground_operations.py \
  --app-url http://127.0.0.1:8000 \
  --acknowledge-no-auto-restart \
  --json-output artifacts/foreground_operations_validation.json \
  --markdown-output artifacts/foreground_operations_validation.md \
  --pretty
```

Generated evidence paths:

- `artifacts/foreground_operations_validation.json`
- `artifacts/foreground_operations_validation.md`

## Expected Result

The expected foreground validation status is `warning`.

That warning is intentional: foreground mode has no automatic restart guarantee
after process failure, terminal logout, host reboot, or an interrupted operator
session. The validation should still require:

- `/healthz` returns `{"status": "ok"}`.
- `/openapi.json` reports the expected app title: `NeX_PCX`.
- Pipeline worker CLI import and argument parsing succeeds.
- Embedding worker CLI import and argument parsing succeeds.

## Stop Signal

Treat `blocked` as a stop signal. Common causes:

- Port `8000` is not serving NeX-PCX.
- Another FastAPI application owns port `8000`.
- The foreground app process is not running.
- A worker script no longer imports or parses arguments.

## Operator Note

Foreground mode is suitable for supervised pre-CX experimentation. For
continuous unattended operation, promote the same launch configuration to
systemd or another process supervisor and re-run app-host service restart
validation.
