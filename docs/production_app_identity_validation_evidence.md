# Production App Identity Validation Evidence

Date: 2026-07-17

## Scope

Slice 288 added an application identity check to operations startup validation.
The check prevents a false-positive startup pass when another FastAPI service
responds to `/healthz` on the expected port.

## Guardrail

When `--app-url` is provided, startup validation now checks:

- `GET /healthz` returns `status=ok`
- `GET /openapi.json` has `info.title=NeX_PCX`

The second check is reported as `app_identity`.

## Blocked Port Check

Validation against `http://127.0.0.1:8000` returned `blocked` because NeX-PCX was
not reachable on that port:

- Overall status: `blocked`
- Check count: `7`
- Failed checks: `2`
- Failed check codes: `app_healthz`, `app_identity`

Earlier in Slice 287, port `8000` had served a different app title
(`Hermes Agent Hub`). At the time of this validation the port returned
connection refused. Both cases are now blocked before go-live.

## NeX-PCX Validation Port Check

NeX-PCX was launched on `http://127.0.0.1:18080` for verification.

Startup validation result:

- Overall status: `ready`
- Check count: `7`
- Passed checks: `7`
- App identity title: `NeX_PCX`
- Provider routes checked: `4`
- Provider route preflight passed: `4`

Production validation was also run with the same app URL and completed with exit
code `0`.

## Operator Note

Before production traffic is pointed at port `8000`, confirm that the process
bound to that port is NeX-PCX:

```bash
curl -fsS http://127.0.0.1:8000/openapi.json | jq '.info.title'
```

The expected value is:

```json
"NeX_PCX"
```
