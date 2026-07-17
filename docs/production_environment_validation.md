# NeX_PCX Production Environment Final Validation

This validation aggregates the runtime configuration audit, startup validation,
and go-live readiness checklist under stricter production guards.

Run it before generating the final go-live evidence bundle:

```bash
NEX_PCX_ENV=production \
NEX_PCX_EMBEDDING_PROVIDER_MODE=remote \
NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS=true \
./.venv/bin/python scripts/validate_production_environment.py \
  --database-url postgresql://nex_pcx_app:<password>@127.0.0.1:5432/nex_pcx_app \
  --expected-database-name nex_pcx_app \
  --app-url http://127.0.0.1:8000 \
  --json-output artifacts/production_environment_validation.json \
  --markdown-output artifacts/production_environment_validation.md \
  --pretty
```

Use `--run-provider-preflight` only when the operator wants to persist provider
route health and contract snapshots during this validation.

## Production Guards

| Guard | Default |
| --- | --- |
| Environment | `NEX_PCX_ENV` must be `production` |
| Database target | `--expected-database-name` must match when supplied |
| Provider mode | `NEX_PCX_EMBEDDING_PROVIDER_MODE` should be `remote` |
| Route readiness | `NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS` should be `true` |
| App URL | `/healthz` and `/openapi.json` app identity are checked when `--app-url` is supplied |

For local rehearsal, use `--allow-non-production`,
`--allow-non-remote-provider`, or `--allow-route-readiness-disabled`. Those
flags keep the guard in the report as a warning rather than a blocker.

## Status Meaning

| Status | Meaning |
| --- | --- |
| `ready` | Guards and all nested validation sections are ready |
| `warning` | Validation can run, but operator review is required |
| `blocked` | A guard or nested validation section failed |
