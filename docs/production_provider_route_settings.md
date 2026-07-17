# Production Provider Route Settings

Date: 2026-07-17

## Scope

Slice 283 registered the DGX Spark remote embedding provider routes in the
production database and enabled the default provider route preflight schedule.

This operation applies operational database settings only. It does not start
remote provider processes on the DGX host and it does not record known-failing
preflight snapshots while provider ports are offline.

## Applied Routes

The production database `nex_pcx_app` now has four active provider routes:

| Route ID | Profile | Provider | Base URL | Timeout | Active | Health Check |
| ---: | --- | --- | --- | ---: | --- | --- |
| `1` | `kure_v1_1024` | `kure-primary` | `http://192.168.20.243:9101` | `120s` | `true` | `true` |
| `2` | `bge_m3_1024` | `bge-primary` | `http://192.168.20.243:9102` | `120s` | `true` | `true` |
| `3` | `qwen3_4b_1000` | `qwen-primary` | `http://192.168.20.243:9103` | `300s` | `true` | `true` |
| `4` | `qwen3_4b_2560` | `qwen-primary` | `http://192.168.20.243:9103` | `300s` | `true` | `true` |

Registration command:

```bash
./.venv/bin/python scripts/verify_dgx_provider_route_registration.py \
  --database-url "postgresql://nex_pcx_app:***@127.0.0.1:5432/nex_pcx_app" \
  --apply \
  --json
```

Verification result:

- `passed`: `true`
- `expected_route_count`: `4`
- `verified_count`: `4`
- `missing_count`: `0`
- `mismatched_count`: `0`
- `applied`: `true`

## Preflight Schedule

The default provider route preflight schedule is enabled:

| Schedule | Enabled | Interval | Last Status | Run Count | Failure Count |
| --- | --- | ---: | --- | ---: | ---: |
| `default_provider_route_preflight` | `true` | `60m` | `never_run` | `0` | `0` |

## Validation Result

Production validation was re-run after route registration and schedule
activation.

- Overall status: `blocked`
- Guard checks: `5` checked, `5` passed
- Startup validation:
  - Database connectivity: passed
  - Alembic revision: passed
  - Application `/healthz`: passed
- Go-live readiness: blocked

Go-live readiness now has one remaining failed check:

- `provider_route_readiness`: `0/4` active provider routes are ready.
- `needs_preflight_count`: `4`
- `provider_preflight_schedule`: passed with `1/1` schedules enabled.

## Current External Blocker

The DGX provider ports were checked before preflight:

- `http://192.168.20.243:9101/healthz`: connection refused
- `http://192.168.20.243:9102/healthz`: connection refused
- `http://192.168.20.243:9103/healthz`: connection refused

Because the provider processes are not listening yet, preflight was not run in
this step. Running it now would only record expected failure snapshots.

## Next Commands After Provider Startup

After starting the DGX provider processes, run:

```bash
./.venv/bin/python scripts/preflight_provider_routes.py \
  --database-url "postgresql://nex_pcx_app:***@127.0.0.1:5432/nex_pcx_app"
```

Then re-run production validation:

```bash
env NEX_PCX_ENV=production \
  NEX_PCX_EMBEDDING_PROVIDER_MODE=remote \
  NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS=true \
  NEX_PCX_EMBEDDING_ROUTE_READINESS_FAILURE_MODE=defer \
  NEX_PCX_DATABASE_URL="postgresql://nex_pcx_app:***@127.0.0.1:5432/nex_pcx_app" \
  ./.venv/bin/python scripts/validate_production_environment.py \
    --expected-database-name nex_pcx_app \
    --app-url http://127.0.0.1:8000
```
