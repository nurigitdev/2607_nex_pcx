# Production Remote Provider Startup Evidence

Date: 2026-07-17

## Scope

Slice 284 started the DGX Spark remote embedding providers, ran production
provider preflight checks, ran the scheduled provider preflight runner, and
confirmed production validation readiness.

This is an operational startup evidence document. It records the runtime state
observed after manual background launch through SSH. For durable long-running
operation, replace the background launches with reviewed systemd units from the
remote GPU provider setup flow.

## Remote Provider Processes

Host: `nexpcx@192.168.20.243`

| Provider | Profile(s) | Port | PID | Health |
| --- | --- | ---: | ---: | --- |
| `kure` | `kure_v1_1024` | `9101` | `622267` | `ready` |
| `bge` | `bge_m3_1024` | `9102` | `622885` | `ready` |
| `qwen` | `qwen3_4b_1000`, `qwen3_4b_2560` | `9103` | `623353` | `ready` |

The provider ports are listening on `0.0.0.0`:

- `0.0.0.0:9101`
- `0.0.0.0:9102`
- `0.0.0.0:9103`

Health metadata:

- KURE: `provider_model_id=local-kure-v1`, `dimension=1024`, `device=cuda:0`
- BGE: `provider_model_id=local-bge-m3`, `dimension=1024`, `device=cuda:0`
- Qwen: `provider_model_id=local-qwen3-embedding-4b`, profile dimensions
  `qwen3_4b_1000=1000`, `qwen3_4b_2560=2560`, `device=cuda:0`

Logs are on the DGX host:

- `/home/nexpcx/2607_nex_pcx/logs/providers/kure.log`
- `/home/nexpcx/2607_nex_pcx/logs/providers/bge.log`
- `/home/nexpcx/2607_nex_pcx/logs/providers/qwen.log`

## Provider Preflight

Manual provider preflight against the production database passed:

```bash
./.venv/bin/python scripts/preflight_provider_routes.py \
  --database-url "postgresql://nex_pcx_app:***@127.0.0.1:5432/nex_pcx_app"
```

Result:

- `route_count`: `4`
- `passed_count`: `4`
- `failed_count`: `0`
- `sample_set_name`: `default_route_contract`

## Scheduled Preflight

The scheduled preflight runner also passed:

```bash
./.venv/bin/python scripts/run_scheduled_provider_preflight.py \
  --database-url "postgresql://nex_pcx_app:***@127.0.0.1:5432/nex_pcx_app" \
  --limit 20
```

Result:

- `run_count`: `1`
- `failed_count`: `0`
- `schedule_name`: `default_provider_route_preflight`
- `last_status`: `succeeded`
- `run_count` on schedule: `1`
- `failure_count` on schedule: `0`
- `next_run_at`: `2026-07-17 16:47:23+09`

Latest persisted route snapshots:

| Route ID | Profile | Health Snapshot | Health | Contract Snapshot | Contract | Dimension |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| `2` | `bge_m3_1024` | `5` | `ready` | `5` | `passed` | `1024` |
| `1` | `kure_v1_1024` | `6` | `ready` | `6` | `passed` | `1024` |
| `3` | `qwen3_4b_1000` | `7` | `ready` | `7` | `passed` | `1000` |
| `4` | `qwen3_4b_2560` | `8` | `ready` | `8` | `passed` | `2560` |

## Final Production Validation

Final validation command:

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

Final result:

- Overall status: `ready`
- Guard checks: `5` checked, `5` passed
- Runtime config audit: `ready`
- Operations startup validation: `ready`
- Go-live readiness: `ready`
- Go-live readiness checks: `11` checked, `11` passed
- Provider route readiness: `4/4` active provider routes ready
- Provider preflight schedule: `1/1` enabled

## Operator Notes

The providers are currently running as background processes, not managed systemd
services. They will not automatically restart after a host reboot or process
crash.

Stop commands if a manual shutdown is needed:

```bash
ssh nexpcx@192.168.20.243 \
  'pkill -TERM -f "uvicorn app.embedding_provider_service:app --host 0.0.0.0 --port 9101"'
ssh nexpcx@192.168.20.243 \
  'pkill -TERM -f "uvicorn app.embedding_provider_service:app --host 0.0.0.0 --port 9102"'
ssh nexpcx@192.168.20.243 \
  'pkill -TERM -f "uvicorn app.embedding_provider_service:app --host 0.0.0.0 --port 9103"'
```

Recommended hardening follow-up:

1. Generate and review remote GPU provider systemd units.
2. Install the units on the DGX host.
3. Restart providers through systemd.
4. Re-run scheduled preflight and production validation.
