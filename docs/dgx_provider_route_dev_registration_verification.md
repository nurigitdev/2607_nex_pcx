# DGX Provider Route Dev Registration Verification

Date: 2026-07-14

Slice 204 added and ran a verification path for registering the DGX Spark remote embedding
provider endpoints in the NeX_PCX development database.

## Expected Routes

| Profile | Provider | Base URL | Timeout | Active | Health check |
| --- | --- | --- | ---: | --- | --- |
| `kure_v1_1024` | `kure-primary` | `http://192.168.20.243:9101` | `120s` | `true` | `true` |
| `bge_m3_1024` | `bge-primary` | `http://192.168.20.243:9102` | `120s` | `true` | `true` |
| `qwen3_4b_1000` | `qwen-primary` | `http://192.168.20.243:9103` | `300s` | `true` | `true` |
| `qwen3_4b_2560` | `qwen-primary` | `http://192.168.20.243:9103` | `300s` | `true` | `true` |

## Verification Command

The verification was run against the development database with `--apply`, which upserts the
expected DGX routes and then reads the database back to verify the persisted values:

```bash
./.venv/bin/python scripts/verify_dgx_provider_route_registration.py \
  --database-url "postgresql://nex_pcx_dev:***@127.0.0.1:5432/nex_pcx_dev" \
  --apply \
  --json
```

## Passing Result

- `passed`: `true`
- `expected_route_count`: `4`
- `verified_count`: `4`
- `missing_count`: `0`
- `mismatched_count`: `0`
- `applied`: `true`

Persisted route IDs:

| Profile | Provider | Route ID |
| --- | --- | ---: |
| `kure_v1_1024` | `kure-primary` | `2` |
| `bge_m3_1024` | `bge-primary` | `3` |
| `qwen3_4b_1000` | `qwen-primary` | `4` |
| `qwen3_4b_2560` | `qwen-primary` | `5` |

The verification checked:

- `provider_mode`
- `provider_base_url`
- `timeout_seconds`
- `priority`
- `is_active`
- `health_check_enabled`
- `runtime_metadata.preset_name`
- `runtime_metadata.backend`
- `runtime_metadata.model_key`
- `runtime_metadata.provider_model_id`
- `runtime_metadata.default_port`
- `runtime_metadata.script`

## Follow-up

This slice verifies route registration only. Route preflight and readiness should be run
after the DGX providers are started or supervised by their deployment process.
