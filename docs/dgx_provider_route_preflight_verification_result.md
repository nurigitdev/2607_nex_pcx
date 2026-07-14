# DGX Provider Route Preflight Verification Result

Date: 2026-07-14

Slice 205 added and ran a managed DGX route preflight verification runner. The runner
launches each remote embedding provider over SSH, waits for `/healthz`, runs preflight
against the registered development DB route profiles, records health/contract snapshots,
and stops the provider before moving to the next one.

- `passed`: `true`
- `database_url`: `postgresql://nex_pcx_dev:***@127.0.0.1:5432/nex_pcx_dev`
- `active_only`: `true`
- `total_elapsed_seconds`: `84.27`
- providers executed: `3`

## Provider Results

| Provider | Passed | Base URL | Health | Profiles | Error |
| --- | --- | --- | --- | ---: | --- |
| `kure` | `true` | `http://192.168.20.243:9101` | `true` | `1/1` | `` |
| `bge` | `true` | `http://192.168.20.243:9102` | `true` | `1/1` | `` |
| `qwen` | `true` | `http://192.168.20.243:9103` | `true` | `2/2` | `` |

## Profile Preflight Results

| Provider | Profile | Passed | Routes | Passed | Failed | Contract Snapshots | Health Snapshots | Error |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `kure` | `kure_v1_1024` | `true` | `1` | `1` | `0` | `1` | `1` | `` |
| `bge` | `bge_m3_1024` | `true` | `1` | `1` | `0` | `2` | `2` | `` |
| `qwen` | `qwen3_4b_1000` | `true` | `1` | `1` | `0` | `3` | `3` | `` |
| `qwen` | `qwen3_4b_2560` | `true` | `1` | `1` | `0` | `4` | `4` | `` |

Post-run cleanup was confirmed for all three provider ports.
