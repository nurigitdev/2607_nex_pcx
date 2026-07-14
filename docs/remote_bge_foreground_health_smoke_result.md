# Remote BGE Foreground Health Smoke Result

Date: 2026-07-14

Slice 197 validated the first real foreground launch of the BGE remote embedding provider
on the DGX Spark host.

## Environment

| Item | Value |
| --- | --- |
| GPU host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Remote workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider | `bge` |
| Port | `9102` |
| Model key | `bge_m3` |
| Profile | `bge_m3_1024` |
| Device | `cuda:0` |

## Preconditions

Remote readiness passed before launch:

- `ready`: `true`
- required checks passed: `10`
- required checks failed: `0`
- optional checks failed: `0`
- port `9102` had no existing listener
- model directory existed at `/home/nexpcx/2607_nex_pcx/models/bge_m3`

The remote `.venv` already included the model runtime packages installed during the KURE
foreground smoke setup.

## Passing Smoke

Command:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider bge \
  --startup-timeout-seconds 300 \
  --poll-interval-seconds 2 \
  --json
```

Observed result:

- `passed`: `true`
- `health_status_code`: `200`
- `health_attempts`: `6`
- `pre_launch_health_reachable`: `false`
- `remote_stop_attempted`: `true`
- `remote_stop_exit_code`: `0`
- `post_stop_health_reachable`: `false`
- `elapsed_seconds`: about `21`

Health payload matched:

- `ready`: `true`
- `provider_type`: `remote`
- `provider_model_id`: `local-bge-m3`
- `model_key`: `bge_m3`
- `profile_names`: `["bge_m3_1024"]`
- `dimension`: `1024`
- `device`: `cuda:0`

Post-smoke cleanup verification:

- `curl -fsS http://192.168.20.243:9102/healthz` returned connection refused.
- Remote `ss` showed no listener on port `9102`.
