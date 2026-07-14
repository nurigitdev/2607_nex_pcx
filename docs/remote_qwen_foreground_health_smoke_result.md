# Remote Qwen Foreground Health Smoke Result

Date: 2026-07-14

Slice 198 validated the first real foreground launch of the Qwen remote embedding provider
on the DGX Spark host. This provider serves two embedding profiles from one loaded model.

## Environment

| Item | Value |
| --- | --- |
| GPU host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Remote workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider | `qwen` |
| Port | `9103` |
| Model key | `qwen3_embedding_4b` |
| Profiles | `qwen3_4b_1000`, `qwen3_4b_2560` |
| Device | `cuda:0` |

## Preconditions

Remote readiness passed before launch:

- `ready`: `true`
- required checks passed: `10`
- required checks failed: `0`
- optional checks failed: `0`
- port `9103` had no existing listener
- model directory existed at `/home/nexpcx/2607_nex_pcx/models/qwen3_embedding_4b`

## Passing Smoke

Command:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider qwen \
  --startup-timeout-seconds 900 \
  --poll-interval-seconds 5 \
  --json
```

Observed result after adding dual-profile dimension validation to the smoke runner:

- `passed`: `true`
- `health_status_code`: `200`
- `health_attempts`: `5`
- `pre_launch_health_reachable`: `false`
- `remote_stop_attempted`: `true`
- `remote_stop_exit_code`: `0`
- `post_stop_health_reachable`: `false`
- `elapsed_seconds`: about `31`

Health payload matched:

- `ready`: `true`
- `provider_type`: `remote`
- `provider_model_id`: `local-qwen3-embedding-4b`
- `model_key`: `qwen3_embedding_4b`
- `profile_names`: `["qwen3_4b_1000", "qwen3_4b_2560"]`
- `dimension`: `null`, because this provider exposes multiple output dimensions
- `device`: `cuda:0`
- `runtime_metadata.profile_dimensions`: `{"qwen3_4b_1000": 1000, "qwen3_4b_2560": 2560}`

Startup log showed both checkpoint shards loaded before uvicorn became ready.

Post-smoke cleanup verification:

- `curl -fsS http://192.168.20.243:9103/healthz` returned connection refused.
- Remote `ss` showed no listener on port `9103`.
