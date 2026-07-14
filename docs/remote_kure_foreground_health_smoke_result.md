# Remote KURE Foreground Health Smoke Result

Date: 2026-07-14

Slice 196 validated the first real foreground launch of the KURE remote embedding provider
on the DGX Spark host.

## Environment

| Item | Value |
| --- | --- |
| GPU host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Remote workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider | `kure` |
| Port | `9101` |
| Model key | `kure_v1` |
| Profile | `kure_v1_1024` |
| Device | `cuda:0` |

## Runtime Dependency Note

The first foreground launch failed because the remote `.venv` did not include
`sentence_transformers`. The remote provider venv was updated with:

```bash
cd /home/nexpcx/2607_nex_pcx
/home/nexpcx/2607_nex_pcx/.venv/bin/python -m pip install -e ".[models]"
```

Import verification after installation:

- `sentence_transformers`: `5.6.0`
- `torch`: `2.13.0+cu130`

## Passing Smoke

Command:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider kure \
  --startup-timeout-seconds 300 \
  --poll-interval-seconds 2 \
  --json
```

Observed result:

- `passed`: `true`
- `health_status_code`: `200`
- `health_attempts`: `5`
- `pre_launch_health_reachable`: `false`
- `remote_stop_attempted`: `true`
- `remote_stop_exit_code`: `0`
- `post_stop_health_reachable`: `false`
- `elapsed_seconds`: about `20`

Health payload matched:

- `ready`: `true`
- `provider_type`: `remote`
- `provider_model_id`: `local-kure-v1`
- `model_key`: `kure_v1`
- `profile_names`: `["kure_v1_1024"]`
- `dimension`: `1024`
- `device`: `cuda:0`

Post-smoke cleanup verification:

- `curl -fsS http://192.168.20.243:9101/healthz` returned connection refused.
- Remote `ss` showed no listener on port `9101`.
