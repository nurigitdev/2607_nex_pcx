# Production Remote Reranker User systemd Evidence

Date: 2026-07-30

## Scope

Slice 416 prepares the Qwen3-Reranker-4B remote provider for the same user-level
systemd operating model already used by the three DGX remote embedding
providers.

## Generated Capability

New setup command:

```bash
./.venv/bin/python scripts/setup_remote_reranker_provider.py \
  --route-host 192.168.20.243 \
  --env-dir /home/nexpcx/2607_nex_pcx/deployment/env \
  --systemd-dir /home/nexpcx/.config/systemd/user \
  --user-systemd \
  --write-files
```

Expected generated files:

- `/home/nexpcx/2607_nex_pcx/deployment/env/nex-pcx-reranker-provider.env`
- `/home/nexpcx/.config/systemd/user/nex-pcx-reranker-provider.service`
- `/home/nexpcx/.config/systemd/user/nex-pcx-reranker-provider.README.md`

The environment file contains provider-local runtime settings only:

- `NEX_PCX_RERANKER_PROVIDER_BACKEND=qwen_reranker`
- `NEX_PCX_RERANKER_PROVIDER_MODEL_ID=Qwen/Qwen3-Reranker-4B`
- `NEX_PCX_RERANKER_PROVIDER_PROFILE_NAME=qwen3_reranker_4b`
- `NEX_PCX_RERANKER_PROVIDER_DEVICE=cuda:0`
- `NEX_PCX_RERANKER_PROVIDER_MODELS_DIR=/home/nexpcx/2607_nex_pcx/models`
- `NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME=qwen3_reranker_4b`

It intentionally does not include the NeX-PCX application database URL or
provider route secrets.

## Operator Commands

Run on DGX:

```bash
systemctl --user daemon-reload
systemctl --user enable --now nex-pcx-reranker-provider.service
systemctl --user status nex-pcx-reranker-provider.service --no-pager
journalctl --user -u nex-pcx-reranker-provider.service -n 100 --no-pager
```

Run from the app host after the service starts:

```bash
curl -fsS http://192.168.20.243:9104/healthz
./.venv/bin/python scripts/run_remote_reranker_request_smoke.py \
  --base-url http://192.168.20.243:9104 \
  --markdown-output artifacts/remote_reranker_request_smoke.md
```

## Application Operations Status

The remote reranker operations status now includes user systemd evidence fields
alongside process and health checks:

- `provider.systemd_unit_name`
- `command_observation.values.systemd_unit`
- `command_observation.values.systemd_active`
- `command_observation.values.systemd_enabled`
- `command_observation.values.systemd_main_pid`

These fields allow the admin UI/API to distinguish a manually launched process
from a user-systemd-managed process.

## Live Registration Evidence

DGX SSH was reachable as `nexpcx@192.168.20.243`, and the service was generated,
enabled, and started with `systemctl --user`.

| Check | Value |
| --- | --- |
| Unit | `nex-pcx-reranker-provider.service` |
| Loaded | `loaded (/home/nexpcx/.config/systemd/user/nex-pcx-reranker-provider.service; enabled; preset: enabled)` |
| Active | `active (running)` |
| Main PID | `7469` |
| Startup timestamp | `2026-07-30 10:21:24 KST` |
| `/healthz` | `ready=true` |
| Provider type | `remote` |
| Provider model | `Qwen/Qwen3-Reranker-4B` |
| Reranker profile | `qwen3_reranker_4b` |
| Backend | `qwen_reranker` |
| Device | `cuda:0` |
| Model directory exists | `true` |
| User linger | `Linger=yes` |

Service status excerpt:

```text
● nex-pcx-reranker-provider.service - NeX-PCX reranker provider (qwen-reranker-primary)
     Loaded: loaded (/home/nexpcx/.config/systemd/user/nex-pcx-reranker-provider.service; enabled; preset: enabled)
     Active: active (running) since Thu 2026-07-30 10:21:24 KST
   Main PID: 7469 (python)
```

Health payload excerpt:

```json
{
  "ready": true,
  "provider_type": "remote",
  "provider_model_id": "Qwen/Qwen3-Reranker-4B",
  "reranker_profile_name": "qwen3_reranker_4b",
  "device": "cuda:0",
  "runtime_metadata": {
    "service": "nex_pcx_reranker_provider_service",
    "backend": "qwen_reranker",
    "models_dir": "/home/nexpcx/2607_nex_pcx/models",
    "model_dir": "/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_4b",
    "model_dir_exists": true
  }
}
```

## Request Smoke Evidence

The app host request smoke passed against `http://192.168.20.243:9104/v1/rerank`.

| Metric | Value |
| --- | ---: |
| Candidate count | `3` |
| Returned count | `2` |
| Request elapsed | `3117 ms` |
| Provider elapsed | `2999 ms` |

Score preview:

| Rank | Candidate | Score | Source Rank |
| ---: | --- | ---: | ---: |
| `1` | `candidate-1` | `8.756176` | `1` |
| `2` | `candidate-2` | `6.445219` | `2` |

## Operations Status Evidence

The NeX-PCX operations status probe returned `ready` and confirmed that process
PID and user systemd MainPID match:

```json
{
  "status_code": 200,
  "operations_status": "ready",
  "pid": "7469",
  "systemd": {
    "systemd_unit": "nex-pcx-reranker-provider.service",
    "systemd_active": "active",
    "systemd_enabled": "enabled",
    "systemd_main_pid": "7469"
  },
  "health_ok": true
}
```
