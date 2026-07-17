# Production Remote Provider User Systemd Evidence

Date: 2026-07-17

## Scope

Slice 285 moved the DGX Spark remote embedding providers from manual background
processes to user-level systemd services under the `nexpcx` account.

System-level installation was not possible in this session because the remote
account requires a sudo password. The user systemd manager is running, so the
providers were installed with `systemctl --user`.

## Generated Files

Host: `nexpcx@192.168.20.243`

Environment files:

- `/home/nexpcx/2607_nex_pcx/deployment/env/nex-pcx-embedding-provider-kure.env`
- `/home/nexpcx/2607_nex_pcx/deployment/env/nex-pcx-embedding-provider-bge.env`
- `/home/nexpcx/2607_nex_pcx/deployment/env/nex-pcx-embedding-provider-qwen.env`

User systemd unit files:

- `/home/nexpcx/.config/systemd/user/nex-pcx-embedding-provider-kure.service`
- `/home/nexpcx/.config/systemd/user/nex-pcx-embedding-provider-bge.service`
- `/home/nexpcx/.config/systemd/user/nex-pcx-embedding-provider-qwen.service`

The unit generator now supports:

```bash
./.venv/bin/python scripts/setup_remote_gpu_provider.py \
  --provider qwen \
  --route-host 192.168.20.243 \
  --provider-model-id local-qwen3-embedding-4b \
  --env-dir /home/nexpcx/2607_nex_pcx/deployment/env \
  --systemd-dir /home/nexpcx/.config/systemd/user \
  --user-systemd \
  --write-files
```

## Service Status

The providers are enabled and active under `systemctl --user`:

| Service | Port | PID | Status |
| --- | ---: | ---: | --- |
| `nex-pcx-embedding-provider-kure.service` | `9101` | `627545` | `active/running` |
| `nex-pcx-embedding-provider-bge.service` | `9102` | `627758` | `active/running` |
| `nex-pcx-embedding-provider-qwen.service` | `9103` | `627971` | `active/running` |

All provider `/healthz` checks returned `ready`:

- KURE: `local-kure-v1`, dimension `1024`, device `cuda:0`
- BGE: `local-bge-m3`, dimension `1024`, device `cuda:0`
- Qwen: `local-qwen3-embedding-4b`, profile dimensions `1000` and `2560`,
  device `cuda:0`

## Validation

Manual provider preflight after systemd transition:

- `route_count`: `4`
- `passed_count`: `4`
- `failed_count`: `0`
- Latest health snapshots: `9` through `12`, all `ready`
- Latest contract snapshots: `9` through `12`, all `passed`

Final production validation:

- Overall status: `ready`
- Guard checks: `5` checked, `5` passed
- Runtime config audit: `ready`
- Operations startup validation: `ready`
- Go-live readiness: `ready`
- Go-live readiness checks: `11` checked, `11` passed
- Provider route readiness: `4/4` active provider routes ready

## Remaining Hardening Note

`loginctl show-user nexpcx -p Linger` currently returns `Linger=no`.

The services are managed by user systemd while the `nexpcx` user manager is
running. For reboot/logout-resilient operation, an administrator should run one
of these hardening paths:

1. Enable user lingering:

   ```bash
   sudo loginctl enable-linger nexpcx
   ```

2. Or install the generated provider units as system-level services under
   `/etc/systemd/system`.

After either hardening path, re-run provider preflight and production validation.
