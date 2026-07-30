# Remote Reranker User systemd Service

Slice 416 aligns the DGX reranker provider with the existing remote embedding
provider operating model: a user-level systemd service owned by `nexpcx`.

## Defaults

| Item | Value |
| --- | --- |
| Host | `nexpcx@192.168.20.243` |
| Workdir | `/home/nexpcx/2607_nex_pcx` |
| Service | `nex-pcx-reranker-provider.service` |
| Port | `9104` |
| ASGI app | `app.reranker_provider_service:app` |
| Backend | `qwen_reranker` |
| Model directory | `/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_4b` |
| Device | `cuda:0` |

The generated environment file intentionally contains only provider-local
runtime settings. It must not contain the NeX-PCX application database URL or
other app-host secrets.

## Generate Files

Run this on the DGX host after the latest source is available under
`/home/nexpcx/2607_nex_pcx`:

```bash
cd /home/nexpcx/2607_nex_pcx
./.venv/bin/python scripts/setup_remote_reranker_provider.py \
  --route-host 192.168.20.243 \
  --env-dir /home/nexpcx/2607_nex_pcx/deployment/env \
  --systemd-dir /home/nexpcx/.config/systemd/user \
  --user-systemd \
  --write-files
```

Expected files:

- `/home/nexpcx/2607_nex_pcx/deployment/env/nex-pcx-reranker-provider.env`
- `/home/nexpcx/.config/systemd/user/nex-pcx-reranker-provider.service`
- `/home/nexpcx/.config/systemd/user/nex-pcx-reranker-provider.README.md`

## Start And Inspect

```bash
systemctl --user daemon-reload
systemctl --user enable --now nex-pcx-reranker-provider.service
systemctl --user status nex-pcx-reranker-provider.service --no-pager
journalctl --user -u nex-pcx-reranker-provider.service -n 100 --no-pager
```

If a foreground/background reranker process is already using port `9104`, stop it
before enabling the systemd unit.

## Health Evidence

From the NeX-PCX app host or DGX host:

```bash
curl -fsS http://192.168.20.243:9104/healthz
./.venv/bin/python scripts/run_remote_reranker_request_smoke.py \
  --base-url http://192.168.20.243:9104 \
  --markdown-output artifacts/remote_reranker_request_smoke.md
```

From the NeX-PCX app host, the operations status API can also report the
configured runtime mode and user systemd observation values:

```bash
curl -fsS http://127.0.0.1:8000/admin/api/remote-reranker/operations-status
```

The payload should include:

- process `status` and `pid`
- `provider.systemd_unit_name`
- `command_observation.values.systemd_active`
- `command_observation.values.systemd_enabled`
- `command_observation.values.systemd_main_pid`
- health contract metadata for provider type, model, profile, backend, and device

## Linger Hardening

User-level systemd services depend on the `nexpcx` user manager. Before relying
on automatic restart after logout or reboot, confirm lingering:

```bash
loginctl show-user nexpcx -p Linger
```

If it returns `Linger=no`, ask a host administrator to run:

```bash
sudo loginctl enable-linger nexpcx
```

