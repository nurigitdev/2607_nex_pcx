# Remote GPU Provider Deployment Playbook

Slice 191 turns the remote embedding provider deployment flow into an operator-ready
checklist. The first target environment is:

- GPU server: NVIDIA DGX Spark
- Host/IP: `192.168.20.243`
- SSH user: `nexpcx`
- Work directory: `/home/nexpcx/2607_nex_pcx`
- Virtual environment: `/home/nexpcx/2607_nex_pcx/.venv`
- Model bundle directory: `/home/nexpcx/2607_nex_pcx/models`

The provider process should run on the GPU server. The NeX_PCX app and workers register
routes that point to the GPU server URL.

## Security And Access Rules

- Prefer SSH keys for operator access. Do not store SSH passwords in git, env files, or
  setup scripts.
- Keep provider ports reachable only from the NeX_PCX app/worker network.
- Keep database URLs on the NeX_PCX app host, not in provider systemd units.
- Do not download models during production provider startup. Copy a verified `models/`
  bundle before starting services.
- Use route preflight checks before activating provider routes for ingestion.

## Port And Route Contract

Current provider presets:

| Provider | Default port | Profile routes | Backend |
| --- | ---: | --- | --- |
| `kure` | `9101` | `kure_v1_1024` | `sentence_transformers` |
| `bge` | `9102` | `bge_m3_1024` | `sentence_transformers` |
| `qwen` | `9103` | `qwen3_4b_1000`, `qwen3_4b_2560` | `qwen_embedding` |

The Qwen provider intentionally serves two NeX_PCX profile routes from one loaded model
and one port. Register both profiles with the same base URL:

```text
http://192.168.20.243:9103
```

## Server Readiness Checklist

Run these checks on the GPU server:

```bash
ssh nexpcx@192.168.20.243
cd /home/nexpcx/2607_nex_pcx
nvidia-smi
./.venv/bin/python --version
./.venv/bin/python -m pip list
find models -maxdepth 2 -type f | head
```

Expected model directories:

- `/home/nexpcx/2607_nex_pcx/models/kure_v1`
- `/home/nexpcx/2607_nex_pcx/models/bge_m3`
- `/home/nexpcx/2607_nex_pcx/models/qwen3_embedding_4b`

Use the existing model checker when the runtime dependencies are installed:

```bash
./.venv/bin/python scripts/check_embedding_models.py --dry-run
```

From the NeX_PCX app host, run the read-only remote readiness checker before installing or
starting provider services:

```bash
./.venv/bin/python scripts/check_remote_gpu_provider_host.py --dry-run
./.venv/bin/python scripts/check_remote_gpu_provider_host.py --json
```

The checker connects through `ssh -o BatchMode=yes`, so it will fail fast instead of
prompting for a password. It verifies SSH identity, work directory, virtualenv Python,
provider imports, `nvidia-smi`, expected model directories, configured provider ports, and
the setup dry-run for each provider preset.

Common readiness failures:

- `runtime_dependency_import` fails with `No module named 'fastapi'`: install the app
  runtime dependencies in the remote `.venv`.
- `source_tree_shape` fails: sync the current NeX_PCX source tree, including
  `pyproject.toml`, `app/core/`, and provider scripts.
- `provider_service_import` fails with `No module named 'app.core'`: copy or sync the full
  `app/core/` package and install the source into the remote `.venv`.
- `setup_script_exists` or `setup_dry_run_*` fails: update or copy the latest NeX_PCX
  source files to the remote work directory before generating provider service files.

Use `docs/remote_gpu_provider_source_sync_runtime_guide.md` for the source sync and
runtime dependency setup procedure.

## Generate Provider Service Files

`scripts/setup_remote_gpu_provider.py` is designed to be run on the GPU server. It does not
SSH anywhere and does not modify `/etc/systemd/system` directly. By default it prints a
plan. Add `--write-files` to create files below the repo's `deployment/` directory.

Qwen example:

```bash
cd /home/nexpcx/2607_nex_pcx
./.venv/bin/python scripts/setup_remote_gpu_provider.py \
  --provider qwen \
  --route-host 192.168.20.243 \
  --provider-model-id dgx-spark-qwen3-embedding-4b-2026-07 \
  --write-files
```

KURE example:

```bash
./.venv/bin/python scripts/setup_remote_gpu_provider.py \
  --provider kure \
  --route-host 192.168.20.243 \
  --provider-model-id dgx-spark-kure-v1-2026-07 \
  --write-files
```

BGE-M3 example:

```bash
./.venv/bin/python scripts/setup_remote_gpu_provider.py \
  --provider bge \
  --route-host 192.168.20.243 \
  --provider-model-id dgx-spark-bge-m3-2026-07 \
  --write-files
```

Generated files:

- `deployment/env/nex-pcx-embedding-provider-<provider>.env`
- `deployment/systemd/nex-pcx-embedding-provider-<provider>.service`

Review these files before installing the unit:

```bash
cat deployment/env/nex-pcx-embedding-provider-qwen.env
cat deployment/systemd/nex-pcx-embedding-provider-qwen.service
```

## Install And Start Systemd Units

Run these commands on the GPU server after reviewing generated files:

```bash
sudo cp deployment/systemd/nex-pcx-embedding-provider-qwen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nex-pcx-embedding-provider-qwen.service
sudo systemctl status nex-pcx-embedding-provider-qwen.service
journalctl -u nex-pcx-embedding-provider-qwen.service -n 100 --no-pager
```

Repeat for `kure` and `bge` after their generated service files are reviewed.

For a quick foreground smoke run without systemd:

```bash
./.venv/bin/python scripts/run_embedding_provider.py \
  --provider qwen \
  --host 0.0.0.0 \
  --port 9103 \
  --device cuda:0 \
  --models-dir /home/nexpcx/2607_nex_pcx/models \
  --provider-model-id dgx-spark-qwen3-embedding-4b-2026-07
```

For an operator-ready command sequence, generate a foreground smoke plan from the app host:

```bash
./.venv/bin/python scripts/plan_remote_provider_foreground_smoke.py --provider kure
```

See `docs/remote_provider_foreground_launch_smoke_plan.md` for the full foreground launch,
health check, and Ctrl-C shutdown sequence.

To run the KURE foreground launch and health smoke as one managed command from the app host:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider kure \
  --json
```

## Provider Health Smoke Test

From the NeX_PCX app host:

```bash
curl -fsS http://192.168.20.243:9103/healthz
curl -fsS http://192.168.20.243:9101/healthz
curl -fsS http://192.168.20.243:9102/healthz
```

For Qwen, confirm:

- `ready` is `true`
- `provider_model_id` matches the DGX Spark model ID
- `device` is `cuda:0`
- `profile_names` includes both `qwen3_4b_1000` and `qwen3_4b_2560`

## Register Routes In NeX_PCX

Run this on the NeX_PCX app host, not inside the provider systemd unit:

```bash
export NEX_PCX_DATABASE_URL="postgresql://USER:PASSWORD@127.0.0.1:5432/nex_pcx_dev"

./.venv/bin/python scripts/register_embedding_provider_routes.py \
  --provider qwen \
  --base-url http://192.168.20.243:9103 \
  --database-url "$NEX_PCX_DATABASE_URL"
```

Register KURE and BGE similarly:

```bash
./.venv/bin/python scripts/register_embedding_provider_routes.py \
  --provider kure \
  --base-url http://192.168.20.243:9101 \
  --database-url "$NEX_PCX_DATABASE_URL"

./.venv/bin/python scripts/register_embedding_provider_routes.py \
  --provider bge \
  --base-url http://192.168.20.243:9102 \
  --database-url "$NEX_PCX_DATABASE_URL"
```

## Preflight And Activation

After registering routes:

```bash
./.venv/bin/python scripts/preflight_provider_routes.py \
  --database-url "$NEX_PCX_DATABASE_URL"
```

Then review:

- `/admin/embedding-provider-routes`
- Provider route health
- Contract check result
- Readiness status
- Change audit log

Only activate ingestion workers after the route readiness panel is passing for the target
profiles.

## Rollback

To stop a provider on the GPU server:

```bash
sudo systemctl disable --now nex-pcx-embedding-provider-qwen.service
```

To remove a unit:

```bash
sudo rm /etc/systemd/system/nex-pcx-embedding-provider-qwen.service
sudo systemctl daemon-reload
```

In NeX_PCX, prefer deactivating the provider route instead of deleting it. This keeps audit
history, contract snapshots, and search experiment reproducibility intact.
