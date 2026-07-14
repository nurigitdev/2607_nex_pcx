# Remote GPU Provider Source Sync And Runtime Setup Guide

Slice 193 documents the safe path for bringing the DGX Spark provider host up to the
current NeX_PCX source level and installing runtime dependencies without touching copied
model bundles.

Target host:

- SSH target: `nexpcx@192.168.20.243`
- Remote work directory: `/home/nexpcx/2607_nex_pcx`
- Remote virtualenv: `/home/nexpcx/2607_nex_pcx/.venv`
- Remote model bundle: `/home/nexpcx/2607_nex_pcx/models`

## Current Readiness Interpretation

After installing `fastapi` and `uvicorn`, the read-only readiness check confirms:

- SSH identity works as `nexpcx` on `spark-1856`.
- The work directory exists.
- The remote venv Python is `Python 3.12.3`.
- `nvidia-smi` sees `NVIDIA GB10` with driver `580.159.03`.
- `models/kure_v1`, `models/bge_m3`, and `models/qwen3_embedding_4b` exist.

Remaining failures mean the source tree is behind the app host:

- `provider_runtime_import`: `ModuleNotFoundError: No module named 'app'`
- `setup_script_exists`: `scripts/setup_remote_gpu_provider.py` missing
- `setup_dry_run_*`: cannot run until the latest setup script is present

So the next step is source synchronization, then package installation from the synced
source.

## Source Sync Rules

Preserve these remote paths:

- `.venv/`
- `models/`
- `storage/`
- `.env`
- `deployment/env/`
- any local provider logs or operator notes

Do not copy a local virtualenv to the GPU server. Reinstall packages into the remote
`.venv` so compiled wheels match the DGX Spark platform.

Before syncing, confirm the local source has been committed and pushed:

```bash
git status --short
git log -1 --oneline
git push
```

## Option A: Git Pull On The GPU Host

Use this path when `/home/nexpcx/2607_nex_pcx` is a git clone.

```bash
ssh nexpcx@192.168.20.243
cd /home/nexpcx/2607_nex_pcx
git remote -v
git status --short
git rev-parse --short HEAD
git fetch origin
git pull --ff-only origin master
git rev-parse --short HEAD
```

If `git status --short` shows local changes, stop and inspect them. Do not use
`git reset --hard` unless the operator explicitly decides those remote changes are
disposable.

## Option B: Rsync From The App Host

Use this path when the GPU directory is not a git clone or GitHub access is not configured
on the GPU host.

First run a dry-run:

```bash
rsync -av --dry-run \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'models/' \
  --exclude 'storage/' \
  --exclude 'deployment/env/' \
  --exclude '.env' \
  ./ nexpcx@192.168.20.243:/home/nexpcx/2607_nex_pcx/
```

Review the file list. Then repeat without `--dry-run`:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'models/' \
  --exclude 'storage/' \
  --exclude 'deployment/env/' \
  --exclude '.env' \
  ./ nexpcx@192.168.20.243:/home/nexpcx/2607_nex_pcx/
```

Use `--delete` only after a separate review, because a bad exclude list can remove useful
remote-only files.

## Runtime Dependency Setup

After the latest source exists on the GPU host:

```bash
ssh nexpcx@192.168.20.243
cd /home/nexpcx/2607_nex_pcx
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
```

Install model runtime packages when provider execution will load local models:

```bash
./.venv/bin/python -m pip install -e ".[models]"
```

If the GPU host cannot download packages directly, build or collect a wheelhouse on a
connected machine that matches the target platform as closely as possible, copy it to the
GPU host, and install with `--no-index --find-links`. Do not rely on the app host's local
`.venv` directory as a portable artifact.

## Post-Install Verification

Run these on the GPU host:

```bash
cd /home/nexpcx/2607_nex_pcx
./.venv/bin/python -c "import fastapi, uvicorn; import app.embedding_provider_service; print('provider_import_ok')"
./.venv/bin/python scripts/setup_remote_gpu_provider.py --provider qwen --route-host 192.168.20.243 --json
```

Then run the full read-only check from the app host:

```bash
./.venv/bin/python scripts/check_remote_gpu_provider_host.py --json
```

Expected result before starting services:

- all required checks pass
- `port_listener_snapshot` may be empty if providers have not been started yet
- `ready` is `true`

## Evidence To Keep

Record these values with the deployment notes:

- Local commit SHA pushed to GitHub
- Remote commit SHA or rsync timestamp
- Remote Python version
- Installed package command used: `pip install -e .` or `pip install -e ".[models]"`
- Readiness checker JSON output after sync
- Model bundle path and model directory names

This keeps the provider launch, route registration, and later search experiments
reproducible.
