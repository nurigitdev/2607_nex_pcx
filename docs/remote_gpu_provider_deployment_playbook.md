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

If the GPU account does not have passwordless sudo, generate user-level systemd
units instead:

```bash
./.venv/bin/python scripts/setup_remote_gpu_provider.py \
  --provider qwen \
  --route-host 192.168.20.243 \
  --provider-model-id local-qwen3-embedding-4b \
  --env-dir /home/nexpcx/2607_nex_pcx/deployment/env \
  --systemd-dir /home/nexpcx/.config/systemd/user \
  --user-systemd \
  --write-files

systemctl --user daemon-reload
systemctl --user enable --now nex-pcx-embedding-provider-qwen.service
systemctl --user status nex-pcx-embedding-provider-qwen.service --no-pager
```

User-level systemd units are useful for an operator-owned GPU host, but they
depend on the user's systemd manager. If `loginctl show-user nexpcx -p Linger`
returns `Linger=no`, ask an administrator to enable lingering or install the
system-level units before relying on automatic restart after logout or reboot.

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

Use the same runner for BGE on port `9102`:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider bge \
  --json
```

Use a longer startup timeout for the shared Qwen provider on port `9103`:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider qwen \
  --startup-timeout-seconds 900 \
  --poll-interval-seconds 5 \
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

## Provider Embedding Request Smoke Test

After a provider reports healthy, run the request-level smoke runner from the app host:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider kure \
  --json
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider bge \
  --json
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider qwen \
  --timeout-seconds 300 \
  --json
```

The runner validates `/v1/embeddings` response shape, provider model ID, provider type,
input count, row count, and output dimension for each selected profile.

The first passing KURE request smoke result is captured in
`docs/remote_kure_embedding_request_smoke_result.md`.
The first passing BGE request smoke result is captured in
`docs/remote_bge_embedding_request_smoke_result.md`.
The first passing Qwen dual-profile request smoke result is captured in
`docs/remote_qwen_embedding_request_smoke_result.md`.

To verify all remote providers in one sequential run, use the suite runner. It launches
KURE, BGE, and Qwen one at a time, validates real embedding responses, and stops each
provider before moving on:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke_suite.py \
  --json \
  --markdown-output artifacts/remote_embedding_request_smoke_suite.md
```

The first passing suite result is captured in
`docs/remote_embedding_request_smoke_suite_result.md`.

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

For the standard DGX Spark development host, verify and upsert all four expected profile
routes in one command:

```bash
./.venv/bin/python scripts/verify_dgx_provider_route_registration.py \
  --database-url "$NEX_PCX_DATABASE_URL" \
  --apply \
  --json
```

The first passing development DB verification result is captured in
`docs/dgx_provider_route_dev_registration_verification.md`.

## Preflight And Activation

After registering routes, use the managed DGX verification runner when providers are not
already supervised. It launches KURE, BGE, and Qwen sequentially, runs profile-scoped route
preflight checks against the NeX-PCX database, records health/contract snapshots, and stops
each foreground provider:

```bash
./.venv/bin/python scripts/run_dgx_provider_route_preflight_verification.py \
  --database-url "$NEX_PCX_DATABASE_URL" \
  --json \
  --markdown-output artifacts/dgx_provider_route_preflight_verification.md
```

The first passing development DB run is captured in
`docs/dgx_provider_route_preflight_verification_result.md`.

After route preflight passes, run a worker E2E smoke before larger benchmark ingestion.
This creates a temporary chunk, creates one embedding job per profile, processes those
jobs through route-aware workers with the readiness gate enabled, verifies vector table
persistence, and removes the fixture by default:

```bash
./.venv/bin/python scripts/run_dgx_remote_embedding_worker_e2e_smoke.py \
  --database-url "$NEX_PCX_DATABASE_URL" \
  --json \
  --markdown-output artifacts/dgx_remote_embedding_worker_e2e_smoke.md
```

The first passing development DB worker E2E smoke is captured in
`docs/dgx_remote_embedding_worker_e2e_smoke_result.md`.

After the worker E2E smoke passes, run the small corpus ingestion benchmark before
moving to larger document sets. It creates a temporary multi-chunk corpus, creates
one embedding job per chunk/profile pair, launches KURE, BGE, and Qwen sequentially,
and summarizes vector persistence plus provider/worker elapsed metadata:

```bash
./.venv/bin/python scripts/run_dgx_small_corpus_embedding_benchmark.py \
  --database-url "$NEX_PCX_DATABASE_URL" \
  --chunk-count 3 \
  --persist-result \
  --json \
  --markdown-output artifacts/dgx_small_corpus_embedding_benchmark.md
```

The first passing development DB benchmark is captured in
`docs/dgx_small_corpus_embedding_benchmark_result.md`.
The first persisted benchmark DB verification is captured in
`docs/dgx_ingestion_benchmark_persistence_result.md`.

If providers are already running under systemd or another supervisor, run the standalone
preflight command instead:

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
