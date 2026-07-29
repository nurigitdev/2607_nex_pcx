# DGX Provider/vLLM Snapshot Collection Runner

Slice 414 adds `scripts/collect_dgx_snapshots.py`, a bounded foreground runner
that collects both operational snapshot families needed by the monitoring UI:

- vLLM Runtime snapshots from the DGX vLLM `/metrics` endpoint
- Provider Resource snapshots from the DGX host process table through SSH

The runner writes one evidence JSON/Markdown pair and persists rows into the
existing `vllm_runtime_metric_snapshots` and `provider_resource_snapshots`
tables. It does not restart providers and does not store prompts, document text,
API keys, or database URLs in the evidence payload.

## One-shot Production Collection

Run this from the app host before opening the vLLM Runtime and Provider Resource
menus:

```bash
NEX_PCX_DATABASE_URL='postgresql://nex_pcx_app:<password>@127.0.0.1:5432/nex_pcx_app' \
./.venv/bin/python scripts/collect_dgx_snapshots.py \
  --component all \
  --host 192.168.20.243 \
  --ssh-user nexpcx \
  --remote-workdir /home/nexpcx/2607_nex_pcx \
  --vllm-base-url http://192.168.20.243:12000 \
  --vllm-provider-name dgx_vllm_qwen36_27b_nvfp4 \
  --vllm-model-id /home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4 \
  --json-output artifacts/dgx_snapshot_collection.json \
  --markdown-output artifacts/dgx_snapshot_collection.md \
  --pretty
```

## Bounded Foreground Collection

For a short live observation window, repeat the collection with a fixed cycle
count:

```bash
NEX_PCX_DATABASE_URL='postgresql://nex_pcx_app:<password>@127.0.0.1:5432/nex_pcx_app' \
./.venv/bin/python scripts/collect_dgx_snapshots.py \
  --max-cycles 5 \
  --interval-seconds 60 \
  --pretty
```

This is intentionally bounded. For unattended operation, wrap the one-shot
command with an external scheduler such as cron or systemd timer.

## Component-specific Collection

Collect only vLLM metrics:

```bash
./.venv/bin/python scripts/collect_dgx_snapshots.py \
  --component vllm \
  --pretty
```

Collect only provider resources:

```bash
./.venv/bin/python scripts/collect_dgx_snapshots.py \
  --component provider-resource \
  --provider all \
  --pretty
```

When running directly on the DGX host, use `--provider-local-only` so the
provider resource probe reads the local process table instead of using SSH.

## Status Semantics

- `completed`: all requested snapshots were persisted and observed status was normal.
- `attention`: snapshots were persisted, but one observed provider status was
  warning, critical, or unknown.
- `partial`: at least one component persisted snapshots and at least one
  component failed to collect.
- `failed`: no expected snapshot payload was collected.
- `blocked`: the plan could not run, usually because `NEX_PCX_DATABASE_URL` was
  missing.

The provider resource probe can return a non-zero process exit when it observes
a critical provider state. The collection runner treats that as `attention`
when snapshot persistence succeeds, because the monitor UI still needs those
critical rows to explain the operational state.
