# DGX Provider Resource Probe Script

Slice 409 adds a read-only probe for checking provider process resource usage on
the DGX-Spark host. It complements provider health checks and vLLM `/metrics` by
capturing host/process-level resource evidence.

## Default Provider Inventory

| Provider | Type | Port | Process Match |
| --- | --- | ---: | --- |
| `kure-primary` | embedding | `9101` | `embedding_provider_service` |
| `bge-primary` | embedding | `9102` | `embedding_provider_service` |
| `qwen-primary` | embedding | `9103` | `embedding_provider_service` |
| `qwen-reranker-primary` | reranker | `9104` | `reranker_provider_service` |
| `dgx_vllm_qwen36_27b_nvfp4` | vLLM | `12000` | `vllm` |

## Dry Run

```bash
./.venv/bin/python scripts/probe_provider_resources.py \
  --dry-run \
  --provider all \
  --pretty
```

Dry-run output lists the provider inventory and read-only commands without
touching the host:

- `ps -eo pid=,ppid=,user=,rss=,vsz=,pcpu=,etimes=,args=`
- `ss -ltnp`
- `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits`
- `/proc/meminfo`

## Local DGX Execution

Run this command on the DGX host when the repository has been copied there:

```bash
./.venv/bin/python scripts/probe_provider_resources.py \
  --local-only \
  --provider all \
  --json-output artifacts/dgx_provider_resources.json \
  --markdown-output artifacts/dgx_provider_resources.md \
  --pretty
```

The script is read-only. It does not stop, restart, or reconfigure any provider.

## SSH Delegated Execution

From an app host that can SSH to DGX-Spark and where the same script exists in
the remote workdir:

```bash
./.venv/bin/python scripts/probe_provider_resources.py \
  --ssh-user nexpcx \
  --host 192.168.20.243 \
  --remote-workdir /home/nexpcx/2607_nex_pcx \
  --provider all \
  --json-output artifacts/dgx_provider_resources.json \
  --markdown-output artifacts/dgx_provider_resources.md \
  --pretty
```

The SSH path runs the probe remotely with `--local-only` and returns JSON to the
caller.

## Evidence Fields

Each provider snapshot includes:

- provider name/type/host/port/model id
- listener match confidence: `port`, `command`, or `missing`
- process PID, parent PID, user, CPU percent, uptime, RSS/VMS memory
- GPU process memory when `nvidia-smi` exposes the PID
- host RAM available percent and swap used percent
- readiness status and reason codes such as `process_not_found`,
  `port_not_listening`, `system_ram_pressure`, `swap_pressure`,
  `ram_budget_pressure`, or `gpu_memory_pressure`

The command preview is redacted for tokens that look like secrets and the full
command is represented only by SHA-256 hash.
