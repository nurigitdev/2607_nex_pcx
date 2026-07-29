# DGX Provider Resource Monitor Strategy

Slice 408 defines the operator-facing resource monitoring strategy for the
shared DGX-Spark host before adding probes, persistence, APIs, or Dashboard UI.

## Goal

NeX-PCX already tracks provider HTTP health and vLLM serving metrics. That is
not enough when the same 128 GB DGX-Spark host runs multiple heavy providers:

- remote embedding providers served by FastAPI
- remote reranker provider served by FastAPI
- vLLM OpenAI-compatible generation runtime

Operators need to know which provider process is consuming RAM, GPU memory, or
swap before deciding whether to retry jobs, drain workers, restart a provider,
or reduce generation/chunk workload.

## Monitoring Boundary

Provider health and provider resource monitoring are separate signals.

| Signal | Source | Meaning |
| --- | --- | --- |
| HTTP health | `/healthz`, `/metrics`, `/v1/chat/completions` smoke | Provider contract is reachable |
| vLLM runtime metrics | vLLM Prometheus `/metrics` | KV cache, queue, token throughput, request latency |
| Host resource snapshot | `psutil`, process table, `nvidia-smi` | RAM RSS, GPU memory, CPU, swap, uptime, process binding |

The resource monitor should never infer answer quality or search quality. It
only explains whether provider processes have enough host resources to keep
serving requests.

## Provider Inventory Contract

The monitor must accept an explicit provider inventory so process matching is
repeatable and safe:

| Field | Purpose |
| --- | --- |
| `provider_name` | Stable NeX-PCX provider name |
| `provider_type` | `embedding`, `reranker`, or `vllm` |
| `host` | DGX host address, initially `192.168.20.243` |
| `port` | Expected listening port |
| `process_match` | Command substring or executable basename used only as a hint |
| `model_id` | Optional model path or model key |

Port binding should be the primary local process match on the DGX host. Command
matching is a fallback for stopped or reconfigured providers and should be
reported as lower confidence.

## Snapshot Contract

Each provider resource snapshot should include:

- provider identity: name, type, host, port, model id
- process identity: pid, parent pid, user, command basename, command hash or
  redacted command preview
- host resource counters: RSS bytes, VMS bytes, CPU percent, process uptime
- GPU resource counters: GPU index, GPU UUID/name when available, used GPU
  memory bytes, utilization percent when available
- system pressure counters: total/available RAM, total/used swap, swap percent
- status: `ok`, `warning`, `critical`, or `unknown`
- reason codes such as `process_not_found`, `port_not_listening`,
  `ram_pressure`, `gpu_memory_pressure`, `swap_pressure`, `collector_failed`
- collection metadata: collected_at, elapsed_ms, collector host, error message

Secrets, prompt text, document text, request payloads, and API keys must never
be stored in resource snapshots.

## Threshold Direction

Initial thresholds should be conservative and configurable later:

| Signal | Warning | Critical |
| --- | --- | --- |
| System RAM available | below 20% | below 10% |
| System swap used | above 1 GB or 10% | above 8 GB or 30% |
| Provider RAM RSS | above configured per-provider budget | above hard budget |
| GPU memory used | above 85% | above 95% |
| Process missing | warning when optional, critical when active/default |

For vLLM, KV cache readiness remains in `vllm_runtime_metric_snapshots`.
Provider resource snapshots should correlate with it rather than replace it.

## Implementation Roadmap

1. **Slice 409: Remote Provider Resource Probe Script**
   - Add a read-only probe that runs on the DGX host or app host with SSH/manual
     output support.
   - Collect process, port, RAM, GPU memory, and swap information as JSON and
     Markdown evidence.

2. **Slice 410: Provider Resource Snapshot Schema Migration**
   - Add a `provider_resource_snapshots` table for normalized resource evidence.

3. **Slice 411: Provider Resource Snapshot API + UI**
   - Persist/list recent snapshots and show provider-level status details.

4. **Slice 412: Dashboard Provider Resource Card**
   - Surface latest provider resource readiness beside vLLM runtime readiness.

## Operating Principle

When a provider is unhealthy and resource pressure is high, operations should
first identify the pressure source, then choose a response:

- pause or drain NeX-PCX workers
- stop optional providers
- reduce batch size or token budget
- restart the specific provider process
- defer large Qwen embedding/reranking/generation runs until memory recovers
