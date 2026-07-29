# vLLM Runtime Metrics Scraper

Slice 402 adds a read-only runtime metrics scraper for the DGX-Spark vLLM server.
It complements generation run metrics by observing the serving process itself:
KV cache pressure, queue depth, token counters, request counters, and latency
histogram averages.

## Command

```bash
./.venv/bin/python scripts/scrape_vllm_runtime_metrics.py \
  --base-url http://192.168.20.243:12000 \
  --provider-name dgx_vllm_qwen36_27b_nvfp4 \
  --model-id /home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4 \
  --pretty
```

For offline parser checks, pass a saved Prometheus text file:

```bash
./.venv/bin/python scripts/scrape_vllm_runtime_metrics.py \
  --sample-file artifacts/vllm_metrics.prom \
  --json-output artifacts/vllm_runtime_metrics.json \
  --markdown-output artifacts/vllm_runtime_metrics.md \
  --pretty
```

## Normalized Fields

- `kv_cache_usage_percent`: max observed vLLM KV/GPU cache usage percent.
- `running_requests`, `waiting_requests`, `swapped_requests`: current serving queue pressure.
- `waiting_requests_by_reason`: reason-labeled waiting request counts when vLLM exposes them.
- `prompt_tokens_total`, `generation_tokens_total`: cumulative token counters.
- `prefix_cache_hit_rate`: prefix cache hits divided by prefix cache queries.
- `num_preemptions_total`: cumulative preemption count when exposed.
- `average_time_to_first_token_seconds`: histogram mean TTFT.
- `average_inter_token_latency_seconds`: histogram mean inter-token latency.
- `average_e2e_request_latency_seconds`: histogram mean end-to-end request latency.

This slice does not persist snapshots yet. The next natural slice is a database
snapshot table plus API/UI panels for trend and threshold monitoring.
