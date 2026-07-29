# vLLM Runtime Metric Snapshot API/UI

Slice 404 exposes the persisted vLLM runtime metric snapshots created by the
Slice 403 scraper.

## API

```text
GET /api/admin/vllm-runtime-metrics/snapshots
```

Query parameters:

- `provider_name`: optional exact provider name filter, such as
  `dgx_vllm_qwen36_27b_nvfp4`.
- `limit`: recent snapshot count. Default is `50`; maximum is `500`.
- `include_raw_samples`: include raw Prometheus sample payloads when `true`.

The response includes:

- `summary`: snapshot count, latest/max KV cache pressure, waiting requests,
  and average TTFT/E2E latency.
- `snapshots`: normalized rows from `vllm_runtime_metric_snapshots`.

## UI

Open the operator page at:

```text
GET /admin/vllm-runtime-metrics
```

The page shows filter controls, summary cards, a recent snapshot table, and raw
JSON evidence. It is read-only; metric collection still runs through
`scripts/scrape_vllm_runtime_metrics.py --persist`.
