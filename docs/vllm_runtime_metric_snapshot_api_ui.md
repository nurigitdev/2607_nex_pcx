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
- `readiness`: latest-snapshot readiness status, badge class, reason codes,
  threshold defaults, and per-signal actual/warning/critical values.
- `snapshots`: normalized rows from `vllm_runtime_metric_snapshots`.

## UI

Open the operator page at:

```text
GET /admin/vllm-runtime-metrics
```

The page shows filter controls, a readiness badge, threshold summary, summary
cards, a recent snapshot table, and raw JSON evidence. It is read-only; metric
collection still runs through `scripts/scrape_vllm_runtime_metrics.py
--persist`.

The readiness badge uses the latest snapshot only:

- `unknown`: no runtime snapshots are available.
- `ok`: the latest snapshot is fresh and below all configured thresholds.
- `warning`: one or more warning thresholds are breached.
- `critical`: at least one critical threshold is breached, including missing
  vLLM metrics or stale snapshots older than the critical threshold.
