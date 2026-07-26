# Generation Provider Metric Snapshot API

Slice 345 persists provider metrics for mock generation runs and exposes an
operations-oriented snapshot API.

## Mock Persistence

`execute_mock_generation_run(...)` now stores a normalized metrics payload under
`generation_runs.response_metadata.provider_metrics`.

The payload follows `generation_provider_metrics_v1` and includes provider,
model, finish reason, token counts, elapsed time, retry count, and success/error
fields.

## API

`GET /api/admin/generation-provider-metrics/snapshot`

Query parameters:

- `limit`: number of recent generation runs to inspect. Default is `50`; maximum
  is `500`.

The response contains:

- `summary`: run count, metric-present count, succeeded/failed/no-answer count,
  total tokens, average elapsed, and average provider elapsed
- `runs`: recent generation run metric rows

Older rows without `provider_metrics` are still included with
`metric_present=false` so operators can see migration/backfill gaps.
