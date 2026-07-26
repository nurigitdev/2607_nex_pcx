# Generation Provider Metric Snapshot UI

Slice 346 adds a read-only operator page for generation provider metrics.

## Page

Open the operator page at `GET /admin/generation-provider-metrics`.

```text
GET /admin/generation-provider-metrics
```

Query parameters:

- `limit`: number of recent generation runs to inspect. Default is `50`; maximum
  is `500`.

The UI shows:

- Summary cards for run count, metrics-present count, success/failure,
  no-answer, total tokens, average elapsed, and average provider elapsed.
- A recent run table linking each generation run to its detail page.
- Raw snapshot JSON for operator evidence and troubleshooting.

The page uses the same snapshot source as:

```text
GET /api/admin/generation-provider-metrics/snapshot
```

Rows without `response_metadata.provider_metrics` remain visible with a missing
metric badge so operators can spot older runs or migration gaps.
