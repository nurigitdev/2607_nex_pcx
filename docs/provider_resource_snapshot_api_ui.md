# Provider Resource Snapshot API + UI

Slice 411 persists DGX provider resource probe results and exposes them through
an operator API and UI.

## Persist Probe Results

Run the probe from the app host and persist the returned provider snapshots:

```bash
NEX_PCX_DATABASE_URL='postgresql://...' \
./.venv/bin/python scripts/probe_provider_resources.py \
  --ssh-user nexpcx \
  --provider all \
  --persist \
  --json-output artifacts/dgx_provider_resources.json \
  --markdown-output artifacts/dgx_provider_resources.md
```

`--persist` is intentionally rejected with `--dry-run`. SSH mode executes the
read-only process probe on the DGX host and persists the received JSON on the
app host.

## API

```http
GET /api/admin/provider-resource-snapshots
```

Query parameters:

- `provider_name`: optional exact provider name
- `provider_type`: optional `embedding`, `reranker`, or `vllm`
- `host`: optional exact DGX host
- `status`: optional `ok`, `warning`, `critical`, or `unknown`
- `limit`: recent snapshot count, default `50`, max `500`
- `include_raw_snapshot`: include normalized raw probe snapshot JSON

Response shape:

- `summary`: snapshot count, latest provider/status, status counts, max RAM,
  max GPU memory, and max swap percent
- `snapshots`: normalized rows from `provider_resource_snapshots`

## UI

```http
GET /admin/provider-resources
```

The page provides:

- provider/type/host/status filters
- summary cards for latest provider readiness and max resource pressure
- provider snapshot table with PID, RAM RSS, GPU memory, CPU, swap, and reason
  codes
- bounded raw JSON evidence for troubleshooting

No prompt text, document text, request payload, API key, or secret is persisted
or displayed.
