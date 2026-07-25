# Remote Reranker Operations Status API/UI

Slice 331 adds a read-only operator surface for the DGX Qwen3-Reranker-4B
provider.

## Operator UI

Open the admin page:

```text
/admin/reranker-provider
```

The page shows:

- App reranker runtime mode, configured remote URL, and timeout.
- DGX remote process status, PID file, and log file path.
- `/healthz` readiness, HTTP status, runtime metadata, and contract mismatches.
- Optional `/v1/rerank` request smoke result with latency, count, rank, and score
  preview.

Use the request smoke button only when the provider has enough GPU memory for a
small rerank call.

## API

Status and health only:

```text
GET /api/admin/reranker-provider/status
```

Status, health, and request smoke:

```text
GET /api/admin/reranker-provider/status?request_smoke=true
```

The API returns HTTP `200` only when process status, health, contract checks, and
any requested request smoke pass. It returns HTTP `503` with the same structured
payload when the provider is stopped, unreachable, mismatched, or app runtime is
misconfigured.

## Default DGX Target

| Item | Value |
| --- | --- |
| SSH target | `nexpcx@192.168.20.243` |
| Workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider port | `9104` |
| Provider name | `qwen-reranker-primary` |
| Model | `Qwen/Qwen3-Reranker-4B` |
| Profile | `qwen3_reranker_4b` |
| Backend | `qwen_reranker` |
| Device | `cuda:0` |

When `NEX_PCX_REMOTE_RERANKER_PROVIDER_URL` is configured, the operations status
uses its host and port as the HTTP target while retaining the standard DGX SSH
target for process status.

## Slice 331 Evidence

The Slice 331 live status check against the DGX reranker returned:

| Item | Result |
| --- | --- |
| Operations status | `ready` |
| Process status | `running` |
| PID | `2437559` |
| Health HTTP status | `200` |
| Request smoke | `passed=true` |
| Request elapsed | `293 ms` |
| Provider elapsed | `279 ms` |
| Returned count | `2` |
| Top score preview | `8.756176` |

Playwright captures:

- Desktop: `/tmp/nex_pcx_slice331_reranker_provider.png`
- Mobile: `/tmp/nex_pcx_slice331_reranker_provider_mobile.png`
