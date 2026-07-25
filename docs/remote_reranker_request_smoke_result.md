# Remote Reranker Request Smoke Result

Slice 328 ran `/v1/rerank` against the live DGX Qwen3-Reranker-4B provider after
foreground health passed.

| Item | Result |
| --- | --- |
| Generated at | `2026-07-25` |
| Provider name | `qwen-reranker-primary` |
| Endpoint | `http://192.168.20.243:9104/v1/rerank` |
| Provider model | `Qwen/Qwen3-Reranker-4B` |
| Reranker profile | `qwen3_reranker_4b` |
| Expected backend | `qwen_reranker` |
| Expected device | `cuda:0` |
| Provider type | `remote` |
| Retrieval strategy | `reranked` |
| Candidate count | `3` |
| Returned count | `2` |
| Top K | `2` |
| Request elapsed | `2325 ms` |
| Provider elapsed | `2209 ms` |
| Contract mismatches | none |

## Result Score Preview

| Rank | Candidate | Score | Source Rank |
| ---: | --- | ---: | ---: |
| `1` | `candidate-1` | `8.756176` | `1` |
| `2` | `candidate-2` | `6.445219` | `2` |

## Runtime Metadata

```json
{
  "service": "nex_pcx_reranker_provider_service",
  "backend": "qwen_reranker",
  "device": "cuda:0",
  "model_source": "/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_4b",
  "elapsed_ms": 2209,
  "input_count": 3
}
```

## Operator Note

This evidence proves the remote reranker can load from the DGX local model bundle,
serve the HTTP contract, return finite rerank scores, and shut down cleanly after
the foreground smoke session.
