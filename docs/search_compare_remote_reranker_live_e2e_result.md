# Search Compare Remote Reranker Live E2E Result

Slice 329 validated the full NeX-PCX Search Compare path against the live DGX
Qwen3-Reranker-4B provider.

## Summary

| Item | Result |
| --- | --- |
| Generated at | `2026-07-25` |
| Target DB | `nex_pcx_dev` |
| DB migration prerequisite | `20260725_0032` |
| Search profile | `reranked_vector_cosine` |
| Source vector profile | `qwen3_4b_2560` |
| Chunk policy | `heading_512_64` |
| Query | `사내 문서 검색 권한과 업무 규칙` |
| Actor user ID | `1` |
| Requested scope | `company` |
| Remote reranker URL | `http://192.168.20.243:9104` |
| Remote reranker model | `Qwen/Qwen3-Reranker-4B` |
| Remote reranker profile | `qwen3_reranker_4b` |
| Remote reranker backend | `qwen_reranker` |
| Remote reranker device | `cuda:0` |
| Search log ID | `24` |
| Profile status | `succeeded` |
| Result count | `3` |
| Candidate count | `12` |
| Search Compare elapsed | `7467 ms` |
| Request elapsed | `7626 ms` |
| Reranker provider elapsed | `7090 ms` |
| Contract mismatches | none |

## Prerequisite Finding

The first live E2E attempt failed before successful reranking because the development
database was still at Alembic revision `20260720_0031`; the
`reranked_vector_cosine` row was missing from `search_profiles`.

Applying `20260725_0032` added the required reranked search profile seed and allowed
the live Search Compare path to complete successfully.

## Reranked Result Preview

| Rank | Chunk | Rerank Score | Source Rank | Source Score | Document | File |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| `1` | `54` | `-3.430655` | `2` | `0.39820148084193174` | `live-smoke-dev-readme-retry` | `live-smoke-dev-readme-retry.md` |
| `2` | `50` | `-6.552152` | `8` | `0.2939935129834619` | `live-smoke-dev-readme-retry` | `live-smoke-dev-readme-retry.md` |
| `3` | `45` | `-8.91459` | `1` | `0.4337836757061492` | `live-smoke-dev-readme-retry` | `live-smoke-dev-readme-retry.md` |

## Profile Runtime Metadata

```json
{
  "provider_type": "rerank",
  "provider_model_id": "Qwen/Qwen3-Reranker-4B",
  "runtime_source": "local_reranker_contract",
  "query_embedding_bridge": true,
  "retrieval_strategy": "reranked",
  "search_profile_name": "reranked_vector_cosine",
  "reranked_vector_profile_name": "qwen3_4b_2560",
  "source_vector_profile_name": "qwen3_4b_2560",
  "candidate_top_k": 12,
  "candidate_multiplier": 4,
  "provider_runtime_mode": "remote",
  "provider_runtime_base_url": "http://192.168.20.243:9104",
  "provider_runtime_timeout_seconds": 300.0,
  "reranker_profile_name": "qwen3_reranker_4b",
  "reranker_model_id": "Qwen/Qwen3-Reranker-4B",
  "reranker_provider_type": "remote",
  "candidate_count": 12,
  "reranker_runtime_metadata": {
    "service": "nex_pcx_reranker_provider_service",
    "backend": "qwen_reranker",
    "device": "cuda:0",
    "model_source": "/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_4b",
    "elapsed_ms": 7090,
    "input_count": 12
  }
}
```

## Source Query Runtime Evidence

The source vector search used the DGX Qwen embedding provider route:

```json
{
  "profile_name": "qwen3_4b_2560",
  "dimension": 2560,
  "provider_type": "remote",
  "provider_model_id": "local-qwen3-embedding-4b",
  "provider_elapsed_ms": 86,
  "provider_runtime_base_url": "http://192.168.20.243:9103",
  "embedding_table_name": "chunk_embeddings_qwen3_4b_2560",
  "embedding_storage_type": "halfvec",
  "provider_route_id": 5,
  "provider_route_name": "qwen-primary"
}
```

## Lifecycle Evidence

- The DGX reranker foreground process served `GET /healthz` and `POST /v1/rerank`.
- Search Compare called `/v1/rerank` through the remote runtime configuration.
- The provider was stopped after the live E2E smoke.
- Post-stop port check for `192.168.20.243:9104` returned connection refused.
