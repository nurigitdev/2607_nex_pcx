# Remote Qwen Embedding Request Smoke Result

Date: 2026-07-14

Slice 202 validated the first real dual-profile `/v1/embeddings` request against the
shared Qwen remote embedding provider on the DGX Spark host.

## Environment

| Item | Value |
| --- | --- |
| GPU host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Remote workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider | `qwen` |
| Port | `9103` |
| Model key | `qwen3_embedding_4b` |
| Profiles | `qwen3_4b_1000`, `qwen3_4b_2560` |
| Device | `cuda:0` |

## Passing Request Smoke

The provider was launched on the remote host, `/healthz` was confirmed healthy, and then
the app host executed:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider qwen \
  --timeout-seconds 300 \
  --json
```

Observed result:

- `passed`: `true`
- `embeddings_url`: `http://192.168.20.243:9103/v1/embeddings`
- `total_elapsed_ms`: `3330`
- `provider_model_id`: `local-qwen3-embedding-4b`
- `provider_type`: `remote`
- `mismatches`: `[]`

Profile observations:

| Profile | Dimension | Storage type | Request elapsed | Provider elapsed | Embeddings |
| --- | ---: | --- | ---: | ---: | ---: |
| `qwen3_4b_1000` | `1000` | `vector` | `3180 ms` | `3160 ms` | `1` |
| `qwen3_4b_2560` | `2560` | `halfvec` | `150 ms` | `134 ms` | `1` |

Runtime metadata matched:

- `adapter`: `qwen_embedding`
- `model_source`: `/home/nexpcx/2607_nex_pcx/models/qwen3_embedding_4b`
- `pooling_strategy`: `sentence-transformers-truncate-dim`
- `normalize_embeddings`: `true`
- `device`: `cuda:0`
- `shared_model_cache_key`: `/home/nexpcx/2607_nex_pcx/models/qwen3_embedding_4b:cuda:0:`

The same shared model cache key was reported by both profiles, confirming that one loaded
Qwen model served both output-dimension profiles. The JSON report included only a
three-value preview from each vector. The full embedding payload was not persisted in the
repository.

Post-smoke cleanup verification:

- `curl -fsS http://192.168.20.243:9103/healthz` returned connection refused after
  shutdown completed.
- Remote `ss` showed no listener on port `9103`.
