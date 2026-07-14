# Remote BGE Embedding Request Smoke Result

Date: 2026-07-14

Slice 201 validated the first real `/v1/embeddings` request against the BGE-M3 remote
embedding provider on the DGX Spark host.

## Environment

| Item | Value |
| --- | --- |
| GPU host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Remote workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider | `bge` |
| Port | `9102` |
| Model key | `bge_m3` |
| Profile | `bge_m3_1024` |
| Device | `cuda:0` |

## Passing Request Smoke

The provider was launched on the remote host, `/healthz` was confirmed healthy, and then
the app host executed:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider bge \
  --timeout-seconds 120 \
  --json
```

Observed result:

- `passed`: `true`
- `embeddings_url`: `http://192.168.20.243:9102/v1/embeddings`
- `total_elapsed_ms`: `707`
- `request_elapsed_ms`: `707`
- `provider_elapsed_ms`: `688`
- `provider_model_id`: `local-bge-m3`
- `provider_type`: `remote`
- `dimension`: `1024`
- `input_count`: `1`
- `embedding_count`: `1`
- `mismatches`: `[]`
- `error`: `null`

Embedding vector preview:

| Profile | First 3 values from first vector |
| --- | --- |
| `bge_m3_1024` | `[-0.043258894234895706, 0.036100007593631744, -0.005832674913108349]` |

Runtime metadata matched:

- `adapter`: `sentence_transformers`
- `profile_name`: `bge_m3_1024`
- `model_source`: `/home/nexpcx/2607_nex_pcx/models/bge_m3`
- `normalize_embeddings`: `true`
- `device`: `cuda:0`

The JSON report included only a three-value preview from the first vector. The full
embedding payload was not persisted in the repository.

Post-smoke cleanup verification:

- `curl -fsS http://192.168.20.243:9102/healthz` returned connection refused after
  shutdown completed.
- Remote `ss` showed no listener on port `9102`.
