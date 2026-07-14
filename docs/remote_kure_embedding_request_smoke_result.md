# Remote KURE Embedding Request Smoke Result

Date: 2026-07-14

Slice 200 validated the first real `/v1/embeddings` request against the KURE remote
embedding provider on the DGX Spark host.

## Environment

| Item | Value |
| --- | --- |
| GPU host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Remote workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider | `kure` |
| Port | `9101` |
| Model key | `kure_v1` |
| Profile | `kure_v1_1024` |
| Device | `cuda:0` |

## Passing Request Smoke

The provider was launched on the remote host, `/healthz` was confirmed healthy, and then
the app host executed:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider kure \
  --timeout-seconds 120 \
  --json
```

Observed result:

- `passed`: `true`
- `embeddings_url`: `http://192.168.20.243:9101/v1/embeddings`
- `total_elapsed_ms`: `914`
- `request_elapsed_ms`: `913`
- `provider_elapsed_ms`: `896`
- `provider_model_id`: `local-kure-v1`
- `provider_type`: `remote`
- `dimension`: `1024`
- `input_count`: `1`
- `embedding_count`: `1`
- `mismatches`: `[]`
- `error`: `null`

Embedding vector preview:

| Profile | First 3 values from first vector |
| --- | --- |
| `kure_v1_1024` | `[-0.04624427482485771, 0.043011296540498734, -0.005476752761751413]` |

Runtime metadata matched:

- `adapter`: `sentence_transformers`
- `profile_name`: `kure_v1_1024`
- `model_source`: `/home/nexpcx/2607_nex_pcx/models/kure_v1`
- `normalize_embeddings`: `true`
- `device`: `cuda:0`

The JSON report included only a three-value preview from the first vector. The full
embedding payload was not persisted in the repository.

Post-smoke cleanup verification:

- `curl -fsS http://192.168.20.243:9101/healthz` returned connection refused after
  shutdown completed.
- Remote `ss` showed no listener on port `9101`.
