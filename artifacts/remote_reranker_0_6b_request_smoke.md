# Remote Reranker Request Smoke Result

- `passed`: `true`
- `provider_name`: `qwen-reranker-primary`
- `rerank_url`: `http://192.168.20.243:9104/v1/rerank`
- `provider_model_id`: `Qwen/Qwen3-Reranker-0.6B`
- `reranker_profile_name`: `qwen3_reranker_0_6b`
- `candidate_count`: `3`
- `returned_count`: `2`
- `request_elapsed_ms`: `1542`
- `provider_elapsed_ms`: `1526`

## Result Score Preview

| Rank | Candidate | Score | Source Rank |
| ---: | --- | ---: | ---: |
| `1` | `candidate-1` | `8.75` | `1` |
| `2` | `candidate-2` | `5.75` | `2` |

## Runtime Metadata

```json
{
  "service": "nex_pcx_reranker_provider_service",
  "backend": "qwen_reranker",
  "device": "cuda:0",
  "requested_torch_dtype": "bfloat16",
  "loaded_parameter_dtype": "bfloat16",
  "model_source": "/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_0_6b",
  "elapsed_ms": 1526,
  "input_count": 3
}
```
