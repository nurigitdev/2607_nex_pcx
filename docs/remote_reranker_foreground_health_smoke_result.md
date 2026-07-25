# Remote Reranker Foreground Health Smoke Result

Slice 328 validated the DGX-Spark foreground launch path for the Qwen3-Reranker-4B
remote provider.

| Item | Result |
| --- | --- |
| Generated at | `2026-07-25` |
| DGX host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider port | `9104` |
| Provider name | `qwen-reranker-primary` |
| Backend | `qwen_reranker` |
| Model | `Qwen/Qwen3-Reranker-4B` |
| Profile | `qwen3_reranker_4b` |
| Device | `cuda:0` |
| Health status | PASS |
| Health attempts | `7` |
| Foreground elapsed | `47.27s` |
| Shutdown confirmed | yes |
| Post-stop port state | closed |

## Readiness Notes

- The DGX model directory existed at `models/qwen3_reranker_4b`.
- `fastapi`, `uvicorn`, `sentence-transformers`, and runtime dependencies were available in the remote venv.
- `app/reranker_provider_service.py` and `app/core/rerankers.py` were copied to the DGX workdir before launch because the model directory was present but the reranker service sources were missing.

## Health Contract

`GET /healthz` returned HTTP 200 with:

```json
{
  "ready": true,
  "provider_type": "remote",
  "provider_model_id": "Qwen/Qwen3-Reranker-4B",
  "reranker_profile_name": "qwen3_reranker_4b",
  "device": "cuda:0",
  "runtime_metadata": {
    "service": "nex_pcx_reranker_provider_service",
    "backend": "qwen_reranker",
    "model_dir_exists": true
  }
}
```

## Lifecycle Evidence

- Pre-launch health check: unreachable, as expected.
- Launch: uvicorn served `app.reranker_provider_service:app` on `0.0.0.0:9104`.
- Model load: checkpoint shards loaded `2/2`.
- Stop: foreground process received shutdown, completed application shutdown, and exited.
- Post-stop health check: unreachable, confirming the foreground provider was no longer serving.
