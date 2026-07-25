# Remote Reranker Foreground Launch And Health Smoke

Slice 324 adds the first operator-safe foreground launch path for the Qwen3-Reranker-4B
remote provider. It follows the embedding provider pattern but targets
`app.reranker_provider_service:app` and validates the reranker `/healthz` contract.

## Defaults

| Item | Value |
| --- | --- |
| DGX host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider port | `9104` |
| Provider backend | `qwen_reranker` |
| Model directory | `/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_4b` |
| ASGI app | `app.reranker_provider_service:app` |

## Generate A Plan

```bash
./.venv/bin/python scripts/plan_remote_reranker_foreground_smoke.py
./.venv/bin/python scripts/plan_remote_reranker_foreground_smoke.py --json
```

The plan prints:

- remote readiness command
- interactive foreground SSH launch command
- remote port listener check
- app-host `/healthz` command
- Ctrl-C shutdown instruction

## Run Automated Health Smoke

```bash
./.venv/bin/python scripts/run_remote_reranker_foreground_smoke.py --json
```

For first model load on DGX, keep a generous startup timeout:

```bash
./.venv/bin/python scripts/run_remote_reranker_foreground_smoke.py \
  --startup-timeout-seconds 900 \
  --poll-interval-seconds 5 \
  --json
```

## Expected Health Contract

The smoke passes when `/healthz` returns HTTP 200 with:

- `ready=true`
- `provider_type=remote`
- `provider_model_id=Qwen/Qwen3-Reranker-4B`
- `reranker_profile_name=qwen3_reranker_4b`
- `device=cuda:0`
- `runtime_metadata.service=nex_pcx_reranker_provider_service`
- `runtime_metadata.backend=qwen_reranker`
- `runtime_metadata.model_dir_exists=true`

## Failure Triage

| Symptom | First check |
| --- | --- |
| Port already active | Stop the existing process or use `--port <new-port>`. |
| Import error | Confirm `fastapi`, `uvicorn`, `sentence-transformers`, and model runtime packages. |
| Missing model directory | Confirm `models/qwen3_reranker_4b` exists on DGX. |
| Slow startup | Increase `--startup-timeout-seconds`; first model load may be slow. |
| Health mismatch | Compare the JSON payload with the expected health contract above. |

After this health smoke passes, run the request smoke for `/v1/rerank`:

```bash
./.venv/bin/python scripts/run_remote_reranker_request_smoke.py \
  --markdown-output artifacts/remote_reranker_request_smoke.md
```

See `docs/remote_reranker_request_smoke_runner.md` for request payload and evidence details.
