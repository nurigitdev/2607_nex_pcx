# DGX vLLM Generation Smoke Runner

Slice 348 adds a live smoke runner for the DGX-Spark vLLM generation runtime.

## Target Runtime

- Host: `192.168.20.243`
- Port: `12000`
- Endpoint: `/v1/chat/completions`
- Model id: `/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4`
- Serving max model length label: `200k`

The API key is never passed as a CLI argument and must be provided through:

`NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`

## Command

```bash
NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY='<secret>' \
  ./.venv/bin/python scripts/run_dgx_vllm_generation_smoke.py \
  --markdown-output docs/dgx_vllm_generation_smoke_result.md
```

The generated markdown evidence records the environment variable name and
whether it was configured, but not the secret value.

By default the runner adds:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

This keeps the smoke request short and ensures the Qwen runtime returns final
assistant content instead of spending the smoke token budget on reasoning text.
Use `--enable-thinking` only when explicitly testing thinking-mode behavior.

## Pass Criteria

The smoke passes when the client receives:

- HTTP 2xx response parsed through `GenerationProviderMetrics`
- non-empty answer text
- finish reason
- total token count
- `provider_metrics.succeeded=true`
- final assistant content with Qwen thinking disabled by default

The response model id is recorded for inspection. Strict response model matching
can be enabled with `--require-response-model-match`, but it is not required by
default because vLLM can expose an alias through `--served-model-name`.

## Evidence Fields

- chat completions URL
- model id requested by NeX-PCX
- API key environment variable name and configured flag
- HTTP status
- response model id and response id
- finish reason
- request/provider elapsed milliseconds
- prompt/completion/total token usage
- answer preview
- whether thinking was disabled for the smoke request
- provider metrics JSON
