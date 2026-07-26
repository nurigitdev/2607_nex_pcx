# Generation Provider Runtime Config API

Slice 349 exposes the DB-backed generation provider configuration that will
drive vLLM generation runs after the mock-first phase.

## API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/admin/generation-provider-configs` | List active or all generation provider configs with normalized runtime metadata |
| `GET /api/admin/generation-provider-configs/default` | Read the active default provider config and runtime contract |
| `POST /api/admin/generation-provider-configs/seed-dgx-vllm` | Upsert DGX-Spark Qwen3.6 vLLM defaults |

The list endpoint accepts `include_inactive=true|false`; the default includes
inactive rows so operators can inspect disabled runtime candidates.

## DGX vLLM Seed Defaults

The seed endpoint uses these safe defaults unless overridden in the request:

- provider name: `dgx_vllm_qwen36_27b_nvfp4`
- provider mode: `remote_openai_compatible`
- base URL: `http://192.168.20.243:12000`
- model id: `/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4`
- endpoint contract: `/v1/chat/completions`
- API key reference: `NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY`
- timeout: `300` seconds
- `extra_body`: `{"chat_template_kwargs": {"enable_thinking": false}}`

By default the seed endpoint does not switch the app away from the mock
provider. Operators must set `is_default=true` explicitly when they want the
remote vLLM provider to become the active default.

## Secret Handling

Generation provider config rows must not persist secret values. The API stores
only the environment variable name in `runtime_options.api_key_env`; the actual
API key remains outside the database and is supplied by process environment.

Responses include:

- `api_key_env`: the expected environment variable name
- `api_key_configured`: whether the current app settings include that variable
  for the known DGX key name
- `secret_policy`: a fixed statement that secret values are environment-only

The response payload redacts sensitive option keys such as `api_key`,
`authorization`, `token`, `password`, and `secret` if they ever appear in older
or manually edited rows.

## Runtime Contract

Each provider payload includes a normalized `runtime_config` block:

- validity flag and validation error, if any
- provider mode
- remote base URL
- remote timeout seconds
- model id
- max tokens
- temperature
- top_p
- configured header names, without header values
- extra body options such as Qwen thinking-mode control

This gives later generation execution slices a single source of truth for
operator-visible runtime settings while keeping live secrets out of API and
documentation evidence.
