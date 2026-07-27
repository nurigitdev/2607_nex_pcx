# Remote Generation Run API

Slice 352 exposes the DB-backed remote vLLM executor through HTTP.

## Endpoint

`POST /api/search/logs/{search_log_id}/generation-runs/remote`

Query parameters match the mock generation run API:

- `max_context_chars`: retrieval context budget, default `12000`
- `include_neighbors`: whether neighboring chunks are included, default `true`
- `max_items`: maximum retrieval candidates, default `8`

The endpoint builds a `RetrievalContextPackage`, resolves the active default
`generation_provider_configs` row, and calls `execute_remote_generation_run(...)`.

## Runtime Config

The default provider must use `remote_openai_compatible` mode. Provider base URL,
model id, timeout, token limit, sampling settings, headers, and `extra_body` come
from the DB provider config.

API keys remain environment-only. The endpoint reads `runtime_options.api_key_env`
from the provider config, resolves that environment variable at request time, and
passes the secret only to the runtime executor. Secret values are not returned in
the API payload and are not stored in `generation_runs`.

## Responses

Successful HTTP execution returns `201` with the existing generation execution
payload:

- `provider`: DB provider config snapshot
- `prompt_package`: reproducible OpenAI-compatible prompt package
- `run`: persisted `generation_runs` row
- `citations`: persisted citation trace

Provider failures are persisted as `status=failed` generation runs when the remote
executor reaches the provider boundary. Configuration and retrieval validation
errors return `400`; missing retrieval context returns `404`; missing database
configuration returns `503`.
