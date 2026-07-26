# OpenAI-Compatible vLLM Client Foundation

Slice 347 adds the first remote generation client boundary without requiring a
live DGX-Spark vLLM server during CI.

## Runtime Config

`GenerationProviderRuntimeConfig` normalizes:

- provider mode: `mock` or `remote_openai_compatible`
- remote base URL and timeout
- optional API key and custom headers
- model id, max tokens, temperature, and top_p
- runtime options copied from `generation_provider_configs`

Remote mode requires a base URL. The mock mode remains handled by the
deterministic mock executor until a later slice wires remote execution into
`generation_runs`.

## Request Contract

`OpenAICompatibleGenerationProviderClient.complete(...)` posts to:

`POST /v1/chat/completions`

The payload is intentionally small and vLLM-friendly:

- `model`
- `messages`
- `max_tokens`
- `temperature`
- `top_p`
- `stream=false`
- optional `stop`
- optional `user` trace id

Prompt preview output can be converted through
`generation_chat_request_from_openai_messages(...)`.

## Response Contract

The client returns:

- answer text from `choices[0].message.content`
- finish reason
- response id and model id
- prompt/completion/total token usage
- elapsed/provider latency
- `GenerationProviderMetrics`
- raw response summary for later generation run metadata

Text extraction supports both plain string content and list-style text parts.

## Failure Handling

The client preserves failure evidence through `GenerationProviderRequestError`.

- transport failure: request error with no HTTP status
- invalid JSON: `error_code=invalid_json`
- non-object JSON: `error_code=invalid_response`
- HTTP error payload: parsed OpenAI-style `error.code` and `error.message`
- invalid success payload: parsed metrics plus original payload

This lets the future remote generation executor persist failed vLLM attempts
with the same provider metrics shape used by mock generation runs.
