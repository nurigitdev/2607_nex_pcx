# Generation Provider Strategy + vLLM Runtime Contract

Slice 336 defines the generation-provider direction before adding generation
run persistence or remote LLM calls.

## Strategy

- NeX_PCX treats generation as a grounded experiment stage after retrieval,
  citation readiness, and retrieval confidence checks.
- Local development and CI use a deterministic `mock` generation provider.
- Remote LLM execution uses an OpenAI-compatible vLLM server directly instead
  of wrapping the LLM in another FastAPI provider process.
- The default remote model candidate is `nvidia/Qwen3.5-122B-A10B-NVFP4`.
- Remote DGX-Spark smoke tests are deferred until the GPU host is reachable;
  schema, API, and UI contracts must still be testable with the mock provider.

## Provider Modes

| Mode | Purpose |
| --- | --- |
| `mock` | Deterministic local/test generation contract validation |
| `remote_openai_compatible` | vLLM `/v1/chat/completions` runtime |

## Runtime Settings

The generation runtime must be configurable without code changes:

- provider mode
- provider base URL
- model id
- API key usage flag or secret reference
- request timeout
- max tokens
- temperature
- top_p
- prompt version
- context character budget

## Request Contract

The first remote endpoint is `/v1/chat/completions`.

The request should include:

- `model`
- `messages`
- `max_tokens`
- `temperature`
- `top_p`
- optional stop sequences
- trace/runtime metadata where supported

The response metadata persisted by NeX_PCX should include:

- answer text
- finish reason
- token usage when provided
- model id
- provider latency
- prompt version and rendered prompt hash
- guardrail status

## Guardrails

Generation must not pass arbitrary weak context to the LLM.

- `answerable` retrieval confidence can proceed.
- `low_confidence` or `no_relevant_context` must produce a blocked or
  no-answer run record.
- citation readiness `failed` must block grounded answer generation until the
  operator explicitly chooses an override in a later slice.

## Next Slices

1. Generation run schema and retrieval package linkage.
2. Prompt package builder.
3. Mock generation API and UI.
4. OpenAI-compatible vLLM client foundation.
5. Remote generation executor wiring from retrieval package to vLLM client.
6. Remote vLLM smoke when DGX-Spark is reachable.
