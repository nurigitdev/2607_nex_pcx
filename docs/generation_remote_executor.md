# Remote vLLM Generation Executor Foundation

Slice 351 adds the first DB-persisted remote generation execution path.

## Execution Contract

`execute_remote_generation_run(...)` accepts a `RetrievalContextPackage`, loads
the active default `generation_provider_configs` row, and requires the provider
mode to be `remote_openai_compatible`.

For answerable context, the executor:

- builds the same `GenerationPromptPackage` used by mock generation
- converts prompt messages into an OpenAI-compatible chat request
- applies model id, max tokens, temperature, top_p, and `extra_body` from the
  provider config
- calls the injected or configured generation provider client
- persists the answer, finish reason, token counts, latency, provider metrics,
  response id, and provider model id into `generation_runs`
- stores one `generation_run_citations` row per included context candidate
- stores `response_metadata.answer_quality` so citation usage, empty answers,
  unexpected no-answer text, and invented citation keys are measurable after a
  successful provider call

## Guardrail Path

If retrieval confidence or citation readiness blocks generation, the executor
does not call the remote provider. It persists:

- `status=no_answer`
- `guardrail_status=no_answer`
- `finish_reason=guardrail_no_answer`
- deterministic no-answer text
- `response_metadata.skipped_provider_call=true`
- `response_metadata.answer_quality.status=passed` when the persisted no-answer
  text reflects the guardrail decision

## Failure Path

When the remote provider raises `GenerationProviderRequestError`, the executor
persists a failed generation run instead of losing the attempt.

The failed run stores:

- `status=failed`
- provider error message
- parsed provider metrics when available
- provider error payload
- prompt/context hash and runtime request metadata
- `response_metadata.answer_quality.status=not_evaluated`

Provider execution success and answer quality are intentionally separated. A
remote call can persist `status=succeeded` while
`response_metadata.answer_quality.status=failed` records that the answer omitted
all expected citation keys.

## Test Strategy

CI uses an injected fake provider, not DGX-Spark. This keeps regression tests
deterministic while proving the persistence contract for success, failure, and
guardrail skip branches.
