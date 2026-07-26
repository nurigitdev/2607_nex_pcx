# Generation Provider Metrics Contract

Slice 344 defines the metrics contract for future vLLM generation execution.

## Parser

`parse_openai_chat_completion_metrics(...)`

The parser accepts an OpenAI-compatible chat completion response payload and
normalizes:

- response id and model id
- HTTP status code
- finish reason from `choices[0].finish_reason`
- prompt/completion/total token usage
- total and provider latency
- retry count
- error code and message from OpenAI-style `error`
- lightweight response metadata such as object, created, service tier,
  system fingerprint, and choice count

## Output

`generation_provider_metrics_payload(...)` serializes the parsed snapshot for
future generation run metadata, API responses, and operations panels.

The contract is intentionally independent from the network client so that mock
tests can validate vLLM response handling before the DGX runtime is reachable.
