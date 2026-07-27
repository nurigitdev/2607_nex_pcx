# Generation Mock Executor

Slice 339 adds the first executable generation path without requiring DGX-Spark or
vLLM availability.

## Purpose

- Validate generation run persistence before remote LLM integration.
- Keep retrieval confidence and citation readiness guardrails observable.
- Store deterministic answer/no-answer evidence in the same tables that remote
  providers will use later.

## Execution Contract

Input is a `RetrievalContextPackage`.

The executor builds a `GenerationPromptPackage`, loads the active default
`generation_provider_configs` row, and writes one `generation_runs` row.

For answerable context:

- `status`: `succeeded`
- `guardrail_status`: `allowed`
- `finish_reason`: `mock_completed`
- `answer_text`: deterministic Korean answer with inline citation key
- `generation_run_citations`: one row per included context candidate
- `response_metadata.answer_quality`: post-generation citation and no-answer
  quality assessment

For blocked context:

- `status`: `no_answer`
- `guardrail_status`: `no_answer`
- `finish_reason`: `guardrail_no_answer`
- `answer_text`: deterministic no-answer text
- `generation_run_citations`: no citation rows
- `response_metadata.answer_quality.status`: `passed` when the no-answer text
  reflects the guardrail decision

## Remote vLLM Handoff

The mock executor intentionally uses OpenAI-compatible prompt metadata:

- `messages`
- `prompt_hash`
- `context_hash`
- `model_id`
- `request_metadata`
- `response_metadata`

The remote vLLM executor should replace only the answer call path. It should keep
the same run and citation repository contract so mock, staging, and live DGX
evidence remain comparable.
