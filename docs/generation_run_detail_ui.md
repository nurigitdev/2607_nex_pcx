# Generation Run Detail UI

Slice 342 adds a stable detail page for persisted generation runs.

## Route

`GET /generation/runs/{generation_run_id}`

The page reads the stored `generation_runs` record and its
`generation_run_citations` rows. It does not re-run retrieval or generation.

## Detail Contract

The detail screen shows:

- answer text and run status
- provider name, provider mode, and model id
- prompt version, prompt hash, context hash, and retrieval package key
- token and latency counters
- citation trace rows with source anchor payloads
- request, response, and guardrail metadata JSON previews

This keeps mock and future vLLM runs reviewable through the same UI contract.
