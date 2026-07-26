# Generation Run API

Slice 340 exposes the mock generation executor through HTTP so retrieval context
packages can be promoted into persisted generation evidence.

## Create Mock Generation Run

`POST /api/search/logs/{search_log_id}/generation-runs/mock`

Query parameters match the retrieval context package controls:

- `max_context_chars`
- `include_neighbors`
- `max_items`

Response status is `201 Created`.

The response contains:

- `provider`: active default generation provider snapshot
- `prompt_package`: OpenAI-compatible messages and prompt/context hashes
- `run`: persisted `generation_runs` row
- `citations`: persisted citation rows for answerable generation runs

If retrieval context cannot be built for the search log, the endpoint returns
`404`. Invalid context controls return `400`. Missing database configuration
returns `503`.

## Read Generation Run

`GET /api/generation/runs/{generation_run_id}`

The response contains:

- `run`: persisted generation run detail
- `citations`: citation rows ordered by citation index

This endpoint is intentionally provider-neutral. Mock and future remote vLLM runs
share the same detail response shape.
