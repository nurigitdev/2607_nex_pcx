# Generation Run UI MVP

Slice 341 adds the first user-facing generation screen.

## Page

`GET /generation`

The page supports:

- search log ID selection
- retrieval context controls
- mock generation execution
- generation answer display
- run status, guardrail status, token counts, latency, and citation trace
- answer quality badge and summary panel from `response_metadata.answer_quality`

## Execution

`POST /generation/runs/mock`

The form builds the retrieval context package for the selected search log and
executes the deterministic mock generation path. On success, the user is
redirected back to `/generation` with the new `generation_run_id` selected.

The UI intentionally uses the same stored run data as the API. Future remote
vLLM generation runs should render in this page without changing the result
panel contract.
