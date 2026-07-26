# Generation Prompt Preview API/UI

Slice 343 adds a non-persistent preview path for generation prompts.

## API

`GET /api/search/logs/{search_log_id}/generation-prompt/preview`

Query parameters mirror the generation run creation path:

- `max_context_chars`
- `include_neighbors`
- `max_items`
- `response_language`

The response returns the retrieval context package and the generated prompt
package. It does not create rows in `generation_runs`.

## UI

`GET /generation?search_log_id=...`

When a retrieval context package is loaded, the page renders a prompt preview
panel showing the OpenAI-compatible `messages`, response language,
`prompt_hash`, `context_hash`, citation count, and blocked/block reason state.

This lets operators inspect the exact generation input before mock or vLLM
execution.
