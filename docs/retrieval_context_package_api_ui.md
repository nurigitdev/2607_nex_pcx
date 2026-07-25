# Retrieval Context Package API/UI

Slice 332 adds a read-only package builder that turns stored search results into
generation-ready context.

## API

```http
GET /api/search/logs/{search_log_id}/retrieval-context
```

Query parameters:

- `max_context_chars`: context character budget, default `12000`.
- `include_neighbors`: include previous/current/next chunk context, default `true`.
- `max_items`: maximum unique candidate chunks to package, default `20`.

The response includes:

- search log query, actor, permission scope, profiles, chunk policy, and runtime metadata.
- included and excluded candidates.
- citation keys such as `RCP-001`.
- source labels with file, page, slide, sheet, or cell information when available.
- chunk text, source anchors, artifact/block IDs, truncation state, and budget summary.
- `generation_context_text`, ready to pass to a later generation smoke path.

## UI

Open:

```text
/search/context
```

The page lets operators select a `search_log_id`, adjust the context budget,
toggle neighbor chunks, inspect citations, and view the raw package JSON.

## Current Boundary

This slice does not call an LLM and does not persist a separate generation run.
It standardizes the retrieval-to-generation handoff contract so that later
generation slices can reuse the same evidence package.
