# Citation Coverage + Source Anchor Readiness

Slice 333 adds a read-only readiness check for retrieval context packages before
they are passed to a generation provider.

## API

```http
GET /api/search/logs/{search_log_id}/citation-readiness
```

Query parameters match the retrieval context package API:

- `max_context_chars`
- `include_neighbors`
- `max_items`

The response reports:

- overall status: `ready`, `warning`, or `failed`
- source anchor coverage percent
- citation-ready percent
- candidate-level citation key, chunk id, source label, and issue codes

## Issue Codes

- `candidate_excluded`: candidate was not included in the generation context.
- `missing_citation_key`: included candidate has no stable citation key.
- `missing_document_identity`: no document title or original file name.
- `missing_chunk_identity`: no chunk id or chunk policy name.
- `missing_generation_text`: included candidate has no context text.
- `weak_source_anchor`: no source anchor, location hint, or lineage reference.
- `missing_artifact_block_reference`: no artifact/block lineage reference.
- `context_truncated`: context text was truncated by the character budget.

## UI

Open:

```text
/search/citation-readiness
```

The page shows recent search logs, lets the operator run a readiness check for a
specific `search_log_id`, and links back to the corresponding retrieval context
package.
