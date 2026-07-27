# Direct Generation Query Orchestrator API

Slice 359 adds the first direct query-to-generation API.

## Purpose

Earlier generation flows required an existing search log ID. Direct generation
lets an operator submit a prompt query and have NeX_PCX orchestrate the RAG path:

1. run search with the requested search profiles and permission scope;
2. persist the search log and search log results;
3. build a retrieval context package from that search log;
4. execute mock or remote OpenAI-compatible generation;
5. persist the generation run, citation trace, and answer quality metadata.

## API

`POST /api/generation/direct-runs`

Request fields:

- `query_text`: user prompt query.
- `actor_user_id`: user identity used by permission-aware search.
- `requested_search_scope`: default `company`.
- `provider_mode`: `mock` or `remote_openai_compatible`.
- `top_k`, `profiles`, `chunk_policy_name`, `document_group`, `file_type`:
  search controls passed through to the search compare runtime.
- `bm25_tokenizer_name`, `hybrid_vector_profile_name`,
  `reranked_vector_profile_name`, `allow_mock_fallback`: runtime strategy controls.
- `max_context_chars`, `include_neighbors`, `max_items`: retrieval context package
  controls.

Response fields:

- `search`: existing search compare payload.
- `retrieval_context`: existing retrieval context package payload.
- `generation`: existing generation execution report payload.
- `links`: direct links to the stored search log, retrieval context, and
  generation run detail page.

Remote mode uses the configured default generation provider and resolves the API
key from environment-backed settings. Secrets are not included in the response.

## CI Contract

Unit tests verify the core orchestration sequence and validation branches.
Integration tests verify that the API can run BM25 search, package context, and
persist a mock generation run in the migrated PostgreSQL test database.

This contract is tracked by SRS FR-075.
