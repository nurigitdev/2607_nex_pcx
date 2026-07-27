# Direct Generation UI MVP

Slice 360 adds the first user-facing direct query generation flow.

## Purpose

The earlier generation screen could create answers only after an operator selected
an existing search log. Direct generation keeps that reproducible path, but adds a
front-door form on `GET /generation` so an operator can enter a prompt query and
let NeX_PCX run:

1. permission-aware search;
2. retrieval context package creation;
3. mock or remote OpenAI-compatible generation;
4. generation result, citation trace, and answer quality display.

## UI Contract

`GET /generation` renders a `data-direct-generation-form` section with these
controls:

- prompt query text;
- search actor and requested search scope;
- generation provider mode: `mock` or `remote_openai_compatible`;
- search profile, top-k, optional chunk policy, file type, document group, and
  BM25 tokenizer;
- retrieval context budget controls.

`POST /generation/direct-runs` accepts the form, calls the Slice 359 direct
generation orchestrator, and redirects back to `/generation` with the created
`search_log_id` and `generation_run_id`. The existing result panel then displays
the generated answer, citation trace, quality badge, and detail/API links.

Remote mode resolves the default generation provider API key through the existing
environment-backed runtime config. The UI defaults to remote only when the
runtime is ready; otherwise it defaults to deterministic `mock`.

## CI Contract

Integration tests verify that the Korean default page renders the direct
generation form, that the form can execute a BM25-backed mock generation run in
the migrated PostgreSQL test database, and that the redirect displays answer
quality and citation evidence.

This contract is tracked by SRS FR-076.
