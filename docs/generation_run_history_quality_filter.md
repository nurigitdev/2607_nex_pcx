# Generation Run History + Quality Filter API/UI

Slice 358 adds a lightweight history surface for stored generation runs.

## Purpose

Generation run detail pages are useful for one run at a time, but operators also
need a way to find quality warnings and failures across recent runs. The history
contract exposes answer quality status as a first-class filter while keeping the
existing provider metrics page focused on operational token and latency metrics.

## API

`GET /api/generation/runs`

Query parameters:

- `limit`: number of recent runs to return. Default is `50`; maximum is `500`.
- `answer_quality_status`: `all`, `passed`, `warning`, `failed`,
  `not_evaluated`, or `not_available`.
- `provider_mode`: `all`, `mock`, or `remote_openai_compatible`.
- `run_status`: `all` or a persisted generation run status such as `succeeded`,
  `failed`, or `no_answer`.

The response returns normalized filters, a quality summary, and recent run rows.
Each row includes provider identity, run status, answer quality status, citation
coverage percent, citation counts, token count, latency, and timestamps.

## UI

`GET /generation/runs`

The page provides:

- quality, provider mode, run status, and limit controls;
- answer quality summary cards;
- a recent run table with detail links to `/generation/runs/{generation_run_id}`;
- raw JSON evidence from the same API payload shape.

Korean remains the default UI language, with matching English labels.

## CI Contract

Integration tests verify API filtering, invalid filter handling, and the HTML
history screen. Documentation tests verify that FR-074 and this contract remain
documented.
