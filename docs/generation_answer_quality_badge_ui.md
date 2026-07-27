# Generation Answer Quality Badge + Detail UI

Slice 357 promotes the answer quality metadata from raw JSON evidence into
operator-readable UI panels.

## Purpose

Slice 356 stores `response_metadata.answer_quality` for mock and remote vLLM
generation runs. The UI now surfaces that contract directly so a reviewer can
quickly decide whether a generated answer is usable as grounded output before
opening the raw metadata.

## Pages

`GET /generation`

- Adds an answer quality badge beside run status and guardrail status.
- Adds an answer quality summary panel under the run metrics.
- Shows quality status, expected citation count, used citation count, citation
  coverage percent, and reason count.

`GET /generation/runs/{generation_run_id}`

- Adds an answer quality detail panel before the answer text.
- Shows quality status, citation coverage, expected/used/missing/unknown
  citation keys, and translated reason codes.
- Keeps `response_metadata` and `guardrail_metadata` JSON previews available as
  reproducibility evidence.

## Status Mapping

| Metadata status | UI badge |
| --- | --- |
| `passed` | Success |
| `warning` | Warning |
| `failed` | Danger |
| `not_evaluated` | Secondary |
| missing metadata | Not Available |

The badge does not replace `generation_runs.status`. Provider execution health,
retrieval guardrail status, and answer quality remain separate signals.

## CI Contract

Integration tests verify that a mock generation run exposes the answer quality
summary on `/generation` and the answer quality detail panel on
`/generation/runs/{generation_run_id}`. Documentation tests verify that the SRS
tracks FR-073 and that this UI contract remains documented.
