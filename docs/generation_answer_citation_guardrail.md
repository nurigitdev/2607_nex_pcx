# Generation Answer Citation Guardrail + Quality Metadata

Slice 356 adds a post-generation answer quality assessment for mock and remote
vLLM generation runs.

## Purpose

The retrieval and prompt guardrails decide whether generation is allowed before
the provider call. The answer quality guardrail evaluates the provider output
after generation so operators can distinguish these cases:

- the provider returned an empty answer;
- the answer says it cannot answer even though answerable context was provided;
- the answer uses no expected citation key such as `[RCP-001]`;
- the answer uses only some expected citation keys;
- the answer invents an unknown citation key;
- the provider failed, so answer quality was not evaluated.

## Metadata Contract

The executor stores `response_metadata.answer_quality` for every mock and remote
generation path. The contract version is `generation_answer_quality_v1`. A
summary is also mirrored into `guardrail_metadata`:

```json
{
  "contract_version": "generation_answer_quality_v1",
  "status": "passed",
  "answer_present": true,
  "no_answer_detected": false,
  "requires_citation": true,
  "expected_citation_keys": ["RCP-001"],
  "cited_citation_keys": ["RCP-001"],
  "recognized_citation_keys": ["RCP-001"],
  "missing_citation_keys": [],
  "unrecognized_citation_keys": [],
  "citation_coverage_percent": 100.0,
  "reason_codes": []
}
```

## Status Rules

| Status | Meaning |
| --- | --- |
| `passed` | The answer satisfies the minimal grounded-answer checks. |
| `warning` | The answer is usable but has partial citation coverage or unknown citation keys. |
| `failed` | The answer is empty, unexpectedly no-answer, or missing all required citations. |
| `not_evaluated` | The provider failed before an answer was produced. |

This quality status does not rewrite `generation_runs.status`. For example, a
remote provider can return HTTP 200 and create `status=succeeded`, while
`answer_quality.status=failed` records that the answer did not cite any expected
source. This separation keeps provider execution health and grounded answer
quality independently measurable.

## CI Contract

Unit tests cover the assessment rules directly. Integration tests verify that
mock success, no-answer guardrail, remote success, remote provider failure, and
remote success without citations all persist the expected metadata.
