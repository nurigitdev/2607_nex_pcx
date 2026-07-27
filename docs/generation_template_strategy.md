# Generation Template Strategy

## Purpose

Generation templates make the output structure explicit without changing the retrieval context, embedding profile, reranker, or LLM provider. They let NeX_PCX compare whether the same grounded context is better consumed as a short answer, report, proposal, summary, or meeting note.

This strategy supports FR-077 in the SRS and keeps template choice reproducible in generation run metadata.

## MVP Template Set

| Template key | Document type | Output format | Primary use |
| --- | --- | --- | --- |
| `grounded_answer` | grounded_answer | Markdown | Default cited answer for direct generation and search-log generation |
| `report` | report | Markdown | Internal report draft with purpose, findings, evidence, risks, and next actions |
| `proposal` | proposal | Markdown | Proposal draft with background, recommendation, scope, execution plan, and expected impact |
| `summary` | summary | Markdown | Concise executive or working-level summary of retrieved evidence |
| `meeting_minutes` | meeting_minutes | Markdown | Meeting note draft with agenda, decisions, action items, and evidence references |

## Template Contract

`generation_templates` stores reusable template metadata:

- `template_key`, `template_name`, `template_version`, `document_type`, `language`, and `output_format`
- `section_schema` for expected Markdown sections and optional required/optional flags
- `system_instruction` and `user_instruction_suffix` for prompt injection
- `style_guidance` for tone, density, and audience hints
- `citation_policy` for citation count, placement, and no-answer behavior
- `is_active`, `is_default`, `created_at`, and `updated_at`

The prompt package must include a template snapshot in request metadata. The snapshot must be sufficient to explain which template shaped the answer even if the template is later edited.

`prompt_hash` must be computed from the rendered messages plus template snapshot. The same retrieval context with a different template must produce a different prompt hash.

Generation runs must persist the selected template key/version/document type/output format in request metadata. Generated Markdown is the canonical first artifact for later export paths.

## Prompt Guardrails

- The LLM may reorganize the answer into the selected section schema, but factual claims must remain grounded in retrieval context citations.
- If retrieval context is blocked, low confidence, or citation readiness has failed, the template must not encourage invented content.
- Empty or unsupported sections should be marked as unavailable rather than filled with plausible text.
- Citation rules remain stronger than style guidance.

## Slice Sequence

- Slice 363: add `generation_templates` schema, migration, and seed defaults.
- Slice 364: add repository access and template-aware prompt package injection.
- Slice 365: add generation template selector UI.
- Slice 366: add report-style generation smoke and quality metadata.
- Slice 367: add Markdown export for generated answers.
