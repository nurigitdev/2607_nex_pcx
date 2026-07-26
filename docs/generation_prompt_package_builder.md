# Generation Prompt Package Builder

Slice 338 converts a retrieval context package into an OpenAI-compatible chat
message package.

## Input

The builder receives a `RetrievalContextPackage` created from a saved search log.
It uses:

- query text
- retrieval package key
- search log id
- requested/effective search scope
- chunk policy
- included context text
- citation keys
- retrieval confidence status

## Output

The builder returns:

- `prompt_version`
- `response_language`
- OpenAI-compatible `messages`
- `citation_keys`
- `context_text`
- `prompt_hash`
- `context_hash`
- `blocked`
- `block_reason`

## Guardrail Behavior

The prompt builder does not call an LLM.

If retrieval confidence is not `answerable`, or if no included context is
available, the package is marked as blocked. The user message then describes why
generation is blocked and instructs a later generation executor to return a
no-answer response instead of sending weak context to a model.

## Default Prompt Rules

- Answer only from the provided retrieval context.
- Use citation keys such as `[RCP-001]`.
- If the context is insufficient, say that the answer cannot be determined from
  the provided documents.
- Do not invent policy, date, amount, role, or source details.
- Use Korean by default.
