# DGX vLLM Generation Run Live E2E Verification

Slice 355 adds a live E2E runner for the full retrieval-to-generation path.
Unlike the direct DGX vLLM smoke runner, this check creates a minimal DB-backed
search fixture, builds a retrieval context package, executes the remote
generation executor, and verifies persisted `generation_runs`,
`generation_run_citations`, and provider metrics.

## Command

```bash
NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY='...' \
NEX_PCX_DATABASE_URL='postgresql://nex_pcx_dev:...@127.0.0.1:5432/nex_pcx_dev' \
./.venv/bin/python scripts/run_dgx_vllm_generation_run_e2e.py \
  --markdown-output docs/dgx_vllm_generation_run_live_e2e_result.md
```

The script defaults to the DGX-Spark vLLM runtime:

| Field | Default |
| --- | --- |
| DGX host | `192.168.20.243` |
| vLLM port | `12000` |
| Provider base URL | `http://192.168.20.243:12000` |
| Chat endpoint | `/v1/chat/completions` |
| Model | `/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4` |
| API key source | `NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY` |
| Timeout | `300` seconds |
| Max tokens | `512` |

## Verification Scope

The runner verifies that:

- a temporary file, document, chunk, search log, and search result can be
  created as a generation-ready fixture;
- retrieval context packaging emits at least one included candidate and citation
  key;
- the default provider can be temporarily set to
  `remote_openai_compatible`;
- `execute_remote_generation_run(...)` calls the configured vLLM provider and
  persists a successful generation run;
- at least one citation row is created and at least one citation key is used in
  the answer;
- provider metrics include HTTP status, provider latency, token usage, finish
  reason, and `succeeded=true`;
- temporary fixture rows are cleaned up and the previous default generation
  provider is restored after success or failure.

## Secret Handling

The API key value is never written to DB, JSON, Markdown, or console output. The
evidence only records the environment variable name and whether it was
configured.

## CI Contract

Automated tests use an injected fake OpenAI-compatible provider, so CI validates
the DB fixture, retrieval package, run persistence, citation trace, redaction,
and cleanup behavior without depending on the live DGX-Spark server.
