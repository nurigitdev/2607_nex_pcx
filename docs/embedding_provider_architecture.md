# Embedding Provider Architecture

Slice 085 separates NeX_PCX orchestration from heavy embedding inference.

NeX_PCX remains responsible for upload, parsing, chunking, PostgreSQL queue state,
permission filtering, pgvector persistence, search logs, and evaluation records. Embedding
calculation can be delegated to either a local smoke adapter or a remote GPU provider.

## Provider Modes

| Mode | Intended use | Notes |
| --- | --- | --- |
| `mock` | deterministic tests and regression fixtures | No model loading. |
| `local` | developer smoke/debug on downloaded `models/` bundle | CPU-only machines should use this sparingly. |
| `remote` | benchmark ingestion and production-like tests | GPU server preloads models and exposes an embedding API. |

## Remote Provider Contract

Request fields:

- `profile_name`
- `model_key`
- `input_type`: `query` or `document`
- `texts`
- `normalize_embeddings`
- `output_dimension`
- `trace_id`

Response fields:

- `embeddings`
- `dimension`
- `provider_model_id`
- `provider_type`
- `elapsed_ms`
- `runtime_metadata`

## Responsibilities

The embedding provider owns model preload, device placement, batching, and raw inference.
The NeX_PCX embedding worker owns queue lease, retry behavior, dimension validation,
pgvector storage, and experiment metadata.

Qwen3 1000 and Qwen3 2560 profiles should share the same remote model service while
preserving separate output-dimension and storage-type metadata.

See `docs/gpu_embedding_provider_deployment.md` for GPU server placement, offline model
bundle handling, provider startup checks, and operational failure modes.
