# Remote Embedding Request Smoke Runner

Slice 199 adds a reusable request-level smoke runner for remote embedding providers. It
assumes the provider is already running, for example from a foreground smoke session,
systemd service, or another managed launch path.

Use this runner after `/healthz` is healthy to confirm that `/v1/embeddings` returns a
valid provider contract response.

The first passing KURE request smoke result from the DGX Spark host is captured in
`docs/remote_kure_embedding_request_smoke_result.md`.
The first passing BGE request smoke result is captured in
`docs/remote_bge_embedding_request_smoke_result.md`.
The first passing Qwen dual-profile request smoke result is captured in
`docs/remote_qwen_embedding_request_smoke_result.md`.
The first passing sequential suite result is captured in
`docs/remote_embedding_request_smoke_suite_result.md`.

## Dry Run

Generate the default KURE request plan:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider kure \
  --dry-run \
  --json
```

Generate the Qwen dual-profile plan:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider qwen \
  --dry-run \
  --json
```

Qwen defaults to both `qwen3_4b_1000` and `qwen3_4b_2560`. Use `--profile-name` to
target only one profile.

## Execute Against A Running Provider

KURE:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider kure \
  --timeout-seconds 120 \
  --json
```

BGE:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider bge \
  --timeout-seconds 120 \
  --json
```

Qwen:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider qwen \
  --timeout-seconds 300 \
  --json
```

Override the endpoint when testing another host or tunnel:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke.py \
  --provider qwen \
  --base-url http://gpu-provider.internal:9103 \
  --json
```

## Validation

For each profile, the runner validates:

- provider model ID
- provider type
- output dimension
- input count
- embedding row count
- finite embedding values through the shared remote provider client

The JSON report includes only a small preview from the first embedding vector, not the full
embedding payload.

## Run The Full Remote Suite

Slice 203 adds a sequential suite runner that launches each provider, waits for `/healthz`,
runs the request-level smoke check, and stops the provider before moving to the next one.
By default it runs KURE, BGE, then the Qwen dual-profile provider:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke_suite.py \
  --json
```

Preview the complete launch/request plan without opening SSH sessions:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke_suite.py \
  --dry-run \
  --json
```

Write a compact Markdown report with provider outcomes and vector previews:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke_suite.py \
  --json \
  --markdown-output artifacts/remote_embedding_request_smoke_suite.md
```

Run only one provider when narrowing down a failure:

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke_suite.py \
  --provider qwen \
  --qwen-request-timeout-seconds 300 \
  --json
```
