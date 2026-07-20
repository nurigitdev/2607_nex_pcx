# Production Upload Queue Auto-Processing Smoke

Date: 2026-07-20

## Scope

Slice 300 adds a smoke runner that verifies the operational path from upload to
queued pipeline work and foreground worker runtime visibility.

The smoke runner can:

- read foreground worker runtime status,
- upload a small Markdown document through `/api/files`,
- capture the returned `pipeline_job_id`,
- poll `/api/pipeline/jobs/{job_id}`, and
- write JSON/Markdown evidence.

## Command

Dry-run:

```bash
./.venv/bin/python scripts/run_upload_queue_auto_processing_smoke.py \
  --dry-run \
  --json-output artifacts/upload_queue_auto_processing_smoke.json \
  --markdown-output artifacts/upload_queue_auto_processing_smoke.md \
  --pretty
```

Live smoke after the supervised foreground app is running:

```bash
./.venv/bin/python scripts/run_upload_queue_auto_processing_smoke.py \
  --app-url http://127.0.0.1:8000 \
  --poll-attempts 12 \
  --poll-interval-seconds 5 \
  --json-output artifacts/upload_queue_auto_processing_smoke.json \
  --markdown-output artifacts/upload_queue_auto_processing_smoke.md \
  --pretty
```

## Operator Note

During DGX memory or swap pressure, use `--dry-run` only. A live smoke can
trigger pipeline and embedding worker cycles when the foreground supervisor is
running.
