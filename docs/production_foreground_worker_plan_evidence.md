# Production Foreground Worker Plan Evidence

Date: 2026-07-17

## Scope

Slice 293 adds a foreground worker command plan for supervised pre-CX
operation. The plan renders bounded commands for pipeline and embedding workers
without claiming queue work during plan generation.

## Command

```bash
./.venv/bin/python scripts/render_foreground_worker_plan.py \
  --json-output artifacts/foreground_worker_plan.json \
  --markdown-output artifacts/foreground_worker_plan.md \
  --pretty
```

Generated evidence paths:

- `artifacts/foreground_worker_plan.json`
- `artifacts/foreground_worker_plan.md`

## Plan Contents

The generated plan includes:

- read-only `--help` checks for pipeline and embedding worker scripts,
- a pipeline worker command that claims at most one pipeline job,
- an embedding worker command that uses provider routes and a bounded
  `--limit`, and
- lease and chunk-policy parameters that an operator can review before use.

## Operator Note

Run the help checks first. Then run the bounded worker commands only from a
supervised terminal, after foreground web validation and provider readiness have
passed. Stop and inspect logs if any command returns a failed job or provider
route error.
