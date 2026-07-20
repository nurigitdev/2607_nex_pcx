# Production Foreground Worker Runner Evidence

Date: 2026-07-20

## Scope

Slice 297 adds a bounded foreground worker runner with a provider resource
guard. The runner is intended for supervised pre-CX operation when web service
foreground launch is used instead of a registered service manager.

The runner standardizes:

- bounded pipeline worker execution,
- active provider route health checks with short guard timeouts,
- mock provider route blocking for foreground operation,
- profile-level token guards for resource-heavy providers,
- profile-by-profile embedding worker execution, and
- JSON/Markdown evidence output.

## Command

Dry-run:

```bash
./.venv/bin/python scripts/run_foreground_workers.py \
  --dry-run \
  --json-output artifacts/foreground_worker_runner.json \
  --markdown-output artifacts/foreground_worker_runner.md \
  --pretty
```

Bounded foreground run:

```bash
./.venv/bin/python scripts/run_foreground_workers.py \
  --pipeline-limit 1 \
  --embedding-limit-per-profile 5 \
  --guard-health-timeout-seconds 5 \
  --max-provider-health-elapsed-ms 5000 \
  --json-output artifacts/foreground_worker_runner.json \
  --markdown-output artifacts/foreground_worker_runner.md \
  --pretty
```

Generated evidence paths:

- `artifacts/foreground_worker_runner.json`
- `artifacts/foreground_worker_runner.md`

## Resource Guard Defaults

The default guard applies conservative token limits to Qwen profiles:

- `qwen3_4b_1000=1200`
- `qwen3_4b_2560=1200`

This keeps large Qwen chunks queued when DGX memory or swap pressure is a
concern. After the DGX host is stable, an operator can explicitly remove the
default guard for a supervised retry:

```bash
./.venv/bin/python scripts/run_foreground_workers.py \
  --no-default-qwen-token-guard \
  --pipeline-limit 0 \
  --embedding-limit-per-profile 1 \
  --pretty
```

## Operator Note

Run the dry-run first. If profiles are skipped, keep them queued and inspect DGX
memory/swap before removing the guard. The runner passes the database URL to
child workers through the environment and does not include raw database
credentials in command evidence.
