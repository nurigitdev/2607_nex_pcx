# Production Foreground App Worker Supervisor Evidence

Date: 2026-07-20

## Scope

Slice 298 adds a foreground supervisor runner that starts the NeX-PCX web app
and continuously runs bounded worker cycles in the same supervised foreground
session.

The supervisor is intended to close the operational gap where uploads create
queued jobs but pipeline and embedding workers only run after a manual command.

The supervisor standardizes:

- foreground web process launch,
- supervisor PID and web PID evidence,
- recurring bounded calls to `run_foreground_workers.py`,
- conservative Qwen token guards inherited from the worker runner,
- graceful termination of the web process when the supervisor exits, and
- JSON/Markdown evidence output.

## Commands

Dry-run:

```bash
./.venv/bin/python scripts/run_foreground_app_worker_supervisor.py \
  --dry-run \
  --json-output artifacts/foreground_app_worker_supervisor.json \
  --markdown-output artifacts/foreground_app_worker_supervisor.md \
  --pretty
```

Foreground supervised run:

```bash
./.venv/bin/python scripts/run_foreground_app_worker_supervisor.py \
  --pipeline-limit 1 \
  --embedding-limit-per-profile 5 \
  --worker-cycle-interval-seconds 5 \
  --json-output artifacts/foreground_app_worker_supervisor.json \
  --markdown-output artifacts/foreground_app_worker_supervisor.md \
  --pretty
```

Stop the supervised foreground session:

```bash
./.venv/bin/python scripts/stop_foreground_production_app.py \
  --pid-file artifacts/foreground_app_worker_supervisor.pid \
  --expected-command-marker run_foreground_app_worker_supervisor.py \
  --json-output artifacts/foreground_production_shutdown.json \
  --markdown-output artifacts/foreground_production_shutdown.md \
  --pretty
```

Generated evidence paths:

- `artifacts/foreground_app_worker_supervisor.json`
- `artifacts/foreground_app_worker_supervisor.md`
- `artifacts/foreground_app_worker_supervisor.pid`

## Operator Note

Use the supervisor runner as the default foreground production launcher. Keep
the standalone `run_foreground_production_app.py` path only for diagnostics
where queue processing must intentionally remain disabled.

Do not disable the default Qwen token guard while the DGX host is under memory
or swap pressure.
