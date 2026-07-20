# Production Foreground Worker Runtime Visibility

Date: 2026-07-20

## Scope

Slice 299 adds a read-only runtime visibility report for foreground worker
operation.

The report reads:

- `artifacts/foreground_app_worker_supervisor.json`
- `artifacts/foreground_worker_runner.json`
- `artifacts/foreground_app_worker_supervisor.pid`
- `artifacts/foreground_production_launch.pid`

It summarizes whether the supervisor evidence is running, whether the
supervisor and web PID files point to live processes, and whether the latest
worker cycle or worker runner evidence recorded failures.

## API

```bash
curl -fsS http://127.0.0.1:8000/api/admin/foreground-worker-runtime
```

Status meanings:

- `ready`: supervisor evidence is running and both supervisor/web processes are alive.
- `warning`: evidence is planned, missing, or not actively running.
- `blocked`: running evidence has stale processes or worker failures are recorded.

## Operator Note

Use this endpoint after upload testing. A `warning` state is acceptable while
the app is intentionally stopped or only dry-run evidence exists. A `blocked`
state should stop go-live until stale processes or failed worker cycles are
resolved.
