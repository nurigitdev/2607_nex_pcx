# Slice 296: Foreground Production Shutdown Evidence

This Slice adds a foreground production stop runner for controlled pre-CX
operation when systemd service registration is intentionally deferred.

The runner standardizes:

- PID file parsing from `artifacts/foreground_production_launch.pid`
- process existence checks
- command guard checks for `uvicorn` and `app.main:create_app`
- `SIGTERM` shutdown
- post-stop port release evidence
- JSON/Markdown evidence output
- separate append-only shutdown log

Dry-run evidence command used during implementation:

```bash
./.venv/bin/python scripts/stop_foreground_production_app.py \
  --dry-run \
  --json-output artifacts/foreground_production_shutdown.json \
  --markdown-output artifacts/foreground_production_shutdown.md \
  --pretty
```

Actual foreground stop command for production operation:

```bash
./.venv/bin/python scripts/stop_foreground_production_app.py \
  --json-output artifacts/foreground_production_shutdown.json \
  --markdown-output artifacts/foreground_production_shutdown.md \
  --pid-file artifacts/foreground_production_launch.pid \
  --log-file artifacts/foreground_production_shutdown.log \
  --pretty
```

Operator notes:

- Run with `--dry-run` first to verify the PID file and command guard.
- Actual stop sends `SIGTERM` only after the process guard passes.
- If the PID file is stale, the runner records `no_process` and sends no signal.
- If the observed command does not match NeX-PCX foreground markers, the runner
  records `blocked` and sends no signal.
