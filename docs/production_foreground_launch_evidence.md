# Slice 295: Foreground Production Launch Evidence

This Slice adds a foreground production launch runner for controlled pre-CX
operation when systemd service registration is intentionally deferred.

The runner standardizes:

- pre-launch checks for workdir, Python executable, database URL presence, port
  availability, and distinct PID/log paths
- Uvicorn foreground launch command
- PID file path
- append-only log file path
- JSON/Markdown evidence output
- no database URL value exposure in evidence files

Dry-run evidence command used during implementation:

```bash
./.venv/bin/python scripts/run_foreground_production_app.py \
  --dry-run \
  --allow-missing-database-url \
  --json-output artifacts/foreground_production_launch.json \
  --markdown-output artifacts/foreground_production_launch.md \
  --pretty
```

Actual foreground launch command for production operation:

```bash
./.venv/bin/python scripts/run_foreground_production_app.py \
  --json-output artifacts/foreground_production_launch.json \
  --markdown-output artifacts/foreground_production_launch.md \
  --pid-file artifacts/foreground_production_launch.pid \
  --log-file artifacts/foreground_production_launch.log \
  --pretty
```

Operator notes:

- Run the actual launch command in a supervised terminal.
- Keep `NEX_PCX_DATABASE_URL` configured in the shell before actual launch.
- Keep the terminal alive while foreground mode is active.
- Use the PID file for stop verification and the log file for incident review.
