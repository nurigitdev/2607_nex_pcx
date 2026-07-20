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

Slice 301 adds `NEX_PCX_FOREGROUND_*` runtime defaults so foreground operation
does not depend on remembering every worker flag at launch time.

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
  --json-output artifacts/foreground_app_worker_supervisor.json \
  --markdown-output artifacts/foreground_app_worker_supervisor.md \
  --pretty
```

Recommended foreground defaults:

| Environment Variable | Default | Meaning |
| --- | ---: | --- |
| `NEX_PCX_FOREGROUND_PIPELINE_LIMIT` | `1` | Pipeline jobs claimed per cycle |
| `NEX_PCX_FOREGROUND_EMBEDDING_LIMIT_PER_PROFILE` | `5` | Embedding jobs claimed per profile |
| `NEX_PCX_FOREGROUND_WORKER_CYCLE_INTERVAL_SECONDS` | `5` | Delay between worker cycles |
| `NEX_PCX_FOREGROUND_CHECK_PORT_AVAILABLE` | `true` | Fail planning if the web port is already reachable |
| `NEX_PCX_FOREGROUND_NO_DEFAULT_QWEN_TOKEN_GUARD` | `false` | Keep Qwen token guard enabled by default |

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
