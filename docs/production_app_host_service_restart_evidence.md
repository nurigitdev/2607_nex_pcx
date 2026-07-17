# Production App Host Service Restart Evidence

Date: 2026-07-17

## Scope

Slice 290 added an app-host managed service restart validation runner. The goal
is to distinguish a manually launched foreground Uvicorn process from a
systemd-managed web and worker service set that can survive failure, logout, or
reboot according to the host policy.

## Validation Runner

Read-only validation:

```bash
./.venv/bin/python scripts/validate_app_host_service_restart.py \
  --scope user \
  --app-url http://127.0.0.1:8000 \
  --json-output artifacts/app_host_service_restart_validation.json \
  --markdown-output artifacts/app_host_service_restart_validation.md \
  --pretty
```

Controlled restart validation:

```bash
./.venv/bin/python scripts/validate_app_host_service_restart.py \
  --scope user \
  --app-url http://127.0.0.1:8000 \
  --restart-web \
  --json-output artifacts/app_host_service_restart_validation.json \
  --markdown-output artifacts/app_host_service_restart_validation.md \
  --pretty
```

Use `--scope system` when `nex-pcx-web.service`,
`nex-pcx-pipeline-worker.service`, and `nex-pcx-embedding-worker.service` are
installed as system-level units.

## Current Host Result

The current development app host does not expose a usable user systemd bus in
this session:

```text
Failed to connect to bus: No data available
```

The validation runner therefore records the current state as `blocked` instead
of treating the foreground port `8000` launch as operationally managed.

Generated evidence paths:

- `artifacts/app_host_service_restart_validation.json`
- `artifacts/app_host_service_restart_validation.md`

## Operator Interpretation

- `ready`: systemd is reachable, all configured units are loaded and active, the
  units have restart policies, and app health/identity checks pass when
  `--app-url` is supplied.
- `warning`: units are reachable and active but one or more restart policies are
  missing.
- `blocked`: systemd is unreachable, units are missing or inactive, restart
  command failed, or the app health/identity check failed.

## Required Follow-up Before Go-Live

Install reviewed app-host units through one of these paths:

- User units under `systemctl --user`, with lingering enabled or an equivalent
  session manager guarantee.
- System units under `/etc/systemd/system`, managed by the host administrator.

After installation, re-run the validation runner with `--app-url
http://127.0.0.1:8000`. Use `--restart-web` during a maintenance window to
prove restart survival end to end.
