# Remote Provider Foreground Launch Smoke And Health Smoke

Slice 195 defines the first safe launch path after the DGX Spark readiness checker reports
`ready=true`.
Slice 196 adds an executable foreground smoke that starts the remote provider over SSH,
polls `/healthz` from the app host, and stops the foreground process after the check.
The first passing KURE result is captured in
`docs/remote_kure_foreground_health_smoke_result.md`.
The first passing BGE result is captured in
`docs/remote_bge_foreground_health_smoke_result.md`.
The first passing Qwen dual-profile result is captured in
`docs/remote_qwen_foreground_health_smoke_result.md`.

Use a foreground launch before systemd so the operator can see startup logs, model load
errors, and shutdown behavior directly. Start with `kure` or `bge` before `qwen`; Qwen is
larger and serves two profiles from one process.

## Preconditions

From the app host:

```bash
./.venv/bin/python scripts/check_remote_gpu_provider_host.py --provider kure --json
```

Expected result:

- `ready` is `true`
- all required checks pass
- provider ports may still be empty because no provider is running yet

## Generate A Foreground Plan

Default safe plan, using KURE on port `9101`:

```bash
./.venv/bin/python scripts/plan_remote_provider_foreground_smoke.py
./.venv/bin/python scripts/plan_remote_provider_foreground_smoke.py --json
```

Generate plans for other providers:

```bash
./.venv/bin/python scripts/plan_remote_provider_foreground_smoke.py --provider bge
./.venv/bin/python scripts/plan_remote_provider_foreground_smoke.py --provider qwen
```

The helper prints:

- readiness command to run from the app host
- foreground SSH launch command
- remote port listener check
- app-host `/healthz` command
- Ctrl-C shutdown instruction

## Run The Automated KURE Smoke

After reviewing the generated plan, run the automated smoke from the app host:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider kure \
  --json
```

The runner uses the same launch plan, but removes the interactive SSH pseudo-terminal flag
so it can manage the foreground process as a child process. It reports:

- the exact non-interactive SSH launch command
- health attempts, status code, and parsed health payload
- mismatches against the expected model key, profile names, provider model ID, and device
- process shutdown result and whether `/healthz` became unreachable after stop
- stdout/stderr tails for startup failure triage

For slower first model loads, increase the startup timeout:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider kure \
  --startup-timeout-seconds 300
```

For Qwen, use a longer startup timeout because the provider loads a larger model and serves
two output profiles from one process:

```bash
./.venv/bin/python scripts/run_remote_provider_foreground_smoke.py \
  --provider qwen \
  --startup-timeout-seconds 900 \
  --poll-interval-seconds 5
```

## Foreground Launch Sequence

1. Run the readiness command from the plan.
2. Open a terminal dedicated to the provider process.
3. Run the `ssh -t ...` foreground launch command from the plan.
4. Leave that SSH session open while the provider loads.
5. From the app host, run the port check and health check commands from the plan.
6. Stop the provider with Ctrl-C in the foreground SSH session.
7. Confirm the port listener disappears.

For KURE, the launch command has this shape:

```bash
ssh -t nexpcx@192.168.20.243 \
  'cd /home/nexpcx/2607_nex_pcx && NEX_PCX_PROVIDER_BACKEND=sentence_transformers ... \
  /home/nexpcx/2607_nex_pcx/.venv/bin/python -m uvicorn \
  app.embedding_provider_service:app --host 0.0.0.0 --port 9101'
```

Expected health URL:

```bash
curl -fsS http://192.168.20.243:9101/healthz
```

## Success Criteria

The smoke passes when:

- the foreground process starts without import errors
- the configured port is listening
- `/healthz` returns HTTP 200
- health response reports the expected `model_key`, `profile_names`, `provider_model_id`,
  `device`, and readiness state
- Ctrl-C stops the process cleanly
- the automated runner confirms `/healthz` is no longer reachable after shutdown

## Failure Triage

| Symptom | First check |
| --- | --- |
| Import error at startup | Re-run `check_remote_gpu_provider_host.py --provider <name> --json`. |
| Port already in use | Check `ss -ltnH` output and stop the old foreground/systemd process. |
| `/healthz` unreachable | Confirm firewall/routing from app host to `192.168.20.243:<port>`. |
| Model load error | Confirm `models/<model_key>` directory and model runtime packages. |
| Slow Qwen startup | Validate KURE/BGE first, then start Qwen with a longer observation window. |
| Automated smoke passes health but fails stop confirmation | Check for an older provider process already bound to the same port. |

After a successful foreground smoke, move to provider health smoke tests and then systemd
service generation.
