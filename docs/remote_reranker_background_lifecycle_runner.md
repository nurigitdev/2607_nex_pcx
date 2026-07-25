# Remote Reranker Background Lifecycle Runner

Slice 330 adds an operator-safe background lifecycle runner for the DGX
Qwen3-Reranker-4B provider. It standardizes the manual SSH background launch
used during live smoke testing.

## Default Target

| Item | Value |
| --- | --- |
| DGX host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Workdir | `/home/nexpcx/2607_nex_pcx` |
| Provider port | `9104` |
| Provider name | `qwen-reranker-primary` |
| Backend | `qwen_reranker` |
| Model | `Qwen/Qwen3-Reranker-4B` |
| Profile | `qwen3_reranker_4b` |
| Device | `cuda:0` |
| PID file | `/home/nexpcx/2607_nex_pcx/run/remote_reranker_provider_9104.pid` |
| Log file | `/home/nexpcx/2607_nex_pcx/logs/remote_reranker_provider_9104.log` |

## Commands

Dry-run the generated SSH commands:

```bash
./.venv/bin/python scripts/run_remote_reranker_background.py status --dry-run --json
```

Start the provider in the background. If a matching uvicorn process is already
running, the runner returns `already_running` and refreshes the PID file instead
of launching a duplicate process.

```bash
./.venv/bin/python scripts/run_remote_reranker_background.py start --json
```

Check status and health:

```bash
./.venv/bin/python scripts/run_remote_reranker_background.py status --json
```

Run status, `/healthz`, and `/v1/rerank` request smoke in one evidence report:

```bash
./.venv/bin/python scripts/run_remote_reranker_background.py smoke \
  --request-timeout-seconds 300 \
  --json-output artifacts/remote_reranker_background_status.json \
  --markdown-output artifacts/remote_reranker_background_status.md
```

Stop the provider when the operator intentionally wants to release GPU memory:

```bash
./.venv/bin/python scripts/run_remote_reranker_background.py stop --json
```

## Current Status Evidence

The Slice 330 smoke was run while the remote reranker was intentionally left
running for follow-up search work.

| Item | Result |
| --- | --- |
| Action | `smoke` |
| Passed | `true` |
| Status | `running` |
| PID | `2437559` |
| Base URL | `http://192.168.20.243:9104` |
| Health | `ready=true` |
| Request smoke | `passed=true` |
| Request elapsed | `308 ms` |
| Provider elapsed | `280 ms` |
| Candidate count | `3` |
| Returned count | `2` |

## Request Score Preview

| Rank | Candidate | Score | Source Rank |
| ---: | --- | ---: | ---: |
| `1` | `candidate-1` | `8.756176` | `1` |
| `2` | `candidate-2` | `6.445219` | `2` |

## Runtime Metadata

```json
{
  "service": "nex_pcx_reranker_provider_service",
  "backend": "qwen_reranker",
  "device": "cuda:0",
  "model_source": "/home/nexpcx/2607_nex_pcx/models/qwen3_reranker_4b",
  "elapsed_ms": 280,
  "input_count": 3
}
```

## Operational Notes

- Use `smoke` before Search Compare live testing so PID/log evidence and request
  latency are captured together.
- Use `start` repeatedly without risk of launching a duplicate provider.
- Use `stop` only when intentionally releasing DGX GPU memory; otherwise keep
  the provider running for reranked search experiments.
