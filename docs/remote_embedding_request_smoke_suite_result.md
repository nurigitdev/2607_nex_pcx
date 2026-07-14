# Remote Embedding Request Smoke Suite Result

Date: 2026-07-14

Slice 203 validated the first sequential suite run across the KURE, BGE-M3, and shared
Qwen remote embedding providers on the DGX Spark host.

## Environment

| Item | Value |
| --- | --- |
| GPU host | `192.168.20.243` |
| SSH user | `nexpcx` |
| Remote workdir | `/home/nexpcx/2607_nex_pcx` |
| Device | `cuda:0` |
| Providers | `kure`, `bge`, `qwen` |
| Ports | `9101`, `9102`, `9103` |

## Command

```bash
./.venv/bin/python scripts/run_remote_provider_embedding_smoke_suite.py \
  --json \
  --markdown-output /tmp/nex_pcx_slice203_remote_embedding_request_smoke_suite.md
```

## Passing Suite Result

- `passed`: `true`
- `total_elapsed_seconds`: `82.87`
- providers executed: `3`
- post-run health checks: all provider ports returned connection refused
- remote `ss`: no listeners remained on `9101`, `9102`, or `9103`

Provider summary:

| Provider | Passed | Base URL | Health attempts | Request cases | Cleanup |
| --- | --- | --- | ---: | ---: | --- |
| `kure` | `true` | `http://192.168.20.243:9101` | `7` | `1` | stopped |
| `bge` | `true` | `http://192.168.20.243:9102` | `6` | `1` | stopped |
| `qwen` | `true` | `http://192.168.20.243:9103` | `11` | `2` | stopped |

Request timing:

| Provider | Profile | Request elapsed | Provider elapsed | Dimension | Embeddings |
| --- | --- | ---: | ---: | ---: | ---: |
| `kure` | `kure_v1_1024` | `826 ms` | `780 ms` | `1024` | `1` |
| `bge` | `bge_m3_1024` | `755 ms` | `730 ms` | `1024` | `1` |
| `qwen` | `qwen3_4b_1000` | `2559 ms` | `2504 ms` | `1000` | `1` |
| `qwen` | `qwen3_4b_2560` | `249 ms` | `194 ms` | `2560` | `1` |

Embedding vector preview:

| Provider | Profile | First 3 values from first vector |
| --- | --- | --- |
| `kure` | `kure_v1_1024` | `[-0.04624427482485771, 0.043011296540498734, -0.005476752761751413]` |
| `bge` | `bge_m3_1024` | `[-0.043258894234895706, 0.036100007593631744, -0.005832674913108349]` |
| `qwen` | `qwen3_4b_1000` | `[-0.00022586015984416008, -0.004938103724271059, -0.03558120131492615]` |
| `qwen` | `qwen3_4b_2560` | `[-0.00013867051166016608, -0.0030318291392177343, -0.021845655515789986]` |

The suite runner still emits full JSON to stdout for automation. Text inputs are masked in
the plan as `<text:n>`, and only the three-value vector previews are retained in Markdown.
