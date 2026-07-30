# Qwen Provider BF16 Runtime Dtype Memory Evidence

Date: 2026-07-30

## Scope

Slice 417 applies explicit BF16 runtime dtype settings only to the Qwen provider
family whose local checkpoints were previously confirmed as BF16:

- Qwen3-Embedding-4B remote embedding provider on port `9103`
- Qwen3-Reranker-4B remote reranker provider on port `9104`

KURE-v1 and bge-m3 were previously observed as FP32 checkpoints, so they do not
receive a default dtype override.

## Runtime Configuration

Generated DGX env files include the following provider-local settings:

| Service | Env key | Value |
| --- | --- | --- |
| `nex-pcx-embedding-provider-qwen.service` | `NEX_PCX_PROVIDER_TORCH_DTYPE` | `bfloat16` |
| `nex-pcx-reranker-provider.service` | `NEX_PCX_RERANKER_PROVIDER_TORCH_DTYPE` | `bfloat16` |

The env files intentionally contain provider-local runtime metadata only. They
do not include the NeX-PCX application database URL or route secrets.

## Health Evidence

Both providers report the requested dtype and the loaded model parameter dtype.

| Provider | Requested dtype | Loaded parameter dtype | Health |
| --- | --- | --- | --- |
| Qwen embedding | `bfloat16` | `bfloat16` | `ready=true` |
| Qwen reranker | `bfloat16` | `bfloat16` | `ready=true` |

Embedding health metadata excerpt:

```json
{
  "runtime_metadata": {
    "requested_torch_dtype": "bfloat16",
    "requested_torch_dtypes": ["bfloat16"],
    "loaded_parameter_dtypes": ["bfloat16"],
    "adapter_runtime_metadata": {
      "qwen3_4b_1000": {
        "requested_torch_dtype": "bfloat16",
        "loaded_parameter_dtype": "bfloat16",
        "shared_model_cache_key": "/home/nexpcx/2607_nex_pcx/models/qwen3_embedding_4b:cuda:0:bfloat16"
      },
      "qwen3_4b_2560": {
        "requested_torch_dtype": "bfloat16",
        "loaded_parameter_dtype": "bfloat16",
        "shared_model_cache_key": "/home/nexpcx/2607_nex_pcx/models/qwen3_embedding_4b:cuda:0:bfloat16"
      }
    }
  }
}
```

Reranker health metadata excerpt:

```json
{
  "runtime_metadata": {
    "requested_torch_dtype": "bfloat16",
    "loaded_parameter_dtype": "bfloat16",
    "model_dir_exists": true
  }
}
```

## Request Smoke Evidence

Qwen embedding request smoke passed for both profiles.

| Profile | Dimension | Request elapsed | Provider elapsed | Preview |
| --- | ---: | ---: | ---: | --- |
| `qwen3_4b_1000` | `1000` | `2892 ms` | `2874 ms` | `[-0.000216, -0.004639, -0.034668]` |
| `qwen3_4b_2560` | `2560` | `59 ms` | `49 ms` | `[-0.000134, -0.002853, -0.021362]` |

Qwen reranker request smoke passed.

| Metric | Value |
| --- | ---: |
| Candidate count | `3` |
| Returned count | `2` |
| Request elapsed | `2306 ms` |
| Provider elapsed | `2293 ms` |

Score preview:

| Rank | Candidate | Score | Source rank |
| ---: | --- | ---: | ---: |
| `1` | `candidate-1` | `8.6875` | `1` |
| `2` | `candidate-2` | `6.5` | `2` |

## Memory Evidence

Memory was captured with:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

DGX-Spark uses unified memory, so `used_memory` should be interpreted together
with process RSS and system memory pressure. It is still useful as a comparable
before/after runtime memory signal for the same provider process.

| Provider | Before PID | Before MiB | After PID | After MiB | Delta MiB | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen embedding | `1959` | `15622` | `21819` | `7968` | `-7654` | `49.0%` |
| Qwen reranker | `7469` | `15891` | `21821` | `8057` | `-7834` | `49.3%` |
| Combined Qwen providers | - | `31513` | - | `16025` | `-15488` | `49.1%` |

Reference vLLM process during the same after capture:

| PID | Process | Used MiB |
| ---: | --- | ---: |
| `13769` | `VLLM::EngineCore` | `83762` |

## Acceptance

- Qwen embedding and reranker provider env files contain explicit BF16 dtype settings.
- `/healthz` reports both requested dtype and loaded parameter dtype as `bfloat16`.
- Embedding and reranker request smoke checks pass after restart.
- Memory evidence shows the Qwen provider family dropped from `31513 MiB` to
  `16025 MiB`, about a `49.1%` reduction for the measured provider processes.
