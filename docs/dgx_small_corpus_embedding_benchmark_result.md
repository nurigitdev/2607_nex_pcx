# DGX Small Corpus 4-Profile Ingestion Benchmark Result

- `passed`: `true`
- `database_url`: `postgresql://nex_pcx_dev:***@127.0.0.1:5432/nex_pcx_dev`
- `chunk_count`: `3`
- `expected_job_count`: `12`
- `preflight_before_worker`: `true`
- `cleanup_attempted`: `true`
- `cleanup_confirmed`: `true`
- `total_elapsed_seconds`: `87.39`
- providers executed: `3`

## Fixture

- `benchmark_run_key`: `dgx-small-corpus-1784068050768`
- `file_id`: `49`
- `document_id`: `49`
- `chunk_ids`: `39, 40, 41`
- `job_count`: `12`

## Provider Results

| Provider | Passed | Base URL | Health | Profiles | Jobs | Error |
| --- | --- | --- | --- | ---: | ---: | --- |
| `kure` | `true` | `http://192.168.20.243:9101` | `true` | `1/1` | `3` | `` |
| `bge` | `true` | `http://192.168.20.243:9102` | `true` | `1/1` | `3` | `` |
| `qwen` | `true` | `http://192.168.20.243:9103` | `true` | `2/2` | `6` | `` |

## Profile Benchmark Summary

| Provider | Profile | Passed | Jobs | Vectors | Vector Table | Dimension | Route | Avg Provider ms | Avg Worker ms | Max Worker ms | Error Count |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `kure` | `kure_v1_1024` | `true` | `3/3` | `3` | `chunk_embeddings_kure_v1_1024` | `1024` | `2` | `27.33` | `63.67` | `78` | `0` |
| `bge` | `bge_m3_1024` | `true` | `3/3` | `3` | `chunk_embeddings_bge_m3_1024` | `1024` | `3` | `22.67` | `56.67` | `68` | `0` |
| `qwen` | `qwen3_4b_1000` | `true` | `3/3` | `3` | `chunk_embeddings_qwen3_4b_1000` | `1000` | `4` | `173.67` | `206.67` | `236` | `0` |
| `qwen` | `qwen3_4b_2560` | `true` | `3/3` | `3` | `chunk_embeddings_qwen3_4b_2560` | `2560` | `5` | `158.67` | `195.67` | `197` | `0` |

## Job-Level Evidence

| Provider | Profile | Chunk | Job | Status | Dimension | Provider Model | Provider ms | Worker ms | Error |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| `kure` | `kure_v1_1024` | `39` | `12` | `succeeded` | `1024` | `local-kure-v1` | `44` | `78` | `` |
| `kure` | `kure_v1_1024` | `40` | `13` | `succeeded` | `1024` | `local-kure-v1` | `18` | `59` | `` |
| `kure` | `kure_v1_1024` | `41` | `14` | `succeeded` | `1024` | `local-kure-v1` | `20` | `54` | `` |
| `bge` | `bge_m3_1024` | `39` | `15` | `succeeded` | `1024` | `local-bge-m3` | `30` | `68` | `` |
| `bge` | `bge_m3_1024` | `40` | `16` | `succeeded` | `1024` | `local-bge-m3` | `18` | `50` | `` |
| `bge` | `bge_m3_1024` | `41` | `17` | `succeeded` | `1024` | `local-bge-m3` | `20` | `52` | `` |
| `qwen` | `qwen3_4b_1000` | `39` | `18` | `succeeded` | `1000` | `local-qwen3-embedding-4b` | `201` | `236` | `` |
| `qwen` | `qwen3_4b_1000` | `40` | `19` | `succeeded` | `1000` | `local-qwen3-embedding-4b` | `162` | `193` | `` |
| `qwen` | `qwen3_4b_1000` | `41` | `20` | `succeeded` | `1000` | `local-qwen3-embedding-4b` | `158` | `191` | `` |
| `qwen` | `qwen3_4b_2560` | `39` | `21` | `succeeded` | `2560` | `local-qwen3-embedding-4b` | `158` | `197` | `` |
| `qwen` | `qwen3_4b_2560` | `40` | `22` | `succeeded` | `2560` | `local-qwen3-embedding-4b` | `160` | `195` | `` |
| `qwen` | `qwen3_4b_2560` | `41` | `23` | `succeeded` | `2560` | `local-qwen3-embedding-4b` | `158` | `195` | `` |
