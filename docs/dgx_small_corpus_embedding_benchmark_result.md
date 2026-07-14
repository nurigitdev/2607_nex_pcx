# DGX Small Corpus 4-Profile Ingestion Benchmark Result

- `passed`: `true`
- `database_url`: `postgresql://nex_pcx_dev:***@127.0.0.1:5432/nex_pcx_dev`
- `chunk_count`: `3`
- `expected_job_count`: `12`
- `preflight_before_worker`: `true`
- `cleanup_attempted`: `true`
- `cleanup_confirmed`: `true`
- `total_elapsed_seconds`: `87.80`
- providers executed: `3`

## Fixture

- `benchmark_run_key`: `dgx-small-corpus-1784070146912`
- `file_id`: `50`
- `document_id`: `50`
- `chunk_ids`: `42, 43, 44`
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
| `kure` | `kure_v1_1024` | `true` | `3/3` | `3` | `chunk_embeddings_kure_v1_1024` | `1024` | `2` | `28.33` | `63.00` | `78` | `0` |
| `bge` | `bge_m3_1024` | `true` | `3/3` | `3` | `chunk_embeddings_bge_m3_1024` | `1024` | `3` | `22.67` | `55.67` | `60` | `0` |
| `qwen` | `qwen3_4b_1000` | `true` | `3/3` | `3` | `chunk_embeddings_qwen3_4b_1000` | `1000` | `4` | `176.33` | `213.67` | `226` | `0` |
| `qwen` | `qwen3_4b_2560` | `true` | `3/3` | `3` | `chunk_embeddings_qwen3_4b_2560` | `2560` | `5` | `165.33` | `204.33` | `213` | `0` |

## Job-Level Evidence

| Provider | Profile | Chunk | Job | Status | Dimension | Provider Model | Provider ms | Worker ms | Error |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| `kure` | `kure_v1_1024` | `42` | `24` | `succeeded` | `1024` | `local-kure-v1` | `46` | `78` | `` |
| `kure` | `kure_v1_1024` | `43` | `25` | `succeeded` | `1024` | `local-kure-v1` | `19` | `55` | `` |
| `kure` | `kure_v1_1024` | `44` | `26` | `succeeded` | `1024` | `local-kure-v1` | `20` | `56` | `` |
| `bge` | `bge_m3_1024` | `42` | `27` | `succeeded` | `1024` | `local-bge-m3` | `30` | `60` | `` |
| `bge` | `bge_m3_1024` | `43` | `28` | `succeeded` | `1024` | `local-bge-m3` | `18` | `48` | `` |
| `bge` | `bge_m3_1024` | `44` | `29` | `succeeded` | `1024` | `local-bge-m3` | `20` | `59` | `` |
| `qwen` | `qwen3_4b_1000` | `42` | `30` | `succeeded` | `1000` | `local-qwen3-embedding-4b` | `190` | `226` | `` |
| `qwen` | `qwen3_4b_1000` | `43` | `31` | `succeeded` | `1000` | `local-qwen3-embedding-4b` | `171` | `203` | `` |
| `qwen` | `qwen3_4b_1000` | `44` | `32` | `succeeded` | `1000` | `local-qwen3-embedding-4b` | `168` | `212` | `` |
| `qwen` | `qwen3_4b_2560` | `42` | `33` | `succeeded` | `2560` | `local-qwen3-embedding-4b` | `169` | `206` | `` |
| `qwen` | `qwen3_4b_2560` | `43` | `34` | `succeeded` | `2560` | `local-qwen3-embedding-4b` | `169` | `213` | `` |
| `qwen` | `qwen3_4b_2560` | `44` | `35` | `succeeded` | `2560` | `local-qwen3-embedding-4b` | `158` | `194` | `` |
