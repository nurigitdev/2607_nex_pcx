# DGX Remote Embedding Worker E2E Smoke Result

Date: 2026-07-14

Slice 206 added and ran a managed DGX remote embedding worker E2E smoke. The runner
creates a temporary chunk, creates one embedding job for each active experiment profile,
launches the matching remote provider over SSH, refreshes route preflight snapshots,
processes the job through the route-aware worker with the readiness gate enabled, verifies
vector persistence, and cleans up the fixture.

- `passed`: `true`
- `database_url`: `postgresql://nex_pcx_dev:***@127.0.0.1:5432/nex_pcx_dev`
- `preflight_before_worker`: `true`
- `cleanup_attempted`: `true`
- `cleanup_confirmed`: `true`
- `total_elapsed_seconds`: `84.66`
- providers executed: `3`

## Fixture

- `smoke_run_key`: `dgx-worker-e2e-1784025442070`
- `file_id`: `48`
- `document_id`: `48`
- `chunk_id`: `38`

## Provider Results

| Provider | Passed | Base URL | Health | Profiles | Error |
| --- | --- | --- | --- | ---: | --- |
| `kure` | `true` | `http://192.168.20.243:9101` | `true` | `1/1` | `` |
| `bge` | `true` | `http://192.168.20.243:9102` | `true` | `1/1` | `` |
| `qwen` | `true` | `http://192.168.20.243:9103` | `true` | `2/2` | `` |

## Worker Profile Results

| Provider | Profile | Passed | Job | Status | Vector Table | Dimension | Route | Health Snapshot | Contract Snapshot | Provider Model | Provider ms | Worker ms | Error |
| --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| `kure` | `kure_v1_1024` | `true` | `8` | `succeeded` | `chunk_embeddings_kure_v1_1024` | `1024` | `2` | `5` | `5` | `local-kure-v1` | `48` | `144` | `` |
| `bge` | `bge_m3_1024` | `true` | `9` | `succeeded` | `chunk_embeddings_bge_m3_1024` | `1024` | `3` | `6` | `6` | `local-bge-m3` | `30` | `74` | `` |
| `qwen` | `qwen3_4b_1000` | `true` | `10` | `succeeded` | `chunk_embeddings_qwen3_4b_1000` | `1000` | `4` | `7` | `7` | `local-qwen3-embedding-4b` | `435` | `532` | `` |
| `qwen` | `qwen3_4b_2560` | `true` | `11` | `succeeded` | `chunk_embeddings_qwen3_4b_2560` | `2560` | `5` | `8` | `8` | `local-qwen3-embedding-4b` | `170` | `270` | `` |

Post-run cleanup was confirmed through deletion of the temporary file fixture, which
cascades to the temporary document, chunk, embedding jobs, and vector rows.
