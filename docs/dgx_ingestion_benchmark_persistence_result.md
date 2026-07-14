# DGX Ingestion Benchmark Persistence Result

- `passed`: `true`
- `benchmark_run_id`: `1`
- `benchmark_run_key`: `dgx-small-corpus-1784070146912`
- `created_by`: `slice-208`
- `profile_count`: `4`
- `job_result_count`: `12`
- `expected_job_count`: `12`
- `vector_count`: `12`
- `cleanup_confirmed`: `true`

Slice 208 added `dgx_ingestion_benchmark_runs`,
`dgx_ingestion_benchmark_profile_results`, and
`dgx_ingestion_benchmark_job_results`.

The first persisted development DB run was created with:

```bash
./.venv/bin/python scripts/run_dgx_small_corpus_embedding_benchmark.py \
  --database-url postgresql://nex_pcx_dev:***@127.0.0.1:5432/nex_pcx_dev \
  --chunk-count 3 \
  --persist-result \
  --created-by slice-208 \
  --json \
  --markdown-output docs/dgx_small_corpus_embedding_benchmark_result.md
```

The persisted detail was verified through
`get_dgx_ingestion_benchmark_detail(...)`:

```text
{'run_id': 1, 'passed': True, 'profiles': 4, 'jobs': 12, 'expected': 12, 'vectors': 12, 'created_by': 'slice-208'}
```
