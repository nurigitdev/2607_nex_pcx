# Search Compare Remote Reranker E2E Smoke

Slice 326 adds an executable smoke runner for the full Search Compare reranked profile
path:

1. Search Compare receives a query for `reranked_vector_cosine`.
2. NeX-PCX creates the query embedding for the selected source vector profile.
3. NeX-PCX retrieves vector candidates from pgvector.
4. NeX-PCX calls the remote reranker provider.
5. NeX-PCX stores reranked search log results and runtime metadata.

## Default Contract

| Item | Value |
| --- | --- |
| Search profile | `reranked_vector_cosine` |
| Source vector profile | `qwen3_4b_2560` |
| Chunk policy | `heading_512_64` |
| Remote reranker URL | `http://192.168.20.243:9104` |
| Reranker model | `Qwen/Qwen3-Reranker-4B` |
| Reranker profile | `qwen3_reranker_4b` |
| Expected backend | `qwen_reranker` |
| Expected device | `cuda:0` |

## Prerequisites

- Apply database migrations for the target DB.
- Ensure the source vector profile has embedded chunks for the selected chunk policy.
- Start the DGX embedding providers needed for query embedding.
- Start the DGX reranker provider on port `9104`.
- Register active embedding provider routes for the selected vector profile.

## Dry Run

```bash
./.venv/bin/python scripts/run_search_compare_remote_reranker_e2e_smoke.py --dry-run
```

The dry-run output redacts the query text and masks database passwords.

## Development DB Smoke

```bash
export NEX_PCX_DATABASE_URL='postgresql://nex_pcx_dev:nuri1004@127.0.0.1:5432/nex_pcx_dev'

./.venv/bin/python scripts/run_search_compare_remote_reranker_e2e_smoke.py \
  --query-text "사내 문서 검색 권한과 업무 규칙" \
  --actor-user-id 1 \
  --requested-search-scope company \
  --reranked-vector-profile-name qwen3_4b_2560 \
  --remote-reranker-base-url http://192.168.20.243:9104 \
  --markdown-output artifacts/search_compare_remote_reranker_e2e_smoke.md
```

Use `--document-group` or `--file-type` when limiting the smoke to a known fixture corpus.

## Validated Evidence

The runner checks:

- the `reranked_vector_cosine` profile succeeded
- at least one reranked result was returned
- profile runtime metadata records `provider_runtime_mode=remote`
- profile runtime metadata records the expected remote reranker base URL and timeout
- reranker model/profile/provider values match the expected Qwen reranker contract
- reranker runtime metadata includes service/backend/device evidence
- each returned result is `retrieval_strategy=reranked`
- each returned result preserves source vector profile, source rank, source score, and candidate count
- each returned rerank score is finite

The markdown report includes:

- pass/fail status
- redacted database URL
- remote reranker base URL
- search log ID
- result count and elapsed time
- reranked result score preview
- profile runtime metadata JSON
- mismatches or error details

## Failure Triage

| Symptom | First check |
| --- | --- |
| `database_url is required` | Set `NEX_PCX_DATABASE_URL` or pass `--database-url`. |
| `query_embedding_failed` | Check active embedding provider route registration and route readiness. |
| `reranked_search_failed` | Check reranker `/healthz`, `/v1/rerank`, timeout, and model/profile values. |
| `result_count` is zero | Confirm the selected vector profile and chunk policy have indexed chunks. |
| runtime metadata mismatch | Compare the markdown metadata block with the expected contract above. |
