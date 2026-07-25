# Remote Reranker Request Smoke Runner

Slice 325 adds a request-level smoke runner for the DGX Qwen3-Reranker-4B provider.
It validates the provider `/v1/rerank` contract after the foreground health smoke has
already confirmed `/healthz`.

## Default Target

| Item | Value |
| --- | --- |
| Host | `192.168.20.243` |
| Port | `9104` |
| Base URL | `http://192.168.20.243:9104` |
| Endpoint | `/v1/rerank` |
| Provider model | `Qwen/Qwen3-Reranker-4B` |
| Reranker profile | `qwen3_reranker_4b` |
| Expected backend | `qwen_reranker` |
| Expected device | `cuda:0` |

## Dry Run

Use dry-run mode first to confirm the request contract without contacting the DGX
provider:

```bash
./.venv/bin/python scripts/run_remote_reranker_request_smoke.py --dry-run
./.venv/bin/python scripts/run_remote_reranker_request_smoke.py --dry-run --json
```

The dry-run payload redacts query and candidate text, while preserving candidate keys,
source profile, retrieval strategy, and top-k values.

## Request Smoke

Start the remote reranker provider first, then run:

```bash
./.venv/bin/python scripts/run_remote_reranker_request_smoke.py \
  --markdown-output artifacts/remote_reranker_request_smoke.md
```

For custom text:

```bash
./.venv/bin/python scripts/run_remote_reranker_request_smoke.py \
  --query-text "사내 문서 검색 권한" \
  --candidate-text "사내 공통 규칙 문서는 모든 직원에게 공개된다." \
  --candidate-text "개인 문서는 작성자와 권한 있는 상위 조직 사용자만 검색한다." \
  --candidate-text "provider 상태는 관리자 화면에서 점검한다." \
  --markdown-output artifacts/remote_reranker_request_smoke.md
```

## Validated Contract

The runner checks:

- response `provider_type=remote`
- response `reranker_model_id=Qwen/Qwen3-Reranker-4B`
- response `reranker_profile_name=qwen3_reranker_4b`
- response `retrieval_strategy=reranked`
- request `candidate_count`, `returned_count`, and `top_k`
- result rank sequence and finite rerank scores
- runtime metadata `service=nex_pcx_reranker_provider_service`
- runtime metadata `backend=qwen_reranker`
- runtime metadata `device=cuda:0`

## Markdown Evidence

The markdown report includes:

- pass/fail status
- rerank endpoint and provider identifiers
- request and provider latency
- rank, candidate key, score, and source rank preview
- runtime metadata JSON
- mismatch or error details when the smoke fails

This smoke runner intentionally does not launch or stop the remote provider. Use
`scripts/run_remote_reranker_foreground_smoke.py` for foreground lifecycle validation.

After this request smoke passes, run
`scripts/run_search_compare_remote_reranker_e2e_smoke.py` to validate the full
Search Compare `reranked_vector_cosine` path against the remote reranker provider.
