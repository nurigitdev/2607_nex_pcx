# NeX_PCX

FastAPI + Bootstrap 기반 pre-CX RAG 실험 플랫폼입니다.

## Slice 001

Slice 001은 Project Skeleton + Quality Gate를 목표로 합니다.

- FastAPI 앱 생성
- Bootstrap 기반 dashboard empty state
- `/healthz` smoke endpoint
- pytest/ruff/black/coverage quality gate

## Local Commands

```bash
./.venv/bin/pip install -e ".[dev]"
bash scripts/quality_gate.sh
./.venv/bin/uvicorn app.main:app --reload
```

## PostgreSQL Test Database

Slice 002 uses a local PostgreSQL test database. Docker is not required.

Set the test database URL before running integration checks:

```bash
export NEX_PCX_TEST_DATABASE_URL="postgresql://USER:PASSWORD@127.0.0.1:5432/nex_pcx_test"
bash scripts/check_pgvector.sh
./.venv/bin/python -m pytest tests/integration
```

The integration tests create `pgvector` extension in the test database and verify both
`vector` and `halfvec(2560)` support.

## Migrations

Alembic migrations read the database URL from `NEX_PCX_DATABASE_URL`.

```bash
export NEX_PCX_DATABASE_URL="postgresql://USER:PASSWORD@127.0.0.1:5432/nex_pcx_dev"
bash scripts/migrate.sh upgrade head
bash scripts/migrate.sh current
```

The initial migration enables the PostgreSQL `vector` extension. Its downgrade is a no-op
because the extension may be shared by later objects or pre-existing local database setup.

## Admin Logs

Application logs are stored in `app_logs` and controlled by `app_log_settings`.
The default retention setting keeps 7 days of logs:

```text
logging_enabled=true
min_log_level=INFO
log_retention_days=7
admin_log_page_size=100
```

The log viewer is available at `/admin/logs` after `NEX_PCX_DATABASE_URL` points to a
migrated database.

## File Upload API

Slice 006 adds a local upload pipeline for supported MVP file types:

- `.pdf`
- `.docx`
- `.hwpx`
- `.pptx`
- `.xlsx`
- `.md`

Uploaded files are stored under `NEX_PCX_UPLOAD_STORAGE_DIR` and metadata is persisted to
`files` and `documents`. The API detects duplicate uploads by SHA-256 checksum and returns
the existing metadata instead of creating another database row.

```bash
export NEX_PCX_DATABASE_URL="postgresql://USER:PASSWORD@127.0.0.1:5432/nex_pcx_dev"
export NEX_PCX_UPLOAD_STORAGE_DIR="storage/uploads"
curl -F "file=@README.md" \
  -F "document_group=default" \
  -F "security_level=internal" \
  http://127.0.0.1:8000/api/files
```

## Markdown Parser Foundation

Slice 008 adds the first parser foundation for Phase 3. `MarkdownParser` converts `.md`
content into structured blocks with heading paths, line ranges, and block metadata. It
preserves fenced code blocks and GitHub-style pipe tables as single blocks so the next
chunking slice can keep those structures intact.

Parser regression fixtures live under `tests/fixtures/` and are included in the quality
gate through `tests/regression`.

## Core Metadata Schema

The core metadata migration creates the first MVP data tables:

- `files`
- `documents`
- `chunk_policies`
- `embedding_profiles`

It also seeds the default `heading_512_64` chunk policy and the four initial embedding
profiles used by the experiment bench.

## Identity and Permission Schema

Slice 009 adds the Phase 2.5 identity and permission metadata foundation:

- `app_users`
- `org_units`
- `user_org_memberships`
- `files.uploaded_by_user_id`
- `documents.owner_user_id`
- `documents.owner_org_unit_id`
- `documents.access_scope`
- `documents.permission_metadata`

The migration seeds a small permission simulation graph with member, team lead, group lead,
and admin accounts. Integration tests verify seed rows, hierarchy, FK links, access scope
defaults, check constraints, and membership cascade behavior.

## Pipeline Job Queue Schema

Slice 010 adds the PostgreSQL-backed queue foundation for document ingestion:

- `pipeline_jobs`
- `pipeline_job_events`
- queued/running/succeeded/failed/canceled/skipped status tracking
- upload/parsing/chunking/embedding/vector-indexing stage tracking
- worker lease metadata for `FOR UPDATE SKIP LOCKED` claim queries
- progress, retry, error, and append-only event metadata

Integration tests verify defaults, constraints, FK cascade behavior, claim indexes, and
locked-row skipping for concurrent worker claims.

## Pipeline Job Repository

Slice 011 adds repository helpers for the pipeline queue lifecycle:

- create queued pipeline jobs and append created events
- claim queued or lease-expired jobs with worker lease metadata
- heartbeat running jobs
- update stage progress
- mark jobs succeeded or failed
- requeue failed/canceled jobs for retry

The repository keeps lifecycle SQL in one module so upload APIs, worker runners, and the
future Job Monitor UI can share the same state transitions.

## Upload API Queue Integration

Slice 012 connects file upload success to the pipeline queue. New uploads create a
`document_ingestion` job in `queued` status and return `pipeline_job_id` plus a compact
`pipeline_job` payload from `/api/files`. Duplicate checksum uploads keep returning the
existing file metadata and do not enqueue another pipeline job.

## UI i18n Foundation

Slice 054 adds a JSON-backed UI translation foundation. Korean is the default language,
and English is available through `?lang=en` or the `nex_pcx_lang` cookie. Shared templates
receive `t("translation.key")`, `current_language`, `language_options`, and `language_url`
from the FastAPI template context.

Locale files live under `app/locales/`. New UI work should add labels to both
`ko.json` and `en.json` and render user-facing text through `t()`.

## Embedding Model Distribution

Slice 082 adds the local model bundle foundation. Runtime workers should load models from
`NEX_PCX_MODELS_DIR` instead of downloading them during application startup. The default
bundle root is `models/`, and downloaded model files are ignored by git.

Install the optional model tooling and inspect the download plan:

```bash
./.venv/bin/pip install -e ".[models]"
./.venv/bin/python scripts/download_embedding_models.py --dry-run
```

Download all model repositories into the default bundle root:

```bash
./.venv/bin/python scripts/download_embedding_models.py
```

After downloading, run a local smoke check for the SentenceTransformers-backed models:

```bash
./.venv/bin/python scripts/check_embedding_models.py --dry-run
./.venv/bin/python scripts/check_embedding_models.py --model kure_v1 --json
```

The 4 embedding profiles map to 3 downloaded model directories:

- `models/kure_v1` for `nlpai-lab/KURE-v1` / `kure_v1_1024`
- `models/bge_m3` for `BAAI/bge-m3` / `bge_m3_1024`
- `models/qwen3_embedding_4b` for `Qwen/Qwen3-Embedding-4B` /
  `qwen3_4b_1000`, `qwen3_4b_2560`

Production and customer-site installs should receive a verified copy of `models/` and run
workers in offline mode rather than downloading model files from the public internet.

## Embedding Provider Architecture

Slice 085 separates heavy embedding inference from the main NeX_PCX application. The app
continues to own queue orchestration, pgvector persistence, permission-aware search, and
experiment logging. Embedding calculation can run through:

- `mock` provider for deterministic tests
- local model adapter for small smoke/debug runs
- remote GPU provider API for benchmark ingestion and production-like experiments

The remote provider should preload downloaded model bundles on a GPU server and expose a
stable request/response contract. See `docs/embedding_provider_architecture.md` for the
initial provider boundary and metadata requirements,
`docs/gpu_embedding_provider_deployment.md` for the deployment checklist, and
`docs/provider_operations_playbook.md` for the provider route preflight, readiness,
contract snapshot, sample set, and alert acknowledgement workflow.

Runtime provider selection is controlled by:

- `NEX_PCX_EMBEDDING_PROVIDER_MODE`: `mock` by default, or `remote`
- `NEX_PCX_REMOTE_EMBEDDING_PROVIDER_URL`: required when provider mode is `remote`
- `NEX_PCX_REMOTE_EMBEDDING_PROVIDER_TIMEOUT_SECONDS`: remote request timeout, default `30.0`
- `/admin/embedding-provider-routes`: profile-specific provider routes used by the worker
  before falling back to the runtime settings above

Process one pending embedding job with route-aware provider selection:

```bash
./.venv/bin/python scripts/process_embedding_job.py --provider-mode mock
./.venv/bin/python scripts/process_embedding_job.py \
  --provider-mode remote \
  --remote-provider-url http://127.0.0.1:9000
```

Process a bounded batch of pending embedding jobs in one worker run:

```bash
./.venv/bin/python scripts/process_embedding_job.py --provider-mode mock --limit 10
```

Batch runs persist a compact summary in `embedding_worker_batch_runs`; the script JSON
response includes `batch_run_id`, processed/succeeded/failed/deferred/idle counts, stop
reason, elapsed time, and per-job result details for operator review.
Operators can review the same history at `/admin/embedding-batch-runs` or through
`/api/admin/embedding-batch-runs`, then requeue retryable failed jobs from a selected
batch run with `/api/admin/embedding-batch-runs/{batch_run_id}/retry-failed`.
Queue backlog by profile is summarized on `/admin/embedding-jobs` and exposed through
`/api/admin/embedding-jobs/backlog-summary`.

Force the legacy runtime-config-only path when you want to ignore route records:

```bash
./.venv/bin/python scripts/process_embedding_job.py \
  --provider-source runtime \
  --provider-mode mock
```

Provider health is exposed at `/api/embedding/providers/health`.
Route health aggregation is exposed at `/api/admin/embedding-provider-routes/health`.
Remote GPU host setup is documented in `docs/remote_gpu_provider_deployment_playbook.md`,
with generated env/systemd files available through
`scripts/setup_remote_gpu_provider.py`.
Remote source sync and `.venv` runtime setup are documented in
`docs/remote_gpu_provider_source_sync_runtime_guide.md`.
Remote foreground provider smoke planning is documented in
`docs/remote_provider_foreground_launch_smoke_plan.md`.
Run `scripts/run_remote_provider_foreground_smoke.py --provider kure --json` from the
app host to start KURE over SSH, verify `/healthz`, and stop the foreground process.
The first KURE DGX Spark smoke result is recorded in
`docs/remote_kure_foreground_health_smoke_result.md`.
The first BGE DGX Spark smoke result is recorded in
`docs/remote_bge_foreground_health_smoke_result.md`.
The first Qwen DGX Spark dual-profile smoke result is recorded in
`docs/remote_qwen_foreground_health_smoke_result.md`.
Remote embedding request smoke checks are documented in
`docs/remote_embedding_request_smoke_runner.md`.
Run `scripts/run_remote_provider_embedding_smoke_suite.py --json` to launch KURE, BGE,
and Qwen sequentially, validate real embedding responses, and stop each remote provider.
The first passing suite result is recorded in
`docs/remote_embedding_request_smoke_suite_result.md`.
The first KURE DGX Spark request smoke result is recorded in
`docs/remote_kure_embedding_request_smoke_result.md`.
The first BGE DGX Spark request smoke result is recorded in
`docs/remote_bge_embedding_request_smoke_result.md`.
The first Qwen DGX Spark dual-profile request smoke result is recorded in
`docs/remote_qwen_embedding_request_smoke_result.md`.
Run `scripts/check_remote_gpu_provider_host.py` from the app host for a read-only
readiness report against the DGX Spark provider server.

Run due scheduled route preflight checks after enabling rows in
`embedding_provider_preflight_schedules`:

```bash
./.venv/bin/python scripts/run_scheduled_provider_preflight.py --limit 20
```

Provider route health snapshots, contract snapshots, and preflight run history share a
30-day retention setting stored in `app_log_settings`. Operators can preview or execute
cleanup through `/api/admin/embedding-provider-routes/cleanup`.

Run the standalone embedding provider skeleton for contract tests:

```bash
./.venv/bin/uvicorn app.embedding_provider_service:app --host 127.0.0.1 --port 9000
```

Run the provider with a preloaded local SentenceTransformers model:

```bash
NEX_PCX_PROVIDER_BACKEND=sentence_transformers \
NEX_PCX_PROVIDER_MODEL_KEY=kure_v1 \
NEX_PCX_PROVIDER_PROFILE_NAMES=kure_v1_1024 \
NEX_PCX_PROVIDER_MODEL_ID=local-kure-v1 \
./.venv/bin/uvicorn app.embedding_provider_service:app --host 127.0.0.1 --port 9000
```

Run the Qwen provider with one loaded model serving both Qwen profiles:

```bash
NEX_PCX_PROVIDER_BACKEND=qwen_embedding \
NEX_PCX_PROVIDER_MODEL_KEY=qwen3_embedding_4b \
NEX_PCX_PROVIDER_PROFILE_NAMES=qwen3_4b_1000,qwen3_4b_2560 \
NEX_PCX_PROVIDER_MODEL_ID=local-qwen3-embedding-4b \
./.venv/bin/uvicorn app.embedding_provider_service:app --host 127.0.0.1 --port 9103
```

Register both `qwen3_4b_1000` and `qwen3_4b_2560` routes with the same provider
base URL, such as `http://127.0.0.1:9103`. The request profile and output
dimension select the storage profile while the provider keeps one Qwen model loaded.

Provider launch and route registration helpers are available for local operations:

```bash
./.venv/bin/python scripts/run_embedding_provider.py --provider qwen --dry-run --json
./.venv/bin/python scripts/run_embedding_provider.py --provider qwen

./.venv/bin/python scripts/register_embedding_provider_routes.py \
  --provider qwen \
  --database-url "$NEX_PCX_DATABASE_URL" \
  --dry-run \
  --json
./.venv/bin/python scripts/register_embedding_provider_routes.py \
  --provider qwen \
  --database-url "$NEX_PCX_DATABASE_URL"
```

Default local ports are `9101` for KURE, `9102` for BGE-M3, and `9103` for the
shared Qwen provider.
