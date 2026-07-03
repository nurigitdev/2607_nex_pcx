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

## Core Metadata Schema

The core metadata migration creates the first MVP data tables:

- `files`
- `documents`
- `chunk_policies`
- `embedding_profiles`

It also seeds the default `heading_512_64` chunk policy and the four initial embedding
profiles used by the experiment bench.
