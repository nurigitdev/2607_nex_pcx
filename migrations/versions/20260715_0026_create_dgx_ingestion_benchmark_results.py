"""Create DGX ingestion benchmark result persistence schema.

Revision ID: 20260715_0026
Revises: 20260713_0025
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260715_0026"
down_revision: str | None = "20260713_0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE dgx_ingestion_benchmark_runs (
            benchmark_run_id BIGSERIAL PRIMARY KEY,
            benchmark_run_key TEXT NOT NULL UNIQUE
                CHECK (length(btrim(benchmark_run_key)) > 0),
            script_name TEXT NOT NULL
                CHECK (length(btrim(script_name)) > 0),
            provider_names JSONB NOT NULL
                CHECK (jsonb_typeof(provider_names) = 'array'),
            profile_names JSONB NOT NULL
                CHECK (jsonb_typeof(profile_names) = 'array'),
            chunk_count INT NOT NULL CHECK (chunk_count > 0),
            expected_job_count INT NOT NULL CHECK (expected_job_count >= 0),
            processed_count INT NOT NULL CHECK (processed_count >= 0),
            succeeded_count INT NOT NULL CHECK (succeeded_count >= 0),
            failed_count INT NOT NULL CHECK (failed_count >= 0),
            vector_count INT NOT NULL CHECK (vector_count >= 0),
            passed BOOLEAN NOT NULL DEFAULT false,
            preflight_before_worker BOOLEAN NOT NULL DEFAULT true,
            active_only_preflight BOOLEAN NOT NULL DEFAULT true,
            cleanup_attempted BOOLEAN NOT NULL DEFAULT false,
            cleanup_confirmed BOOLEAN NOT NULL DEFAULT false,
            total_elapsed_seconds DOUBLE PRECISION NOT NULL
                CHECK (total_elapsed_seconds >= 0),
            total_provider_elapsed_ms INT
                CHECK (total_provider_elapsed_ms IS NULL OR total_provider_elapsed_ms >= 0),
            total_worker_elapsed_ms INT
                CHECK (total_worker_elapsed_ms IS NULL OR total_worker_elapsed_ms >= 0),
            fixture_file_id BIGINT,
            fixture_document_id BIGINT,
            fixture_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(fixture_chunk_ids) = 'array'),
            plan_payload JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(plan_payload) = 'object'),
            fixture_payload JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(fixture_payload) = 'object'),
            report_payload JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(report_payload) = 'object'),
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_dgx_ingestion_benchmark_runs_created
        ON dgx_ingestion_benchmark_runs (created_at DESC, benchmark_run_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_dgx_ingestion_benchmark_runs_passed_created
        ON dgx_ingestion_benchmark_runs (passed, created_at DESC)
        """)

    op.execute("""
        CREATE TABLE dgx_ingestion_benchmark_profile_results (
            benchmark_profile_id BIGSERIAL PRIMARY KEY,
            benchmark_run_id BIGINT NOT NULL
                REFERENCES dgx_ingestion_benchmark_runs(benchmark_run_id)
                ON DELETE CASCADE,
            provider TEXT NOT NULL CHECK (length(btrim(provider)) > 0),
            profile_name TEXT NOT NULL CHECK (length(btrim(profile_name)) > 0),
            expected_job_count INT NOT NULL CHECK (expected_job_count >= 0),
            processed_count INT NOT NULL CHECK (processed_count >= 0),
            succeeded_count INT NOT NULL CHECK (succeeded_count >= 0),
            failed_count INT NOT NULL CHECK (failed_count >= 0),
            vector_count INT NOT NULL CHECK (vector_count >= 0),
            passed BOOLEAN NOT NULL DEFAULT false,
            vector_table_name TEXT,
            vector_dimension INT CHECK (vector_dimension IS NULL OR vector_dimension > 0),
            vector_storage_type TEXT,
            provider_route_id BIGINT,
            provider_route_name TEXT,
            provider_runtime_base_url TEXT,
            provider_model_id TEXT,
            provider_type TEXT,
            readiness_status TEXT,
            readiness_health_snapshot_id BIGINT,
            readiness_contract_snapshot_id BIGINT,
            total_provider_elapsed_ms INT
                CHECK (total_provider_elapsed_ms IS NULL OR total_provider_elapsed_ms >= 0),
            avg_provider_elapsed_ms DOUBLE PRECISION
                CHECK (avg_provider_elapsed_ms IS NULL OR avg_provider_elapsed_ms >= 0),
            max_provider_elapsed_ms INT
                CHECK (max_provider_elapsed_ms IS NULL OR max_provider_elapsed_ms >= 0),
            total_worker_elapsed_ms INT
                CHECK (total_worker_elapsed_ms IS NULL OR total_worker_elapsed_ms >= 0),
            avg_worker_elapsed_ms DOUBLE PRECISION
                CHECK (avg_worker_elapsed_ms IS NULL OR avg_worker_elapsed_ms >= 0),
            max_worker_elapsed_ms INT
                CHECK (max_worker_elapsed_ms IS NULL OR max_worker_elapsed_ms >= 0),
            errors JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(errors) = 'array'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (benchmark_run_id, profile_name)
        )
        """)
    op.execute("""
        CREATE INDEX idx_dgx_ingestion_benchmark_profiles_profile
        ON dgx_ingestion_benchmark_profile_results (
            profile_name,
            benchmark_run_id DESC
        )
        """)
    op.execute("""
        CREATE INDEX idx_dgx_ingestion_benchmark_profiles_provider
        ON dgx_ingestion_benchmark_profile_results (
            provider,
            benchmark_run_id DESC
        )
        """)

    op.execute("""
        CREATE TABLE dgx_ingestion_benchmark_job_results (
            benchmark_job_result_id BIGSERIAL PRIMARY KEY,
            benchmark_run_id BIGINT NOT NULL
                REFERENCES dgx_ingestion_benchmark_runs(benchmark_run_id)
                ON DELETE CASCADE,
            benchmark_profile_id BIGINT NOT NULL
                REFERENCES dgx_ingestion_benchmark_profile_results(benchmark_profile_id)
                ON DELETE CASCADE,
            provider TEXT NOT NULL CHECK (length(btrim(provider)) > 0),
            profile_name TEXT NOT NULL CHECK (length(btrim(profile_name)) > 0),
            source_job_id BIGINT,
            source_chunk_id BIGINT,
            processed BOOLEAN NOT NULL DEFAULT false,
            job_status TEXT CHECK (
                job_status IS NULL OR
                job_status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')
            ),
            vector_table_name TEXT,
            vector_dimension INT CHECK (vector_dimension IS NULL OR vector_dimension > 0),
            vector_storage_type TEXT,
            provider_route_id BIGINT,
            provider_route_name TEXT,
            provider_runtime_base_url TEXT,
            provider_model_id TEXT,
            provider_type TEXT,
            provider_elapsed_ms INT
                CHECK (provider_elapsed_ms IS NULL OR provider_elapsed_ms >= 0),
            worker_elapsed_ms INT
                CHECK (worker_elapsed_ms IS NULL OR worker_elapsed_ms >= 0),
            readiness_status TEXT,
            readiness_health_snapshot_id BIGINT,
            readiness_contract_snapshot_id BIGINT,
            message TEXT,
            error TEXT,
            passed BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_dgx_ingestion_benchmark_jobs_run_profile
        ON dgx_ingestion_benchmark_job_results (
            benchmark_run_id,
            profile_name,
            benchmark_job_result_id
        )
        """)
    op.execute("""
        CREATE INDEX idx_dgx_ingestion_benchmark_jobs_source_job
        ON dgx_ingestion_benchmark_job_results (source_job_id)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS dgx_ingestion_benchmark_job_results")
    op.execute("DROP TABLE IF EXISTS dgx_ingestion_benchmark_profile_results")
    op.execute("DROP TABLE IF EXISTS dgx_ingestion_benchmark_runs")
