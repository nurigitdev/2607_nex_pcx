"""Create search experiment run schema.

Revision ID: 20260712_0024
Revises: 20260712_0023
Create Date: 2026-07-12
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260712_0024"
down_revision: str | None = "20260712_0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE search_experiment_runs (
            experiment_run_id BIGSERIAL PRIMARY KEY,
            run_name TEXT NOT NULL CHECK (length(btrim(run_name)) > 0),
            query_text TEXT NOT NULL CHECK (length(btrim(query_text)) > 0),
            normalized_query_text TEXT,
            actor_user_id BIGINT REFERENCES app_users(user_id),
            requested_search_scope TEXT
                CHECK (
                    requested_search_scope IS NULL
                    OR requested_search_scope IN ('mine', 'team', 'managed_org', 'company')
                ),
            effective_search_scope TEXT
                CHECK (
                    effective_search_scope IS NULL
                    OR effective_search_scope IN ('mine', 'team', 'managed_org', 'company')
                ),
            document_group TEXT,
            file_type TEXT,
            chunk_policy_name TEXT REFERENCES chunk_policies(chunk_policy_name),
            strategy_name TEXT NOT NULL DEFAULT 'vector_cosine'
                CHECK (length(btrim(strategy_name)) > 0),
            similarity_metric TEXT NOT NULL DEFAULT 'cosine'
                CHECK (similarity_metric IN ('cosine', 'l2', 'inner_product')),
            top_k INT NOT NULL DEFAULT 5 CHECK (top_k > 0),
            score_threshold DOUBLE PRECISION,
            profile_names JSONB NOT NULL
                CHECK (jsonb_typeof(profile_names) = 'array'),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'canceled')),
            total_profile_count INT NOT NULL DEFAULT 0 CHECK (total_profile_count >= 0),
            completed_profile_count INT NOT NULL DEFAULT 0 CHECK (completed_profile_count >= 0),
            result_count INT NOT NULL DEFAULT 0 CHECK (result_count >= 0),
            failure_count INT NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
            total_elapsed_ms INT CHECK (
                total_elapsed_ms IS NULL OR total_elapsed_ms >= 0
            ),
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            error_message TEXT,
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_search_experiment_runs_created_at
        ON search_experiment_runs (created_at DESC, experiment_run_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_search_experiment_runs_status
        ON search_experiment_runs (status, created_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_search_experiment_runs_strategy
        ON search_experiment_runs (strategy_name, created_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_search_experiment_runs_actor
        ON search_experiment_runs (actor_user_id, created_at DESC)
        """)

    op.execute("""
        CREATE TABLE search_experiment_profile_runs (
            experiment_profile_run_id BIGSERIAL PRIMARY KEY,
            experiment_run_id BIGINT NOT NULL
                REFERENCES search_experiment_runs(experiment_run_id) ON DELETE CASCADE,
            profile_name TEXT NOT NULL REFERENCES embedding_profiles(profile_name),
            search_log_id BIGINT REFERENCES search_logs(search_log_id) ON DELETE SET NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')),
            result_count INT NOT NULL DEFAULT 0 CHECK (result_count >= 0),
            top_score DOUBLE PRECISION,
            average_score DOUBLE PRECISION,
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            error_message TEXT,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (experiment_run_id, profile_name)
        )
        """)
    op.execute("""
        CREATE INDEX idx_search_experiment_profile_runs_run
        ON search_experiment_profile_runs (experiment_run_id, profile_name)
        """)
    op.execute("""
        CREATE INDEX idx_search_experiment_profile_runs_profile_status
        ON search_experiment_profile_runs (profile_name, status)
        """)
    op.execute("""
        CREATE INDEX idx_search_experiment_profile_runs_search_log
        ON search_experiment_profile_runs (search_log_id)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS search_experiment_profile_runs")
    op.execute("DROP TABLE IF EXISTS search_experiment_runs")
