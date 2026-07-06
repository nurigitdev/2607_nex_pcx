"""Create search log and feedback schema.

Revision ID: 20260707_0008
Revises: 20260706_0007
Create Date: 2026-07-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260707_0008"
down_revision: str | None = "20260706_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE search_logs (
            search_log_id BIGSERIAL PRIMARY KEY,
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
            permission_filter_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(permission_filter_metadata) = 'object'),
            document_group TEXT,
            file_type TEXT,
            chunk_policy_name TEXT REFERENCES chunk_policies(chunk_policy_name),
            top_k INT NOT NULL CHECK (top_k > 0),
            similarity_metric TEXT NOT NULL DEFAULT 'cosine'
                CHECK (similarity_metric IN ('cosine', 'l2', 'inner_product')),
            profiles JSONB NOT NULL
                CHECK (jsonb_typeof(profiles) = 'array'),
            query_runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(query_runtime_metadata) = 'object'),
            total_elapsed_ms INT CHECK (
                total_elapsed_ms IS NULL OR total_elapsed_ms >= 0
            ),
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("CREATE INDEX idx_search_logs_created_at ON search_logs (created_at DESC)")
    op.execute("CREATE INDEX idx_search_logs_actor ON search_logs (actor_user_id, created_at DESC)")
    op.execute("""
        CREATE INDEX idx_search_logs_scope
        ON search_logs (requested_search_scope, effective_search_scope)
        """)
    op.execute("CREATE INDEX idx_search_logs_chunk_policy ON search_logs (chunk_policy_name)")

    op.execute("""
        CREATE TABLE search_log_results (
            search_log_result_id BIGSERIAL PRIMARY KEY,
            search_log_id BIGINT NOT NULL
                REFERENCES search_logs(search_log_id) ON DELETE CASCADE,
            profile_name TEXT NOT NULL REFERENCES embedding_profiles(profile_name),
            rank INT NOT NULL CHECK (rank > 0),
            chunk_id BIGINT NOT NULL REFERENCES chunks(chunk_id),
            distance DOUBLE PRECISION,
            score DOUBLE PRECISION,
            profile_elapsed_ms INT CHECK (
                profile_elapsed_ms IS NULL OR profile_elapsed_ms >= 0
            ),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (search_log_id, profile_name, rank)
        )
        """)
    op.execute("""
        CREATE INDEX idx_search_log_results_log_profile
        ON search_log_results (search_log_id, profile_name, rank)
        """)
    op.execute("CREATE INDEX idx_search_log_results_chunk ON search_log_results (chunk_id)")

    op.execute("""
        CREATE TABLE search_result_feedback (
            feedback_id BIGSERIAL PRIMARY KEY,
            search_log_result_id BIGINT NOT NULL
                REFERENCES search_log_results(search_log_result_id) ON DELETE CASCADE,
            relevance_label TEXT NOT NULL
                CHECK (
                    relevance_label IN (
                        'correct',
                        'partial',
                        'wrong',
                        'duplicate',
                        'insufficient_context'
                    )
                ),
            comment TEXT,
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_search_result_feedback_result
        ON search_result_feedback (search_log_result_id)
        """)
    op.execute("""
        CREATE INDEX idx_search_result_feedback_user
        ON search_result_feedback (created_by_user_id, created_at DESC)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS search_result_feedback")
    op.execute("DROP TABLE IF EXISTS search_log_results")
    op.execute("DROP TABLE IF EXISTS search_logs")
