"""Create golden evaluation result schema.

Revision ID: 20260707_0010
Revises: 20260707_0009
Create Date: 2026-07-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260707_0010"
down_revision: str | None = "20260707_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE golden_evaluation_runs (
            evaluation_run_id BIGSERIAL PRIMARY KEY,
            question_set_id BIGINT NOT NULL
                REFERENCES golden_question_sets(question_set_id) ON DELETE CASCADE,
            run_name TEXT NOT NULL CHECK (length(btrim(run_name)) > 0),
            profile_name TEXT NOT NULL REFERENCES embedding_profiles(profile_name),
            chunk_policy_name TEXT REFERENCES chunk_policies(chunk_policy_name),
            similarity_metric TEXT NOT NULL DEFAULT 'cosine'
                CHECK (similarity_metric IN ('cosine', 'l2', 'inner_product')),
            top_k INT NOT NULL DEFAULT 5 CHECK (top_k > 0),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
            question_count INT NOT NULL DEFAULT 0 CHECK (question_count >= 0),
            recall_question_count INT NOT NULL DEFAULT 0 CHECK (recall_question_count >= 0),
            ndcg_question_count INT NOT NULL DEFAULT 0 CHECK (ndcg_question_count >= 0),
            no_answer_question_count INT NOT NULL DEFAULT 0
                CHECK (no_answer_question_count >= 0),
            hidden_violation_count INT NOT NULL DEFAULT 0 CHECK (hidden_violation_count >= 0),
            mean_recall_at_k DOUBLE PRECISION
                CHECK (mean_recall_at_k IS NULL OR mean_recall_at_k BETWEEN 0 AND 1),
            mean_reciprocal_rank DOUBLE PRECISION
                CHECK (
                    mean_reciprocal_rank IS NULL
                    OR mean_reciprocal_rank BETWEEN 0 AND 1
                ),
            mean_ndcg DOUBLE PRECISION
                CHECK (mean_ndcg IS NULL OR mean_ndcg BETWEEN 0 AND 1),
            no_answer_success_rate DOUBLE PRECISION
                CHECK (
                    no_answer_success_rate IS NULL
                    OR no_answer_success_rate BETWEEN 0 AND 1
                ),
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            error_message TEXT,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (question_set_id, run_name)
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_evaluation_runs_set
        ON golden_evaluation_runs (question_set_id, created_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_golden_evaluation_runs_profile_status
        ON golden_evaluation_runs (profile_name, status)
        """)
    op.execute("""
        CREATE INDEX idx_golden_evaluation_runs_created_at
        ON golden_evaluation_runs (created_at DESC)
        """)

    op.execute("""
        CREATE TABLE golden_evaluation_results (
            evaluation_result_id BIGSERIAL PRIMARY KEY,
            evaluation_run_id BIGINT NOT NULL
                REFERENCES golden_evaluation_runs(evaluation_run_id) ON DELETE CASCADE,
            question_id BIGINT NOT NULL
                REFERENCES golden_questions(question_id) ON DELETE CASCADE,
            search_log_id BIGINT REFERENCES search_logs(search_log_id) ON DELETE SET NULL,
            top_k INT NOT NULL CHECK (top_k > 0),
            visible_expected_count INT NOT NULL DEFAULT 0
                CHECK (visible_expected_count >= 0),
            retrieved_count INT NOT NULL DEFAULT 0 CHECK (retrieved_count >= 0),
            matched_visible_count INT NOT NULL DEFAULT 0
                CHECK (matched_visible_count >= 0),
            hidden_violation_count INT NOT NULL DEFAULT 0 CHECK (hidden_violation_count >= 0),
            matched_chunk_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[],
            hidden_violation_chunk_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[],
            recall_at_k DOUBLE PRECISION
                CHECK (recall_at_k IS NULL OR recall_at_k BETWEEN 0 AND 1),
            reciprocal_rank DOUBLE PRECISION
                CHECK (reciprocal_rank IS NULL OR reciprocal_rank BETWEEN 0 AND 1),
            dcg DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (dcg >= 0),
            ideal_dcg DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (ideal_dcg >= 0),
            ndcg DOUBLE PRECISION
                CHECK (ndcg IS NULL OR ndcg BETWEEN 0 AND 1),
            no_answer_success BOOLEAN,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (evaluation_run_id, question_id)
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_evaluation_results_run
        ON golden_evaluation_results (evaluation_run_id)
        """)
    op.execute("""
        CREATE INDEX idx_golden_evaluation_results_question
        ON golden_evaluation_results (question_id)
        """)
    op.execute("""
        CREATE INDEX idx_golden_evaluation_results_search_log
        ON golden_evaluation_results (search_log_id)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS golden_evaluation_results")
    op.execute("DROP TABLE IF EXISTS golden_evaluation_runs")
