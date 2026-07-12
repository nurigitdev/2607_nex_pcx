"""Create golden batch metric snapshot schema.

Revision ID: 20260713_0025
Revises: 20260712_0024
Create Date: 2026-07-13
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260713_0025"
down_revision: str | None = "20260712_0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE golden_search_experiment_batch_metric_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            batch_key TEXT NOT NULL CHECK (length(btrim(batch_key)) > 0),
            question_set_id BIGINT NOT NULL,
            question_set_name TEXT NOT NULL DEFAULT '',
            batch_prefix TEXT NOT NULL CHECK (length(btrim(batch_prefix)) > 0),
            strategy_name TEXT NOT NULL CHECK (length(btrim(strategy_name)) > 0),
            top_k INT NOT NULL CHECK (top_k > 0),
            score_threshold DOUBLE PRECISION,
            chunk_policy_name TEXT,
            profile_names JSONB NOT NULL
                CHECK (jsonb_typeof(profile_names) = 'array'),
            batch_status TEXT NOT NULL
                CHECK (batch_status IN ('pending', 'running', 'succeeded', 'failed', 'canceled')),
            batch_question_count INT NOT NULL CHECK (batch_question_count >= 0),
            batch_succeeded_count INT NOT NULL CHECK (batch_succeeded_count >= 0),
            batch_failed_count INT NOT NULL CHECK (batch_failed_count >= 0),
            batch_running_count INT NOT NULL CHECK (batch_running_count >= 0),
            total_result_count INT NOT NULL CHECK (total_result_count >= 0),
            average_result_count DOUBLE PRECISION NOT NULL CHECK (average_result_count >= 0),
            total_elapsed_ms INT NOT NULL CHECK (total_elapsed_ms >= 0),
            average_elapsed_ms DOUBLE PRECISION CHECK (
                average_elapsed_ms IS NULL OR average_elapsed_ms >= 0
            ),
            evaluated_row_count INT NOT NULL CHECK (evaluated_row_count >= 0),
            recall_question_count INT NOT NULL CHECK (recall_question_count >= 0),
            ndcg_question_count INT NOT NULL CHECK (ndcg_question_count >= 0),
            no_answer_question_count INT NOT NULL CHECK (no_answer_question_count >= 0),
            hidden_violation_count INT NOT NULL CHECK (hidden_violation_count >= 0),
            mean_recall_at_k DOUBLE PRECISION,
            mean_reciprocal_rank DOUBLE PRECISION,
            mean_ndcg DOUBLE PRECISION,
            no_answer_success_rate DOUBLE PRECISION,
            source_first_experiment_run_id BIGINT NOT NULL,
            source_last_experiment_run_id BIGINT NOT NULL,
            source_first_created_at TIMESTAMPTZ NOT NULL,
            source_last_updated_at TIMESTAMPTZ NOT NULL,
            metric_payload JSONB NOT NULL CHECK (jsonb_typeof(metric_payload) = 'object'),
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_batch_metric_snapshots_batch_time
        ON golden_search_experiment_batch_metric_snapshots (
            batch_key,
            created_at DESC,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_batch_metric_snapshots_question_set_time
        ON golden_search_experiment_batch_metric_snapshots (
            question_set_id,
            created_at DESC,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE TABLE golden_search_experiment_batch_profile_metric_snapshots (
            snapshot_profile_metric_id BIGSERIAL PRIMARY KEY,
            snapshot_id BIGINT NOT NULL
                REFERENCES golden_search_experiment_batch_metric_snapshots(snapshot_id)
                ON DELETE CASCADE,
            profile_name TEXT NOT NULL,
            question_count INT NOT NULL CHECK (question_count >= 0),
            recall_question_count INT NOT NULL CHECK (recall_question_count >= 0),
            ndcg_question_count INT NOT NULL CHECK (ndcg_question_count >= 0),
            no_answer_question_count INT NOT NULL CHECK (no_answer_question_count >= 0),
            hidden_violation_count INT NOT NULL CHECK (hidden_violation_count >= 0),
            mean_recall_at_k DOUBLE PRECISION,
            mean_reciprocal_rank DOUBLE PRECISION,
            mean_ndcg DOUBLE PRECISION,
            no_answer_success_rate DOUBLE PRECISION,
            total_result_count INT NOT NULL CHECK (total_result_count >= 0),
            average_result_count DOUBLE PRECISION,
            average_elapsed_ms DOUBLE PRECISION,
            UNIQUE (snapshot_id, profile_name)
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_batch_profile_metric_snapshots_profile
        ON golden_search_experiment_batch_profile_metric_snapshots (
            profile_name,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE TABLE golden_search_experiment_batch_question_metric_snapshots (
            snapshot_question_metric_id BIGSERIAL PRIMARY KEY,
            snapshot_id BIGINT NOT NULL
                REFERENCES golden_search_experiment_batch_metric_snapshots(snapshot_id)
                ON DELETE CASCADE,
            question_id BIGINT NOT NULL,
            question_text TEXT NOT NULL CHECK (length(btrim(question_text)) > 0),
            profile_name TEXT NOT NULL,
            experiment_run_id BIGINT NOT NULL,
            search_log_id BIGINT NOT NULL,
            top_k INT NOT NULL CHECK (top_k > 0),
            result_count INT NOT NULL CHECK (result_count >= 0),
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            visible_expected_count INT NOT NULL CHECK (visible_expected_count >= 0),
            retrieved_count INT NOT NULL CHECK (retrieved_count >= 0),
            matched_visible_count INT NOT NULL CHECK (matched_visible_count >= 0),
            hidden_violation_count INT NOT NULL CHECK (hidden_violation_count >= 0),
            matched_chunk_ids JSONB NOT NULL
                CHECK (jsonb_typeof(matched_chunk_ids) = 'array'),
            hidden_violation_chunk_ids JSONB NOT NULL
                CHECK (jsonb_typeof(hidden_violation_chunk_ids) = 'array'),
            recall_at_k DOUBLE PRECISION,
            reciprocal_rank DOUBLE PRECISION,
            dcg DOUBLE PRECISION NOT NULL,
            ideal_dcg DOUBLE PRECISION NOT NULL,
            ndcg DOUBLE PRECISION,
            no_answer_success BOOLEAN,
            UNIQUE (snapshot_id, question_id, profile_name)
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_batch_question_metric_snapshots_question
        ON golden_search_experiment_batch_question_metric_snapshots (
            question_id,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_batch_question_metric_snapshots_profile
        ON golden_search_experiment_batch_question_metric_snapshots (
            profile_name,
            snapshot_id DESC
        )
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS golden_search_experiment_batch_question_metric_snapshots")
    op.execute("DROP TABLE IF EXISTS golden_search_experiment_batch_profile_metric_snapshots")
    op.execute("DROP TABLE IF EXISTS golden_search_experiment_batch_metric_snapshots")
