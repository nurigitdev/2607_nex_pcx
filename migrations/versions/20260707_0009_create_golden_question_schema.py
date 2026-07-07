"""Create golden question evaluation schema.

Revision ID: 20260707_0009
Revises: 20260707_0008
Create Date: 2026-07-07
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260707_0009"
down_revision: str | None = "20260707_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE golden_question_sets (
            question_set_id BIGSERIAL PRIMARY KEY,
            set_name TEXT NOT NULL UNIQUE CHECK (length(btrim(set_name)) > 0),
            description TEXT,
            is_active BOOLEAN NOT NULL DEFAULT true,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_by_user_id BIGINT REFERENCES app_users(user_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_question_sets_active
        ON golden_question_sets (is_active, set_name)
        """)

    op.execute("""
        CREATE TABLE golden_questions (
            question_id BIGSERIAL PRIMARY KEY,
            question_set_id BIGINT NOT NULL
                REFERENCES golden_question_sets(question_set_id) ON DELETE CASCADE,
            question_text TEXT NOT NULL CHECK (length(btrim(question_text)) > 0),
            normalized_question_text TEXT,
            question_type TEXT NOT NULL DEFAULT 'single_fact'
                CHECK (
                    question_type IN (
                        'single_fact',
                        'section',
                        'comparison',
                        'no_answer',
                        'table_figure'
                    )
                ),
            actor_user_id BIGINT REFERENCES app_users(user_id),
            requested_search_scope TEXT
                CHECK (
                    requested_search_scope IS NULL
                    OR requested_search_scope IN ('mine', 'team', 'managed_org', 'company')
                ),
            document_group TEXT,
            file_type TEXT,
            chunk_policy_name TEXT REFERENCES chunk_policies(chunk_policy_name),
            top_k INT NOT NULL DEFAULT 5 CHECK (top_k > 0),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_by_user_id BIGINT REFERENCES app_users(user_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (
                question_set_id,
                question_text,
                actor_user_id,
                requested_search_scope
            )
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_questions_set
        ON golden_questions (question_set_id, question_id)
        """)
    op.execute("""
        CREATE INDEX idx_golden_questions_actor_scope
        ON golden_questions (actor_user_id, requested_search_scope)
        """)
    op.execute("""
        CREATE INDEX idx_golden_questions_chunk_policy
        ON golden_questions (chunk_policy_name)
        """)

    op.execute("""
        CREATE TABLE golden_question_expected_targets (
            expected_target_id BIGSERIAL PRIMARY KEY,
            question_id BIGINT NOT NULL
                REFERENCES golden_questions(question_id) ON DELETE CASCADE,
            chunk_id BIGINT REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            expected_heading_path TEXT[],
            expectation_type TEXT NOT NULL DEFAULT 'visible'
                CHECK (expectation_type IN ('visible', 'hidden')),
            relevance_grade INT NOT NULL DEFAULT 3
                CHECK (relevance_grade BETWEEN 0 AND 3),
            notes TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (chunk_id IS NOT NULL OR expected_heading_path IS NOT NULL),
            UNIQUE (question_id, chunk_id, expectation_type),
            UNIQUE (question_id, expected_heading_path, expectation_type)
        )
        """)
    op.execute("""
        CREATE INDEX idx_golden_question_expected_targets_question
        ON golden_question_expected_targets (question_id)
        """)
    op.execute("""
        CREATE INDEX idx_golden_question_expected_targets_chunk
        ON golden_question_expected_targets (chunk_id)
        """)
    op.execute("""
        CREATE INDEX idx_golden_question_expected_targets_type
        ON golden_question_expected_targets (expectation_type)
        """)
    op.execute("""
        CREATE INDEX idx_golden_question_expected_targets_heading_path
        ON golden_question_expected_targets USING GIN (expected_heading_path)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS golden_question_expected_targets")
    op.execute("DROP TABLE IF EXISTS golden_questions")
    op.execute("DROP TABLE IF EXISTS golden_question_sets")
