"""Create BM25 search profile and keyword index schema.

Revision ID: 20260720_0030
Revises: 20260715_0029
Create Date: 2026-07-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260720_0030"
down_revision: str | None = "20260715_0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE search_profiles (
            search_profile_name TEXT PRIMARY KEY
                CHECK (length(btrim(search_profile_name)) > 0),
            profile_kind TEXT NOT NULL
                CHECK (profile_kind IN ('embedding', 'keyword', 'hybrid')),
            embedding_profile_name TEXT REFERENCES embedding_profiles(profile_name),
            strategy_name TEXT NOT NULL CHECK (length(btrim(strategy_name)) > 0),
            display_name TEXT NOT NULL CHECK (length(btrim(display_name)) > 0),
            is_active BOOLEAN NOT NULL DEFAULT true,
            runtime_parameters JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_parameters) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (
                (
                    profile_kind = 'embedding'
                    AND embedding_profile_name IS NOT NULL
                )
                OR (
                    profile_kind <> 'embedding'
                    AND embedding_profile_name IS NULL
                )
            )
        )
        """)
    op.execute("""
        CREATE INDEX idx_search_profiles_kind_active
        ON search_profiles (profile_kind, is_active, search_profile_name)
        """)
    op.execute("""
        CREATE INDEX idx_search_profiles_embedding_profile
        ON search_profiles (embedding_profile_name)
        WHERE embedding_profile_name IS NOT NULL
        """)
    op.execute("""
        INSERT INTO search_profiles (
            search_profile_name,
            profile_kind,
            embedding_profile_name,
            strategy_name,
            display_name,
            is_active,
            runtime_parameters
        )
        SELECT
            profile_name,
            'embedding',
            profile_name,
            'vector_cosine',
            profile_name,
            is_active,
            jsonb_build_object(
                'source', 'embedding_profiles',
                'similarity_metric', 'cosine',
                'storage_type', storage_type,
                'dimension', dimension
            )
        FROM embedding_profiles
        ON CONFLICT (search_profile_name) DO NOTHING
        """)
    op.execute("""
        INSERT INTO search_profiles (
            search_profile_name,
            profile_kind,
            embedding_profile_name,
            strategy_name,
            display_name,
            is_active,
            runtime_parameters
        )
        VALUES
            (
                'bm25_keyword',
                'keyword',
                NULL,
                'bm25_keyword',
                'BM25 Keyword',
                false,
                '{
                    "planned": true,
                    "index_source": "chunks.chunk_text",
                    "tokenizer": "unicode_word_v1",
                    "scoring": "okapi_bm25",
                    "k1": 1.2,
                    "b": 0.75
                }'::jsonb
            ),
            (
                'hybrid_keyword_vector',
                'hybrid',
                NULL,
                'hybrid_keyword_vector',
                'Hybrid Keyword + Vector',
                false,
                '{
                    "planned": true,
                    "keyword_strategy": "bm25_keyword",
                    "vector_strategy": "vector_cosine",
                    "fusion": "rrf",
                    "rrf_k": 60
                }'::jsonb
            )
        ON CONFLICT (search_profile_name) DO NOTHING
        """)

    op.execute("""
        CREATE TABLE chunk_keyword_terms (
            chunk_id BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            chunk_policy_name TEXT NOT NULL REFERENCES chunk_policies(chunk_policy_name),
            tokenizer_name TEXT NOT NULL DEFAULT 'unicode_word_v1'
                CHECK (length(btrim(tokenizer_name)) > 0),
            term TEXT NOT NULL CHECK (length(btrim(term)) > 0),
            term_frequency INT NOT NULL CHECK (term_frequency > 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (chunk_id, tokenizer_name, term)
        )
        """)
    op.execute("""
        CREATE INDEX idx_chunk_keyword_terms_policy_term
        ON chunk_keyword_terms (chunk_policy_name, tokenizer_name, term)
        """)

    op.execute("""
        CREATE TABLE chunk_keyword_statistics (
            chunk_policy_name TEXT NOT NULL REFERENCES chunk_policies(chunk_policy_name),
            tokenizer_name TEXT NOT NULL DEFAULT 'unicode_word_v1'
                CHECK (length(btrim(tokenizer_name)) > 0),
            term TEXT NOT NULL CHECK (length(btrim(term)) > 0),
            document_frequency INT NOT NULL CHECK (document_frequency >= 0),
            corpus_chunk_count INT NOT NULL CHECK (corpus_chunk_count >= 0),
            average_document_length NUMERIC(12, 4) NOT NULL
                CHECK (average_document_length >= 0),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (chunk_policy_name, tokenizer_name, term)
        )
        """)
    op.execute("""
        CREATE INDEX idx_chunk_keyword_statistics_updated
        ON chunk_keyword_statistics (updated_at DESC)
        """)

    op.execute("""
        ALTER TABLE search_logs
        ADD COLUMN strategy_name TEXT NOT NULL DEFAULT 'vector_cosine'
            CHECK (length(btrim(strategy_name)) > 0)
        """)
    op.execute("""
        CREATE INDEX idx_search_logs_strategy
        ON search_logs (strategy_name, created_at DESC)
        """)
    op.execute("""
        ALTER TABLE search_logs
        DROP CONSTRAINT IF EXISTS search_logs_similarity_metric_check
        """)
    op.execute("""
        ALTER TABLE search_logs
        ADD CONSTRAINT search_logs_similarity_metric_check
        CHECK (similarity_metric IN ('cosine', 'l2', 'inner_product', 'bm25'))
        """)

    op.execute("""
        ALTER TABLE search_log_results
        DROP CONSTRAINT IF EXISTS search_log_results_profile_name_fkey
        """)
    op.execute("""
        ALTER TABLE search_log_results
        ADD COLUMN search_profile_name TEXT
            REFERENCES search_profiles(search_profile_name),
        ADD COLUMN retrieval_strategy TEXT
            CHECK (
                retrieval_strategy IS NULL
                OR length(btrim(retrieval_strategy)) > 0
            ),
        ADD COLUMN score_components JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(score_components) = 'object')
        """)
    op.execute("""
        UPDATE search_log_results
        SET search_profile_name = profile_name,
            retrieval_strategy = 'vector_cosine'
        WHERE profile_name IN (
            SELECT search_profile_name FROM search_profiles
        )
        """)
    op.execute("""
        CREATE INDEX idx_search_log_results_search_profile
        ON search_log_results (search_profile_name, rank)
        """)
    op.execute("""
        CREATE INDEX idx_search_log_results_retrieval_strategy
        ON search_log_results (retrieval_strategy, rank)
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_search_log_results_retrieval_strategy")
    op.execute("DROP INDEX IF EXISTS idx_search_log_results_search_profile")
    op.execute("""
        ALTER TABLE search_log_results
        DROP COLUMN IF EXISTS score_components,
        DROP COLUMN IF EXISTS retrieval_strategy,
        DROP COLUMN IF EXISTS search_profile_name
        """)
    op.execute("""
        DELETE FROM search_log_results
        WHERE profile_name NOT IN (
            SELECT profile_name FROM embedding_profiles
        )
        """)
    op.execute("""
        ALTER TABLE search_log_results
        ADD CONSTRAINT search_log_results_profile_name_fkey
        FOREIGN KEY (profile_name) REFERENCES embedding_profiles(profile_name)
        """)

    op.execute("DROP INDEX IF EXISTS idx_search_logs_strategy")
    op.execute("""
        ALTER TABLE search_logs
        DROP CONSTRAINT IF EXISTS search_logs_similarity_metric_check
        """)
    op.execute("""
        UPDATE search_logs
        SET similarity_metric = 'cosine'
        WHERE similarity_metric = 'bm25'
        """)
    op.execute("""
        ALTER TABLE search_logs
        ADD CONSTRAINT search_logs_similarity_metric_check
        CHECK (similarity_metric IN ('cosine', 'l2', 'inner_product'))
        """)
    op.execute("ALTER TABLE search_logs DROP COLUMN IF EXISTS strategy_name")

    op.execute("DROP TABLE IF EXISTS chunk_keyword_statistics")
    op.execute("DROP TABLE IF EXISTS chunk_keyword_terms")
    op.execute("DROP TABLE IF EXISTS search_profiles")
