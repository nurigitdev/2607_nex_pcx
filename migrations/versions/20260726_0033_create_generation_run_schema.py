"""Create generation run schema.

Revision ID: 20260726_0033
Revises: 20260725_0032
Create Date: 2026-07-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260726_0033"
down_revision: str | None = "20260725_0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE generation_provider_configs (
            provider_config_id BIGSERIAL PRIMARY KEY,
            provider_name TEXT NOT NULL UNIQUE CHECK (length(btrim(provider_name)) > 0),
            provider_mode TEXT NOT NULL
                CHECK (provider_mode IN ('mock', 'remote_openai_compatible')),
            provider_base_url TEXT,
            model_id TEXT NOT NULL CHECK (length(btrim(model_id)) > 0),
            is_default BOOLEAN NOT NULL DEFAULT false,
            is_active BOOLEAN NOT NULL DEFAULT true,
            request_timeout_seconds INT NOT NULL DEFAULT 120
                CHECK (request_timeout_seconds > 0),
            max_tokens INT NOT NULL DEFAULT 1024 CHECK (max_tokens > 0),
            temperature DOUBLE PRECISION NOT NULL DEFAULT 0.2
                CHECK (temperature >= 0 AND temperature <= 2),
            top_p DOUBLE PRECISION NOT NULL DEFAULT 0.9
                CHECK (top_p > 0 AND top_p <= 1),
            runtime_options JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_options) = 'object'),
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (
                provider_mode <> 'remote_openai_compatible'
                OR (
                    provider_base_url IS NOT NULL
                    AND length(btrim(provider_base_url)) > 0
                )
            )
        )
        """)
    op.execute("""
        CREATE UNIQUE INDEX idx_generation_provider_configs_default
        ON generation_provider_configs (is_default)
        WHERE is_default
        """)
    op.execute("""
        CREATE INDEX idx_generation_provider_configs_active
        ON generation_provider_configs (is_active, provider_mode, provider_name)
        """)
    op.execute("""
        INSERT INTO generation_provider_configs (
            provider_name,
            provider_mode,
            model_id,
            is_default,
            runtime_options
        )
        VALUES (
            'mock_qwen36_27b_nvfp4',
            'mock',
            'nvidia/Qwen3.6-27B-NVFP4',
            true,
            '{
                "contract": "openai_chat_completions",
                "endpoint": "/v1/chat/completions",
                "deterministic": true,
                "slice": 337
            }'::jsonb
        )
        ON CONFLICT (provider_name) DO NOTHING
        """)

    op.execute("""
        CREATE TABLE generation_runs (
            generation_run_id BIGSERIAL PRIMARY KEY,
            search_log_id BIGINT NOT NULL
                REFERENCES search_logs(search_log_id) ON DELETE CASCADE,
            retrieval_package_key TEXT NOT NULL CHECK (length(btrim(retrieval_package_key)) > 0),
            provider_config_id BIGINT
                REFERENCES generation_provider_configs(provider_config_id) ON DELETE SET NULL,
            provider_name TEXT NOT NULL CHECK (length(btrim(provider_name)) > 0),
            provider_mode TEXT NOT NULL
                CHECK (provider_mode IN ('mock', 'remote_openai_compatible')),
            model_id TEXT NOT NULL CHECK (length(btrim(model_id)) > 0),
            prompt_version TEXT NOT NULL DEFAULT 'grounded_answer_v1'
                CHECK (length(btrim(prompt_version)) > 0),
            prompt_hash TEXT,
            context_hash TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (
                    status IN (
                        'pending',
                        'blocked',
                        'running',
                        'succeeded',
                        'failed',
                        'canceled',
                        'no_answer'
                    )
                ),
            guardrail_status TEXT NOT NULL DEFAULT 'allowed'
                CHECK (guardrail_status IN ('allowed', 'blocked', 'no_answer')),
            retrieval_confidence_status TEXT NOT NULL
                CHECK (
                    retrieval_confidence_status IN (
                        'answerable',
                        'low_confidence',
                        'no_relevant_context',
                        'failed'
                    )
                ),
            citation_readiness_status TEXT NOT NULL
                CHECK (citation_readiness_status IN ('ready', 'warning', 'failed')),
            query_text TEXT NOT NULL CHECK (length(btrim(query_text)) > 0),
            answer_text TEXT,
            finish_reason TEXT,
            input_token_count INT CHECK (
                input_token_count IS NULL OR input_token_count >= 0
            ),
            output_token_count INT CHECK (
                output_token_count IS NULL OR output_token_count >= 0
            ),
            total_token_count INT CHECK (
                total_token_count IS NULL OR total_token_count >= 0
            ),
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            request_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(request_metadata) = 'object'),
            response_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(response_metadata) = 'object'),
            guardrail_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(guardrail_metadata) = 'object'),
            error_message TEXT,
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id) ON DELETE SET NULL,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_generation_runs_search_log
        ON generation_runs (search_log_id, created_at DESC, generation_run_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_generation_runs_status_time
        ON generation_runs (status, created_at DESC, generation_run_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_generation_runs_provider_time
        ON generation_runs (provider_name, model_id, created_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_generation_runs_package_key
        ON generation_runs (retrieval_package_key)
        """)

    op.execute("""
        CREATE TABLE generation_run_citations (
            generation_run_citation_id BIGSERIAL PRIMARY KEY,
            generation_run_id BIGINT NOT NULL
                REFERENCES generation_runs(generation_run_id) ON DELETE CASCADE,
            citation_key TEXT NOT NULL CHECK (length(btrim(citation_key)) > 0),
            citation_index INT NOT NULL CHECK (citation_index > 0),
            search_log_result_id BIGINT
                REFERENCES search_log_results(search_log_result_id) ON DELETE SET NULL,
            chunk_id BIGINT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
            document_id BIGINT REFERENCES documents(document_id) ON DELETE SET NULL,
            file_id BIGINT REFERENCES files(file_id) ON DELETE SET NULL,
            source_label TEXT NOT NULL DEFAULT '',
            source_anchor JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(source_anchor) = 'object'),
            citation_payload JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(citation_payload) = 'object'),
            was_cited BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (generation_run_id, citation_key),
            UNIQUE (generation_run_id, citation_index)
        )
        """)
    op.execute("""
        CREATE INDEX idx_generation_run_citations_run
        ON generation_run_citations (generation_run_id, citation_index)
        """)
    op.execute("""
        CREATE INDEX idx_generation_run_citations_chunk
        ON generation_run_citations (chunk_id)
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS generation_run_citations")
    op.execute("DROP TABLE IF EXISTS generation_runs")
    op.execute("DROP TABLE IF EXISTS generation_provider_configs")
