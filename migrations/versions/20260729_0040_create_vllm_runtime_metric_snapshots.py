"""Create vLLM runtime metric snapshots.

Revision ID: 20260729_0040
Revises: 20260728_0039
Create Date: 2026-07-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260729_0040"
down_revision: str | None = "20260728_0039"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE vllm_runtime_metric_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            provider_name TEXT NOT NULL CHECK (length(btrim(provider_name)) > 0),
            provider_base_url TEXT NOT NULL CHECK (length(btrim(provider_base_url)) > 0),
            model_id TEXT,
            sampled_at TIMESTAMPTZ NOT NULL,
            scrape_elapsed_ms INTEGER
                CHECK (scrape_elapsed_ms IS NULL OR scrape_elapsed_ms >= 0),
            raw_text_bytes INTEGER NOT NULL DEFAULT 0 CHECK (raw_text_bytes >= 0),
            metric_count INTEGER NOT NULL DEFAULT 0 CHECK (metric_count >= 0),
            vllm_metric_count INTEGER NOT NULL DEFAULT 0 CHECK (vllm_metric_count >= 0),
            metric_names JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(metric_names) = 'array'),
            kv_cache_usage_ratio DOUBLE PRECISION
                CHECK (kv_cache_usage_ratio IS NULL OR kv_cache_usage_ratio >= 0),
            kv_cache_usage_percent DOUBLE PRECISION
                CHECK (kv_cache_usage_percent IS NULL OR kv_cache_usage_percent >= 0),
            cpu_cache_usage_ratio DOUBLE PRECISION
                CHECK (cpu_cache_usage_ratio IS NULL OR cpu_cache_usage_ratio >= 0),
            cpu_cache_usage_percent DOUBLE PRECISION
                CHECK (cpu_cache_usage_percent IS NULL OR cpu_cache_usage_percent >= 0),
            running_requests INTEGER
                CHECK (running_requests IS NULL OR running_requests >= 0),
            waiting_requests INTEGER
                CHECK (waiting_requests IS NULL OR waiting_requests >= 0),
            swapped_requests INTEGER
                CHECK (swapped_requests IS NULL OR swapped_requests >= 0),
            waiting_requests_by_reason JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(waiting_requests_by_reason) = 'object'),
            request_success_total BIGINT
                CHECK (request_success_total IS NULL OR request_success_total >= 0),
            prompt_tokens_total BIGINT
                CHECK (prompt_tokens_total IS NULL OR prompt_tokens_total >= 0),
            generation_tokens_total BIGINT
                CHECK (generation_tokens_total IS NULL OR generation_tokens_total >= 0),
            prompt_tokens_cached_total BIGINT
                CHECK (prompt_tokens_cached_total IS NULL OR prompt_tokens_cached_total >= 0),
            prefix_cache_hits_total BIGINT
                CHECK (prefix_cache_hits_total IS NULL OR prefix_cache_hits_total >= 0),
            prefix_cache_queries_total BIGINT
                CHECK (prefix_cache_queries_total IS NULL OR prefix_cache_queries_total >= 0),
            prefix_cache_hit_rate DOUBLE PRECISION
                CHECK (prefix_cache_hit_rate IS NULL OR prefix_cache_hit_rate >= 0),
            num_preemptions_total BIGINT
                CHECK (num_preemptions_total IS NULL OR num_preemptions_total >= 0),
            average_time_to_first_token_seconds DOUBLE PRECISION
                CHECK (
                    average_time_to_first_token_seconds IS NULL
                    OR average_time_to_first_token_seconds >= 0
                ),
            average_inter_token_latency_seconds DOUBLE PRECISION
                CHECK (
                    average_inter_token_latency_seconds IS NULL
                    OR average_inter_token_latency_seconds >= 0
                ),
            average_e2e_request_latency_seconds DOUBLE PRECISION
                CHECK (
                    average_e2e_request_latency_seconds IS NULL
                    OR average_e2e_request_latency_seconds >= 0
                ),
            average_request_queue_time_seconds DOUBLE PRECISION
                CHECK (
                    average_request_queue_time_seconds IS NULL
                    OR average_request_queue_time_seconds >= 0
                ),
            average_request_prefill_time_seconds DOUBLE PRECISION
                CHECK (
                    average_request_prefill_time_seconds IS NULL
                    OR average_request_prefill_time_seconds >= 0
                ),
            average_request_decode_time_seconds DOUBLE PRECISION
                CHECK (
                    average_request_decode_time_seconds IS NULL
                    OR average_request_decode_time_seconds >= 0
                ),
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            raw_samples JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(raw_samples) = 'array'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_vllm_runtime_metric_snapshots_sampled_at
        ON vllm_runtime_metric_snapshots (sampled_at DESC, snapshot_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_vllm_runtime_metric_snapshots_provider_sampled
        ON vllm_runtime_metric_snapshots (
            provider_name,
            sampled_at DESC,
            snapshot_id DESC
        )
        """)
    op.execute("""
        CREATE INDEX idx_vllm_runtime_metric_snapshots_kv_usage
        ON vllm_runtime_metric_snapshots (kv_cache_usage_percent DESC)
        WHERE kv_cache_usage_percent IS NOT NULL
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_vllm_runtime_metric_snapshots_kv_usage")
    op.execute("DROP INDEX IF EXISTS idx_vllm_runtime_metric_snapshots_provider_sampled")
    op.execute("DROP INDEX IF EXISTS idx_vllm_runtime_metric_snapshots_sampled_at")
    op.execute("DROP TABLE IF EXISTS vllm_runtime_metric_snapshots")
