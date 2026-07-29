"""Add vLLM runtime readiness threshold settings.

Revision ID: 20260729_0041
Revises: 20260729_0040
Create Date: 2026-07-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260729_0041"
down_revision: str | None = "20260729_0040"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO app_log_settings
            (setting_name, setting_value, value_type, description)
        VALUES
            (
                'vllm_runtime_stale_snapshot_warning_minutes',
                '10',
                'text',
                'vLLM runtime snapshot age in minutes that triggers a warning'
            ),
            (
                'vllm_runtime_stale_snapshot_critical_minutes',
                '30',
                'text',
                'vLLM runtime snapshot age in minutes that triggers a critical signal'
            ),
            (
                'vllm_runtime_kv_cache_warning_percent',
                '80',
                'text',
                'vLLM KV cache usage percent that triggers a warning'
            ),
            (
                'vllm_runtime_kv_cache_critical_percent',
                '90',
                'text',
                'vLLM KV cache usage percent that triggers a critical signal'
            ),
            (
                'vllm_runtime_waiting_requests_warning',
                '1',
                'int',
                'Waiting request count that triggers a vLLM runtime warning'
            ),
            (
                'vllm_runtime_waiting_requests_critical',
                '5',
                'int',
                'Waiting request count that triggers a vLLM runtime critical signal'
            ),
            (
                'vllm_runtime_swapped_requests_warning',
                '1',
                'int',
                'Swapped request count that triggers a vLLM runtime warning'
            ),
            (
                'vllm_runtime_swapped_requests_critical',
                '3',
                'int',
                'Swapped request count that triggers a vLLM runtime critical signal'
            ),
            (
                'vllm_runtime_preemptions_warning_total',
                '1',
                'int',
                'Preemption total that triggers a vLLM runtime warning'
            ),
            (
                'vllm_runtime_preemptions_critical_total',
                '10',
                'int',
                'Preemption total that triggers a vLLM runtime critical signal'
            ),
            (
                'vllm_runtime_ttft_warning_seconds',
                '2',
                'text',
                'Average time to first token in seconds that triggers a warning'
            ),
            (
                'vllm_runtime_ttft_critical_seconds',
                '5',
                'text',
                'Average time to first token in seconds that triggers a critical signal'
            ),
            (
                'vllm_runtime_e2e_latency_warning_seconds',
                '30',
                'text',
                'Average end-to-end request latency in seconds that triggers a warning'
            ),
            (
                'vllm_runtime_e2e_latency_critical_seconds',
                '60',
                'text',
                'Average end-to-end request latency in seconds that triggers a critical signal'
            )
        ON CONFLICT (setting_name) DO NOTHING
        """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM app_log_settings
        WHERE setting_name IN (
            'vllm_runtime_stale_snapshot_warning_minutes',
            'vllm_runtime_stale_snapshot_critical_minutes',
            'vllm_runtime_kv_cache_warning_percent',
            'vllm_runtime_kv_cache_critical_percent',
            'vllm_runtime_waiting_requests_warning',
            'vllm_runtime_waiting_requests_critical',
            'vllm_runtime_swapped_requests_warning',
            'vllm_runtime_swapped_requests_critical',
            'vllm_runtime_preemptions_warning_total',
            'vllm_runtime_preemptions_critical_total',
            'vllm_runtime_ttft_warning_seconds',
            'vllm_runtime_ttft_critical_seconds',
            'vllm_runtime_e2e_latency_warning_seconds',
            'vllm_runtime_e2e_latency_critical_seconds'
        )
        """)
