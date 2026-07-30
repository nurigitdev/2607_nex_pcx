"""Align DGX vLLM runtime defaults to Qwen3.5 122B.

Revision ID: 20260730_0043
Revises: 20260729_0042
Create Date: 2026-07-30
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260730_0043"
down_revision: str | None = "20260729_0042"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        DO $$
        DECLARE
            previous_default_provider TEXT;
        BEGIN
            SELECT provider_name
            INTO previous_default_provider
            FROM generation_provider_configs
            WHERE is_default
            ORDER BY provider_config_id
            LIMIT 1;

            INSERT INTO generation_provider_configs (
                provider_name,
                provider_mode,
                provider_base_url,
                model_id,
                is_default,
                is_active,
                request_timeout_seconds,
                max_tokens,
                temperature,
                top_p,
                runtime_options,
                created_by
            )
            VALUES (
                'mock_qwen35_122b_a10b_nvfp4',
                'mock',
                NULL,
                'nvidia/Qwen3.5-122B-A10B-NVFP4',
                false,
                true,
                120,
                1024,
                0.2,
                0.9,
                jsonb_build_object(
                    'contract',
                    'openai_chat_completions',
                    'endpoint',
                    '/v1/chat/completions',
                    'deterministic',
                    true,
                    'slice',
                    415
                ),
                'slice_415_migration'
            )
            ON CONFLICT (provider_name) DO UPDATE
            SET provider_mode = EXCLUDED.provider_mode,
                provider_base_url = EXCLUDED.provider_base_url,
                model_id = EXCLUDED.model_id,
                is_active = EXCLUDED.is_active,
                request_timeout_seconds = EXCLUDED.request_timeout_seconds,
                max_tokens = EXCLUDED.max_tokens,
                temperature = EXCLUDED.temperature,
                top_p = EXCLUDED.top_p,
                runtime_options = EXCLUDED.runtime_options,
                created_by = COALESCE(
                    generation_provider_configs.created_by,
                    EXCLUDED.created_by
                ),
                updated_at = now();

            INSERT INTO generation_provider_configs (
                provider_name,
                provider_mode,
                provider_base_url,
                model_id,
                is_default,
                is_active,
                request_timeout_seconds,
                max_tokens,
                temperature,
                top_p,
                runtime_options,
                created_by
            )
            VALUES (
                'dgx_vllm_qwen35_122b_a10b_nvfp4',
                'remote_openai_compatible',
                'http://192.168.20.243:12000',
                '/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4',
                false,
                false,
                300,
                4096,
                0.2,
                0.9,
                jsonb_build_object(
                    'contract',
                    'openai_chat_completions',
                    'endpoint',
                    '/v1/chat/completions',
                    'api_key_env',
                    'NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY',
                    'extra_body',
                    jsonb_build_object(
                        'chat_template_kwargs',
                        jsonb_build_object('enable_thinking', false)
                    ),
                    'serving_max_model_len',
                    '200k',
                    'min_max_tokens',
                    4096,
                    'long_form_max_tokens',
                    8192,
                    'long_form_document_types',
                    jsonb_build_array('proposal', 'report'),
                    'secret_storage',
                    'environment_variable_only',
                    'slice',
                    415
                ),
                'slice_415_migration'
            )
            ON CONFLICT (provider_name) DO UPDATE
            SET provider_mode = EXCLUDED.provider_mode,
                provider_base_url = EXCLUDED.provider_base_url,
                model_id = EXCLUDED.model_id,
                is_active = EXCLUDED.is_active,
                request_timeout_seconds = EXCLUDED.request_timeout_seconds,
                max_tokens = EXCLUDED.max_tokens,
                temperature = EXCLUDED.temperature,
                top_p = EXCLUDED.top_p,
                runtime_options = EXCLUDED.runtime_options,
                created_by = COALESCE(
                    generation_provider_configs.created_by,
                    EXCLUDED.created_by
                ),
                updated_at = now();

            UPDATE generation_provider_configs
            SET is_default = false,
                is_active = false,
                updated_at = now()
            WHERE provider_name IN (
                'mock_qwen36_27b_nvfp4',
                'dgx_vllm_qwen36_27b_nvfp4'
            );

            IF previous_default_provider = 'mock_qwen36_27b_nvfp4' THEN
                UPDATE generation_provider_configs
                SET is_default = true,
                    is_active = true,
                    updated_at = now()
                WHERE provider_name = 'mock_qwen35_122b_a10b_nvfp4';
            ELSIF previous_default_provider = 'dgx_vllm_qwen36_27b_nvfp4' THEN
                UPDATE generation_provider_configs
                SET is_default = true,
                    is_active = true,
                    updated_at = now()
                WHERE provider_name = 'dgx_vllm_qwen35_122b_a10b_nvfp4';
            END IF;
        END $$;
        """)


def downgrade() -> None:
    op.execute("""
        DO $$
        DECLARE
            current_default_provider TEXT;
        BEGIN
            SELECT provider_name
            INTO current_default_provider
            FROM generation_provider_configs
            WHERE is_default
            ORDER BY provider_config_id
            LIMIT 1;

            UPDATE generation_provider_configs
            SET is_default = false,
                is_active = false,
                updated_at = now()
            WHERE provider_name IN (
                'mock_qwen35_122b_a10b_nvfp4',
                'dgx_vllm_qwen35_122b_a10b_nvfp4'
            );

            IF current_default_provider = 'mock_qwen35_122b_a10b_nvfp4' THEN
                UPDATE generation_provider_configs
                SET is_default = true,
                    is_active = true,
                    updated_at = now()
                WHERE provider_name = 'mock_qwen36_27b_nvfp4';
            ELSIF current_default_provider = 'dgx_vllm_qwen35_122b_a10b_nvfp4' THEN
                UPDATE generation_provider_configs
                SET is_default = true,
                    is_active = true,
                    updated_at = now()
                WHERE provider_name = 'dgx_vllm_qwen36_27b_nvfp4';
            END IF;
        END $$;
        """)
