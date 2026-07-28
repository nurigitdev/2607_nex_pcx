"""Improve DGX generation token budget and template required sections."""

from alembic import op

revision = "20260728_0037"
down_revision = "20260728_0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        UPDATE generation_provider_configs
        SET max_tokens = GREATEST(max_tokens, 4096),
            runtime_options = runtime_options
                || jsonb_build_object(
                    'min_max_tokens',
                    4096,
                    'long_form_max_tokens',
                    8192,
                    'long_form_document_types',
                    jsonb_build_array('proposal', 'report'),
                    'slice',
                    372
                ),
            updated_at = now()
        WHERE provider_name = 'dgx_vllm_qwen36_27b_nvfp4'
          AND provider_mode = 'remote_openai_compatible'
        """)
    op.execute("""
        UPDATE generation_templates AS gt
        SET section_schema = (
                SELECT jsonb_agg(
                    jsonb_set(section, '{required}', 'true'::jsonb, true)
                    ORDER BY ord
                )
                FROM jsonb_array_elements(gt.section_schema) WITH ORDINALITY AS item(section, ord)
            ),
            updated_at = now()
        WHERE template_key IN ('report', 'proposal')
        """)


def downgrade() -> None:
    op.execute("""
        UPDATE generation_templates AS gt
        SET section_schema = (
                SELECT jsonb_agg(
                    jsonb_set(
                        section,
                        '{required}',
                        CASE
                            WHEN gt.template_key = 'report'
                             AND section ->> 'key' IN ('background', 'risks', 'next_steps')
                                THEN 'false'::jsonb
                            WHEN gt.template_key = 'proposal'
                             AND section ->> 'key' IN ('plan', 'impact')
                                THEN 'false'::jsonb
                            ELSE 'true'::jsonb
                        END,
                        true
                    )
                    ORDER BY ord
                )
                FROM jsonb_array_elements(gt.section_schema) WITH ORDINALITY AS item(section, ord)
            ),
            updated_at = now()
        WHERE template_key IN ('report', 'proposal')
        """)
    op.execute("""
        UPDATE generation_provider_configs
        SET max_tokens = LEAST(max_tokens, 1024),
            runtime_options = runtime_options
                - 'min_max_tokens'
                - 'long_form_max_tokens'
                - 'long_form_document_types',
            updated_at = now()
        WHERE provider_name = 'dgx_vllm_qwen36_27b_nvfp4'
          AND provider_mode = 'remote_openai_compatible'
        """)
