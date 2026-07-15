"""Seed local extraction profiles.

Revision ID: 20260715_0028
Revises: 20260715_0027
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260715_0028"
down_revision: str | None = "20260715_0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LOCAL_EXTRACTION_PROFILE_NAMES = (
    "local_markdown_default",
    "local_plain_text_default",
    "local_pdf_text_default",
    "local_docx_default",
    "local_pptx_default",
    "local_xlsx_default",
    "local_hwpx_default",
)


def upgrade() -> None:
    op.execute("""
        INSERT INTO extraction_profiles (
            extraction_profile_name,
            extractor_name,
            extractor_version,
            provider_mode,
            supported_file_types,
            default_options,
            is_active
        )
        VALUES
            (
                'local_markdown_default',
                'local_markdown',
                '0.1.0',
                'local',
                ARRAY['md'],
                '{"normalize_line_endings": true, "preserve_code_blocks": true}'::jsonb,
                true
            ),
            (
                'local_plain_text_default',
                'local_plain_text',
                '0.1.0',
                'local',
                ARRAY['txt', 'text'],
                '{"normalize_line_endings": true}'::jsonb,
                true
            ),
            (
                'local_pdf_text_default',
                'local_pdf_text',
                '0.1.0',
                'local',
                ARRAY['pdf'],
                '{"text_layer_only": true, "ocr_enabled": false}'::jsonb,
                true
            ),
            (
                'local_docx_default',
                'local_docx',
                '0.1.0',
                'local',
                ARRAY['docx'],
                '{"preserve_headings": true, "preserve_tables": true}'::jsonb,
                true
            ),
            (
                'local_pptx_default',
                'local_pptx',
                '0.1.0',
                'local',
                ARRAY['pptx'],
                '{"preserve_slide_boundaries": true, "preserve_tables": true}'::jsonb,
                true
            ),
            (
                'local_xlsx_default',
                'local_xlsx',
                '0.1.0',
                'local',
                ARRAY['xlsx'],
                '{"preserve_sheet_boundaries": true, "emit_markdown_tables": true}'::jsonb,
                true
            ),
            (
                'local_hwpx_default',
                'local_hwpx',
                '0.1.0',
                'local',
                ARRAY['hwpx'],
                '{"preserve_sections": true, "preserve_tables": true}'::jsonb,
                true
            )
        ON CONFLICT (extraction_profile_name) DO UPDATE
        SET extractor_name = EXCLUDED.extractor_name,
            extractor_version = EXCLUDED.extractor_version,
            provider_mode = EXCLUDED.provider_mode,
            supported_file_types = EXCLUDED.supported_file_types,
            default_options = EXCLUDED.default_options,
            is_active = EXCLUDED.is_active,
            updated_at = now()
        """)


def downgrade() -> None:
    profile_names = ", ".join(
        f"'{profile_name}'" for profile_name in LOCAL_EXTRACTION_PROFILE_NAMES
    )
    op.execute(f"""
        DELETE FROM extraction_profiles
        WHERE extraction_profile_name IN ({profile_names})
        """)
