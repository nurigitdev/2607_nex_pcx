"""Add generation template version lineage metadata.

Revision ID: 20260728_0035
Revises: 20260727_0034
Create Date: 2026-07-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260728_0035"
down_revision: str | None = "20260727_0034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE generation_templates
        ADD COLUMN template_family TEXT,
        ADD COLUMN clone_source_template_id BIGINT
            REFERENCES generation_templates(generation_template_id) ON DELETE SET NULL,
        ADD COLUMN change_note TEXT NOT NULL DEFAULT ''
        """)
    op.execute("""
        UPDATE generation_templates
        SET template_family = template_key
        WHERE template_family IS NULL
        """)
    op.execute("""
        ALTER TABLE generation_templates
        ADD CONSTRAINT generation_templates_family_non_empty_check
        CHECK (template_family IS NULL OR length(btrim(template_family)) > 0)
        """)
    op.execute("""
        CREATE INDEX idx_generation_templates_family_version
        ON generation_templates (template_family, language, template_version, template_key)
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_generation_templates_family_version")
    op.execute("""
        ALTER TABLE generation_templates
        DROP CONSTRAINT IF EXISTS generation_templates_family_non_empty_check
        """)
    op.execute("""
        ALTER TABLE generation_templates
        DROP COLUMN IF EXISTS change_note,
        DROP COLUMN IF EXISTS clone_source_template_id,
        DROP COLUMN IF EXISTS template_family
        """)
