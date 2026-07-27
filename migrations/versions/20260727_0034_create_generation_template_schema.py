"""Create generation template schema.

Revision ID: 20260727_0034
Revises: 20260726_0033
Create Date: 2026-07-27
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_0034"
down_revision: str | None = "20260726_0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE generation_templates (
            generation_template_id BIGSERIAL PRIMARY KEY,
            template_key TEXT NOT NULL UNIQUE
                CHECK (length(btrim(template_key)) > 0),
            template_name TEXT NOT NULL CHECK (length(btrim(template_name)) > 0),
            template_version TEXT NOT NULL DEFAULT 'v1'
                CHECK (length(btrim(template_version)) > 0),
            document_type TEXT NOT NULL
                CHECK (
                    document_type IN (
                        'grounded_answer',
                        'report',
                        'proposal',
                        'summary',
                        'meeting_minutes'
                    )
                ),
            language TEXT NOT NULL DEFAULT 'ko'
                CHECK (length(btrim(language)) > 0),
            output_format TEXT NOT NULL DEFAULT 'markdown'
                CHECK (output_format IN ('markdown')),
            section_schema JSONB NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(section_schema) = 'array'),
            system_instruction TEXT NOT NULL
                CHECK (length(btrim(system_instruction)) > 0),
            user_instruction_suffix TEXT NOT NULL DEFAULT '',
            style_guidance JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(style_guidance) = 'object'),
            citation_policy JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(citation_policy) = 'object'),
            is_default BOOLEAN NOT NULL DEFAULT false,
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_by TEXT,
            created_by_user_id BIGINT REFERENCES app_users(user_id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (NOT is_default OR is_active)
        )
        """)
    op.execute("""
        CREATE UNIQUE INDEX idx_generation_templates_default_language
        ON generation_templates (language, is_default)
        WHERE is_default
        """)
    op.execute("""
        CREATE INDEX idx_generation_templates_active_type
        ON generation_templates (is_active, document_type, template_key)
        """)

    op.execute("""
        INSERT INTO generation_templates (
            template_key,
            template_name,
            template_version,
            document_type,
            language,
            output_format,
            section_schema,
            system_instruction,
            user_instruction_suffix,
            style_guidance,
            citation_policy,
            is_default,
            is_active
        )
        VALUES
            (
                'grounded_answer',
                '근거 기반 답변',
                'v1',
                'grounded_answer',
                'ko',
                'markdown',
                '[
                    {"key": "answer", "heading": "답변", "required": true},
                    {"key": "evidence", "heading": "근거", "required": true},
                    {"key": "limits", "heading": "한계", "required": false}
                ]'::jsonb,
                '검색 근거에 기반해 간결하고 검증 가능한 답변을 작성한다.',
                '답변에는 관련 citation key를 포함하고, 근거가 부족하면 부족하다고 명시한다.',
                '{"tone": "concise", "audience": "internal", "density": "balanced"}'::jsonb,
                '{"required": true, "placement": "inline_or_bullet", "minimum_citations": 1}'::jsonb,
                true,
                true
            ),
            (
                'report',
                '보고서 초안',
                'v1',
                'report',
                'ko',
                'markdown',
                '[
                    {"key": "title", "heading": "제목", "required": true},
                    {"key": "overview", "heading": "요약", "required": true},
                    {"key": "background", "heading": "배경", "required": false},
                    {"key": "findings", "heading": "주요 내용", "required": true},
                    {"key": "evidence", "heading": "근거", "required": true},
                    {"key": "risks", "heading": "리스크", "required": false},
                    {"key": "next_steps", "heading": "후속 조치", "required": false}
                ]'::jsonb,
                '검색 근거를 바탕으로 내부 보고서 형식의 Markdown 초안을 작성한다.',
                '각 주요 주장에는 citation key를 붙이고, 확인되지 않은 내용은 추정으로 작성하지 않는다.',
                '{"tone": "formal", "audience": "manager", "density": "detailed"}'::jsonb,
                '{"required": true, "placement": "per_section", "minimum_citations": 2}'::jsonb,
                false,
                true
            ),
            (
                'proposal',
                '제안서 초안',
                'v1',
                'proposal',
                'ko',
                'markdown',
                '[
                    {"key": "purpose", "heading": "제안 목적", "required": true},
                    {"key": "current_state", "heading": "현황", "required": true},
                    {"key": "recommendation", "heading": "제안 내용", "required": true},
                    {"key": "plan", "heading": "추진 방안", "required": false},
                    {"key": "impact", "heading": "기대 효과", "required": false},
                    {"key": "evidence", "heading": "근거", "required": true}
                ]'::jsonb,
                '검색 근거를 바탕으로 실행 가능한 제안서 형식의 Markdown 초안을 작성한다.',
                '근거가 없는 효과나 일정은 단정하지 말고 확인 필요 항목으로 분리한다.',
                '{"tone": "persuasive", "audience": "decision_maker", "density": "balanced"}'::jsonb,
                '{"required": true, "placement": "per_section", "minimum_citations": 2}'::jsonb,
                false,
                true
            ),
            (
                'summary',
                '요약문',
                'v1',
                'summary',
                'ko',
                'markdown',
                '[
                    {"key": "key_points", "heading": "핵심 요약", "required": true},
                    {"key": "details", "heading": "상세 요약", "required": false},
                    {"key": "evidence", "heading": "참고 근거", "required": true}
                ]'::jsonb,
                '검색 근거를 바탕으로 핵심만 빠르게 파악할 수 있는 요약문을 작성한다.',
                '중복 설명을 줄이고 중요한 근거 citation을 우선 포함한다.',
                '{"tone": "concise", "audience": "working_level", "density": "compact"}'::jsonb,
                '{"required": true, "placement": "bullet", "minimum_citations": 1}'::jsonb,
                false,
                true
            ),
            (
                'meeting_minutes',
                '회의록 초안',
                'v1',
                'meeting_minutes',
                'ko',
                'markdown',
                '[
                    {"key": "agenda", "heading": "안건", "required": true},
                    {"key": "discussion", "heading": "논의 내용", "required": true},
                    {"key": "decisions", "heading": "결정 사항", "required": false},
                    {"key": "actions", "heading": "액션 아이템", "required": false},
                    {"key": "evidence", "heading": "근거", "required": true}
                ]'::jsonb,
                '검색 근거를 바탕으로 회의록 형식의 Markdown 초안을 작성한다.',
                '결정 사항과 액션 아이템은 근거가 확인될 때만 작성하고 불명확하면 확인 필요로 표시한다.',
                '{"tone": "neutral", "audience": "team", "density": "balanced"}'::jsonb,
                '{"required": true, "placement": "section_footer", "minimum_citations": 1}'::jsonb,
                false,
                true
            )
        ON CONFLICT (template_key) DO UPDATE
        SET
            template_name = EXCLUDED.template_name,
            template_version = EXCLUDED.template_version,
            document_type = EXCLUDED.document_type,
            language = EXCLUDED.language,
            output_format = EXCLUDED.output_format,
            section_schema = EXCLUDED.section_schema,
            system_instruction = EXCLUDED.system_instruction,
            user_instruction_suffix = EXCLUDED.user_instruction_suffix,
            style_guidance = EXCLUDED.style_guidance,
            citation_policy = EXCLUDED.citation_policy,
            is_default = EXCLUDED.is_default,
            is_active = EXCLUDED.is_active,
            updated_at = now()
        """)

    op.execute("""
        ALTER TABLE generation_runs
        ADD COLUMN generation_template_id BIGINT
            REFERENCES generation_templates(generation_template_id) ON DELETE SET NULL
        """)
    op.execute("""
        CREATE INDEX idx_generation_runs_template_time
        ON generation_runs (generation_template_id, created_at DESC)
        WHERE generation_template_id IS NOT NULL
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_generation_runs_template_time")
    op.execute("""
        ALTER TABLE generation_runs
        DROP COLUMN IF EXISTS generation_template_id
        """)
    op.execute("DROP TABLE IF EXISTS generation_templates")
