"""Seed document summary generation template presets.

Revision ID: 20260728_0038
Revises: 20260728_0037
Create Date: 2026-07-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260728_0038"
down_revision: str | None = "20260728_0037"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SUMMARY_TEMPLATE_KEYS = (
    "summary_executive",
    "summary_risk_action",
    "summary_working",
)


def upgrade() -> None:
    op.execute("""
        INSERT INTO generation_templates (
            template_key,
            template_family,
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
            is_active,
            change_note,
            created_by
        )
        VALUES
            (
                'summary_executive',
                'summary',
                '임원 보고 요약',
                'v1',
                'summary',
                'ko',
                'markdown',
                '[
                    {"key": "executive_summary", "heading": "임원 요약", "required": true},
                    {"key": "key_decisions", "heading": "주요 판단 포인트", "required": true},
                    {"key": "risks", "heading": "리스크", "required": true},
                    {"key": "next_steps", "heading": "후속 조치", "required": true},
                    {"key": "evidence", "heading": "참고 근거", "required": true}
                ]'::jsonb,
                '검색 근거를 바탕으로 임원 보고용 요약을 작성한다.',
                '판단 포인트에는 citation key를 포함하고, 근거 부족 내용은 확인 필요로 분리한다.',
                '{"tone": "formal", "audience": "decision_maker", "density": "compact"}'::jsonb,
                '{"required": true, "placement": "per_section", "minimum_citations": 2}'::jsonb,
                false,
                true,
                'Slice 380 summary preset seed',
                'migration_slice_380'
            ),
            (
                'summary_risk_action',
                'summary',
                '리스크/후속 조치 요약',
                'v1',
                'summary',
                'ko',
                'markdown',
                '[
                    {"key": "key_points", "heading": "핵심 요약", "required": true},
                    {"key": "risks", "heading": "리스크", "required": true},
                    {"key": "actions", "heading": "조치 계획", "required": true},
                    {"key": "open_questions", "heading": "확인 필요", "required": true},
                    {"key": "evidence", "heading": "참고 근거", "required": true}
                ]'::jsonb,
                '검색 근거를 바탕으로 리스크와 후속 조치를 중심으로 실행 가능한 요약을 작성한다.',
                '담당자나 일정이 근거에 없으면 임의로 만들지 말고 확인 필요 항목으로 표시한다.',
                '{"tone": "action", "audience": "team_lead", "density": "balanced"}'::jsonb,
                '{"required": true, "placement": "per_section", "minimum_citations": 2}'::jsonb,
                false,
                true,
                'Slice 380 summary preset seed',
                'migration_slice_380'
            ),
            (
                'summary_working',
                'summary',
                '실무 검토 요약',
                'v1',
                'summary',
                'ko',
                'markdown',
                '[
                    {"key": "key_points", "heading": "핵심 요약", "required": true},
                    {"key": "details", "heading": "상세 요약", "required": true},
                    {"key": "implications", "heading": "실무 영향", "required": true},
                    {"key": "open_questions", "heading": "확인 필요", "required": true},
                    {"key": "evidence", "heading": "참고 근거", "required": true}
                ]'::jsonb,
                '검색 근거를 바탕으로 실무 검토용 요약을 작성한다.',
                '중요한 수치, 조건, 제약 사항은 누락하지 말고 citation key와 함께 정리한다.',
                '{"tone": "neutral", "audience": "working_level", "density": "detailed"}'::jsonb,
                '{"required": true, "placement": "per_section", "minimum_citations": 2}'::jsonb,
                false,
                true,
                'Slice 380 summary preset seed',
                'migration_slice_380'
            )
        ON CONFLICT (template_key) DO UPDATE
        SET
            template_family = EXCLUDED.template_family,
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
            change_note = EXCLUDED.change_note,
            created_by = COALESCE(generation_templates.created_by, EXCLUDED.created_by),
            updated_at = now()
        """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM generation_templates
        WHERE template_key IN (
            'summary_executive',
            'summary_risk_action',
            'summary_working'
        )
          AND created_by = 'migration_slice_380'
        """)
