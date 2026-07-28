from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one
from app.core.generation_templates import (
    GenerationTemplateCloneInput,
    GenerationTemplateInput,
    InvalidGenerationTemplateError,
    clone_generation_template_version,
    get_default_generation_template,
    get_generation_template_by_key,
    list_generation_templates,
    rollback_generation_template_version,
    set_generation_template_active,
    set_generation_template_default,
    upsert_generation_template,
)

pytestmark = pytest.mark.integration


def _create_search_log(database_url: str) -> int:
    with connect(database_url) as conn:
        row = conn.execute("""
            INSERT INTO search_logs (
                query_text,
                normalized_query_text,
                top_k,
                profiles,
                query_runtime_metadata
            )
            VALUES (
                'generation template schema smoke query',
                'generation template schema smoke query',
                3,
                '["reranked_vector_cosine"]'::jsonb,
                '{"test": "generation_template_schema"}'::jsonb
            )
            RETURNING search_log_id
            """).fetchone()
        conn.commit()
    assert row is not None
    return int(row["search_log_id"])


def _template_key(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _cleanup_generation_templates(database_url: str, *template_keys: str) -> None:
    with connect(database_url) as conn:
        conn.execute(
            """
            UPDATE generation_templates
            SET is_default = false,
                updated_at = now()
            WHERE is_default
            """,
        )
        conn.execute(
            """
            UPDATE generation_templates
            SET is_default = true,
                is_active = true,
                updated_at = now()
            WHERE template_key = 'grounded_answer'
            """,
        )
        if template_keys:
            conn.execute(
                """
                DELETE FROM generation_templates
                WHERE template_key = ANY(%s)
                """,
                (list(template_keys),),
            )
        conn.commit()


def test_generation_templates_seed_default_set(migrated_database_url: str) -> None:
    summary = fetch_one(
        migrated_database_url,
        """
        SELECT
            count(*) AS template_count,
            count(*) FILTER (WHERE is_active) AS active_count,
            count(*) FILTER (WHERE is_default) AS default_count
        FROM generation_templates
        """,
    )
    default_template = fetch_one(
        migrated_database_url,
        """
        SELECT
            template_key,
            template_family,
            template_name,
            template_version,
            document_type,
            language,
            output_format,
            section_schema,
            style_guidance,
            citation_policy
        FROM generation_templates
        WHERE is_default
        """,
    )

    assert summary["template_count"] >= 5
    assert summary["active_count"] >= 5
    assert summary["default_count"] == 1
    assert default_template["template_key"] == "grounded_answer"
    assert default_template["template_family"] == "grounded_answer"
    assert default_template["template_version"] == "v1"
    assert default_template["document_type"] == "grounded_answer"
    assert default_template["language"] == "ko"
    assert default_template["output_format"] == "markdown"
    assert default_template["section_schema"][0]["key"] == "answer"
    assert default_template["style_guidance"]["tone"] == "concise"
    assert default_template["citation_policy"]["required"] is True


def test_generation_template_repository_reads_seeded_templates(
    migrated_database_url: str,
) -> None:
    templates = list_generation_templates(migrated_database_url)
    default_template = get_default_generation_template(migrated_database_url)
    report_template = get_generation_template_by_key(migrated_database_url, " REPORT ")
    summary_preset = get_generation_template_by_key(
        migrated_database_url,
        " summary_risk_action ",
    )

    assert len(templates) >= 5
    assert default_template is not None
    assert default_template.template_key == "grounded_answer"
    assert default_template.is_default is True
    assert report_template is not None
    assert report_template.template_key == "report"
    assert report_template.document_type == "report"
    assert report_template.section_schema[0]["key"] == "title"
    assert summary_preset is not None
    assert summary_preset.template_key == "summary_risk_action"
    assert summary_preset.template_family == "summary"
    assert summary_preset.document_type == "summary"
    assert {section["key"] for section in summary_preset.section_schema} >= {
        "risks",
        "actions",
        "evidence",
    }


def test_generation_template_repository_manages_custom_template_lifecycle(
    migrated_database_url: str,
) -> None:
    template_key = _template_key("pytest_template")
    inactive_key = _template_key("pytest_inactive")
    try:
        created = upsert_generation_template(
            migrated_database_url,
            GenerationTemplateInput(
                template_key=f" {template_key.upper()} ",
                template_name="Pytest 보고서",
                template_version="v1",
                document_type="report",
                language="ko",
                section_schema=(
                    {"key": "title", "heading": "제목", "required": True},
                    {"key": "findings", "heading": "주요 내용", "required": True},
                ),
                system_instruction="보고서 형식으로 작성한다.",
                user_instruction_suffix="citation key를 포함한다.",
                style_guidance={"tone": "formal"},
                citation_policy={"required": True, "minimum_citations": 2},
                is_default=False,
                is_active=True,
                created_by="pytest",
            ),
        )
        updated_default = upsert_generation_template(
            migrated_database_url,
            GenerationTemplateInput(
                template_key=template_key,
                template_name="Pytest 보고서 v2",
                template_version="v2",
                document_type="report",
                language="ko",
                section_schema=(
                    {"key": "summary", "heading": "요약", "required": True},
                    {"key": "evidence", "heading": "근거", "required": True},
                ),
                system_instruction="업데이트된 보고서 형식으로 작성한다.",
                style_guidance={"tone": "review"},
                citation_policy={"required": True, "minimum_citations": 1},
                is_default=True,
                is_active=True,
                created_by="pytest",
            ),
        )
        inactive = upsert_generation_template(
            migrated_database_url,
            GenerationTemplateInput(
                template_key=inactive_key,
                template_name="Pytest 비활성 템플릿",
                document_type="summary",
                section_schema=({"key": "summary", "heading": "요약", "required": True},),
                system_instruction="요약한다.",
                is_active=False,
            ),
        )

        current_default = get_default_generation_template(migrated_database_url)
        hidden_inactive = get_generation_template_by_key(migrated_database_url, inactive_key)
        visible_inactive = get_generation_template_by_key(
            migrated_database_url,
            inactive_key,
            include_inactive=True,
        )

        assert created.template_key == template_key
        assert created.template_family == template_key
        assert created.template_name == "Pytest 보고서"
        assert updated_default.template_name == "Pytest 보고서 v2"
        assert updated_default.is_default is True
        assert updated_default.section_schema[0]["key"] == "summary"
        assert current_default is not None
        assert current_default.template_key == template_key
        assert inactive.is_active is False
        assert hidden_inactive is None
        assert visible_inactive is not None

        with pytest.raises(InvalidGenerationTemplateError, match="default"):
            set_generation_template_active(
                migrated_database_url,
                template_key,
                is_active=False,
            )
        with pytest.raises(InvalidGenerationTemplateError, match="inactive"):
            set_generation_template_default(migrated_database_url, inactive_key)

        assert (
            set_generation_template_active(
                migrated_database_url,
                inactive_key,
                is_active=True,
            ).is_active
            is True
        )
        assert (
            set_generation_template_default(migrated_database_url, inactive_key).is_default is True
        )
        assert (
            set_generation_template_active(
                migrated_database_url,
                "missing_generation_template",
                is_active=True,
            )
            is None
        )
        assert (
            set_generation_template_default(
                migrated_database_url,
                "missing_generation_template",
            )
            is None
        )
    finally:
        _cleanup_generation_templates(migrated_database_url, template_key, inactive_key)


def test_generation_template_repository_clones_and_rolls_back_versions(
    migrated_database_url: str,
) -> None:
    template_key = _template_key("pytest_template_family")
    clone_key = _template_key("pytest_template_family_v2")
    try:
        created = upsert_generation_template(
            migrated_database_url,
            GenerationTemplateInput(
                template_key=template_key,
                template_family="pytest_family",
                template_name="Pytest 계보 템플릿",
                template_version="v1",
                document_type="report",
                language="ko",
                section_schema=(
                    {"key": "title", "heading": "제목", "required": True},
                    {"key": "evidence", "heading": "근거", "required": True},
                ),
                system_instruction="계보 테스트 템플릿이다.",
                style_guidance={"tone": "formal"},
                citation_policy={"required": True},
                is_default=True,
                created_by="pytest",
            ),
        )
        cloned = clone_generation_template_version(
            migrated_database_url,
            GenerationTemplateCloneInput(
                source_template_key=template_key,
                target_template_key=clone_key,
                target_template_version="v2",
                target_template_name="Pytest 계보 템플릿 v2",
                change_note="rollback 후보",
                created_by="pytest",
            ),
        )
        rolled_back = rollback_generation_template_version(migrated_database_url, clone_key)
        default_template = get_default_generation_template(migrated_database_url)

        assert created.is_default is True
        assert cloned is not None
        assert cloned.template_family == "pytest_family"
        assert cloned.clone_source_template_id == created.generation_template_id
        assert cloned.change_note == "rollback 후보"
        assert cloned.is_default is False
        assert rolled_back is not None
        assert rolled_back.template_key == clone_key
        assert rolled_back.is_default is True
        assert default_template is not None
        assert default_template.template_key == clone_key

        with pytest.raises(InvalidGenerationTemplateError, match="already exists"):
            clone_generation_template_version(
                migrated_database_url,
                GenerationTemplateCloneInput(
                    source_template_key=template_key,
                    target_template_key=clone_key,
                    target_template_version="v3",
                ),
            )
        assert (
            clone_generation_template_version(
                migrated_database_url,
                GenerationTemplateCloneInput(
                    source_template_key="missing_generation_template",
                    target_template_key=_template_key("pytest_missing_clone"),
                    target_template_version="v1",
                ),
            )
            is None
        )
    finally:
        _cleanup_generation_templates(migrated_database_url, template_key, clone_key)


def test_generation_template_repository_hides_inactive_templates_by_default(
    migrated_database_url: str,
) -> None:
    with connect(migrated_database_url) as conn:
        row = conn.execute("""
            INSERT INTO generation_templates (
                template_key,
                template_name,
                document_type,
                section_schema,
                system_instruction,
                is_active
            )
            VALUES (
                'inactive_repository_template',
                'Inactive Repository Template',
                'summary',
                '[]'::jsonb,
                'inactive repository smoke',
                false
            )
            ON CONFLICT (template_key) DO UPDATE
            SET is_active = false,
                updated_at = now()
            RETURNING template_key
            """).fetchone()
        conn.commit()
    assert row is not None

    hidden = get_generation_template_by_key(
        migrated_database_url,
        "inactive_repository_template",
    )
    visible = get_generation_template_by_key(
        migrated_database_url,
        "inactive_repository_template",
        include_inactive=True,
    )
    active_keys = {
        template.template_key for template in list_generation_templates(migrated_database_url)
    }
    all_keys = {
        template.template_key
        for template in list_generation_templates(migrated_database_url, include_inactive=True)
    }

    assert hidden is None
    assert visible is not None
    assert visible.is_active is False
    assert "inactive_repository_template" not in active_keys
    assert "inactive_repository_template" in all_keys


@pytest.mark.parametrize(
    ("template_key", "expected_section_key"),
    [
        ("report", "findings"),
        ("proposal", "recommendation"),
        ("summary", "key_points"),
        ("meeting_minutes", "actions"),
    ],
)
def test_non_default_generation_templates_seed_section_contracts(
    migrated_database_url: str,
    template_key: str,
    expected_section_key: str,
) -> None:
    template = fetch_one(
        migrated_database_url,
        """
        SELECT template_key, document_type, section_schema, citation_policy
        FROM generation_templates
        WHERE template_key = %s
        """,
        (template_key,),
    )

    section_keys = {section["key"] for section in template["section_schema"]}

    assert template["template_key"] == template_key
    assert template["document_type"] == template_key
    assert expected_section_key in section_keys
    assert template["citation_policy"]["required"] is True
    if template_key in {"report", "proposal"}:
        assert all(section["required"] is True for section in template["section_schema"])


@pytest.mark.parametrize(
    ("template_key", "expected_section_key"),
    [
        ("summary_executive", "executive_summary"),
        ("summary_risk_action", "actions"),
        ("summary_working", "implications"),
    ],
)
def test_summary_generation_template_presets_are_seeded(
    migrated_database_url: str,
    template_key: str,
    expected_section_key: str,
) -> None:
    template = fetch_one(
        migrated_database_url,
        """
        SELECT
            template_key,
            template_family,
            document_type,
            section_schema,
            style_guidance,
            citation_policy,
            is_default,
            is_active,
            change_note
        FROM generation_templates
        WHERE template_key = %s
        """,
        (template_key,),
    )

    section_keys = {section["key"] for section in template["section_schema"]}

    assert template["template_key"] == template_key
    assert template["template_family"] == "summary"
    assert template["document_type"] == "summary"
    assert template["is_default"] is False
    assert template["is_active"] is True
    assert expected_section_key in section_keys
    assert "evidence" in section_keys
    assert all(section["required"] is True for section in template["section_schema"])
    assert template["style_guidance"]["audience"]
    assert template["citation_policy"]["minimum_citations"] == 2
    assert template["change_note"] == "Slice 380 summary preset seed"


def test_generation_run_can_link_generation_template(
    migrated_database_url: str,
) -> None:
    search_log_id = _create_search_log(migrated_database_url)
    provider = fetch_one(
        migrated_database_url,
        """
        SELECT provider_config_id, provider_name, provider_mode, model_id
        FROM generation_provider_configs
        WHERE is_default
        """,
    )
    template = fetch_one(
        migrated_database_url,
        """
        SELECT generation_template_id, template_key
        FROM generation_templates
        WHERE template_key = 'report'
        """,
    )

    with connect(migrated_database_url) as conn:
        run = conn.execute(
            """
            INSERT INTO generation_runs (
                search_log_id,
                retrieval_package_key,
                generation_template_id,
                provider_config_id,
                provider_name,
                provider_mode,
                model_id,
                retrieval_confidence_status,
                citation_readiness_status,
                query_text,
                request_metadata
            )
            VALUES (
                %s,
                'package-template-schema',
                %s,
                %s,
                %s,
                %s,
                %s,
                'answerable',
                'ready',
                'generation template schema smoke query',
                '{"template_key": "report"}'::jsonb
            )
            RETURNING generation_run_id
            """,
            (
                search_log_id,
                template["generation_template_id"],
                provider["provider_config_id"],
                provider["provider_name"],
                provider["provider_mode"],
                provider["model_id"],
            ),
        ).fetchone()
        conn.commit()

    assert run is not None
    stored = fetch_one(
        migrated_database_url,
        """
        SELECT gt.template_key, gr.request_metadata
        FROM generation_runs gr
        JOIN generation_templates gt
          ON gt.generation_template_id = gr.generation_template_id
        WHERE gr.generation_run_id = %s
        """,
        (run["generation_run_id"],),
    )

    assert stored["template_key"] == "report"
    assert stored["request_metadata"]["template_key"] == "report"


def test_generation_template_default_must_be_unique_per_language(
    migrated_database_url: str,
) -> None:
    with connect(migrated_database_url) as conn:
        with pytest.raises(errors.UniqueViolation):
            conn.execute("""
                INSERT INTO generation_templates (
                    template_key,
                    template_name,
                    document_type,
                    section_schema,
                    system_instruction,
                    is_default
                )
                VALUES (
                    'duplicate_default_template',
                    'Duplicate Default',
                    'summary',
                    '[]'::jsonb,
                    'duplicate default smoke',
                    true
                )
                """)
        conn.rollback()


def test_generation_template_default_must_be_active(
    migrated_database_url: str,
) -> None:
    with connect(migrated_database_url) as conn:
        with pytest.raises(errors.CheckViolation):
            conn.execute("""
                INSERT INTO generation_templates (
                    template_key,
                    template_name,
                    document_type,
                    section_schema,
                    system_instruction,
                    is_default,
                    is_active
                )
                VALUES (
                    'inactive_default_template',
                    'Inactive Default',
                    'summary',
                    '[]'::jsonb,
                    'inactive default smoke',
                    true,
                    false
                )
                """)
        conn.rollback()


def test_generation_template_json_shapes_are_constrained(
    migrated_database_url: str,
) -> None:
    with connect(migrated_database_url) as conn:
        with pytest.raises(errors.CheckViolation):
            conn.execute("""
                INSERT INTO generation_templates (
                    template_key,
                    template_name,
                    document_type,
                    section_schema,
                    system_instruction
                )
                VALUES (
                    'invalid_section_schema_template',
                    'Invalid Section Schema',
                    'summary',
                    '{}'::jsonb,
                    'invalid schema smoke'
                )
                """)
        conn.rollback()

        with pytest.raises(errors.CheckViolation):
            conn.execute("""
                INSERT INTO generation_templates (
                    template_key,
                    template_name,
                    document_type,
                    section_schema,
                    style_guidance,
                    system_instruction
                )
                VALUES (
                    'invalid_style_guidance_template',
                    'Invalid Style Guidance',
                    'summary',
                    '[]'::jsonb,
                    '[]'::jsonb,
                    'invalid style smoke'
                )
                """)
        conn.rollback()
