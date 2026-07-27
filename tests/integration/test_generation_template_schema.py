import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

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
    assert default_template["template_version"] == "v1"
    assert default_template["document_type"] == "grounded_answer"
    assert default_template["language"] == "ko"
    assert default_template["output_format"] == "markdown"
    assert default_template["section_schema"][0]["key"] == "answer"
    assert default_template["style_guidance"]["tone"] == "concise"
    assert default_template["citation_policy"]["required"] is True


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
