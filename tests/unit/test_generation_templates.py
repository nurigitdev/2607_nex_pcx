from datetime import UTC, datetime

import pytest

from app.core.generation_templates import (
    DEFAULT_GENERATION_TEMPLATE_KEY,
    GenerationTemplateRecord,
    InvalidGenerationTemplateError,
    default_generation_template_record,
    generation_template_snapshot,
    get_generation_template_by_key,
)


def test_default_generation_template_record_is_korean_markdown_contract() -> None:
    template = default_generation_template_record(language=" KO ")

    assert template.generation_template_id is None
    assert template.template_key == DEFAULT_GENERATION_TEMPLATE_KEY
    assert template.language == "ko"
    assert template.output_format == "markdown"
    assert template.is_default is True
    assert template.is_active is True
    assert template.section_schema[0]["key"] == "answer"
    assert template.citation_policy["required"] is True


def test_generation_template_snapshot_is_reproducible_copy() -> None:
    now = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)
    template = GenerationTemplateRecord(
        generation_template_id=12,
        template_key="summary",
        template_name="요약문",
        template_version="v2",
        document_type="summary",
        language="ko",
        output_format="markdown",
        section_schema=({"key": "key_points", "heading": "핵심 요약", "required": True},),
        system_instruction="요약한다.",
        user_instruction_suffix="짧게 작성한다.",
        style_guidance={"tone": "concise"},
        citation_policy={"required": True},
        is_default=False,
        is_active=True,
        created_by="pytest",
        created_by_user_id=1,
        created_at=now,
        updated_at=now,
    )

    snapshot = generation_template_snapshot(template)
    snapshot["section_schema"][0]["key"] = "mutated"

    assert snapshot["generation_template_id"] == 12
    assert snapshot["template_key"] == "summary"
    assert template.section_schema[0]["key"] == "key_points"
    assert "created_at" not in snapshot


@pytest.mark.parametrize("template_key", ["", "  "])
def test_get_generation_template_by_key_rejects_blank_template_key(
    template_key: str,
) -> None:
    with pytest.raises(InvalidGenerationTemplateError, match="template_key"):
        get_generation_template_by_key("postgresql://unused/example", template_key)
