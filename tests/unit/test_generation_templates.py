from datetime import UTC, datetime

import pytest

from app.core.generation_templates import (
    DEFAULT_GENERATION_TEMPLATE_KEY,
    GenerationTemplateInput,
    GenerationTemplateRecord,
    InvalidGenerationTemplateError,
    default_generation_template_record,
    generation_template_snapshot,
    get_generation_template_by_key,
    suggest_generation_template_clone_key,
    suggest_generation_template_next_version,
    validate_generation_template_input,
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
        template_family="summary",
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
        clone_source_template_id=3,
        change_note="요약 섹션 보강",
        created_by="pytest",
        created_by_user_id=1,
        created_at=now,
        updated_at=now,
    )

    snapshot = generation_template_snapshot(template)
    snapshot["section_schema"][0]["key"] = "mutated"

    assert snapshot["generation_template_id"] == 12
    assert snapshot["template_key"] == "summary"
    assert snapshot["template_family"] == "summary"
    assert snapshot["clone_source_template_id"] == 3
    assert snapshot["change_note"] == "요약 섹션 보강"
    assert template.section_schema[0]["key"] == "key_points"
    assert "created_at" not in snapshot


def test_generation_template_clone_suggestions_use_family_and_version() -> None:
    template = default_generation_template_record()
    custom_template = GenerationTemplateRecord(
        generation_template_id=9,
        template_key="report_current",
        template_family="executive-report",
        template_name="Executive Report",
        template_version="draftA",
        document_type="report",
        language="en",
        output_format="markdown",
        section_schema=({"key": "summary", "heading": "Summary", "required": True},),
        system_instruction="Write a report.",
        user_instruction_suffix="",
        style_guidance={},
        citation_policy={"required": True},
        is_default=False,
        is_active=True,
        clone_source_template_id=None,
        change_note="",
        created_by=None,
        created_by_user_id=None,
        created_at=None,
        updated_at=None,
    )

    assert suggest_generation_template_next_version(template) == "v2"
    assert suggest_generation_template_clone_key(template) == "grounded_answer_v2"
    assert suggest_generation_template_next_version(custom_template) == "draftA_next"
    assert suggest_generation_template_clone_key(custom_template) == "executive-report_drafta_next"


@pytest.mark.parametrize("template_key", ["", "  "])
def test_get_generation_template_by_key_rejects_blank_template_key(
    template_key: str,
) -> None:
    with pytest.raises(InvalidGenerationTemplateError, match="template_key"):
        get_generation_template_by_key("postgresql://unused/example", template_key)


def test_validate_generation_template_input_normalizes_supported_contract() -> None:
    template_input = GenerationTemplateInput(
        template_key=" Report_Draft-v2 ",
        template_name=" 보고서 초안 ",
        template_version=" v2 ",
        document_type="report",
        language=" KO ",
        section_schema=(
            {"key": "Title", "heading": " 제목 ", "required": True},
            {"key": "findings", "heading": "주요 내용", "required": False},
        ),
        system_instruction=" 보고서로 작성한다. ",
        user_instruction_suffix=" 근거를 표시한다. ",
        style_guidance={"tone": "formal"},
        citation_policy={"required": True},
        is_default=False,
        is_active=True,
        created_by=" pytest ",
        created_by_user_id=7,
        change_note=" 변경 내역 ",
    )

    validated = validate_generation_template_input(template_input)

    assert validated.template_key == "report_draft-v2"
    assert validated.template_name == "보고서 초안"
    assert validated.template_version == "v2"
    assert validated.language == "ko"
    assert validated.section_schema == (
        {"key": "title", "heading": "제목", "required": True},
        {"key": "findings", "heading": "주요 내용", "required": False},
    )
    assert validated.system_instruction == "보고서로 작성한다."
    assert validated.user_instruction_suffix == "근거를 표시한다."
    assert validated.template_family == "report_draft-v2"
    assert validated.change_note == "변경 내역"
    assert validated.created_by == "pytest"
    assert validated.created_by_user_id == 7


@pytest.mark.parametrize(
    ("template_input", "message"),
    [
        (
            GenerationTemplateInput(
                template_key="bad key",
                template_name="Bad",
                document_type="report",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
            ),
            "template_key",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_doc",
                template_name="Bad",
                document_type="unsupported",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
            ),
            "document_type",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_language",
                template_name="Bad",
                document_type="report",
                language="ja",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
            ),
            "language",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_output",
                template_name="Bad",
                document_type="report",
                output_format="html",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
            ),
            "output_format",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_sections",
                template_name="Bad",
                document_type="report",
                section_schema=(),
                system_instruction="Do it",
            ),
            "section_schema",
        ),
        (
            GenerationTemplateInput(
                template_key="duplicate_sections",
                template_name="Bad",
                document_type="report",
                section_schema=(
                    {"key": "title", "heading": "Title", "required": True},
                    {"key": "title", "heading": "Title again", "required": False},
                ),
                system_instruction="Do it",
            ),
            "unique",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_required",
                template_name="Bad",
                document_type="report",
                section_schema=({"key": "title", "heading": "Title", "required": "true"},),
                system_instruction="Do it",
            ),
            "boolean",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_default",
                template_name="Bad",
                document_type="report",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
                is_default=True,
                is_active=False,
            ),
            "default",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_mapping",
                template_name="Bad",
                document_type="report",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
                style_guidance=[],
            ),
            "style_guidance",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_user_id",
                template_name="Bad",
                document_type="report",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
                created_by_user_id=0,
            ),
            "created_by_user_id",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_family",
                template_family="bad family",
                template_name="Bad",
                document_type="report",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
            ),
            "template_family",
        ),
        (
            GenerationTemplateInput(
                template_key="bad_clone_source",
                template_name="Bad",
                document_type="report",
                section_schema=({"key": "title", "heading": "Title", "required": True},),
                system_instruction="Do it",
                clone_source_template_id=0,
            ),
            "clone_source_template_id",
        ),
    ],
)
def test_validate_generation_template_input_rejects_invalid_contracts(
    template_input: GenerationTemplateInput,
    message: str,
) -> None:
    with pytest.raises(InvalidGenerationTemplateError, match=message):
        validate_generation_template_input(template_input)
