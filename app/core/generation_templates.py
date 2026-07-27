"""Generation template repository and prompt snapshot helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect

DEFAULT_GENERATION_TEMPLATE_KEY = "grounded_answer"
DEFAULT_GENERATION_TEMPLATE_NAME = "근거 기반 답변"
DEFAULT_GENERATION_TEMPLATE_VERSION = "v1"
DEFAULT_GENERATION_TEMPLATE_DOCUMENT_TYPE = "grounded_answer"
DEFAULT_GENERATION_TEMPLATE_LANGUAGE = "ko"
GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN = "markdown"
GENERATION_TEMPLATE_DOCUMENT_TYPES = {
    "grounded_answer",
    "report",
    "proposal",
    "summary",
    "meeting_minutes",
}


@dataclass(frozen=True)
class GenerationTemplateRecord:
    generation_template_id: int | None
    template_key: str
    template_name: str
    template_version: str
    document_type: str
    language: str
    output_format: str
    section_schema: tuple[dict[str, Any], ...]
    system_instruction: str
    user_instruction_suffix: str
    style_guidance: dict[str, Any]
    citation_policy: dict[str, Any]
    is_default: bool
    is_active: bool
    created_by: str | None
    created_by_user_id: int | None
    created_at: datetime | None
    updated_at: datetime | None


class InvalidGenerationTemplateError(ValueError):
    """Raised when a generation template input is invalid."""


def _validate_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidGenerationTemplateError(f"{field_name} must not be empty")
    return normalized


def _normalize_template_key(template_key: str) -> str:
    return _validate_non_empty(template_key, "template_key").lower()


def _normalize_language(language: str | None) -> str:
    normalized = (language or DEFAULT_GENERATION_TEMPLATE_LANGUAGE).strip().lower()
    return normalized or DEFAULT_GENERATION_TEMPLATE_LANGUAGE


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _section_schema(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _template_from_row(row: Mapping[str, Any]) -> GenerationTemplateRecord:
    template_id = row.get("generation_template_id")
    return GenerationTemplateRecord(
        generation_template_id=int(template_id) if template_id is not None else None,
        template_key=str(row["template_key"]),
        template_name=str(row["template_name"]),
        template_version=str(row["template_version"]),
        document_type=str(row["document_type"]),
        language=str(row["language"]),
        output_format=str(row["output_format"]),
        section_schema=_section_schema(row.get("section_schema")),
        system_instruction=str(row["system_instruction"]),
        user_instruction_suffix=str(row.get("user_instruction_suffix") or ""),
        style_guidance=_mapping(row.get("style_guidance")),
        citation_policy=_mapping(row.get("citation_policy")),
        is_default=bool(row["is_default"]),
        is_active=bool(row["is_active"]),
        created_by=row.get("created_by"),
        created_by_user_id=row.get("created_by_user_id"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def default_generation_template_record(
    *,
    language: str = DEFAULT_GENERATION_TEMPLATE_LANGUAGE,
) -> GenerationTemplateRecord:
    normalized_language = _normalize_language(language)
    return GenerationTemplateRecord(
        generation_template_id=None,
        template_key=DEFAULT_GENERATION_TEMPLATE_KEY,
        template_name=DEFAULT_GENERATION_TEMPLATE_NAME,
        template_version=DEFAULT_GENERATION_TEMPLATE_VERSION,
        document_type=DEFAULT_GENERATION_TEMPLATE_DOCUMENT_TYPE,
        language=normalized_language,
        output_format=GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN,
        section_schema=(
            {"key": "answer", "heading": "답변", "required": True},
            {"key": "evidence", "heading": "근거", "required": True},
            {"key": "limits", "heading": "한계", "required": False},
        ),
        system_instruction="검색 근거에 기반해 간결하고 검증 가능한 답변을 작성한다.",
        user_instruction_suffix=(
            "답변에는 관련 citation key를 포함하고, 근거가 부족하면 부족하다고 명시한다."
        ),
        style_guidance={"tone": "concise", "audience": "internal", "density": "balanced"},
        citation_policy={
            "required": True,
            "placement": "inline_or_bullet",
            "minimum_citations": 1,
        },
        is_default=True,
        is_active=True,
        created_by=None,
        created_by_user_id=None,
        created_at=None,
        updated_at=None,
    )


def generation_template_snapshot(template: GenerationTemplateRecord) -> dict[str, Any]:
    return {
        "generation_template_id": template.generation_template_id,
        "template_key": template.template_key,
        "template_name": template.template_name,
        "template_version": template.template_version,
        "document_type": template.document_type,
        "language": template.language,
        "output_format": template.output_format,
        "section_schema": [dict(section) for section in template.section_schema],
        "system_instruction": template.system_instruction,
        "user_instruction_suffix": template.user_instruction_suffix,
        "style_guidance": dict(template.style_guidance),
        "citation_policy": dict(template.citation_policy),
        "is_default": template.is_default,
        "is_active": template.is_active,
    }


def list_generation_templates(
    database_url: str,
    *,
    include_inactive: bool = False,
) -> tuple[GenerationTemplateRecord, ...]:
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM generation_templates
            WHERE (%s OR is_active)
            ORDER BY is_default DESC, template_key
            """,
            (include_inactive,),
        ).fetchall()
    return tuple(_template_from_row(dict(row)) for row in rows)


def get_default_generation_template(
    database_url: str,
    *,
    language: str | None = DEFAULT_GENERATION_TEMPLATE_LANGUAGE,
) -> GenerationTemplateRecord | None:
    normalized_language = _normalize_language(language)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM generation_templates
            WHERE is_default
              AND is_active
              AND language = %s
            ORDER BY generation_template_id
            LIMIT 1
            """,
            (normalized_language,),
        ).fetchone()
    return _template_from_row(dict(row)) if row else None


def get_generation_template_by_key(
    database_url: str,
    template_key: str,
    *,
    include_inactive: bool = False,
) -> GenerationTemplateRecord | None:
    normalized_template_key = _normalize_template_key(template_key)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM generation_templates
            WHERE template_key = %s
              AND (%s OR is_active)
            """,
            (normalized_template_key, include_inactive),
        ).fetchone()
    return _template_from_row(dict(row)) if row else None
