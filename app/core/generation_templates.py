"""Generation template repository and prompt snapshot helpers."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

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
GENERATION_TEMPLATE_LANGUAGES = {"ko", "en"}
GENERATION_TEMPLATE_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
GENERATION_TEMPLATE_SECTION_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")


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


@dataclass(frozen=True)
class GenerationTemplateInput:
    template_key: str
    template_name: str
    template_version: str = DEFAULT_GENERATION_TEMPLATE_VERSION
    document_type: str = DEFAULT_GENERATION_TEMPLATE_DOCUMENT_TYPE
    language: str = DEFAULT_GENERATION_TEMPLATE_LANGUAGE
    output_format: str = GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN
    section_schema: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    system_instruction: str = ""
    user_instruction_suffix: str = ""
    style_guidance: Mapping[str, Any] = field(default_factory=dict)
    citation_policy: Mapping[str, Any] = field(default_factory=dict)
    is_default: bool = False
    is_active: bool = True
    created_by: str | None = None
    created_by_user_id: int | None = None


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


def _validate_created_by(created_by: str | None) -> str | None:
    if created_by is None:
        return None
    normalized = created_by.strip()
    return normalized or None


def _validate_created_by_user_id(created_by_user_id: int | None) -> int | None:
    if created_by_user_id is None:
        return None
    if created_by_user_id < 1:
        raise InvalidGenerationTemplateError("created_by_user_id must be positive")
    return created_by_user_id


def _validate_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidGenerationTemplateError(f"{field_name} must be a JSON object")
    return dict(value)


def _validate_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise InvalidGenerationTemplateError(f"{field_name} must be a boolean")


def _validate_section_schema(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvalidGenerationTemplateError("section_schema must be a JSON array")

    sections: list[dict[str, Any]] = []
    section_keys: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise InvalidGenerationTemplateError("section_schema entries must be JSON objects")
        raw_section_key = str(item.get("key") or "")
        section_key = _validate_non_empty(raw_section_key, f"section_schema[{index}].key").lower()
        if GENERATION_TEMPLATE_SECTION_KEY_PATTERN.fullmatch(section_key) is None:
            raise InvalidGenerationTemplateError(
                "section_schema key must use lowercase letters, numbers, and underscores"
            )
        if section_key in section_keys:
            raise InvalidGenerationTemplateError("section_schema keys must be unique")
        section_keys.add(section_key)
        sections.append(
            {
                "key": section_key,
                "heading": _validate_non_empty(
                    str(item.get("heading") or ""),
                    f"section_schema[{index}].heading",
                ),
                "required": _validate_bool(item.get("required", False), "section_schema.required"),
            }
        )

    if not sections:
        raise InvalidGenerationTemplateError("section_schema must contain at least one section")
    return tuple(sections)


def validate_generation_template_input(
    template_input: GenerationTemplateInput,
) -> GenerationTemplateInput:
    template_key = _normalize_template_key(template_input.template_key)
    if GENERATION_TEMPLATE_KEY_PATTERN.fullmatch(template_key) is None:
        raise InvalidGenerationTemplateError(
            "template_key must use 2-64 lowercase letters, numbers, hyphens, or underscores"
        )

    document_type = _validate_non_empty(template_input.document_type, "document_type")
    if document_type not in GENERATION_TEMPLATE_DOCUMENT_TYPES:
        raise InvalidGenerationTemplateError("document_type is not supported")

    language = _normalize_language(template_input.language)
    if language not in GENERATION_TEMPLATE_LANGUAGES:
        raise InvalidGenerationTemplateError("language must be ko or en")

    output_format = _validate_non_empty(template_input.output_format, "output_format").lower()
    if output_format != GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN:
        raise InvalidGenerationTemplateError("output_format must be markdown")

    is_default = bool(template_input.is_default)
    is_active = bool(template_input.is_active)
    if is_default and not is_active:
        raise InvalidGenerationTemplateError("default generation template must be active")

    return GenerationTemplateInput(
        template_key=template_key,
        template_name=_validate_non_empty(template_input.template_name, "template_name"),
        template_version=_validate_non_empty(
            template_input.template_version,
            "template_version",
        ),
        document_type=document_type,
        language=language,
        output_format=output_format,
        section_schema=_validate_section_schema(template_input.section_schema),
        system_instruction=_validate_non_empty(
            template_input.system_instruction,
            "system_instruction",
        ),
        user_instruction_suffix=(template_input.user_instruction_suffix or "").strip(),
        style_guidance=_validate_mapping(template_input.style_guidance, "style_guidance"),
        citation_policy=_validate_mapping(template_input.citation_policy, "citation_policy"),
        is_default=is_default,
        is_active=is_active,
        created_by=_validate_created_by(template_input.created_by),
        created_by_user_id=_validate_created_by_user_id(template_input.created_by_user_id),
    )


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


def upsert_generation_template(
    database_url: str,
    template_input: GenerationTemplateInput,
) -> GenerationTemplateRecord:
    validated = validate_generation_template_input(template_input)

    with connect(database_url) as conn:
        if validated.is_default:
            conn.execute(
                """
                UPDATE generation_templates
                SET is_default = false,
                    updated_at = now()
                WHERE language = %s
                  AND is_default
                """,
                (validated.language,),
            )
        row = conn.execute(
            """
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
                is_active,
                created_by,
                created_by_user_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                created_by = COALESCE(EXCLUDED.created_by, generation_templates.created_by),
                created_by_user_id = COALESCE(
                    EXCLUDED.created_by_user_id,
                    generation_templates.created_by_user_id
                ),
                updated_at = now()
            RETURNING *
            """,
            (
                validated.template_key,
                validated.template_name,
                validated.template_version,
                validated.document_type,
                validated.language,
                validated.output_format,
                Json([dict(section) for section in validated.section_schema]),
                validated.system_instruction,
                validated.user_instruction_suffix,
                Json(dict(validated.style_guidance)),
                Json(dict(validated.citation_policy)),
                validated.is_default,
                validated.is_active,
                validated.created_by,
                validated.created_by_user_id,
            ),
        ).fetchone()
        conn.commit()

    if row is None:
        raise InvalidGenerationTemplateError("generation template was not saved")
    return _template_from_row(dict(row))


def set_generation_template_active(
    database_url: str,
    template_key: str,
    *,
    is_active: bool,
) -> GenerationTemplateRecord | None:
    normalized_template_key = _normalize_template_key(template_key)
    with connect(database_url) as conn:
        current = conn.execute(
            """
            SELECT *
            FROM generation_templates
            WHERE template_key = %s
            FOR UPDATE
            """,
            (normalized_template_key,),
        ).fetchone()
        if current is None:
            conn.rollback()
            return None
        if bool(current["is_default"]) and not is_active:
            conn.rollback()
            raise InvalidGenerationTemplateError("default generation template must remain active")
        row = conn.execute(
            """
            UPDATE generation_templates
            SET is_active = %s,
                updated_at = now()
            WHERE template_key = %s
            RETURNING *
            """,
            (is_active, normalized_template_key),
        ).fetchone()
        conn.commit()
    return _template_from_row(dict(row)) if row else None


def set_generation_template_default(
    database_url: str,
    template_key: str,
) -> GenerationTemplateRecord | None:
    normalized_template_key = _normalize_template_key(template_key)
    with connect(database_url) as conn:
        current = conn.execute(
            """
            SELECT *
            FROM generation_templates
            WHERE template_key = %s
            FOR UPDATE
            """,
            (normalized_template_key,),
        ).fetchone()
        if current is None:
            conn.rollback()
            return None
        if not bool(current["is_active"]):
            conn.rollback()
            raise InvalidGenerationTemplateError("inactive generation template cannot be default")
        conn.execute(
            """
            UPDATE generation_templates
            SET is_default = false,
                updated_at = now()
            WHERE language = %s
              AND is_default
            """,
            (str(current["language"]),),
        )
        row = conn.execute(
            """
            UPDATE generation_templates
            SET is_default = true,
                is_active = true,
                updated_at = now()
            WHERE template_key = %s
            RETURNING *
            """,
            (normalized_template_key,),
        ).fetchone()
        conn.commit()
    return _template_from_row(dict(row)) if row else None
