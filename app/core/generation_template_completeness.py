"""Template completeness checks for generated answers."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.core.generation_answer_quality import CITATION_KEY_PATTERN
from app.core.generation_runs import GenerationRunRecord
from app.core.generation_templates import GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN

GENERATION_TEMPLATE_COMPLETENESS_CONTRACT_VERSION = "generation_template_completeness_v1"
GENERATION_TEMPLATE_COMPLETENESS_PASSED = "passed"
GENERATION_TEMPLATE_COMPLETENESS_WARNING = "warning"
GENERATION_TEMPLATE_COMPLETENESS_FAILED = "failed"
GENERATION_TEMPLATE_COMPLETENESS_NOT_EVALUATED = "not_evaluated"

TEMPLATE_COMPLETENESS_REASON_NO_TEMPLATE_METADATA = "no_template_metadata"
TEMPLATE_COMPLETENESS_REASON_UNSUPPORTED_OUTPUT_FORMAT = "unsupported_output_format"
TEMPLATE_COMPLETENESS_REASON_NO_ANSWER_GUARDRAIL = "no_answer_guardrail"
TEMPLATE_COMPLETENESS_REASON_EMPTY_ANSWER = "empty_answer"
TEMPLATE_COMPLETENESS_REASON_NO_REQUIRED_SECTIONS = "no_required_sections"
TEMPLATE_COMPLETENESS_REASON_MISSING_REQUIRED_SECTION = "missing_required_section"
TEMPLATE_COMPLETENESS_REASON_MISSING_SECTION_CITATION = "missing_section_citation"

_MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*#*\s*$")
_HEADING_NUMBER_PREFIX_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")
_TITLE_SECTION_KEYS = {"title"}


@dataclass(frozen=True)
class TemplateCompletenessSectionCheck:
    key: str
    heading: str
    required: bool
    present: bool
    citation_required: bool
    cited: bool
    citation_keys: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class GenerationTemplateCompletenessAssessment:
    status: str
    template_key: str | None
    template_name: str | None
    template_version: str | None
    document_type: str | None
    output_format: str | None
    required_section_count: int
    present_required_section_count: int
    citation_required_section_count: int
    cited_required_section_count: int
    required_section_coverage_percent: float | None
    required_section_citation_coverage_percent: float | None
    section_checks: tuple[TemplateCompletenessSectionCheck, ...]
    reason_codes: tuple[str, ...]


def assess_generation_template_completeness(
    run: GenerationRunRecord,
) -> GenerationTemplateCompletenessAssessment:
    """Assess whether a generation run followed its stored template section contract."""

    template = _template_snapshot(run)
    if not template:
        return _not_evaluated(
            reason_code=TEMPLATE_COMPLETENESS_REASON_NO_TEMPLATE_METADATA,
        )

    output_format = _template_value(template, "output_format")
    base_kwargs = {
        "template_key": _template_value(template, "template_key"),
        "template_name": _template_value(template, "template_name"),
        "template_version": _template_value(template, "template_version"),
        "document_type": _template_value(template, "document_type"),
        "output_format": output_format,
    }
    if output_format != GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN:
        return _not_evaluated(
            reason_code=TEMPLATE_COMPLETENESS_REASON_UNSUPPORTED_OUTPUT_FORMAT,
            **base_kwargs,
        )

    sections = _template_sections(template)
    required_sections = tuple(section for section in sections if section["required"])
    if not required_sections:
        return GenerationTemplateCompletenessAssessment(
            status=GENERATION_TEMPLATE_COMPLETENESS_WARNING,
            **base_kwargs,
            required_section_count=0,
            present_required_section_count=0,
            citation_required_section_count=0,
            cited_required_section_count=0,
            required_section_coverage_percent=None,
            required_section_citation_coverage_percent=None,
            section_checks=(),
            reason_codes=(TEMPLATE_COMPLETENESS_REASON_NO_REQUIRED_SECTIONS,),
        )

    if run.guardrail_status == "no_answer" or run.status == "no_answer":
        section_checks = tuple(
            _empty_section_check(section, reason_codes=()) for section in sections
        )
        return GenerationTemplateCompletenessAssessment(
            status=GENERATION_TEMPLATE_COMPLETENESS_NOT_EVALUATED,
            **base_kwargs,
            required_section_count=len(required_sections),
            present_required_section_count=0,
            citation_required_section_count=0,
            cited_required_section_count=0,
            required_section_coverage_percent=0.0,
            required_section_citation_coverage_percent=None,
            section_checks=section_checks,
            reason_codes=(TEMPLATE_COMPLETENESS_REASON_NO_ANSWER_GUARDRAIL,),
        )

    answer_text = run.answer_text or ""
    parsed_answer = _parse_markdown_sections(answer_text)
    answer_present = bool(" ".join(answer_text.split()))
    section_checks = tuple(
        _section_check(
            section,
            parsed_answer=parsed_answer,
            citation_policy_required=_citation_policy_required(template),
        )
        for section in sections
    )
    present_required = tuple(check for check in section_checks if check.required and check.present)
    citation_required = tuple(
        check for check in section_checks if check.required and check.citation_required
    )
    cited_required = tuple(check for check in citation_required if check.cited)

    reason_codes: list[str] = []
    if not answer_present:
        reason_codes.append(TEMPLATE_COMPLETENESS_REASON_EMPTY_ANSWER)
    if len(present_required) < len(required_sections):
        reason_codes.append(TEMPLATE_COMPLETENESS_REASON_MISSING_REQUIRED_SECTION)
    if len(cited_required) < len(citation_required):
        reason_codes.append(TEMPLATE_COMPLETENESS_REASON_MISSING_SECTION_CITATION)

    return GenerationTemplateCompletenessAssessment(
        status=_status_for_reason_codes(tuple(reason_codes)),
        **base_kwargs,
        required_section_count=len(required_sections),
        present_required_section_count=len(present_required),
        citation_required_section_count=len(citation_required),
        cited_required_section_count=len(cited_required),
        required_section_coverage_percent=_coverage_percent(
            len(present_required),
            len(required_sections),
        ),
        required_section_citation_coverage_percent=_coverage_percent(
            len(cited_required),
            len(citation_required),
        ),
        section_checks=section_checks,
        reason_codes=tuple(reason_codes),
    )


def generation_template_completeness_payload(
    assessment: GenerationTemplateCompletenessAssessment,
) -> dict[str, object]:
    return {
        "contract_version": GENERATION_TEMPLATE_COMPLETENESS_CONTRACT_VERSION,
        "status": assessment.status,
        "template_key": assessment.template_key,
        "template_name": assessment.template_name,
        "template_version": assessment.template_version,
        "document_type": assessment.document_type,
        "output_format": assessment.output_format,
        "required_section_count": assessment.required_section_count,
        "present_required_section_count": assessment.present_required_section_count,
        "citation_required_section_count": assessment.citation_required_section_count,
        "cited_required_section_count": assessment.cited_required_section_count,
        "required_section_coverage_percent": (assessment.required_section_coverage_percent),
        "required_section_citation_coverage_percent": (
            assessment.required_section_citation_coverage_percent
        ),
        "section_checks": [
            {
                "key": check.key,
                "heading": check.heading,
                "required": check.required,
                "present": check.present,
                "citation_required": check.citation_required,
                "cited": check.cited,
                "citation_keys": list(check.citation_keys),
                "reason_codes": list(check.reason_codes),
            }
            for check in assessment.section_checks
        ],
        "reason_codes": list(assessment.reason_codes),
    }


def _not_evaluated(
    *,
    reason_code: str,
    template_key: str | None = None,
    template_name: str | None = None,
    template_version: str | None = None,
    document_type: str | None = None,
    output_format: str | None = None,
) -> GenerationTemplateCompletenessAssessment:
    return GenerationTemplateCompletenessAssessment(
        status=GENERATION_TEMPLATE_COMPLETENESS_NOT_EVALUATED,
        template_key=template_key,
        template_name=template_name,
        template_version=template_version,
        document_type=document_type,
        output_format=output_format,
        required_section_count=0,
        present_required_section_count=0,
        citation_required_section_count=0,
        cited_required_section_count=0,
        required_section_coverage_percent=None,
        required_section_citation_coverage_percent=None,
        section_checks=(),
        reason_codes=(reason_code,),
    )


def _template_snapshot(run: GenerationRunRecord) -> dict[str, Any]:
    request_template = run.request_metadata.get("generation_template")
    if isinstance(request_template, Mapping):
        return dict(request_template)

    response_template = run.response_metadata.get("template")
    if isinstance(response_template, Mapping):
        template = dict(response_template)
        section_keys = _sequence_of_strings(template.get("template_section_keys"))
        required_section_keys = set(
            _sequence_of_strings(template.get("required_template_section_keys"))
        )
        if section_keys:
            template["section_schema"] = [
                {
                    "key": section_key,
                    "heading": section_key,
                    "required": section_key in required_section_keys,
                }
                for section_key in section_keys
            ]
        return template

    return {}


def _template_value(template: Mapping[str, Any], key: str) -> str | None:
    value = template.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _template_sections(template: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_sections = template.get("section_schema")
    if not isinstance(raw_sections, Sequence) or isinstance(raw_sections, (str, bytes)):
        return ()

    sections: list[dict[str, Any]] = []
    for index, raw_section in enumerate(raw_sections, start=1):
        if not isinstance(raw_section, Mapping):
            continue
        key = str(raw_section.get("key") or f"section_{index}").strip()
        if not key:
            key = f"section_{index}"
        heading = str(raw_section.get("heading") or key).strip() or key
        sections.append(
            {
                "key": key,
                "heading": heading,
                "required": bool(raw_section.get("required")),
            }
        )
    return tuple(sections)


def _citation_policy_required(template: Mapping[str, Any]) -> bool:
    policy = template.get("citation_policy")
    if not isinstance(policy, Mapping):
        return False
    return bool(policy.get("required"))


def _section_check(
    section: Mapping[str, Any],
    *,
    parsed_answer: Mapping[str, str],
    citation_policy_required: bool,
) -> TemplateCompletenessSectionCheck:
    key = str(section["key"])
    heading = str(section["heading"])
    required = bool(section["required"])
    text = _section_text(key, heading, parsed_answer)
    present = text is not None
    citation_required = required and citation_policy_required and key not in _TITLE_SECTION_KEYS
    citation_keys = _unique_nonblank(CITATION_KEY_PATTERN.findall(text or ""))
    cited = bool(citation_keys)
    reason_codes: list[str] = []
    if required and not present:
        reason_codes.append(TEMPLATE_COMPLETENESS_REASON_MISSING_REQUIRED_SECTION)
    if present and citation_required and not citation_keys:
        reason_codes.append(TEMPLATE_COMPLETENESS_REASON_MISSING_SECTION_CITATION)
    return TemplateCompletenessSectionCheck(
        key=key,
        heading=heading,
        required=required,
        present=present,
        citation_required=citation_required,
        cited=cited,
        citation_keys=citation_keys,
        reason_codes=tuple(reason_codes),
    )


def _empty_section_check(
    section: Mapping[str, Any],
    *,
    reason_codes: tuple[str, ...],
) -> TemplateCompletenessSectionCheck:
    return TemplateCompletenessSectionCheck(
        key=str(section["key"]),
        heading=str(section["heading"]),
        required=bool(section["required"]),
        present=False,
        citation_required=False,
        cited=False,
        citation_keys=(),
        reason_codes=reason_codes,
    )


def _parse_markdown_sections(answer_text: str) -> dict[str, str]:
    matches = list(_MARKDOWN_HEADING_PATTERN.finditer(answer_text))
    sections: dict[str, str] = {}
    if not matches:
        return sections

    for index, match in enumerate(matches):
        heading = _normalize_heading(match.group(2))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer_text)
        text = answer_text[match.start() : end].strip()
        sections.setdefault(heading, text)
        if match.group(1) == "#":
            sections.setdefault("__title__", text)
    return sections


def _section_text(
    key: str,
    heading: str,
    parsed_answer: Mapping[str, str],
) -> str | None:
    if key in _TITLE_SECTION_KEYS:
        return parsed_answer.get("__title__")
    return parsed_answer.get(_normalize_heading(heading))


def _normalize_heading(value: str) -> str:
    normalized = " ".join(value.strip().split())
    return _HEADING_NUMBER_PREFIX_PATTERN.sub("", normalized).strip()


def _sequence_of_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return _unique_nonblank([str(item) for item in value])


def _unique_nonblank(values: Sequence[str]) -> tuple[str, ...]:
    normalized: dict[str, None] = {}
    for value in values:
        key = str(value).strip()
        if key:
            normalized[key] = None
    return tuple(normalized)


def _coverage_percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100, 2)


def _status_for_reason_codes(reason_codes: tuple[str, ...]) -> str:
    failed_reasons = {
        TEMPLATE_COMPLETENESS_REASON_EMPTY_ANSWER,
        TEMPLATE_COMPLETENESS_REASON_MISSING_REQUIRED_SECTION,
    }
    if any(reason in failed_reasons for reason in reason_codes):
        return GENERATION_TEMPLATE_COMPLETENESS_FAILED
    if reason_codes:
        return GENERATION_TEMPLATE_COMPLETENESS_WARNING
    return GENERATION_TEMPLATE_COMPLETENESS_PASSED
