"""Answer quality checks for grounded generation runs."""

import re
from dataclasses import dataclass

GENERATION_ANSWER_QUALITY_CONTRACT_VERSION = "generation_answer_quality_v1"
GENERATION_ANSWER_QUALITY_PASSED = "passed"
GENERATION_ANSWER_QUALITY_WARNING = "warning"
GENERATION_ANSWER_QUALITY_FAILED = "failed"
GENERATION_ANSWER_QUALITY_NOT_EVALUATED = "not_evaluated"
GENERATION_ANSWER_QUALITY_STATUSES = {
    GENERATION_ANSWER_QUALITY_PASSED,
    GENERATION_ANSWER_QUALITY_WARNING,
    GENERATION_ANSWER_QUALITY_FAILED,
    GENERATION_ANSWER_QUALITY_NOT_EVALUATED,
}

ANSWER_QUALITY_REASON_PROVIDER_ERROR = "provider_error"
ANSWER_QUALITY_REASON_EMPTY_ANSWER = "empty_answer"
ANSWER_QUALITY_REASON_UNEXPECTED_NO_ANSWER = "unexpected_no_answer"
ANSWER_QUALITY_REASON_GUARDRAIL_NO_ANSWER_NOT_REFLECTED = "guardrail_no_answer_not_reflected"
ANSWER_QUALITY_REASON_MISSING_REQUIRED_CITATION = "missing_required_citation"
ANSWER_QUALITY_REASON_PARTIAL_CITATION_COVERAGE = "partial_citation_coverage"
ANSWER_QUALITY_REASON_UNRECOGNIZED_CITATION_KEY = "unrecognized_citation_key"
ANSWER_QUALITY_REASON_UNEXPECTED_CITATION_IN_NO_ANSWER = "unexpected_citation_in_no_answer"
ANSWER_QUALITY_REASON_UNEXPECTED_CITATION_WITHOUT_CONTEXT = "unexpected_citation_without_context"

NO_ANSWER_MARKERS = (
    "답변할 수 없습니다",
    "확인할 수 없습니다",
    "알 수 없습니다",
    "insufficient context",
    "cannot answer",
    "not enough information",
)
CITATION_KEY_PATTERN = re.compile(r"\[(RCP-\d{3})\]")


@dataclass(frozen=True)
class GenerationAnswerQualityAssessment:
    status: str
    answer_present: bool
    no_answer_detected: bool
    requires_citation: bool
    expected_citation_keys: tuple[str, ...]
    cited_citation_keys: tuple[str, ...]
    recognized_citation_keys: tuple[str, ...]
    missing_citation_keys: tuple[str, ...]
    unrecognized_citation_keys: tuple[str, ...]
    citation_coverage_percent: float | None
    reason_codes: tuple[str, ...]


def assess_generation_answer_quality(
    *,
    answer_text: str | None,
    expected_citation_keys: tuple[str, ...] | list[str],
    guardrail_status: str,
    provider_error: bool = False,
) -> GenerationAnswerQualityAssessment:
    """Assess whether a generated answer satisfies minimal grounded-answer rules."""

    expected_keys = _unique_nonblank(expected_citation_keys)
    normalized_answer = " ".join((answer_text or "").split())
    answer_present = bool(normalized_answer)
    cited_keys = _unique_nonblank(CITATION_KEY_PATTERN.findall(normalized_answer))
    recognized_keys = tuple(key for key in cited_keys if key in expected_keys)
    missing_keys = tuple(key for key in expected_keys if key not in recognized_keys)
    unrecognized_keys = tuple(key for key in cited_keys if key not in expected_keys)
    no_answer_detected = _detect_no_answer(normalized_answer)
    requires_citation = guardrail_status != "no_answer" and bool(expected_keys)

    if provider_error:
        return GenerationAnswerQualityAssessment(
            status=GENERATION_ANSWER_QUALITY_NOT_EVALUATED,
            answer_present=answer_present,
            no_answer_detected=no_answer_detected,
            requires_citation=requires_citation,
            expected_citation_keys=expected_keys,
            cited_citation_keys=cited_keys,
            recognized_citation_keys=recognized_keys,
            missing_citation_keys=missing_keys,
            unrecognized_citation_keys=unrecognized_keys,
            citation_coverage_percent=_citation_coverage_percent(
                expected_keys,
                recognized_keys,
            ),
            reason_codes=(ANSWER_QUALITY_REASON_PROVIDER_ERROR,),
        )

    reason_codes: list[str] = []
    if not answer_present:
        reason_codes.append(ANSWER_QUALITY_REASON_EMPTY_ANSWER)
    if guardrail_status == "no_answer":
        if answer_present and not no_answer_detected:
            reason_codes.append(ANSWER_QUALITY_REASON_GUARDRAIL_NO_ANSWER_NOT_REFLECTED)
        if cited_keys:
            reason_codes.append(ANSWER_QUALITY_REASON_UNEXPECTED_CITATION_IN_NO_ANSWER)
    else:
        if no_answer_detected and expected_keys:
            reason_codes.append(ANSWER_QUALITY_REASON_UNEXPECTED_NO_ANSWER)
        if requires_citation and not recognized_keys:
            reason_codes.append(ANSWER_QUALITY_REASON_MISSING_REQUIRED_CITATION)
        elif requires_citation and missing_keys:
            reason_codes.append(ANSWER_QUALITY_REASON_PARTIAL_CITATION_COVERAGE)
        if unrecognized_keys:
            reason_codes.append(ANSWER_QUALITY_REASON_UNRECOGNIZED_CITATION_KEY)
        if cited_keys and not expected_keys:
            reason_codes.append(ANSWER_QUALITY_REASON_UNEXPECTED_CITATION_WITHOUT_CONTEXT)

    return GenerationAnswerQualityAssessment(
        status=_status_for_reason_codes(tuple(reason_codes)),
        answer_present=answer_present,
        no_answer_detected=no_answer_detected,
        requires_citation=requires_citation,
        expected_citation_keys=expected_keys,
        cited_citation_keys=cited_keys,
        recognized_citation_keys=recognized_keys,
        missing_citation_keys=missing_keys,
        unrecognized_citation_keys=unrecognized_keys,
        citation_coverage_percent=_citation_coverage_percent(expected_keys, recognized_keys),
        reason_codes=tuple(reason_codes),
    )


def generation_answer_quality_payload(
    assessment: GenerationAnswerQualityAssessment,
) -> dict[str, object]:
    return {
        "contract_version": GENERATION_ANSWER_QUALITY_CONTRACT_VERSION,
        "status": assessment.status,
        "answer_present": assessment.answer_present,
        "no_answer_detected": assessment.no_answer_detected,
        "requires_citation": assessment.requires_citation,
        "expected_citation_keys": list(assessment.expected_citation_keys),
        "cited_citation_keys": list(assessment.cited_citation_keys),
        "recognized_citation_keys": list(assessment.recognized_citation_keys),
        "missing_citation_keys": list(assessment.missing_citation_keys),
        "unrecognized_citation_keys": list(assessment.unrecognized_citation_keys),
        "citation_coverage_percent": assessment.citation_coverage_percent,
        "reason_codes": list(assessment.reason_codes),
    }


def _unique_nonblank(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: dict[str, None] = {}
    for value in values:
        key = str(value).strip()
        if key:
            normalized[key] = None
    return tuple(normalized)


def _detect_no_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(marker.lower() in lowered for marker in NO_ANSWER_MARKERS)


def _citation_coverage_percent(
    expected_keys: tuple[str, ...],
    recognized_keys: tuple[str, ...],
) -> float | None:
    if not expected_keys:
        return None
    return round(len(recognized_keys) / len(expected_keys) * 100, 2)


def _status_for_reason_codes(reason_codes: tuple[str, ...]) -> str:
    failed_reasons = {
        ANSWER_QUALITY_REASON_EMPTY_ANSWER,
        ANSWER_QUALITY_REASON_UNEXPECTED_NO_ANSWER,
        ANSWER_QUALITY_REASON_GUARDRAIL_NO_ANSWER_NOT_REFLECTED,
        ANSWER_QUALITY_REASON_MISSING_REQUIRED_CITATION,
    }
    if any(reason in failed_reasons for reason in reason_codes):
        return GENERATION_ANSWER_QUALITY_FAILED
    if reason_codes:
        return GENERATION_ANSWER_QUALITY_WARNING
    return GENERATION_ANSWER_QUALITY_PASSED
