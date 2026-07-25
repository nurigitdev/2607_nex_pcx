"""Confidence guardrails for deciding whether retrieval can support generation."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

RETRIEVAL_CONFIDENCE_ANSWERABLE = "answerable"
RETRIEVAL_CONFIDENCE_LOW = "low_confidence"
RETRIEVAL_CONFIDENCE_NO_CONTEXT = "no_relevant_context"
RETRIEVAL_CONFIDENCE_FAILED = "failed"

DEFAULT_VECTOR_MIN_SCORE = 0.5
DEFAULT_BM25_MIN_SCORE = 0.01
DEFAULT_RERANKER_MIN_SCORE = 0.0


@dataclass(frozen=True)
class RetrievalConfidenceThresholds:
    vector_min_score: float = DEFAULT_VECTOR_MIN_SCORE
    bm25_min_score: float = DEFAULT_BM25_MIN_SCORE
    reranker_min_score: float = DEFAULT_RERANKER_MIN_SCORE


@dataclass(frozen=True)
class RetrievalConfidenceProfileAssessment:
    profile_name: str
    status: str
    answerable: bool
    result_count: int
    top_rank: int | None
    top_chunk_id: int | None
    retrieval_strategy: str | None
    top_score: float | None
    top_source_score: float | None
    top_reranker_score: float | None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalConfidenceAssessment:
    status: str
    answerable: bool
    withhold_generation_context: bool
    profile_count: int
    result_count: int
    answerable_profile_count: int
    low_confidence_profile_count: int
    no_context_profile_count: int
    failed_profile_count: int
    reason_codes: tuple[str, ...]
    thresholds: RetrievalConfidenceThresholds = field(default_factory=RetrievalConfidenceThresholds)
    profiles: tuple[RetrievalConfidenceProfileAssessment, ...] = ()


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _result_rank(result: object) -> int:
    rank = _int_or_none(getattr(result, "rank", None))
    return rank if rank is not None else 0


def _score_components(result: object) -> dict[str, Any]:
    components = getattr(result, "score_components", {})
    return dict(components) if isinstance(components, dict) else {}


def _retrieval_strategy(profile_name: str, result: object, components: dict[str, Any]) -> str:
    strategy = (
        getattr(result, "retrieval_strategy", None)
        or getattr(result, "search_profile_name", None)
        or profile_name
    )
    if strategy:
        return str(strategy)
    if components.get("reranker_model_id") or components.get("raw_cross_encoder_score"):
        return "reranked"
    return profile_name


def _is_reranked(profile_name: str, strategy: str, components: dict[str, Any]) -> bool:
    return (
        profile_name == "reranked_vector_cosine"
        or strategy == "reranked"
        or bool(components.get("reranker_model_id"))
        or "raw_cross_encoder_score" in components
    )


def _is_bm25(profile_name: str, strategy: str) -> bool:
    return profile_name == "bm25_keyword" or strategy == "bm25_keyword"


def _is_hybrid(profile_name: str, strategy: str) -> bool:
    return profile_name == "hybrid_keyword_vector" or strategy == "hybrid_keyword_vector"


def _threshold_reason(
    *,
    score: float | None,
    threshold: float,
    weak_code: str,
    missing_code: str,
) -> str | None:
    if score is None:
        return missing_code
    if score < threshold:
        return weak_code
    return None


def assess_profile_retrieval_confidence(
    profile_name: str,
    results: tuple[object, ...] | list[object],
    *,
    profile_status: str = "succeeded",
    thresholds: RetrievalConfidenceThresholds | None = None,
) -> RetrievalConfidenceProfileAssessment:
    """Assess one profile's top result without mutating or filtering results."""

    active_thresholds = thresholds or RetrievalConfidenceThresholds()
    if profile_status == "failed":
        return RetrievalConfidenceProfileAssessment(
            profile_name=profile_name,
            status=RETRIEVAL_CONFIDENCE_FAILED,
            answerable=False,
            result_count=0,
            top_rank=None,
            top_chunk_id=None,
            retrieval_strategy=None,
            top_score=None,
            top_source_score=None,
            top_reranker_score=None,
            reason_codes=("profile_failed",),
        )

    sorted_results = tuple(sorted(results, key=_result_rank))
    if not sorted_results:
        return RetrievalConfidenceProfileAssessment(
            profile_name=profile_name,
            status=RETRIEVAL_CONFIDENCE_NO_CONTEXT,
            answerable=False,
            result_count=0,
            top_rank=None,
            top_chunk_id=None,
            retrieval_strategy=None,
            top_score=None,
            top_source_score=None,
            top_reranker_score=None,
            reason_codes=("no_results",),
        )

    top_result = sorted_results[0]
    components = _score_components(top_result)
    top_score = _float_or_none(getattr(top_result, "score", None))
    top_source_score = _float_or_none(components.get("source_score"))
    raw_reranker_score = _float_or_none(components.get("raw_cross_encoder_score"))
    top_reranker_score = raw_reranker_score if raw_reranker_score is not None else top_score
    strategy = _retrieval_strategy(profile_name, top_result, components)
    reason_codes: list[str] = []
    answerable = False

    if _is_reranked(profile_name, strategy, components):
        source_reason = _threshold_reason(
            score=top_source_score,
            threshold=active_thresholds.vector_min_score,
            weak_code="weak_source_vector_score",
            missing_code="missing_source_vector_score",
        )
        reranker_reason = _threshold_reason(
            score=top_reranker_score,
            threshold=active_thresholds.reranker_min_score,
            weak_code="weak_reranker_score",
            missing_code="missing_reranker_score",
        )
        answerable = source_reason is None or reranker_reason is None
        if not answerable:
            reason_codes.extend(code for code in (source_reason, reranker_reason) if code)
    elif _is_hybrid(profile_name, strategy):
        source_count = _int_or_none(components.get("source_count")) or 0
        vector_score = _float_or_none(components.get("vector_score"))
        keyword_score = _float_or_none(components.get("keyword_score"))
        answerable = (
            source_count >= 2
            or (vector_score is not None and vector_score >= active_thresholds.vector_min_score)
            or (keyword_score is not None and keyword_score >= active_thresholds.bm25_min_score)
        )
        if not answerable:
            if source_count < 2:
                reason_codes.append("single_hybrid_source")
            if vector_score is None or vector_score < active_thresholds.vector_min_score:
                reason_codes.append("weak_vector_score")
            if keyword_score is None or keyword_score < active_thresholds.bm25_min_score:
                reason_codes.append("weak_keyword_score")
    elif _is_bm25(profile_name, strategy):
        reason = _threshold_reason(
            score=top_score,
            threshold=active_thresholds.bm25_min_score,
            weak_code="weak_bm25_score",
            missing_code="missing_bm25_score",
        )
        answerable = reason is None
        if reason is not None:
            reason_codes.append(reason)
    else:
        reason = _threshold_reason(
            score=top_score,
            threshold=active_thresholds.vector_min_score,
            weak_code="weak_vector_score",
            missing_code="missing_vector_score",
        )
        answerable = reason is None
        if reason is not None:
            reason_codes.append(reason)

    return RetrievalConfidenceProfileAssessment(
        profile_name=profile_name,
        status=RETRIEVAL_CONFIDENCE_ANSWERABLE if answerable else RETRIEVAL_CONFIDENCE_LOW,
        answerable=answerable,
        result_count=len(sorted_results),
        top_rank=_int_or_none(getattr(top_result, "rank", None)),
        top_chunk_id=_int_or_none(getattr(top_result, "chunk_id", None)),
        retrieval_strategy=strategy,
        top_score=top_score,
        top_source_score=top_source_score,
        top_reranker_score=(
            top_reranker_score if _is_reranked(profile_name, strategy, components) else None
        ),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
    )


def assess_retrieval_confidence(
    profile_assessments: tuple[RetrievalConfidenceProfileAssessment, ...],
    *,
    thresholds: RetrievalConfidenceThresholds | None = None,
) -> RetrievalConfidenceAssessment:
    active_thresholds = thresholds or RetrievalConfidenceThresholds()
    result_count = sum(profile.result_count for profile in profile_assessments)
    answerable_profiles = tuple(profile for profile in profile_assessments if profile.answerable)
    low_profiles = tuple(
        profile for profile in profile_assessments if profile.status == RETRIEVAL_CONFIDENCE_LOW
    )
    no_context_profiles = tuple(
        profile
        for profile in profile_assessments
        if profile.status == RETRIEVAL_CONFIDENCE_NO_CONTEXT
    )
    failed_profiles = tuple(
        profile for profile in profile_assessments if profile.status == RETRIEVAL_CONFIDENCE_FAILED
    )
    reason_codes = tuple(
        dict.fromkeys(code for profile in profile_assessments for code in profile.reason_codes)
    )
    if answerable_profiles:
        status = RETRIEVAL_CONFIDENCE_ANSWERABLE
    elif result_count > 0:
        status = RETRIEVAL_CONFIDENCE_LOW
    else:
        status = RETRIEVAL_CONFIDENCE_NO_CONTEXT

    return RetrievalConfidenceAssessment(
        status=status,
        answerable=status == RETRIEVAL_CONFIDENCE_ANSWERABLE,
        withhold_generation_context=status != RETRIEVAL_CONFIDENCE_ANSWERABLE,
        profile_count=len(profile_assessments),
        result_count=result_count,
        answerable_profile_count=len(answerable_profiles),
        low_confidence_profile_count=len(low_profiles),
        no_context_profile_count=len(no_context_profiles),
        failed_profile_count=len(failed_profiles),
        reason_codes=reason_codes,
        thresholds=active_thresholds,
        profiles=profile_assessments,
    )


def assess_search_compare_retrieval_confidence(
    profile_results: tuple[object, ...] | list[object],
    *,
    thresholds: RetrievalConfidenceThresholds | None = None,
) -> RetrievalConfidenceAssessment:
    active_thresholds = thresholds or RetrievalConfidenceThresholds()
    profile_assessments = tuple(
        assess_profile_retrieval_confidence(
            str(getattr(profile, "profile_name", "")),
            tuple(getattr(profile, "results", ())),
            profile_status=str(getattr(profile, "status", "succeeded")),
            thresholds=active_thresholds,
        )
        for profile in profile_results
    )
    return assess_retrieval_confidence(profile_assessments, thresholds=active_thresholds)


def assess_search_log_retrieval_confidence(
    search_log: object,
    result_details: tuple[object, ...] | list[object],
    *,
    thresholds: RetrievalConfidenceThresholds | None = None,
) -> RetrievalConfidenceAssessment:
    active_thresholds = thresholds or RetrievalConfidenceThresholds()
    metadata = getattr(search_log, "query_runtime_metadata", {}) or {}
    profile_failures = metadata.get("profile_failures", {}) if isinstance(metadata, dict) else {}
    failed_profile_names = set(profile_failures) if isinstance(profile_failures, dict) else set()
    profile_names = tuple(str(profile) for profile in getattr(search_log, "profiles", ()))
    grouped_results: dict[str, list[object]] = {profile_name: [] for profile_name in profile_names}
    for detail in result_details:
        result = getattr(detail, "search_log_result", detail)
        profile_name = str(getattr(result, "profile_name", ""))
        grouped_results.setdefault(profile_name, []).append(result)
    profile_names = tuple(dict.fromkeys((*profile_names, *grouped_results)))

    profile_assessments = tuple(
        assess_profile_retrieval_confidence(
            profile_name,
            tuple(grouped_results.get(profile_name, ())),
            profile_status="failed" if profile_name in failed_profile_names else "succeeded",
            thresholds=active_thresholds,
        )
        for profile_name in profile_names
    )
    return assess_retrieval_confidence(profile_assessments, thresholds=active_thresholds)


def retrieval_confidence_profile_payload(
    profile: RetrievalConfidenceProfileAssessment,
) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "status": profile.status,
        "answerable": profile.answerable,
        "result_count": profile.result_count,
        "top_rank": profile.top_rank,
        "top_chunk_id": profile.top_chunk_id,
        "retrieval_strategy": profile.retrieval_strategy,
        "top_score": profile.top_score,
        "top_source_score": profile.top_source_score,
        "top_reranker_score": profile.top_reranker_score,
        "reason_codes": list(profile.reason_codes),
    }


def retrieval_confidence_assessment_payload(
    assessment: RetrievalConfidenceAssessment | None,
) -> dict[str, object]:
    if assessment is None:
        return {}
    return {
        "status": assessment.status,
        "answerable": assessment.answerable,
        "withhold_generation_context": assessment.withhold_generation_context,
        "profile_count": assessment.profile_count,
        "result_count": assessment.result_count,
        "answerable_profile_count": assessment.answerable_profile_count,
        "low_confidence_profile_count": assessment.low_confidence_profile_count,
        "no_context_profile_count": assessment.no_context_profile_count,
        "failed_profile_count": assessment.failed_profile_count,
        "reason_codes": list(assessment.reason_codes),
        "thresholds": {
            "vector_min_score": assessment.thresholds.vector_min_score,
            "bm25_min_score": assessment.thresholds.bm25_min_score,
            "reranker_min_score": assessment.thresholds.reranker_min_score,
        },
        "profiles": [
            retrieval_confidence_profile_payload(profile) for profile in assessment.profiles
        ],
    }
