"""Citation and source-anchor readiness checks for retrieval context packages."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.core.retrieval_context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    DEFAULT_CONTEXT_MAX_ITEMS,
    RetrievalContextCandidate,
    RetrievalContextInput,
    RetrievalContextPackage,
    build_retrieval_context_package,
)

CITATION_READINESS_READY = "ready"
CITATION_READINESS_WARNING = "warning"
CITATION_READINESS_FAILED = "failed"


@dataclass(frozen=True)
class CitationReadinessInput:
    search_log_id: int
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET
    include_neighbors: bool = True
    max_items: int = DEFAULT_CONTEXT_MAX_ITEMS


@dataclass(frozen=True)
class CitationReadinessIssue:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class CitationReadinessCandidate:
    citation_key: str | None
    search_log_result_id: int
    chunk_id: int
    document_id: int
    document_title: str | None
    original_file_name: str
    source_label: str
    included: bool
    status: str
    has_document_identity: bool
    has_chunk_identity: bool
    has_source_anchor: bool
    has_location_hint: bool
    has_lineage_reference: bool
    has_generation_text: bool
    issue_count: int
    issues: tuple[CitationReadinessIssue, ...]


@dataclass(frozen=True)
class CitationReadinessSummary:
    status: str
    total_candidate_count: int
    included_candidate_count: int
    excluded_candidate_count: int
    ready_count: int
    warning_count: int
    failed_count: int
    source_anchor_ready_count: int
    source_anchor_coverage_percent: Decimal
    citation_ready_percent: Decimal
    issue_count: int


@dataclass(frozen=True)
class CitationReadinessReport:
    package: RetrievalContextPackage
    summary: CitationReadinessSummary
    candidates: tuple[CitationReadinessCandidate, ...]


def _percent(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0.00")
    return ((Decimal(numerator) / Decimal(denominator)) * Decimal(100)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def _issue(code: str, severity: str, message: str) -> CitationReadinessIssue:
    return CitationReadinessIssue(code=code, severity=severity, message=message)


def _has_location_hint(candidate: RetrievalContextCandidate) -> bool:
    citation = candidate.citation
    return bool(
        citation.page_no
        or citation.slide_no
        or citation.sheet_name
        or citation.cell_range
        or citation.heading_path
    )


def _has_lineage_reference(candidate: RetrievalContextCandidate) -> bool:
    citation = candidate.citation
    return bool(citation.artifact_id or citation.block_id)


def _has_source_anchor(candidate: RetrievalContextCandidate) -> bool:
    return bool(candidate.citation.source_anchor)


def _candidate_status(issues: tuple[CitationReadinessIssue, ...]) -> str:
    if any(issue.severity == CITATION_READINESS_FAILED for issue in issues):
        return CITATION_READINESS_FAILED
    if any(issue.severity == CITATION_READINESS_WARNING for issue in issues):
        return CITATION_READINESS_WARNING
    return CITATION_READINESS_READY


def assess_citation_candidate(
    candidate: RetrievalContextCandidate,
) -> CitationReadinessCandidate:
    issues: list[CitationReadinessIssue] = []
    citation = candidate.citation

    has_document_identity = bool(citation.document_title or citation.original_file_name)
    has_chunk_identity = citation.chunk_id > 0 and bool(citation.chunk_policy_name)
    has_source_anchor = _has_source_anchor(candidate)
    has_location_hint = _has_location_hint(candidate)
    has_lineage_reference = _has_lineage_reference(candidate)
    has_generation_text = bool(candidate.context_text.strip())

    if not candidate.included:
        issues.append(
            _issue(
                "candidate_excluded",
                CITATION_READINESS_WARNING,
                f"Candidate was excluded from generation context: {candidate.exclusion_reason}",
            )
        )
    if not citation.citation_key and candidate.included:
        issues.append(
            _issue(
                "missing_citation_key",
                CITATION_READINESS_FAILED,
                "Included context requires a stable citation key.",
            )
        )
    if not has_document_identity:
        issues.append(
            _issue(
                "missing_document_identity",
                CITATION_READINESS_FAILED,
                "Citation requires a document title or original file name.",
            )
        )
    if not has_chunk_identity:
        issues.append(
            _issue(
                "missing_chunk_identity",
                CITATION_READINESS_FAILED,
                "Citation requires a chunk id and chunk policy name.",
            )
        )
    if candidate.included and not has_generation_text:
        issues.append(
            _issue(
                "missing_generation_text",
                CITATION_READINESS_FAILED,
                "Included citation has no generation context text.",
            )
        )
    if candidate.included and not (has_source_anchor or has_location_hint or has_lineage_reference):
        issues.append(
            _issue(
                "weak_source_anchor",
                CITATION_READINESS_WARNING,
                "Citation has no source anchor, source lineage reference, or location hint.",
            )
        )
    if candidate.included and not has_lineage_reference:
        issues.append(
            _issue(
                "missing_artifact_block_reference",
                CITATION_READINESS_WARNING,
                "Citation has no artifact_id or block_id lineage reference.",
            )
        )
    if candidate.truncated:
        issues.append(
            _issue(
                "context_truncated",
                CITATION_READINESS_WARNING,
                "Context text was truncated by the package character budget.",
            )
        )

    issue_tuple = tuple(issues)
    return CitationReadinessCandidate(
        citation_key=citation.citation_key,
        search_log_result_id=candidate.primary_result.search_log_result_id,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        document_title=citation.document_title,
        original_file_name=citation.original_file_name,
        source_label=citation.source_label,
        included=candidate.included,
        status=_candidate_status(issue_tuple),
        has_document_identity=has_document_identity,
        has_chunk_identity=has_chunk_identity,
        has_source_anchor=has_source_anchor,
        has_location_hint=has_location_hint,
        has_lineage_reference=has_lineage_reference,
        has_generation_text=has_generation_text,
        issue_count=len(issue_tuple),
        issues=issue_tuple,
    )


def assess_citation_readiness_package(
    package: RetrievalContextPackage,
) -> CitationReadinessReport:
    candidates = tuple(assess_citation_candidate(candidate) for candidate in package.candidates)
    included = tuple(candidate for candidate in candidates if candidate.included)
    ready_count = sum(1 for candidate in candidates if candidate.status == CITATION_READINESS_READY)
    warning_count = sum(
        1 for candidate in candidates if candidate.status == CITATION_READINESS_WARNING
    )
    failed_count = sum(
        1 for candidate in candidates if candidate.status == CITATION_READINESS_FAILED
    )
    source_anchor_ready_count = sum(
        1
        for candidate in included
        if candidate.has_source_anchor
        or candidate.has_location_hint
        or candidate.has_lineage_reference
    )
    issue_count = sum(candidate.issue_count for candidate in candidates)
    if failed_count or not included:
        status = CITATION_READINESS_FAILED
    elif warning_count:
        status = CITATION_READINESS_WARNING
    else:
        status = CITATION_READINESS_READY

    summary = CitationReadinessSummary(
        status=status,
        total_candidate_count=len(candidates),
        included_candidate_count=len(included),
        excluded_candidate_count=len(candidates) - len(included),
        ready_count=ready_count,
        warning_count=warning_count,
        failed_count=failed_count,
        source_anchor_ready_count=source_anchor_ready_count,
        source_anchor_coverage_percent=_percent(source_anchor_ready_count, len(included)),
        citation_ready_percent=_percent(ready_count, len(candidates)),
        issue_count=issue_count,
    )
    return CitationReadinessReport(
        package=package,
        summary=summary,
        candidates=candidates,
    )


def build_citation_readiness_report(
    database_url: str,
    readiness_input: CitationReadinessInput,
) -> CitationReadinessReport | None:
    package = build_retrieval_context_package(
        database_url,
        RetrievalContextInput(
            search_log_id=readiness_input.search_log_id,
            max_context_chars=readiness_input.max_context_chars,
            include_neighbors=readiness_input.include_neighbors,
            max_items=readiness_input.max_items,
        ),
    )
    if package is None:
        return None
    return assess_citation_readiness_package(package)
