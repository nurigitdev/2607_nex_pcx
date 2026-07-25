"""Build generation-ready retrieval context packages from search logs."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from app.core.retrieval_confidence import (
    RetrievalConfidenceAssessment,
    assess_search_log_retrieval_confidence,
)
from app.core.search_logs import (
    SearchLogDetailRecord,
    SearchLogResultDetailRecord,
    SearchLogResultRecord,
    get_search_log_detail,
)
from app.core.search_result_context import (
    SearchResultContextChunk,
    SearchResultSourceContext,
    get_search_result_source_context,
)

DEFAULT_CONTEXT_CHAR_BUDGET = 12_000
MIN_CONTEXT_CHAR_BUDGET = 500
MAX_CONTEXT_CHAR_BUDGET = 50_000
DEFAULT_CONTEXT_MAX_ITEMS = 20
MAX_CONTEXT_MAX_ITEMS = 100
MIN_TRUNCATED_CONTEXT_CHARS = 240

PROFILE_PRIORITY = {
    "reranked_vector_cosine": 0,
    "hybrid_keyword_vector": 1,
    "bm25_keyword": 2,
}


@dataclass(frozen=True)
class RetrievalContextInput:
    search_log_id: int
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET
    include_neighbors: bool = True
    max_items: int = DEFAULT_CONTEXT_MAX_ITEMS


@dataclass(frozen=True)
class RetrievalContextResultReference:
    search_log_result_id: int
    profile_name: str
    search_profile_name: str | None
    retrieval_strategy: str | None
    rank: int
    chunk_id: int
    distance: float | None
    score: float | None
    score_components: dict[str, Any]
    profile_elapsed_ms: int | None
    created_at: datetime


@dataclass(frozen=True)
class RetrievalContextCitation:
    citation_key: str | None
    chunk_id: int
    document_id: int
    file_id: int
    document_title: str | None
    original_file_name: str
    file_ext: str | None
    document_group: str
    chunk_policy_name: str
    chunk_seq: int | None
    heading_path: tuple[str, ...]
    page_no: int | None
    slide_no: int | None
    sheet_name: str | None
    cell_range: str | None
    artifact_id: int | None
    block_id: int | None
    source_anchor: dict[str, Any]
    source_label: str


@dataclass(frozen=True)
class RetrievalContextChunkEntry:
    position: str
    chunk_id: int
    chunk_seq: int
    chunk_text: str
    chunk_preview: str
    char_count: int
    token_count: int | None
    source_anchor: dict[str, Any]


@dataclass(frozen=True)
class RetrievalContextCandidate:
    included: bool
    exclusion_reason: str | None
    citation: RetrievalContextCitation
    primary_result: RetrievalContextResultReference
    supporting_results: tuple[RetrievalContextResultReference, ...]
    chunks: tuple[RetrievalContextChunkEntry, ...]
    context_text: str
    context_char_count: int
    original_context_char_count: int
    truncated: bool = False


@dataclass(frozen=True)
class RetrievalContextSummary:
    candidate_result_count: int
    unique_candidate_count: int
    included_count: int
    excluded_count: int
    duplicate_supporting_result_count: int
    max_context_chars: int
    used_context_chars: int
    remaining_context_chars: int
    truncated_count: int
    source_context_missing_count: int
    include_neighbors: bool
    max_items: int


@dataclass(frozen=True)
class RetrievalContextPackage:
    package_key: str
    search_log: SearchLogDetailRecord
    summary: RetrievalContextSummary
    candidates: tuple[RetrievalContextCandidate, ...]
    generation_context_text: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence_assessment: RetrievalConfidenceAssessment | None = None

    @property
    def included_candidates(self) -> tuple[RetrievalContextCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.included)

    @property
    def excluded_candidates(self) -> tuple[RetrievalContextCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.included)


class InvalidRetrievalContextError(ValueError):
    """Raised when retrieval context package input is invalid."""


def validate_retrieval_context_input(
    context_input: RetrievalContextInput,
) -> RetrievalContextInput:
    if context_input.search_log_id <= 0:
        raise InvalidRetrievalContextError("search_log_id must be greater than 0")
    if context_input.max_context_chars < MIN_CONTEXT_CHAR_BUDGET:
        raise InvalidRetrievalContextError(
            f"max_context_chars must be greater than or equal to {MIN_CONTEXT_CHAR_BUDGET}"
        )
    if context_input.max_context_chars > MAX_CONTEXT_CHAR_BUDGET:
        raise InvalidRetrievalContextError(
            f"max_context_chars must be less than or equal to {MAX_CONTEXT_CHAR_BUDGET}"
        )
    if context_input.max_items <= 0:
        raise InvalidRetrievalContextError("max_items must be greater than 0")
    if context_input.max_items > MAX_CONTEXT_MAX_ITEMS:
        raise InvalidRetrievalContextError(
            f"max_items must be less than or equal to {MAX_CONTEXT_MAX_ITEMS}"
        )
    return context_input


def _result_reference(
    result: SearchLogResultRecord,
) -> RetrievalContextResultReference:
    return RetrievalContextResultReference(
        search_log_result_id=result.search_log_result_id,
        profile_name=result.profile_name,
        search_profile_name=result.search_profile_name,
        retrieval_strategy=result.retrieval_strategy,
        rank=result.rank,
        chunk_id=result.chunk_id,
        distance=result.distance,
        score=result.score,
        score_components=result.score_components,
        profile_elapsed_ms=result.profile_elapsed_ms,
        created_at=result.created_at,
    )


def _profile_sort_name(result: SearchLogResultRecord) -> str:
    return result.search_profile_name or result.retrieval_strategy or result.profile_name


def _profile_priority(result: SearchLogResultRecord) -> tuple[int, str]:
    name = _profile_sort_name(result)
    return (PROFILE_PRIORITY.get(name, 10), name)


def _result_sort_key(result_detail: SearchLogResultDetailRecord) -> tuple[int, str, int, int]:
    result = result_detail.search_log_result
    priority, profile_name = _profile_priority(result)
    return (
        priority,
        profile_name,
        result.rank,
        result.search_log_result_id,
    )


def _chunk_preview(text: str, *, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _context_chunks(
    context: SearchResultSourceContext,
    *,
    include_neighbors: bool,
) -> tuple[SearchResultContextChunk, ...]:
    chunks = tuple(context.chunks)
    if include_neighbors:
        return chunks
    current = tuple(chunk for chunk in chunks if chunk.position == "current")
    return current or chunks[:1]


def _chunk_entry(chunk: SearchResultContextChunk) -> RetrievalContextChunkEntry:
    return RetrievalContextChunkEntry(
        position=chunk.position,
        chunk_id=chunk.chunk_id,
        chunk_seq=chunk.chunk_seq,
        chunk_text=chunk.chunk_text,
        chunk_preview=chunk.chunk_preview,
        char_count=chunk.char_count,
        token_count=chunk.token_count,
        source_anchor=chunk.source_anchor,
    )


def _current_chunk(
    chunks: tuple[SearchResultContextChunk, ...],
) -> SearchResultContextChunk | None:
    return next(
        (chunk for chunk in chunks if chunk.position == "current"), chunks[0] if chunks else None
    )


def _source_label(
    *,
    file_name: str,
    page_no: int | None,
    slide_no: int | None,
    sheet_name: str | None,
    cell_range: str | None,
) -> str:
    parts = [file_name]
    if page_no is not None:
        parts.append(f"p.{page_no}")
    if slide_no is not None:
        parts.append(f"slide {slide_no}")
    if sheet_name:
        parts.append(f"sheet {sheet_name}")
    if cell_range:
        parts.append(f"cell {cell_range}")
    return " / ".join(parts)


def _citation_from_context(
    context: SearchResultSourceContext,
    *,
    citation_key: str | None,
) -> RetrievalContextCitation:
    current = _current_chunk(tuple(context.chunks))
    heading_path = current.heading_path if current is not None else ()
    page_no = current.page_no if current is not None else None
    slide_no = current.slide_no if current is not None else None
    sheet_name = current.sheet_name if current is not None else None
    cell_range = current.cell_range if current is not None else None
    return RetrievalContextCitation(
        citation_key=citation_key,
        chunk_id=context.search_result.chunk_id,
        document_id=context.document.document_id,
        file_id=context.document.file_id,
        document_title=context.document.document_title,
        original_file_name=context.document.original_file_name,
        file_ext=context.document.file_ext,
        document_group=context.document.document_group,
        chunk_policy_name=(current.chunk_policy_name if current else ""),
        chunk_seq=(current.chunk_seq if current else None),
        heading_path=heading_path,
        page_no=page_no,
        slide_no=slide_no,
        sheet_name=sheet_name,
        cell_range=cell_range,
        artifact_id=(current.artifact_id if current else None),
        block_id=(current.block_id if current else None),
        source_anchor=(current.source_anchor if current else {}),
        source_label=_source_label(
            file_name=context.document.original_file_name,
            page_no=page_no,
            slide_no=slide_no,
            sheet_name=sheet_name,
            cell_range=cell_range,
        ),
    )


def _minimal_citation(
    result_detail: SearchLogResultDetailRecord,
    *,
    citation_key: str | None,
) -> RetrievalContextCitation:
    result = result_detail.search_log_result
    return RetrievalContextCitation(
        citation_key=citation_key,
        chunk_id=result.chunk_id,
        document_id=result_detail.document_id,
        file_id=result_detail.file_id,
        document_title=result_detail.document_title,
        original_file_name=result_detail.original_file_name,
        file_ext=result_detail.file_ext,
        document_group=result_detail.document_group,
        chunk_policy_name=result_detail.chunk_policy_name,
        chunk_seq=None,
        heading_path=result_detail.heading_path,
        page_no=result_detail.page_no,
        slide_no=result_detail.slide_no,
        sheet_name=result_detail.sheet_name,
        cell_range=result_detail.cell_range,
        artifact_id=None,
        block_id=None,
        source_anchor={},
        source_label=_source_label(
            file_name=result_detail.original_file_name,
            page_no=result_detail.page_no,
            slide_no=result_detail.slide_no,
            sheet_name=result_detail.sheet_name,
            cell_range=result_detail.cell_range,
        ),
    )


def _render_candidate_context(
    *,
    citation: RetrievalContextCitation,
    primary_result: RetrievalContextResultReference,
    chunks: tuple[RetrievalContextChunkEntry, ...],
) -> str:
    title = citation.document_title or citation.original_file_name
    lines = [
        f"[{citation.citation_key}] {title}",
        f"Source: {citation.source_label}",
        f"Chunk: {citation.chunk_id}",
        f"Profile: {primary_result.profile_name} / Rank: {primary_result.rank}",
    ]
    if citation.heading_path:
        lines.append(f"Heading: {' > '.join(citation.heading_path)}")
    lines.append("")
    for chunk in chunks:
        lines.append(f"{chunk.position}:")
        lines.append(chunk.chunk_text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    if max_chars <= len("... [truncated]"):
        return "", True
    suffix = "\n... [truncated]"
    return text[: max_chars - len(suffix)].rstrip() + suffix, True


def _excluded_candidate(
    result_detail: SearchLogResultDetailRecord,
    *,
    reason: str,
) -> RetrievalContextCandidate:
    result = result_detail.search_log_result
    return RetrievalContextCandidate(
        included=False,
        exclusion_reason=reason,
        citation=_minimal_citation(result_detail, citation_key=None),
        primary_result=_result_reference(result),
        supporting_results=(),
        chunks=(),
        context_text="",
        context_char_count=0,
        original_context_char_count=0,
    )


def _package_key(
    *,
    search_log_id: int,
    context_input: RetrievalContextInput,
    candidates: tuple[RetrievalContextCandidate, ...],
) -> str:
    material = "|".join(
        [
            str(search_log_id),
            str(context_input.max_context_chars),
            str(context_input.include_neighbors),
            str(context_input.max_items),
            ",".join(
                str(candidate.primary_result.search_log_result_id) for candidate in candidates
            ),
        ]
    )
    return sha256(material.encode("utf-8")).hexdigest()[:16]


def build_retrieval_context_package(
    database_url: str,
    context_input: RetrievalContextInput,
) -> RetrievalContextPackage | None:
    """Return a deterministic retrieval context package for generation experiments."""

    validated = validate_retrieval_context_input(context_input)
    search_log = get_search_log_detail(database_url, validated.search_log_id)
    if search_log is None:
        return None

    confidence_assessment = assess_search_log_retrieval_confidence(
        search_log.search_log,
        search_log.results,
    )
    withhold_generation_context = confidence_assessment.withhold_generation_context
    sorted_results = sorted(search_log.results, key=_result_sort_key)
    candidates: list[RetrievalContextCandidate] = []
    candidates_by_chunk_id: dict[int, int] = {}
    used_chars = 0
    included_candidate_count = 0
    source_context_missing_count = 0

    for result_detail in sorted_results:
        result = result_detail.search_log_result
        existing_index = candidates_by_chunk_id.get(result.chunk_id)
        if existing_index is not None:
            existing = candidates[existing_index]
            candidates[existing_index] = RetrievalContextCandidate(
                included=existing.included,
                exclusion_reason=existing.exclusion_reason,
                citation=existing.citation,
                primary_result=existing.primary_result,
                supporting_results=(
                    *existing.supporting_results,
                    _result_reference(result),
                ),
                chunks=existing.chunks,
                context_text=existing.context_text,
                context_char_count=existing.context_char_count,
                original_context_char_count=existing.original_context_char_count,
                truncated=existing.truncated,
            )
            continue

        if len(candidates_by_chunk_id) >= validated.max_items:
            candidates.append(_excluded_candidate(result_detail, reason="max_items_exceeded"))
            continue

        if withhold_generation_context:
            candidates.append(
                _excluded_candidate(
                    result_detail,
                    reason=confidence_assessment.status,
                )
            )
            candidates_by_chunk_id[result.chunk_id] = len(candidates) - 1
            continue

        context = get_search_result_source_context(database_url, result.search_log_result_id)
        if context is None:
            source_context_missing_count += 1
            candidates.append(_excluded_candidate(result_detail, reason="source_context_missing"))
            candidates_by_chunk_id[result.chunk_id] = len(candidates) - 1
            continue

        citation = _citation_from_context(
            context,
            citation_key=f"RCP-{len(candidates_by_chunk_id) + 1:03d}",
        )
        chunk_entries = tuple(
            _chunk_entry(chunk)
            for chunk in _context_chunks(context, include_neighbors=validated.include_neighbors)
        )
        primary_result = _result_reference(result)
        rendered_context = _render_candidate_context(
            citation=citation,
            primary_result=primary_result,
            chunks=chunk_entries,
        )
        original_context_chars = len(rendered_context)
        separator_chars = 2 if included_candidate_count else 0
        remaining_chars = validated.max_context_chars - used_chars - separator_chars
        should_exclude = remaining_chars <= 0 or (
            candidates and remaining_chars < MIN_TRUNCATED_CONTEXT_CHARS
        )
        if should_exclude:
            candidate = RetrievalContextCandidate(
                included=False,
                exclusion_reason="context_budget_exceeded",
                citation=citation,
                primary_result=primary_result,
                supporting_results=(),
                chunks=chunk_entries,
                context_text="",
                context_char_count=0,
                original_context_char_count=original_context_chars,
            )
        else:
            context_text, truncated = _truncate_text(rendered_context, remaining_chars)
            used_chars += separator_chars + len(context_text)
            included_candidate_count += 1
            candidate = RetrievalContextCandidate(
                included=True,
                exclusion_reason=None,
                citation=citation,
                primary_result=primary_result,
                supporting_results=(),
                chunks=chunk_entries,
                context_text=context_text,
                context_char_count=len(context_text),
                original_context_char_count=original_context_chars,
                truncated=truncated,
            )

        candidates.append(candidate)
        candidates_by_chunk_id[result.chunk_id] = len(candidates) - 1

    candidate_tuple = tuple(candidates)
    included_count = sum(1 for candidate in candidate_tuple if candidate.included)
    excluded_count = len(candidate_tuple) - included_count
    duplicate_supporting_count = sum(
        len(candidate.supporting_results) for candidate in candidate_tuple
    )
    generation_text = "\n\n".join(
        candidate.context_text for candidate in candidate_tuple if candidate.included
    )
    summary = RetrievalContextSummary(
        candidate_result_count=len(sorted_results),
        unique_candidate_count=len(candidates_by_chunk_id),
        included_count=included_count,
        excluded_count=excluded_count,
        duplicate_supporting_result_count=duplicate_supporting_count,
        max_context_chars=validated.max_context_chars,
        used_context_chars=len(generation_text),
        remaining_context_chars=max(0, validated.max_context_chars - len(generation_text)),
        truncated_count=sum(1 for candidate in candidate_tuple if candidate.truncated),
        source_context_missing_count=source_context_missing_count,
        include_neighbors=validated.include_neighbors,
        max_items=validated.max_items,
    )
    return RetrievalContextPackage(
        package_key=_package_key(
            search_log_id=search_log.search_log.search_log_id,
            context_input=validated,
            candidates=candidate_tuple,
        ),
        search_log=search_log,
        summary=summary,
        candidates=candidate_tuple,
        generation_context_text=generation_text,
        confidence_assessment=confidence_assessment,
    )
