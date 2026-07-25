from datetime import UTC, datetime

from app.core import citation_readiness
from app.core.citation_readiness import (
    CITATION_READINESS_FAILED,
    CITATION_READINESS_READY,
    CITATION_READINESS_WARNING,
    CitationReadinessInput,
    assess_citation_candidate,
    assess_citation_readiness_package,
    build_citation_readiness_report,
)
from app.core.retrieval_context import (
    RetrievalContextCandidate,
    RetrievalContextCitation,
    RetrievalContextChunkEntry,
    RetrievalContextPackage,
    RetrievalContextResultReference,
    RetrievalContextSummary,
)
from app.core.search_logs import SearchLogDetailRecord, SearchLogRecord

NOW = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def _result() -> RetrievalContextResultReference:
    return RetrievalContextResultReference(
        search_log_result_id=10,
        profile_name="reranked_vector_cosine",
        search_profile_name="reranked_vector_cosine",
        retrieval_strategy="reranked",
        rank=1,
        chunk_id=20,
        distance=None,
        score=0.91,
        score_components={},
        profile_elapsed_ms=8,
        created_at=NOW,
    )


def _citation(**overrides) -> RetrievalContextCitation:
    values = {
        "citation_key": "RCP-001",
        "chunk_id": 20,
        "document_id": 30,
        "file_id": 40,
        "document_title": "Readiness Doc",
        "original_file_name": "readiness.md",
        "file_ext": ".md",
        "document_group": "default",
        "chunk_policy_name": "heading_512_64",
        "chunk_seq": 1,
        "heading_path": ("Readiness",),
        "page_no": 2,
        "slide_no": None,
        "sheet_name": None,
        "cell_range": None,
        "artifact_id": 50,
        "block_id": 60,
        "source_anchor": {"start_line": 3},
        "source_label": "readiness.md / p.2",
    }
    values.update(overrides)
    return RetrievalContextCitation(**values)


def _candidate(**overrides) -> RetrievalContextCandidate:
    values = {
        "included": True,
        "exclusion_reason": None,
        "citation": _citation(),
        "primary_result": _result(),
        "supporting_results": (),
        "chunks": (
            RetrievalContextChunkEntry(
                position="current",
                chunk_id=20,
                chunk_seq=1,
                chunk_text="ready context",
                chunk_preview="ready context",
                char_count=13,
                token_count=2,
                source_anchor={"start_line": 3},
            ),
        ),
        "context_text": "[RCP-001] ready context",
        "context_char_count": 23,
        "original_context_char_count": 23,
        "truncated": False,
    }
    values.update(overrides)
    return RetrievalContextCandidate(**values)


def _search_log() -> SearchLogDetailRecord:
    return SearchLogDetailRecord(
        search_log=SearchLogRecord(
            search_log_id=77,
            query_text="citation readiness",
            normalized_query_text="citation readiness",
            actor_user_id=1,
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={},
            document_group="default",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            strategy_name="reranked_vector_cosine",
            top_k=3,
            similarity_metric="cosine",
            profiles=("reranked_vector_cosine",),
            query_runtime_metadata={},
            total_elapsed_ms=42,
            created_by=None,
            created_by_user_id=None,
            review_tags=(),
            review_memo=None,
            reviewed_by_user_id=None,
            reviewed_at=None,
            created_at=NOW,
        ),
        actor_login_id="pcx.admin",
        actor_display_name="PCX Admin",
        results=(),
    )


def _package(candidates: tuple[RetrievalContextCandidate, ...]) -> RetrievalContextPackage:
    return RetrievalContextPackage(
        package_key="package-key",
        search_log=_search_log(),
        summary=RetrievalContextSummary(
            candidate_result_count=len(candidates),
            unique_candidate_count=len(candidates),
            included_count=sum(1 for candidate in candidates if candidate.included),
            excluded_count=sum(1 for candidate in candidates if not candidate.included),
            duplicate_supporting_result_count=0,
            max_context_chars=12000,
            used_context_chars=100,
            remaining_context_chars=11900,
            truncated_count=sum(1 for candidate in candidates if candidate.truncated),
            source_context_missing_count=0,
            include_neighbors=True,
            max_items=20,
        ),
        candidates=candidates,
        generation_context_text="\n\n".join(candidate.context_text for candidate in candidates),
        generated_at=NOW,
    )


def test_assess_citation_candidate_returns_ready_for_complete_citation() -> None:
    result = assess_citation_candidate(_candidate())

    assert result.status == CITATION_READINESS_READY
    assert result.issue_count == 0
    assert result.has_source_anchor is True
    assert result.has_lineage_reference is True


def test_assess_citation_candidate_warns_for_weak_lineage_and_truncation() -> None:
    candidate = _candidate(
        citation=_citation(
            source_anchor={},
            heading_path=(),
            page_no=None,
            artifact_id=None,
            block_id=None,
            source_label="readiness.md",
        ),
        truncated=True,
    )

    result = assess_citation_candidate(candidate)

    assert result.status == CITATION_READINESS_WARNING
    assert {issue.code for issue in result.issues} == {
        "weak_source_anchor",
        "missing_artifact_block_reference",
        "context_truncated",
    }


def test_assess_citation_candidate_fails_for_missing_key_or_text() -> None:
    candidate = _candidate(
        citation=_citation(citation_key=None, document_title=None, original_file_name=""),
        context_text="",
    )

    result = assess_citation_candidate(candidate)

    assert result.status == CITATION_READINESS_FAILED
    assert {issue.code for issue in result.issues} >= {
        "missing_citation_key",
        "missing_document_identity",
        "missing_generation_text",
    }


def test_assess_citation_candidate_warns_for_excluded_candidate() -> None:
    candidate = _candidate(
        included=False,
        exclusion_reason="context_budget_exceeded",
        context_text="",
    )

    result = assess_citation_candidate(candidate)

    assert result.status == CITATION_READINESS_WARNING
    assert result.issues[0].code == "candidate_excluded"


def test_assess_citation_readiness_package_summarizes_status_and_percentages() -> None:
    ready = _candidate()
    warning = _candidate(
        citation=_citation(artifact_id=None, block_id=None),
    )
    failed = _candidate(
        citation=_citation(citation_key=None),
    )

    report = assess_citation_readiness_package(_package((ready, warning, failed)))

    assert report.summary.status == CITATION_READINESS_FAILED
    assert report.summary.total_candidate_count == 3
    assert report.summary.ready_count == 1
    assert report.summary.warning_count == 1
    assert report.summary.failed_count == 1
    assert str(report.summary.citation_ready_percent) == "33.33"
    assert str(report.summary.source_anchor_coverage_percent) == "100.00"


def test_assess_citation_readiness_package_fails_when_no_context_is_included() -> None:
    excluded = _candidate(included=False, exclusion_reason="max_items_exceeded", context_text="")

    report = assess_citation_readiness_package(_package((excluded,)))

    assert report.summary.status == CITATION_READINESS_FAILED
    assert report.summary.included_candidate_count == 0


def test_build_citation_readiness_report_uses_retrieval_context_package(
    monkeypatch,
) -> None:
    package = _package((_candidate(),))
    monkeypatch.setattr(
        citation_readiness,
        "build_retrieval_context_package",
        lambda _database_url, context_input: package if context_input.search_log_id == 77 else None,
    )

    report = build_citation_readiness_report(
        "postgresql://example/test",
        CitationReadinessInput(search_log_id=77, max_context_chars=3000),
    )
    missing = build_citation_readiness_report(
        "postgresql://example/test",
        CitationReadinessInput(search_log_id=78),
    )

    assert report is not None
    assert report.summary.status == CITATION_READINESS_READY
    assert missing is None
