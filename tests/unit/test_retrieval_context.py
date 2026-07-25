from datetime import UTC, datetime

import pytest

from app.core import retrieval_context as context_builder
from app.core.retrieval_context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    InvalidRetrievalContextError,
    RetrievalContextInput,
    build_retrieval_context_package,
    validate_retrieval_context_input,
)
from app.core.search_logs import (
    SearchLogDetailRecord,
    SearchLogRecord,
    SearchLogResultDetailRecord,
    SearchLogResultRecord,
)
from app.core.search_result_context import (
    SearchResultContextChunk,
    SearchResultContextReference,
    SearchResultSourceContext,
    SearchResultSourceDocument,
)

NOW = datetime(2026, 7, 25, 9, 30, tzinfo=UTC)


def _search_log(results: tuple[SearchLogResultDetailRecord, ...]) -> SearchLogDetailRecord:
    return SearchLogDetailRecord(
        search_log=SearchLogRecord(
            search_log_id=77,
            query_text="생성 단계에 사용할 근거를 찾아줘",
            normalized_query_text="생성 단계에 사용할 근거를 찾아줘",
            actor_user_id=3,
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"scope": "company"},
            document_group="default",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            strategy_name="reranked_vector_cosine",
            top_k=3,
            similarity_metric="cosine",
            profiles=("bge_m3_1024", "reranked_vector_cosine", "bm25_keyword"),
            query_runtime_metadata={"reranker_profile_name": "qwen3_reranker_4b"},
            total_elapsed_ms=123,
            created_by="unit-test",
            created_by_user_id=3,
            review_tags=(),
            review_memo=None,
            reviewed_by_user_id=None,
            reviewed_at=None,
            created_at=NOW,
        ),
        actor_login_id="alice.member",
        actor_display_name="Alice",
        results=results,
    )


def _result_detail(
    *,
    result_id: int,
    chunk_id: int,
    profile_name: str,
    rank: int,
    score: float,
    search_profile_name: str | None = None,
    retrieval_strategy: str | None = None,
) -> SearchLogResultDetailRecord:
    return SearchLogResultDetailRecord(
        search_log_result=SearchLogResultRecord(
            search_log_result_id=result_id,
            search_log_id=77,
            profile_name=profile_name,
            rank=rank,
            chunk_id=chunk_id,
            distance=None,
            score=score,
            search_profile_name=search_profile_name,
            retrieval_strategy=retrieval_strategy,
            score_components={"score": score},
            profile_elapsed_ms=11,
            created_at=NOW,
        ),
        document_id=100 + chunk_id,
        file_id=200 + chunk_id,
        chunk_preview=f"chunk {chunk_id} preview",
        content_hash=f"hash-{chunk_id}",
        chunk_policy_name="heading_512_64",
        heading_path=("Section", f"Chunk {chunk_id}"),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title=f"Document {chunk_id}",
        document_group="default",
        original_file_name=f"document-{chunk_id}.md",
        file_ext=".md",
        feedback=(),
    )


def _context(
    *,
    result_id: int,
    chunk_id: int,
    text: str,
    page_no: int | None = None,
    slide_no: int | None = None,
    sheet_name: str | None = None,
    cell_range: str | None = None,
) -> SearchResultSourceContext:
    document = SearchResultSourceDocument(
        document_id=100 + chunk_id,
        file_id=200 + chunk_id,
        document_title=f"Document {chunk_id}",
        document_group="default",
        document_status="parsed",
        original_file_name=f"document-{chunk_id}.md",
        file_ext=".md",
        storage_path=f"/tmp/document-{chunk_id}.md",
    )
    chunks = (
        SearchResultContextChunk(
            position="previous",
            chunk_id=chunk_id - 1,
            document_id=document.document_id,
            chunk_seq=0,
            chunk_text=f"previous {chunk_id}",
            chunk_preview=f"previous {chunk_id}",
            content_hash=f"prev-{chunk_id}",
            chunk_policy_name="heading_512_64",
            artifact_id=None,
            block_id=None,
            chunk_type="text",
            heading_path=("Section",),
            source_anchor={"line": 1},
            page_no=None,
            slide_no=None,
            sheet_name=None,
            cell_range=None,
            source_char_start=None,
            source_char_end=None,
            token_count=2,
            char_count=len(f"previous {chunk_id}"),
            prev_chunk_id=None,
            next_chunk_id=chunk_id,
            metadata={},
        ),
        SearchResultContextChunk(
            position="current",
            chunk_id=chunk_id,
            document_id=document.document_id,
            chunk_seq=1,
            chunk_text=text,
            chunk_preview=text[:80],
            content_hash=f"current-{chunk_id}",
            chunk_policy_name="heading_512_64",
            artifact_id=300 + chunk_id,
            block_id=400 + chunk_id,
            chunk_type="text",
            heading_path=("Section", f"Chunk {chunk_id}"),
            source_anchor={"line": 2},
            page_no=page_no,
            slide_no=slide_no,
            sheet_name=sheet_name,
            cell_range=cell_range,
            source_char_start=10,
            source_char_end=10 + len(text),
            token_count=8,
            char_count=len(text),
            prev_chunk_id=chunk_id - 1,
            next_chunk_id=chunk_id + 1,
            metadata={},
        ),
        SearchResultContextChunk(
            position="next",
            chunk_id=chunk_id + 1,
            document_id=document.document_id,
            chunk_seq=2,
            chunk_text=f"next {chunk_id}",
            chunk_preview=f"next {chunk_id}",
            content_hash=f"next-{chunk_id}",
            chunk_policy_name="heading_512_64",
            artifact_id=None,
            block_id=None,
            chunk_type="text",
            heading_path=("Section",),
            source_anchor={"line": 3},
            page_no=None,
            slide_no=None,
            sheet_name=None,
            cell_range=None,
            source_char_start=None,
            source_char_end=None,
            token_count=2,
            char_count=len(f"next {chunk_id}"),
            prev_chunk_id=chunk_id,
            next_chunk_id=None,
            metadata={},
        ),
    )
    return SearchResultSourceContext(
        search_result=SearchResultContextReference(
            search_log_result_id=result_id,
            search_log_id=77,
            profile_name="profile",
            rank=1,
            chunk_id=chunk_id,
            distance=None,
            score=0.9,
            profile_elapsed_ms=5,
            created_at=NOW,
        ),
        document=document,
        chunks=chunks,
        source_block=None,
        source_artifact=None,
    )


def test_build_retrieval_context_package_prioritizes_reranked_and_merges_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = (
        _result_detail(result_id=1, chunk_id=10, profile_name="bge_m3_1024", rank=1, score=0.7),
        _result_detail(
            result_id=2,
            chunk_id=10,
            profile_name="reranked_vector_cosine",
            search_profile_name="reranked_vector_cosine",
            retrieval_strategy="reranked",
            rank=1,
            score=0.95,
        ),
        _result_detail(
            result_id=3,
            chunk_id=20,
            profile_name="bm25_keyword",
            search_profile_name="bm25_keyword",
            retrieval_strategy="bm25_keyword",
            rank=1,
            score=0.4,
        ),
    )
    contexts = {
        2: _context(result_id=2, chunk_id=10, text="reranked evidence"),
        3: _context(result_id=3, chunk_id=20, text="keyword evidence", page_no=4),
    }
    monkeypatch.setattr(context_builder, "get_search_log_detail", lambda *_: _search_log(results))
    monkeypatch.setattr(
        context_builder,
        "get_search_result_source_context",
        lambda _database_url, result_id: contexts[result_id],
    )

    package = build_retrieval_context_package(
        "postgresql://example/test",
        RetrievalContextInput(search_log_id=77, max_context_chars=DEFAULT_CONTEXT_CHAR_BUDGET),
    )

    assert package is not None
    assert package.summary.candidate_result_count == 3
    assert package.summary.unique_candidate_count == 2
    assert package.summary.duplicate_supporting_result_count == 1
    assert [item.citation.citation_key for item in package.included_candidates] == [
        "RCP-001",
        "RCP-002",
    ]
    assert package.included_candidates[0].primary_result.profile_name == "reranked_vector_cosine"
    assert package.included_candidates[0].supporting_results[0].profile_name == "bge_m3_1024"
    assert package.included_candidates[1].citation.source_label == "document-20.md / p.4"
    assert "[RCP-001]" in package.generation_context_text
    assert "reranked evidence" in package.generation_context_text


def test_build_retrieval_context_package_can_exclude_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result_detail(
        result_id=4,
        chunk_id=30,
        profile_name="kure_v1_1024",
        rank=1,
        score=0.8,
    )
    monkeypatch.setattr(context_builder, "get_search_log_detail", lambda *_: _search_log((result,)))
    monkeypatch.setattr(
        context_builder,
        "get_search_result_source_context",
        lambda *_: _context(result_id=4, chunk_id=30, text="current only"),
    )

    package = build_retrieval_context_package(
        "postgresql://example/test",
        RetrievalContextInput(search_log_id=77, include_neighbors=False),
    )

    assert package is not None
    assert [chunk.position for chunk in package.included_candidates[0].chunks] == ["current"]
    assert "previous" not in package.generation_context_text
    assert "current only" in package.generation_context_text


def test_build_retrieval_context_package_truncates_first_item_and_excludes_later_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _result_detail(result_id=5, chunk_id=40, profile_name="kure_v1_1024", rank=1, score=0.8)
    second = _result_detail(result_id=6, chunk_id=50, profile_name="bge_m3_1024", rank=1, score=0.7)
    monkeypatch.setattr(
        context_builder, "get_search_log_detail", lambda *_: _search_log((first, second))
    )
    monkeypatch.setattr(
        context_builder,
        "get_search_result_source_context",
        lambda _database_url, result_id: _context(
            result_id=result_id,
            chunk_id=40 if result_id == 5 else 50,
            text="long evidence " * 100,
        ),
    )

    package = build_retrieval_context_package(
        "postgresql://example/test",
        RetrievalContextInput(search_log_id=77, max_context_chars=500),
    )

    assert package is not None
    assert package.summary.included_count == 1
    assert package.summary.excluded_count == 1
    assert package.summary.truncated_count == 1
    assert package.included_candidates[0].truncated is True
    assert package.excluded_candidates[0].exclusion_reason == "context_budget_exceeded"
    assert len(package.generation_context_text) <= 500


def test_build_retrieval_context_package_reports_missing_source_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result_detail(
        result_id=7, chunk_id=60, profile_name="kure_v1_1024", rank=1, score=0.8
    )
    monkeypatch.setattr(context_builder, "get_search_log_detail", lambda *_: _search_log((result,)))
    monkeypatch.setattr(context_builder, "get_search_result_source_context", lambda *_: None)

    package = build_retrieval_context_package(
        "postgresql://example/test",
        RetrievalContextInput(search_log_id=77),
    )

    assert package is not None
    assert package.summary.source_context_missing_count == 1
    assert package.excluded_candidates[0].exclusion_reason == "source_context_missing"
    assert package.excluded_candidates[0].citation.source_label == "document-60.md"


def test_build_retrieval_context_package_returns_none_for_missing_search_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_builder, "get_search_log_detail", lambda *_: None)

    assert (
        build_retrieval_context_package(
            "postgresql://example/test",
            RetrievalContextInput(search_log_id=77),
        )
        is None
    )


@pytest.mark.parametrize(
    "context_input",
    [
        RetrievalContextInput(search_log_id=0),
        RetrievalContextInput(search_log_id=1, max_context_chars=499),
        RetrievalContextInput(search_log_id=1, max_context_chars=50_001),
        RetrievalContextInput(search_log_id=1, max_items=0),
        RetrievalContextInput(search_log_id=1, max_items=101),
    ],
)
def test_validate_retrieval_context_input_rejects_invalid_values(
    context_input: RetrievalContextInput,
) -> None:
    with pytest.raises(InvalidRetrievalContextError):
        validate_retrieval_context_input(context_input)
