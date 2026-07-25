"""Source context lookup for stored search results."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect

CONTEXT_PREVIEW_CHARS = 360


@dataclass(frozen=True)
class SearchResultContextReference:
    search_log_result_id: int
    search_log_id: int
    profile_name: str
    rank: int
    chunk_id: int
    distance: float | None
    score: float | None
    profile_elapsed_ms: int | None
    created_at: datetime


@dataclass(frozen=True)
class SearchResultSourceDocument:
    document_id: int
    file_id: int
    document_title: str | None
    document_group: str
    document_status: str
    original_file_name: str
    file_ext: str | None
    storage_path: str


@dataclass(frozen=True)
class SearchResultContextChunk:
    position: str
    chunk_id: int
    document_id: int
    chunk_seq: int
    chunk_text: str
    chunk_preview: str
    content_hash: str
    chunk_policy_name: str
    artifact_id: int | None
    block_id: int | None
    chunk_type: str
    heading_path: tuple[str, ...]
    source_anchor: dict[str, Any]
    page_no: int | None
    slide_no: int | None
    sheet_name: str | None
    cell_range: str | None
    source_char_start: int | None
    source_char_end: int | None
    token_count: int | None
    char_count: int
    prev_chunk_id: int | None
    next_chunk_id: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResultSourceBlock:
    block_id: int
    artifact_id: int
    document_id: int
    parent_block_id: int | None
    block_seq: int
    block_type: str
    content_preview: str | None
    content_markdown_preview: str | None
    heading_path: tuple[str, ...]
    source_anchor: dict[str, Any]
    page_no: int | None
    slide_no: int | None
    sheet_name: str | None
    cell_range: str | None
    char_start: int | None
    char_end: int | None
    token_count: int | None
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class SearchResultSourceArtifact:
    artifact_id: int
    extraction_run_id: int | None
    file_id: int
    document_id: int | None
    artifact_type: str
    content_preview: str | None
    content_length: int | None
    storage_path: str | None
    content_hash: str | None
    size_bytes: int | None
    language: str | None
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class SearchResultSourceContext:
    search_result: SearchResultContextReference
    document: SearchResultSourceDocument
    chunks: tuple[SearchResultContextChunk, ...]
    source_block: SearchResultSourceBlock | None
    source_artifact: SearchResultSourceArtifact | None


class InvalidSearchResultContextError(ValueError):
    """Raised when search result source context input is invalid."""


def _require_positive_id(value: int, field_name: str) -> None:
    if value <= 0:
        raise InvalidSearchResultContextError(f"{field_name} must be greater than 0")


def _preview(value: str | None, *, limit: int = CONTEXT_PREVIEW_CHARS) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _result_reference_from_row(row: dict[str, Any]) -> SearchResultContextReference:
    return SearchResultContextReference(
        search_log_result_id=int(row["search_log_result_id"]),
        search_log_id=int(row["search_log_id"]),
        profile_name=str(row["profile_name"]),
        rank=int(row["rank"]),
        chunk_id=int(row["chunk_id"]),
        distance=float(row["distance"]) if row.get("distance") is not None else None,
        score=float(row["score"]) if row.get("score") is not None else None,
        profile_elapsed_ms=_optional_int(row.get("profile_elapsed_ms")),
        created_at=row["created_at"],
    )


def _document_from_row(row: dict[str, Any]) -> SearchResultSourceDocument:
    return SearchResultSourceDocument(
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
        document_title=row["document_title"],
        document_group=str(row["document_group"]),
        document_status=str(row["document_status"]),
        original_file_name=str(row["original_file_name"]),
        file_ext=row["file_ext"],
        storage_path=str(row["storage_path"]),
    )


def _chunk_from_row(row: dict[str, Any]) -> SearchResultContextChunk:
    chunk_text = str(row["chunk_text"])
    return SearchResultContextChunk(
        position=str(row["position"]),
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        chunk_seq=int(row["chunk_seq"]),
        chunk_text=chunk_text,
        chunk_preview=_preview(chunk_text) or "",
        content_hash=str(row["content_hash"]),
        chunk_policy_name=str(row["chunk_policy_name"]),
        artifact_id=_optional_int(row.get("artifact_id")),
        block_id=_optional_int(row.get("block_id")),
        chunk_type=str(row["chunk_type"]),
        heading_path=tuple(row["heading_path"] or ()),
        source_anchor=dict(row["source_anchor"] or {}),
        page_no=_optional_int(row.get("page_no")),
        slide_no=_optional_int(row.get("slide_no")),
        sheet_name=row["sheet_name"],
        cell_range=row["cell_range"],
        source_char_start=_optional_int(row.get("source_char_start")),
        source_char_end=_optional_int(row.get("source_char_end")),
        token_count=_optional_int(row.get("token_count")),
        char_count=int(row["char_count"]),
        prev_chunk_id=_optional_int(row.get("prev_chunk_id")),
        next_chunk_id=_optional_int(row.get("next_chunk_id")),
        metadata=dict(row["metadata"] or {}),
    )


def _block_from_row(row: dict[str, Any] | None) -> SearchResultSourceBlock | None:
    if row is None:
        return None
    return SearchResultSourceBlock(
        block_id=int(row["block_id"]),
        artifact_id=int(row["artifact_id"]),
        document_id=int(row["document_id"]),
        parent_block_id=_optional_int(row.get("parent_block_id")),
        block_seq=int(row["block_seq"]),
        block_type=str(row["block_type"]),
        content_preview=_preview(row["content_text"]),
        content_markdown_preview=_preview(row["content_markdown"]),
        heading_path=tuple(row["heading_path"] or ()),
        source_anchor=dict(row["source_anchor"] or {}),
        page_no=_optional_int(row.get("page_no")),
        slide_no=_optional_int(row.get("slide_no")),
        sheet_name=row["sheet_name"],
        cell_range=row["cell_range"],
        char_start=_optional_int(row.get("char_start")),
        char_end=_optional_int(row.get("char_end")),
        token_count=_optional_int(row.get("token_count")),
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def _artifact_from_row(row: dict[str, Any] | None) -> SearchResultSourceArtifact | None:
    if row is None:
        return None
    content_text = row["content_text"]
    return SearchResultSourceArtifact(
        artifact_id=int(row["artifact_id"]),
        extraction_run_id=_optional_int(row.get("extraction_run_id")),
        file_id=int(row["file_id"]),
        document_id=_optional_int(row.get("document_id")),
        artifact_type=str(row["artifact_type"]),
        content_preview=_preview(content_text),
        content_length=len(content_text) if content_text is not None else None,
        storage_path=row["storage_path"],
        content_hash=row["content_hash"],
        size_bytes=_optional_int(row.get("size_bytes")),
        language=row["language"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def _get_target_row(cursor: Any, search_log_result_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT
            slr.search_log_result_id,
            slr.search_log_id,
            slr.profile_name,
            slr.rank,
            slr.chunk_id,
            slr.distance,
            slr.score,
            slr.profile_elapsed_ms,
            slr.created_at,
            c.document_id,
            c.chunk_seq,
            c.chunk_policy_name,
            c.prev_chunk_id,
            c.next_chunk_id,
            c.artifact_id,
            c.block_id,
            d.file_id,
            d.document_title,
            d.document_group,
            d.document_status,
            f.original_file_name,
            f.file_ext,
            f.storage_path
        FROM search_log_results slr
        JOIN chunks c ON c.chunk_id = slr.chunk_id
        JOIN documents d ON d.document_id = c.document_id
        JOIN files f ON f.file_id = d.file_id
        WHERE slr.search_log_result_id = %s
        """,
        (search_log_result_id,),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def _list_context_chunks(
    cursor: Any,
    target: dict[str, Any],
) -> tuple[SearchResultContextChunk, ...]:
    cursor.execute(
        """
        SELECT
            CASE
                WHEN c.chunk_id = %(current_chunk_id)s THEN 'current'
                WHEN %(prev_chunk_id)s::BIGINT IS NOT NULL
                  AND c.chunk_id = %(prev_chunk_id)s::BIGINT
                    THEN 'previous'
                WHEN %(next_chunk_id)s::BIGINT IS NOT NULL
                  AND c.chunk_id = %(next_chunk_id)s::BIGINT
                    THEN 'next'
                WHEN %(prev_chunk_id)s::BIGINT IS NULL
                  AND c.chunk_seq = %(current_chunk_seq)s - 1
                    THEN 'previous'
                WHEN %(next_chunk_id)s::BIGINT IS NULL
                  AND c.chunk_seq = %(current_chunk_seq)s + 1
                    THEN 'next'
                ELSE 'context'
            END AS position,
            c.chunk_id,
            c.document_id,
            c.chunk_seq,
            c.chunk_text,
            c.content_hash,
            c.chunk_policy_name,
            c.artifact_id,
            c.block_id,
            c.chunk_type,
            c.heading_path,
            c.source_anchor,
            c.page_no,
            c.slide_no,
            c.sheet_name,
            c.cell_range,
            c.source_char_start,
            c.source_char_end,
            c.token_count,
            c.char_count,
            c.prev_chunk_id,
            c.next_chunk_id,
            c.metadata
        FROM chunks c
        WHERE c.document_id = %(document_id)s
          AND c.chunk_policy_name = %(chunk_policy_name)s
          AND (
              c.chunk_id = %(current_chunk_id)s
              OR (
                  %(prev_chunk_id)s::BIGINT IS NOT NULL
                  AND c.chunk_id = %(prev_chunk_id)s::BIGINT
              )
              OR (
                  %(next_chunk_id)s::BIGINT IS NOT NULL
                  AND c.chunk_id = %(next_chunk_id)s::BIGINT
              )
              OR (
                  %(prev_chunk_id)s::BIGINT IS NULL
                  AND c.chunk_seq = %(current_chunk_seq)s - 1
              )
              OR (
                  %(next_chunk_id)s::BIGINT IS NULL
                  AND c.chunk_seq = %(current_chunk_seq)s + 1
              )
          )
        ORDER BY c.chunk_seq ASC, c.chunk_id ASC
        """,
        {
            "current_chunk_id": int(target["chunk_id"]),
            "document_id": int(target["document_id"]),
            "chunk_policy_name": str(target["chunk_policy_name"]),
            "current_chunk_seq": int(target["chunk_seq"]),
            "prev_chunk_id": target["prev_chunk_id"],
            "next_chunk_id": target["next_chunk_id"],
        },
    )
    return tuple(_chunk_from_row(dict(row)) for row in cursor.fetchall())


def _get_source_block(
    cursor: Any,
    block_id: int | None,
) -> SearchResultSourceBlock | None:
    if block_id is None:
        return None
    cursor.execute(
        """
        SELECT
            block_id,
            artifact_id,
            document_id,
            parent_block_id,
            block_seq,
            block_type,
            content_text,
            content_markdown,
            heading_path,
            source_anchor,
            page_no,
            slide_no,
            sheet_name,
            cell_range,
            char_start,
            char_end,
            token_count,
            metadata,
            created_at
        FROM document_blocks
        WHERE block_id = %s
        """,
        (block_id,),
    )
    row = cursor.fetchone()
    return _block_from_row(dict(row)) if row else None


def _get_source_artifact(
    cursor: Any,
    artifact_id: int | None,
) -> SearchResultSourceArtifact | None:
    if artifact_id is None:
        return None
    cursor.execute(
        """
        SELECT
            artifact_id,
            extraction_run_id,
            file_id,
            document_id,
            artifact_type,
            content_text,
            storage_path,
            content_hash,
            size_bytes,
            language,
            metadata,
            created_at
        FROM extraction_artifacts
        WHERE artifact_id = %s
        """,
        (artifact_id,),
    )
    row = cursor.fetchone()
    return _artifact_from_row(dict(row)) if row else None


def get_search_result_source_context(
    database_url: str,
    search_log_result_id: int,
) -> SearchResultSourceContext | None:
    """Return source trace context for a stored search result."""

    _require_positive_id(search_log_result_id, "search_log_result_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            target = _get_target_row(cursor, search_log_result_id)
            if target is None:
                return None
            chunks = _list_context_chunks(cursor, target)
            block = _get_source_block(cursor, _optional_int(target.get("block_id")))
            artifact_id = _optional_int(target.get("artifact_id"))
            if artifact_id is None and block is not None:
                artifact_id = block.artifact_id
            artifact = _get_source_artifact(cursor, artifact_id)

    return SearchResultSourceContext(
        search_result=_result_reference_from_row(target),
        document=_document_from_row(target),
        chunks=chunks,
        source_block=block,
        source_artifact=artifact,
    )
