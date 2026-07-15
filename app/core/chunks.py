"""Chunk metadata persistence for parsed documents."""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect

DEFAULT_CHUNK_POLICY_NAME = "heading_512_64"
CHUNK_TYPES = {"text", "table", "image", "figure", "code", "list", "heading"}


@dataclass(frozen=True)
class ChunkInput:
    document_id: int
    chunk_seq: int
    chunk_text: str
    chunk_policy_name: str = DEFAULT_CHUNK_POLICY_NAME
    artifact_id: int | None = None
    block_id: int | None = None
    chunk_type: str = "text"
    content_markdown: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    heading_path: tuple[str, ...] = ()
    source_anchor: dict[str, Any] = field(default_factory=dict)
    page_no: int | None = None
    slide_no: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    source_char_start: int | None = None
    source_char_end: int | None = None
    token_count: int | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    document_id: int
    chunk_seq: int
    chunk_text: str
    content_hash: str
    chunk_policy_name: str
    artifact_id: int | None
    block_id: int | None
    chunk_type: str
    content_markdown: str | None
    parser_name: str | None
    parser_version: str | None
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


class InvalidChunkError(ValueError):
    """Raised when chunk metadata is invalid before reaching the DB."""


def calculate_chunk_content_hash(chunk_text: str) -> str:
    return sha256(chunk_text.encode("utf-8")).hexdigest()


def _require_positive_id(value: int, field_name: str) -> None:
    if value <= 0:
        raise InvalidChunkError(f"{field_name} must be greater than 0")


def _validate_optional_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidChunkError(f"{field_name} must be greater than 0")


def _validate_optional_non_negative(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise InvalidChunkError(f"{field_name} must be greater than or equal to 0")


def validate_chunk_input(chunk_input: ChunkInput) -> str:
    _require_positive_id(chunk_input.document_id, "document_id")
    _validate_optional_positive_id(chunk_input.artifact_id, "artifact_id")
    _validate_optional_positive_id(chunk_input.block_id, "block_id")
    if chunk_input.chunk_seq < 0:
        raise InvalidChunkError("chunk_seq must be greater than or equal to 0")
    if chunk_input.chunk_type not in CHUNK_TYPES:
        raise InvalidChunkError(f"Unsupported chunk_type: {chunk_input.chunk_type}")
    if not chunk_input.chunk_text.strip():
        raise InvalidChunkError("chunk_text is required")
    if not chunk_input.chunk_policy_name.strip():
        raise InvalidChunkError("chunk_policy_name is required")
    if chunk_input.page_no is not None and chunk_input.page_no <= 0:
        raise InvalidChunkError("page_no must be greater than 0")
    if chunk_input.slide_no is not None and chunk_input.slide_no <= 0:
        raise InvalidChunkError("slide_no must be greater than 0")
    if chunk_input.token_count is not None and chunk_input.token_count < 0:
        raise InvalidChunkError("token_count must be greater than or equal to 0")
    _validate_optional_non_negative(chunk_input.source_char_start, "source_char_start")
    _validate_optional_non_negative(chunk_input.source_char_end, "source_char_end")
    if (
        chunk_input.source_char_start is not None
        and chunk_input.source_char_end is not None
        and chunk_input.source_char_end < chunk_input.source_char_start
    ):
        raise InvalidChunkError(
            "source_char_end must be greater than or equal to source_char_start"
        )
    if chunk_input.content_hash is not None and not chunk_input.content_hash.strip():
        raise InvalidChunkError("content_hash must not be blank")
    return chunk_input.content_hash or calculate_chunk_content_hash(chunk_input.chunk_text)


def _row_to_chunk_record(row: dict[str, Any]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        chunk_seq=int(row["chunk_seq"]),
        chunk_text=str(row["chunk_text"]),
        content_hash=str(row["content_hash"]),
        chunk_policy_name=str(row["chunk_policy_name"]),
        artifact_id=int(row["artifact_id"]) if row.get("artifact_id") is not None else None,
        block_id=int(row["block_id"]) if row.get("block_id") is not None else None,
        chunk_type=str(row["chunk_type"]),
        content_markdown=row["content_markdown"],
        parser_name=row["parser_name"],
        parser_version=row["parser_version"],
        heading_path=tuple(row["heading_path"] or ()),
        source_anchor=dict(row["source_anchor"] or {}),
        page_no=int(row["page_no"]) if row.get("page_no") is not None else None,
        slide_no=int(row["slide_no"]) if row.get("slide_no") is not None else None,
        sheet_name=row["sheet_name"],
        cell_range=row["cell_range"],
        source_char_start=(
            int(row["source_char_start"]) if row.get("source_char_start") is not None else None
        ),
        source_char_end=(
            int(row["source_char_end"]) if row.get("source_char_end") is not None else None
        ),
        token_count=int(row["token_count"]) if row.get("token_count") is not None else None,
        char_count=int(row["char_count"]),
        prev_chunk_id=int(row["prev_chunk_id"]) if row.get("prev_chunk_id") is not None else None,
        next_chunk_id=int(row["next_chunk_id"]) if row.get("next_chunk_id") is not None else None,
        metadata=dict(row["metadata"] or {}),
    )


def _select_chunk_columns(alias: str = "chunks") -> str:
    return f"""
        {alias}.chunk_id,
        {alias}.document_id,
        {alias}.chunk_seq,
        {alias}.chunk_text,
        {alias}.content_hash,
        {alias}.chunk_policy_name,
        {alias}.artifact_id,
        {alias}.block_id,
        {alias}.chunk_type,
        {alias}.content_markdown,
        {alias}.parser_name,
        {alias}.parser_version,
        {alias}.heading_path,
        {alias}.source_anchor,
        {alias}.page_no,
        {alias}.slide_no,
        {alias}.sheet_name,
        {alias}.cell_range,
        {alias}.source_char_start,
        {alias}.source_char_end,
        {alias}.token_count,
        {alias}.char_count,
        {alias}.prev_chunk_id,
        {alias}.next_chunk_id,
        {alias}.metadata
    """


def create_chunk_in_connection(
    connection: Connection,
    chunk_input: ChunkInput,
) -> ChunkRecord:
    content_hash = validate_chunk_input(chunk_input)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO chunks (
                document_id,
                artifact_id,
                block_id,
                chunk_seq,
                chunk_type,
                chunk_text,
                content_markdown,
                content_hash,
                chunk_policy_name,
                parser_name,
                parser_version,
                heading_path,
                source_anchor,
                page_no,
                slide_no,
                sheet_name,
                cell_range,
                source_char_start,
                source_char_end,
                token_count,
                char_count,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            RETURNING {_select_chunk_columns()}
            """,
            (
                chunk_input.document_id,
                chunk_input.artifact_id,
                chunk_input.block_id,
                chunk_input.chunk_seq,
                chunk_input.chunk_type,
                chunk_input.chunk_text,
                chunk_input.content_markdown,
                content_hash,
                chunk_input.chunk_policy_name,
                chunk_input.parser_name,
                chunk_input.parser_version,
                list(chunk_input.heading_path) or None,
                Json(chunk_input.source_anchor),
                chunk_input.page_no,
                chunk_input.slide_no,
                chunk_input.sheet_name,
                chunk_input.cell_range,
                chunk_input.source_char_start,
                chunk_input.source_char_end,
                chunk_input.token_count,
                len(chunk_input.chunk_text),
                Json(chunk_input.metadata),
            ),
        )
        return _row_to_chunk_record(dict(cursor.fetchone()))


def create_chunk(database_url: str, chunk_input: ChunkInput) -> ChunkRecord:
    with connect(database_url) as connection:
        return create_chunk_in_connection(connection, chunk_input)


def list_document_chunks(
    database_url: str,
    document_id: int,
    *,
    chunk_policy_name: str | None = None,
) -> list[ChunkRecord]:
    _require_positive_id(document_id, "document_id")
    params: tuple[object, ...]
    where_policy = ""
    params = (document_id,)
    if chunk_policy_name is not None:
        if not chunk_policy_name.strip():
            raise InvalidChunkError("chunk_policy_name must not be blank")
        where_policy = "AND chunk_policy_name = %s"
        params = (document_id, chunk_policy_name)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_chunk_columns()}
                FROM chunks
                WHERE document_id = %s
                  {where_policy}
                ORDER BY chunk_policy_name ASC, chunk_seq ASC
                """,
                params,
            )
            rows = cursor.fetchall()
    return [_row_to_chunk_record(dict(row)) for row in rows]


def replace_document_chunks_in_connection(
    connection: Connection,
    document_id: int,
    chunk_inputs: list[ChunkInput],
    *,
    chunk_policy_name: str = DEFAULT_CHUNK_POLICY_NAME,
) -> list[ChunkRecord]:
    _require_positive_id(document_id, "document_id")
    if not chunk_policy_name.strip():
        raise InvalidChunkError("chunk_policy_name is required")
    sorted_chunk_inputs = sorted(chunk_inputs, key=lambda chunk: chunk.chunk_seq)
    for expected_seq, chunk_input in enumerate(sorted_chunk_inputs):
        if chunk_input.document_id != document_id:
            raise InvalidChunkError("all chunks must belong to the target document_id")
        if chunk_input.chunk_policy_name != chunk_policy_name:
            raise InvalidChunkError("all chunks must use the target chunk_policy_name")
        if chunk_input.chunk_seq != expected_seq:
            raise InvalidChunkError("chunk_seq values must be contiguous from 0")

    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM chunks
            WHERE document_id = %s
              AND chunk_policy_name = %s
            """,
            (document_id, chunk_policy_name),
        )

    records = [
        create_chunk_in_connection(connection, chunk_input) for chunk_input in sorted_chunk_inputs
    ]
    linked_records: list[ChunkRecord] = []
    for index, current in enumerate(records):
        previous = records[index - 1] if index > 0 else None
        following = records[index + 1] if index + 1 < len(records) else None
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE chunks
                SET prev_chunk_id = %s,
                    next_chunk_id = %s
                WHERE chunk_id = %s
                RETURNING {_select_chunk_columns()}
                """,
                (
                    previous.chunk_id if previous else None,
                    following.chunk_id if following else None,
                    current.chunk_id,
                ),
            )
            updated = _row_to_chunk_record(dict(cursor.fetchone()))
        linked_records.append(updated)
    return linked_records


def replace_document_chunks(
    database_url: str,
    document_id: int,
    chunk_inputs: list[ChunkInput],
    *,
    chunk_policy_name: str = DEFAULT_CHUNK_POLICY_NAME,
) -> list[ChunkRecord]:
    with connect(database_url) as connection:
        return replace_document_chunks_in_connection(
            connection,
            document_id,
            chunk_inputs,
            chunk_policy_name=chunk_policy_name,
        )
