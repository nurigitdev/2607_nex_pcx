"""pgvector-backed chunk search helpers."""

from dataclasses import dataclass
from typing import Any

from psycopg import Connection

from app.core.database import connect
from app.core.embedding_vectors import (
    EmbeddingVectorTable,
    InvalidEmbeddingVectorError,
    generate_mock_embedding,
    get_embedding_vector_table,
    vector_to_pg_literal,
)

SUPPORTED_SIMILARITY_METRICS = {"cosine"}
MAX_TOP_K = 100
CHUNK_PREVIEW_CHARS = 240


@dataclass(frozen=True)
class VectorSearchInput:
    query_text: str
    profile_name: str
    top_k: int = 5
    similarity_metric: str = "cosine"
    query_embedding: tuple[float, ...] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


@dataclass(frozen=True)
class VectorSearchResult:
    profile_name: str
    rank: int
    chunk_id: int
    document_id: int
    file_id: int
    distance: float
    score: float
    chunk_text: str
    chunk_preview: str
    content_hash: str
    chunk_policy_name: str
    heading_path: tuple[str, ...]
    page_no: int | None
    slide_no: int | None
    sheet_name: str | None
    cell_range: str | None
    document_title: str | None
    document_group: str
    original_file_name: str
    file_ext: str | None
    embedding_elapsed_ms: int | None


class InvalidVectorSearchError(ValueError):
    """Raised when vector search input is invalid before reaching the DB."""


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidVectorSearchError(f"{field_name} must not be blank")
    return normalized


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise InvalidVectorSearchError("top_k must be greater than 0")
    if top_k > MAX_TOP_K:
        raise InvalidVectorSearchError(f"top_k must be less than or equal to {MAX_TOP_K}")


def _validate_similarity_metric(similarity_metric: str) -> str:
    metric = similarity_metric.strip()
    if metric not in SUPPORTED_SIMILARITY_METRICS:
        raise InvalidVectorSearchError(f"Unsupported similarity metric: {similarity_metric}")
    return metric


def _validate_query_embedding(
    query_input: VectorSearchInput,
    table: EmbeddingVectorTable,
) -> tuple[float, ...]:
    if query_input.query_embedding is None:
        return generate_mock_embedding(
            query_input.query_text,
            profile_name=table.profile_name,
            dimension=table.dimension,
        )
    if len(query_input.query_embedding) != table.dimension:
        raise InvalidVectorSearchError(
            f"{table.profile_name} query embedding requires {table.dimension} dimensions"
        )
    return query_input.query_embedding


def validate_vector_search_input(
    query_input: VectorSearchInput,
) -> tuple[EmbeddingVectorTable, tuple[float, ...]]:
    if not query_input.query_text.strip():
        raise InvalidVectorSearchError("query_text is required")
    _validate_top_k(query_input.top_k)
    _validate_similarity_metric(query_input.similarity_metric)
    _validate_nonblank(query_input.chunk_policy_name, "chunk_policy_name")
    _validate_nonblank(query_input.document_group, "document_group")
    _validate_nonblank(query_input.file_type, "file_type")
    try:
        table = get_embedding_vector_table(query_input.profile_name)
    except InvalidEmbeddingVectorError as exc:
        raise InvalidVectorSearchError(str(exc)) from exc
    return table, _validate_query_embedding(query_input, table)


def _chunk_preview(chunk_text: str) -> str:
    normalized = " ".join(chunk_text.split())
    if len(normalized) <= CHUNK_PREVIEW_CHARS:
        return normalized
    return normalized[: CHUNK_PREVIEW_CHARS - 1].rstrip() + "..."


def _row_to_vector_search_result(
    row: dict[str, Any],
    *,
    profile_name: str,
) -> VectorSearchResult:
    chunk_text = str(row["chunk_text"])
    return VectorSearchResult(
        profile_name=profile_name,
        rank=int(row["rank"]),
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
        distance=float(row["distance"]),
        score=float(row["score"]),
        chunk_text=chunk_text,
        chunk_preview=_chunk_preview(chunk_text),
        content_hash=str(row["content_hash"]),
        chunk_policy_name=str(row["chunk_policy_name"]),
        heading_path=tuple(row["heading_path"] or ()),
        page_no=int(row["page_no"]) if row.get("page_no") is not None else None,
        slide_no=int(row["slide_no"]) if row.get("slide_no") is not None else None,
        sheet_name=row["sheet_name"],
        cell_range=row["cell_range"],
        document_title=row["document_title"],
        document_group=str(row["document_group"]),
        original_file_name=str(row["original_file_name"]),
        file_ext=row["file_ext"],
        embedding_elapsed_ms=(
            int(row["embedding_elapsed_ms"]) if row["embedding_elapsed_ms"] is not None else None
        ),
    )


def _build_search_filters(
    query_input: VectorSearchInput,
) -> tuple[str, list[object]]:
    where_clauses = ["d.document_status = 'active'"]
    params: list[object] = []
    chunk_policy_name = _validate_nonblank(query_input.chunk_policy_name, "chunk_policy_name")
    document_group = _validate_nonblank(query_input.document_group, "document_group")
    file_type = _validate_nonblank(query_input.file_type, "file_type")

    if chunk_policy_name is not None:
        where_clauses.append("c.chunk_policy_name = %s")
        params.append(chunk_policy_name)
    if document_group is not None:
        where_clauses.append("d.document_group = %s")
        params.append(document_group)
    if file_type is not None:
        where_clauses.append("f.file_ext = %s")
        params.append(file_type)

    return " AND ".join(where_clauses), params


def search_similar_chunks_in_connection(
    connection: Connection,
    query_input: VectorSearchInput,
) -> list[VectorSearchResult]:
    table, query_embedding = validate_vector_search_input(query_input)
    query_literal = vector_to_pg_literal(query_embedding)
    cast_type = "halfvec" if table.storage_type == "halfvec" else "vector"
    where_sql, filter_params = _build_search_filters(query_input)

    params: list[object] = [query_literal, *filter_params, query_input.top_k]
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            WITH query_embedding AS (
                SELECT %s::{cast_type} AS embedding
            ),
            ranked_chunks AS (
                SELECT
                    c.chunk_id,
                    c.document_id,
                    d.file_id,
                    c.chunk_text,
                    c.content_hash,
                    c.chunk_policy_name,
                    c.heading_path,
                    c.page_no,
                    c.slide_no,
                    c.sheet_name,
                    c.cell_range,
                    d.document_title,
                    d.document_group,
                    f.original_file_name,
                    f.file_ext,
                    e.elapsed_ms AS embedding_elapsed_ms,
                    e.embedding <=> query_embedding.embedding AS distance
                FROM {table.table_name} e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                CROSS JOIN query_embedding
                WHERE {where_sql}
                ORDER BY distance ASC, c.chunk_id ASC
                LIMIT %s
            )
            SELECT
                row_number() OVER (ORDER BY distance ASC, chunk_id ASC) AS rank,
                chunk_id,
                document_id,
                file_id,
                chunk_text,
                content_hash,
                chunk_policy_name,
                heading_path,
                page_no,
                slide_no,
                sheet_name,
                cell_range,
                document_title,
                document_group,
                original_file_name,
                file_ext,
                embedding_elapsed_ms,
                distance,
                1 - distance AS score
            FROM ranked_chunks
            ORDER BY rank ASC
            """,
            tuple(params),
        )
        rows = cursor.fetchall()

    return [
        _row_to_vector_search_result(dict(row), profile_name=table.profile_name) for row in rows
    ]


def search_similar_chunks(
    database_url: str,
    query_input: VectorSearchInput,
) -> list[VectorSearchResult]:
    with connect(database_url) as connection:
        return search_similar_chunks_in_connection(connection, query_input)
