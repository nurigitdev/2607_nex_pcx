"""BM25 keyword search helpers."""

from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection

from app.core.bm25_keyword_index import (
    DEFAULT_BM25_TOKENIZER_NAME,
    build_bm25_term_frequencies,
    validate_bm25_tokenizer_name,
)
from app.core.chunks import DEFAULT_CHUNK_POLICY_NAME
from app.core.database import connect
from app.core.permissions import PermissionSearchFilter
from app.core.vector_search import CHUNK_PREVIEW_CHARS, MAX_TOP_K

DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75
BM25_SEARCH_PROFILE_NAME = "bm25_keyword"
BM25_RETRIEVAL_STRATEGY = "bm25_keyword"


@dataclass(frozen=True)
class BM25SearchInput:
    query_text: str
    top_k: int = 5
    chunk_policy_name: str = DEFAULT_CHUNK_POLICY_NAME
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME
    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B
    document_group: str | None = None
    file_type: str | None = None
    permission_filter: PermissionSearchFilter | None = None


@dataclass(frozen=True)
class BM25SearchResult:
    search_profile_name: str
    retrieval_strategy: str
    rank: int
    chunk_id: int
    document_id: int
    file_id: int
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
    matched_term_count: int
    document_length: float
    score_components: dict[str, Any] = field(default_factory=dict)


class InvalidBM25SearchError(ValueError):
    """Raised when BM25 search input is invalid before reaching the DB."""


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidBM25SearchError(f"{field_name} must not be blank")
    return normalized


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise InvalidBM25SearchError("top_k must be greater than 0")
    if top_k > MAX_TOP_K:
        raise InvalidBM25SearchError(f"top_k must be less than or equal to {MAX_TOP_K}")


def _validate_bm25_parameters(k1: float, b: float) -> None:
    if k1 <= 0:
        raise InvalidBM25SearchError("k1 must be greater than 0")
    if b < 0 or b > 1:
        raise InvalidBM25SearchError("b must be between 0 and 1")


def validate_bm25_search_input(
    query_input: BM25SearchInput,
) -> dict[str, int]:
    if not query_input.query_text.strip():
        raise InvalidBM25SearchError("query_text is required")
    _validate_top_k(query_input.top_k)
    _validate_bm25_parameters(query_input.k1, query_input.b)
    _validate_nonblank(query_input.chunk_policy_name, "chunk_policy_name")
    _validate_nonblank(query_input.document_group, "document_group")
    _validate_nonblank(query_input.file_type, "file_type")
    try:
        tokenizer_name = validate_bm25_tokenizer_name(query_input.tokenizer_name)
    except ValueError as exc:
        raise InvalidBM25SearchError(str(exc)) from exc
    return build_bm25_term_frequencies(
        query_input.query_text,
        tokenizer_name=tokenizer_name,
    )


def _chunk_preview(chunk_text: str) -> str:
    normalized = " ".join(chunk_text.split())
    if len(normalized) <= CHUNK_PREVIEW_CHARS:
        return normalized
    return normalized[: CHUNK_PREVIEW_CHARS - 1].rstrip() + "..."


def _build_search_filters(
    query_input: BM25SearchInput,
) -> tuple[str, list[object]]:
    where_clauses = ["d.document_status = 'active'"]
    params: list[object] = []
    document_group = _validate_nonblank(query_input.document_group, "document_group")
    file_type = _validate_nonblank(query_input.file_type, "file_type")

    if document_group is not None:
        where_clauses.append("d.document_group = %s")
        params.append(document_group)
    if file_type is not None:
        where_clauses.append("f.file_ext = %s")
        params.append(file_type)
    if query_input.permission_filter is not None:
        where_clauses.append(query_input.permission_filter.where_sql)
        params.extend(query_input.permission_filter.params)

    return " AND ".join(where_clauses), params


def _row_to_bm25_search_result(
    row: dict[str, Any],
    *,
    tokenizer_name: str,
    k1: float,
    b: float,
    query_terms: tuple[str, ...],
) -> BM25SearchResult:
    chunk_text = str(row["chunk_text"])
    matched_term_count = int(row["matched_term_count"])
    document_length = float(row["document_length"])
    return BM25SearchResult(
        search_profile_name=BM25_SEARCH_PROFILE_NAME,
        retrieval_strategy=BM25_RETRIEVAL_STRATEGY,
        rank=int(row["rank"]),
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
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
        matched_term_count=matched_term_count,
        document_length=document_length,
        score_components={
            "tokenizer_name": tokenizer_name,
            "k1": k1,
            "b": b,
            "query_terms": list(query_terms),
            "matched_term_count": matched_term_count,
            "document_length": document_length,
        },
    )


def search_bm25_chunks_in_connection(
    connection: Connection,
    query_input: BM25SearchInput,
) -> list[BM25SearchResult]:
    query_term_frequencies = validate_bm25_search_input(query_input)
    if not query_term_frequencies:
        return []

    chunk_policy_name = _validate_nonblank(query_input.chunk_policy_name, "chunk_policy_name")
    tokenizer_name = validate_bm25_tokenizer_name(query_input.tokenizer_name)
    where_sql, filter_params = _build_search_filters(query_input)
    query_terms = tuple(sorted(query_term_frequencies))
    query_frequencies = tuple(query_term_frequencies[term] for term in query_terms)

    params: list[object] = [
        list(query_terms),
        list(query_frequencies),
        chunk_policy_name,
        tokenizer_name,
        query_input.k1,
        query_input.k1,
        query_input.b,
        query_input.b,
        chunk_policy_name,
        tokenizer_name,
        *filter_params,
        query_input.top_k,
    ]
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            WITH query_terms AS (
                SELECT term, query_frequency
                FROM unnest(%s::text[], %s::int[]) AS query_terms (
                    term,
                    query_frequency
                )
            ),
            chunk_lengths AS (
                SELECT
                    chunk_id,
                    sum(term_frequency)::double precision AS document_length
                FROM chunk_keyword_terms
                WHERE chunk_policy_name = %s
                  AND tokenizer_name = %s
                GROUP BY chunk_id
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
                    count(DISTINCT query_terms.term) AS matched_term_count,
                    max(chunk_lengths.document_length) AS document_length,
                    sum(
                        CASE
                            WHEN stats.average_document_length <= 0 THEN 0
                            ELSE
                                ln(
                                    1.0 + (
                                        (
                                            stats.corpus_chunk_count
                                            - stats.document_frequency
                                            + 0.5
                                        )::double precision
                                        / (
                                            stats.document_frequency + 0.5
                                        )::double precision
                                    )
                                )
                                * (
                                    keyword_terms.term_frequency::double precision
                                    * (%s + 1.0)
                                )
                                / (
                                    keyword_terms.term_frequency::double precision
                                    + %s * (
                                        1.0 - %s
                                        + %s * (
                                            chunk_lengths.document_length
                                            / stats.average_document_length::double precision
                                        )
                                    )
                                )
                                * query_terms.query_frequency
                        END
                    ) AS score
                FROM query_terms
                JOIN chunk_keyword_terms keyword_terms
                  ON keyword_terms.term = query_terms.term
                JOIN chunk_keyword_statistics stats
                  ON stats.chunk_policy_name = keyword_terms.chunk_policy_name
                 AND stats.tokenizer_name = keyword_terms.tokenizer_name
                 AND stats.term = keyword_terms.term
                JOIN chunk_lengths
                  ON chunk_lengths.chunk_id = keyword_terms.chunk_id
                JOIN chunks c ON c.chunk_id = keyword_terms.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                WHERE keyword_terms.chunk_policy_name = %s
                  AND keyword_terms.tokenizer_name = %s
                  AND {where_sql}
                GROUP BY
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
                    f.file_ext
                ORDER BY score DESC, c.chunk_id ASC
                LIMIT %s
            )
            SELECT
                row_number() OVER (ORDER BY score DESC, chunk_id ASC) AS rank,
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
                matched_term_count,
                document_length,
                score
            FROM ranked_chunks
            ORDER BY rank ASC
            """,
            tuple(params),
        )
        rows = cursor.fetchall()

    return [
        _row_to_bm25_search_result(
            dict(row),
            tokenizer_name=tokenizer_name,
            k1=query_input.k1,
            b=query_input.b,
            query_terms=query_terms,
        )
        for row in rows
    ]


def search_bm25_chunks(
    database_url: str,
    query_input: BM25SearchInput,
) -> list[BM25SearchResult]:
    with connect(database_url) as connection:
        return search_bm25_chunks_in_connection(connection, query_input)
