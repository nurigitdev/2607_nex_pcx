"""BM25 keyword index coverage and freshness read-model helpers."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.core.bm25_keyword_index import (
    DEFAULT_BM25_TOKENIZER_NAME,
    validate_bm25_tokenizer_name,
)
from app.core.chunk_policies import _validate_policy_name
from app.core.database import connect
from app.core.document_inventory import (
    _validate_document_group,
    _validate_limit,
    _validate_parse_status,
)
from app.core.embedding_coverage import _coverage_percent

BM25_INDEX_COVERAGE_STATUSES = {
    "complete",
    "partial",
    "missing",
    "stale",
    "not_chunked",
}


@dataclass(frozen=True)
class BM25IndexCoverageRow:
    document_id: int
    file_id: int
    document_title: str | None
    original_file_name: str
    file_ext: str | None
    document_group: str
    parse_status: str
    access_scope: str
    uploaded_at: datetime
    chunk_policy_name: str
    target_token_size: int
    overlap_token_size: int
    split_strategy: str
    tokenizer_name: str
    policy_chunk_count: int
    chunk_count: int
    indexed_chunk_count: int
    missing_chunk_count: int
    term_row_count: int
    statistics_term_count: int
    statistics_corpus_chunk_count: int
    average_document_length: Decimal | None
    coverage_percent: Decimal
    status: str
    latest_term_created_at: datetime | None
    latest_statistics_updated_at: datetime | None


@dataclass(frozen=True)
class BM25IndexCoveragePolicySummary:
    chunk_policy_name: str
    tokenizer_name: str
    document_count: int
    chunked_document_count: int
    complete_document_count: int
    partial_document_count: int
    missing_document_count: int
    stale_document_count: int
    not_chunked_document_count: int
    total_chunk_count: int
    indexed_chunk_count: int
    missing_chunk_count: int
    term_row_count: int
    statistics_term_count: int
    statistics_corpus_chunk_count: int
    coverage_percent: Decimal
    latest_term_created_at: datetime | None
    latest_statistics_updated_at: datetime | None


@dataclass(frozen=True)
class BM25IndexCoverageSummary:
    document_count: int
    policy_count: int
    document_policy_count: int
    total_chunk_count: int
    indexed_chunk_count: int
    missing_chunk_count: int
    term_row_count: int
    statistics_term_count: int
    complete_row_count: int
    attention_row_count: int
    stale_row_count: int
    missing_row_count: int
    coverage_percent: Decimal
    latest_term_created_at: datetime | None
    latest_statistics_updated_at: datetime | None
    policies: tuple[BM25IndexCoveragePolicySummary, ...]


@dataclass(frozen=True)
class BM25IndexCoverageMatrix:
    summary: BM25IndexCoverageSummary
    rows: tuple[BM25IndexCoverageRow, ...]


class InvalidBM25IndexCoverageError(ValueError):
    """Raised when BM25 index coverage query input is invalid."""


def _row_int(row: dict[str, Any], key: str) -> int:
    return int(row[key] or 0)


def _latest_datetime(values: list[datetime | None]) -> datetime | None:
    concrete_values = [value for value in values if value is not None]
    return max(concrete_values) if concrete_values else None


def _status_for_bm25_index_cell(
    *,
    policy_chunk_count: int,
    chunk_count: int,
    indexed_chunk_count: int,
    statistics_term_count: int,
    statistics_corpus_chunk_count: int,
) -> str:
    if chunk_count == 0:
        return "not_chunked"
    if statistics_term_count == 0 and indexed_chunk_count == 0:
        return "missing"
    if statistics_corpus_chunk_count != policy_chunk_count:
        return "stale"
    if indexed_chunk_count >= chunk_count:
        return "complete"
    if indexed_chunk_count > 0:
        return "partial"
    return "missing"


def _row_to_bm25_index_coverage_row(row: dict[str, Any]) -> BM25IndexCoverageRow:
    chunk_count = _row_int(row, "chunk_count")
    policy_chunk_count = _row_int(row, "policy_chunk_count")
    indexed_chunk_count = _row_int(row, "indexed_chunk_count")
    statistics_term_count = _row_int(row, "statistics_term_count")
    statistics_corpus_chunk_count = _row_int(row, "statistics_corpus_chunk_count")
    missing_chunk_count = max(chunk_count - indexed_chunk_count, 0)
    return BM25IndexCoverageRow(
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
        document_title=row["document_title"],
        original_file_name=str(row["original_file_name"]),
        file_ext=row["file_ext"],
        document_group=str(row["document_group"]),
        parse_status=str(row["parse_status"]),
        access_scope=str(row["access_scope"]),
        uploaded_at=row["uploaded_at"],
        chunk_policy_name=str(row["chunk_policy_name"]),
        target_token_size=int(row["target_token_size"]),
        overlap_token_size=int(row["overlap_token_size"]),
        split_strategy=str(row["split_strategy"]),
        tokenizer_name=str(row["tokenizer_name"]),
        policy_chunk_count=policy_chunk_count,
        chunk_count=chunk_count,
        indexed_chunk_count=indexed_chunk_count,
        missing_chunk_count=missing_chunk_count,
        term_row_count=_row_int(row, "term_row_count"),
        statistics_term_count=statistics_term_count,
        statistics_corpus_chunk_count=statistics_corpus_chunk_count,
        average_document_length=row["average_document_length"],
        coverage_percent=_coverage_percent(indexed_chunk_count, chunk_count),
        status=_status_for_bm25_index_cell(
            policy_chunk_count=policy_chunk_count,
            chunk_count=chunk_count,
            indexed_chunk_count=indexed_chunk_count,
            statistics_term_count=statistics_term_count,
            statistics_corpus_chunk_count=statistics_corpus_chunk_count,
        ),
        latest_term_created_at=row["latest_term_created_at"],
        latest_statistics_updated_at=row["latest_statistics_updated_at"],
    )


def _build_policy_summary(
    chunk_policy_name: str,
    rows: list[BM25IndexCoverageRow],
) -> BM25IndexCoveragePolicySummary:
    total_chunk_count = sum(row.chunk_count for row in rows)
    indexed_chunk_count = sum(row.indexed_chunk_count for row in rows)
    return BM25IndexCoveragePolicySummary(
        chunk_policy_name=chunk_policy_name,
        tokenizer_name=rows[0].tokenizer_name if rows else DEFAULT_BM25_TOKENIZER_NAME,
        document_count=len(rows),
        chunked_document_count=sum(1 for row in rows if row.chunk_count > 0),
        complete_document_count=sum(1 for row in rows if row.status == "complete"),
        partial_document_count=sum(1 for row in rows if row.status == "partial"),
        missing_document_count=sum(1 for row in rows if row.status == "missing"),
        stale_document_count=sum(1 for row in rows if row.status == "stale"),
        not_chunked_document_count=sum(1 for row in rows if row.status == "not_chunked"),
        total_chunk_count=total_chunk_count,
        indexed_chunk_count=indexed_chunk_count,
        missing_chunk_count=sum(row.missing_chunk_count for row in rows),
        term_row_count=sum(row.term_row_count for row in rows),
        statistics_term_count=max((row.statistics_term_count for row in rows), default=0),
        statistics_corpus_chunk_count=max(
            (row.statistics_corpus_chunk_count for row in rows),
            default=0,
        ),
        coverage_percent=_coverage_percent(indexed_chunk_count, total_chunk_count),
        latest_term_created_at=_latest_datetime([row.latest_term_created_at for row in rows]),
        latest_statistics_updated_at=_latest_datetime(
            [row.latest_statistics_updated_at for row in rows]
        ),
    )


def _build_summary(
    rows: tuple[BM25IndexCoverageRow, ...],
) -> BM25IndexCoverageSummary:
    rows_by_policy: dict[str, list[BM25IndexCoverageRow]] = {}
    document_ids: set[int] = set()
    for row in rows:
        rows_by_policy.setdefault(row.chunk_policy_name, []).append(row)
        document_ids.add(row.document_id)

    policies = tuple(
        _build_policy_summary(chunk_policy_name, policy_rows)
        for chunk_policy_name, policy_rows in sorted(rows_by_policy.items())
    )
    total_chunk_count = sum(row.chunk_count for row in rows)
    indexed_chunk_count = sum(row.indexed_chunk_count for row in rows)
    complete_row_count = sum(1 for row in rows if row.status == "complete")
    stale_row_count = sum(1 for row in rows if row.status == "stale")
    missing_row_count = sum(1 for row in rows if row.status == "missing")
    partial_row_count = sum(1 for row in rows if row.status == "partial")
    return BM25IndexCoverageSummary(
        document_count=len(document_ids),
        policy_count=len(policies),
        document_policy_count=len(rows),
        total_chunk_count=total_chunk_count,
        indexed_chunk_count=indexed_chunk_count,
        missing_chunk_count=sum(row.missing_chunk_count for row in rows),
        term_row_count=sum(row.term_row_count for row in rows),
        statistics_term_count=sum(policy.statistics_term_count for policy in policies),
        complete_row_count=complete_row_count,
        attention_row_count=stale_row_count + missing_row_count + partial_row_count,
        stale_row_count=stale_row_count,
        missing_row_count=missing_row_count,
        coverage_percent=_coverage_percent(indexed_chunk_count, total_chunk_count),
        latest_term_created_at=_latest_datetime([row.latest_term_created_at for row in rows]),
        latest_statistics_updated_at=_latest_datetime(
            [row.latest_statistics_updated_at for row in rows]
        ),
        policies=policies,
    )


def get_bm25_index_coverage_matrix(
    database_url: str,
    *,
    parse_status: str | None = None,
    document_group: str | None = None,
    chunk_policy_name: str | None = None,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
    limit: int = 100,
) -> BM25IndexCoverageMatrix:
    try:
        validated_limit = _validate_limit(limit, max_limit=200)
        validated_parse_status = _validate_parse_status(parse_status)
        validated_document_group = _validate_document_group(document_group)
        validated_chunk_policy_name = (
            _validate_policy_name(chunk_policy_name) if chunk_policy_name is not None else None
        )
        validated_tokenizer_name = validate_bm25_tokenizer_name(tokenizer_name)
    except ValueError as exc:
        raise InvalidBM25IndexCoverageError(str(exc)) from exc

    document_filters: list[str] = []
    policy_filters: list[str] = []
    document_params: list[object] = []
    policy_params: list[object] = []
    if validated_parse_status is not None:
        document_filters.append("f.parse_status = %s")
        document_params.append(validated_parse_status)
    if validated_document_group is not None:
        document_filters.append("d.document_group = %s")
        document_params.append(validated_document_group)
    if validated_chunk_policy_name is not None:
        policy_filters.append("chunk_policy_name = %s")
        policy_params.append(validated_chunk_policy_name)

    where_clause = f"WHERE {' AND '.join(document_filters)}" if document_filters else ""
    policy_where_clause = f"WHERE {' AND '.join(policy_filters)}" if policy_filters else ""

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                WITH selected_documents AS (
                    SELECT
                        d.document_id,
                        d.file_id,
                        d.document_title,
                        d.document_group,
                        d.access_scope,
                        f.original_file_name,
                        f.file_ext,
                        f.parse_status,
                        f.uploaded_at
                    FROM documents d
                    JOIN files f ON f.file_id = d.file_id
                    {where_clause}
                    ORDER BY f.uploaded_at DESC, d.document_id DESC
                    LIMIT %s
                ),
                selected_policies AS (
                    SELECT
                        chunk_policy_name,
                        target_token_size,
                        overlap_token_size,
                        split_strategy
                    FROM chunk_policies
                    {policy_where_clause}
                    ORDER BY target_token_size ASC, overlap_token_size ASC, chunk_policy_name ASC
                ),
                chunk_stats AS (
                    SELECT
                        c.document_id,
                        c.chunk_policy_name,
                        count(*)::int AS chunk_count
                    FROM chunks c
                    JOIN selected_documents sd ON sd.document_id = c.document_id
                    GROUP BY c.document_id, c.chunk_policy_name
                ),
                policy_chunk_stats AS (
                    SELECT
                        c.chunk_policy_name,
                        count(*)::int AS policy_chunk_count
                    FROM chunks c
                    GROUP BY c.chunk_policy_name
                ),
                term_stats AS (
                    SELECT
                        c.document_id,
                        c.chunk_policy_name,
                        count(DISTINCT ckt.chunk_id)::int AS indexed_chunk_count,
                        count(*)::int AS term_row_count,
                        max(ckt.created_at) AS latest_term_created_at
                    FROM chunk_keyword_terms ckt
                    JOIN chunks c ON c.chunk_id = ckt.chunk_id
                    JOIN selected_documents sd ON sd.document_id = c.document_id
                    WHERE ckt.tokenizer_name = %s
                    GROUP BY c.document_id, c.chunk_policy_name
                ),
                statistics AS (
                    SELECT
                        cks.chunk_policy_name,
                        count(*)::int AS statistics_term_count,
                        max(cks.corpus_chunk_count)::int AS statistics_corpus_chunk_count,
                        max(cks.average_document_length) AS average_document_length,
                        max(cks.updated_at) AS latest_statistics_updated_at
                    FROM chunk_keyword_statistics cks
                    WHERE cks.tokenizer_name = %s
                    GROUP BY cks.chunk_policy_name
                )
                SELECT
                    sd.document_id,
                    sd.file_id,
                    sd.document_title,
                    sd.original_file_name,
                    sd.file_ext,
                    sd.document_group,
                    sd.parse_status,
                    sd.access_scope,
                    sd.uploaded_at,
                    cp.chunk_policy_name,
                    cp.target_token_size,
                    cp.overlap_token_size,
                    cp.split_strategy,
                    %s::text AS tokenizer_name,
                    COALESCE(pcs.policy_chunk_count, 0)::int AS policy_chunk_count,
                    COALESCE(cs.chunk_count, 0)::int AS chunk_count,
                    COALESCE(ts.indexed_chunk_count, 0)::int AS indexed_chunk_count,
                    COALESCE(ts.term_row_count, 0)::int AS term_row_count,
                    COALESCE(st.statistics_term_count, 0)::int AS statistics_term_count,
                    COALESCE(st.statistics_corpus_chunk_count, 0)::int
                        AS statistics_corpus_chunk_count,
                    st.average_document_length,
                    ts.latest_term_created_at,
                    st.latest_statistics_updated_at
                FROM selected_documents sd
                CROSS JOIN selected_policies cp
                LEFT JOIN chunk_stats cs
                  ON cs.document_id = sd.document_id
                 AND cs.chunk_policy_name = cp.chunk_policy_name
                LEFT JOIN policy_chunk_stats pcs
                  ON pcs.chunk_policy_name = cp.chunk_policy_name
                LEFT JOIN term_stats ts
                  ON ts.document_id = sd.document_id
                 AND ts.chunk_policy_name = cp.chunk_policy_name
                LEFT JOIN statistics st ON st.chunk_policy_name = cp.chunk_policy_name
                ORDER BY
                    sd.uploaded_at DESC,
                    sd.document_id DESC,
                    cp.target_token_size ASC,
                    cp.overlap_token_size ASC,
                    cp.chunk_policy_name ASC
                """,
                (
                    *document_params,
                    validated_limit,
                    *policy_params,
                    validated_tokenizer_name,
                    validated_tokenizer_name,
                    validated_tokenizer_name,
                ),
            )
            rows = tuple(_row_to_bm25_index_coverage_row(dict(row)) for row in cursor.fetchall())

    return BM25IndexCoverageMatrix(
        summary=_build_summary(rows),
        rows=rows,
    )
