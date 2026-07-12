"""Embedding coverage matrix read-model helpers."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.core.database import connect
from app.core.document_inventory import (
    _validate_document_group,
    _validate_limit,
    _validate_parse_status,
)
from app.core.embedding_jobs import _validate_profile_name
from app.core.embedding_vectors import EMBEDDING_VECTOR_TABLES

EMBEDDING_COVERAGE_STATUSES = {
    "complete",
    "partial",
    "pending",
    "running",
    "failed",
    "skipped",
    "missing",
    "not_chunked",
}


@dataclass(frozen=True)
class EmbeddingCoverageProfileCell:
    profile_name: str
    model_name: str
    dimension: int
    storage_type: str
    is_active: bool
    chunk_count: int
    job_count: int
    pending_count: int
    running_count: int
    failed_count: int
    retryable_failed_count: int
    exhausted_failed_count: int
    succeeded_job_count: int
    skipped_count: int
    embedded_chunk_count: int
    coverage_percent: Decimal
    status: str
    latest_job_updated_at: datetime | None
    latest_embedding_at: datetime | None
    average_embedding_elapsed_ms: Decimal | None


@dataclass(frozen=True)
class EmbeddingCoverageDocument:
    document_id: int
    file_id: int
    document_title: str | None
    original_file_name: str
    file_ext: str | None
    document_group: str
    parse_status: str
    access_scope: str
    chunk_count: int
    uploaded_at: datetime
    profiles: tuple[EmbeddingCoverageProfileCell, ...]

    @property
    def complete_profile_count(self) -> int:
        return sum(1 for profile in self.profiles if profile.status == "complete")

    @property
    def attention_profile_count(self) -> int:
        return sum(1 for profile in self.profiles if profile.status in {"failed", "partial"})

    @property
    def missing_profile_count(self) -> int:
        return sum(1 for profile in self.profiles if profile.status in {"missing", "not_chunked"})


@dataclass(frozen=True)
class EmbeddingCoverageProfileSummary:
    profile_name: str
    model_name: str
    document_count: int
    complete_document_count: int
    partial_document_count: int
    pending_document_count: int
    running_document_count: int
    failed_document_count: int
    missing_document_count: int
    not_chunked_document_count: int
    total_chunk_count: int
    embedded_chunk_count: int
    coverage_percent: Decimal


@dataclass(frozen=True)
class EmbeddingCoverageSummary:
    document_count: int
    profile_count: int
    total_chunk_count: int
    expected_embedding_count: int
    embedded_chunk_count: int
    complete_cell_count: int
    incomplete_cell_count: int
    attention_cell_count: int
    coverage_percent: Decimal
    profile_summaries: tuple[EmbeddingCoverageProfileSummary, ...]


@dataclass(frozen=True)
class EmbeddingCoverageMatrix:
    summary: EmbeddingCoverageSummary
    documents: tuple[EmbeddingCoverageDocument, ...]


class InvalidEmbeddingCoverageError(ValueError):
    """Raised when embedding coverage query input is invalid."""


def _status_for_cell(
    *,
    chunk_count: int,
    job_count: int,
    pending_count: int,
    running_count: int,
    failed_count: int,
    succeeded_job_count: int,
    skipped_count: int,
    embedded_chunk_count: int,
) -> str:
    if chunk_count == 0:
        return "not_chunked"
    if embedded_chunk_count >= chunk_count:
        return "complete"
    if running_count > 0:
        return "running"
    if failed_count > 0 and pending_count == 0:
        return "failed"
    if pending_count > 0:
        return "pending"
    if embedded_chunk_count > 0 or succeeded_job_count > 0:
        return "partial"
    if job_count > 0 and skipped_count == job_count:
        return "skipped"
    return "missing"


def _coverage_percent(embedded_chunk_count: int, chunk_count: int) -> Decimal:
    if chunk_count <= 0:
        return Decimal("0.00")
    return (Decimal(embedded_chunk_count) * Decimal(100) / Decimal(chunk_count)).quantize(
        Decimal("0.01")
    )


def _overall_coverage_percent(
    *,
    embedded_chunk_count: int,
    expected_embedding_count: int,
) -> Decimal:
    if expected_embedding_count <= 0:
        return Decimal("0.00")
    return (
        Decimal(embedded_chunk_count) * Decimal(100) / Decimal(expected_embedding_count)
    ).quantize(Decimal("0.01"))


def _vector_stats_union_sql() -> str:
    statements = []
    for _profile_name, table in sorted(EMBEDDING_VECTOR_TABLES.items()):
        statements.append(f"""
            SELECT
                %s::text AS profile_name,
                c.document_id,
                count(v.chunk_id)::int AS embedded_chunk_count,
                max(v.created_at) AS latest_embedding_at,
                avg(v.elapsed_ms)::numeric(12,2) AS average_embedding_elapsed_ms
            FROM chunks c
            JOIN {table.table_name} v ON v.chunk_id = c.chunk_id
            GROUP BY c.document_id
            """)
    return "\nUNION ALL\n".join(statements)


def _row_int(row: dict[str, Any], key: str) -> int:
    return int(row[key] or 0)


def _row_to_profile_cell(row: dict[str, Any]) -> EmbeddingCoverageProfileCell:
    chunk_count = _row_int(row, "chunk_count")
    job_count = _row_int(row, "job_count")
    pending_count = _row_int(row, "pending_count")
    running_count = _row_int(row, "running_count")
    failed_count = _row_int(row, "failed_count")
    succeeded_job_count = _row_int(row, "succeeded_job_count")
    skipped_count = _row_int(row, "skipped_count")
    embedded_chunk_count = _row_int(row, "embedded_chunk_count")
    status = _status_for_cell(
        chunk_count=chunk_count,
        job_count=job_count,
        pending_count=pending_count,
        running_count=running_count,
        failed_count=failed_count,
        succeeded_job_count=succeeded_job_count,
        skipped_count=skipped_count,
        embedded_chunk_count=embedded_chunk_count,
    )
    return EmbeddingCoverageProfileCell(
        profile_name=str(row["profile_name"]),
        model_name=str(row["model_name"]),
        dimension=int(row["dimension"]),
        storage_type=str(row["storage_type"]),
        is_active=bool(row["is_active"]),
        chunk_count=chunk_count,
        job_count=job_count,
        pending_count=pending_count,
        running_count=running_count,
        failed_count=failed_count,
        retryable_failed_count=_row_int(row, "retryable_failed_count"),
        exhausted_failed_count=_row_int(row, "exhausted_failed_count"),
        succeeded_job_count=succeeded_job_count,
        skipped_count=skipped_count,
        embedded_chunk_count=embedded_chunk_count,
        coverage_percent=_coverage_percent(embedded_chunk_count, chunk_count),
        status=status,
        latest_job_updated_at=row["latest_job_updated_at"],
        latest_embedding_at=row["latest_embedding_at"],
        average_embedding_elapsed_ms=row["average_embedding_elapsed_ms"],
    )


def _profile_summary(
    profile_name: str,
    model_name: str,
    cells: list[EmbeddingCoverageProfileCell],
) -> EmbeddingCoverageProfileSummary:
    total_chunk_count = sum(cell.chunk_count for cell in cells)
    embedded_chunk_count = sum(cell.embedded_chunk_count for cell in cells)
    return EmbeddingCoverageProfileSummary(
        profile_name=profile_name,
        model_name=model_name,
        document_count=len(cells),
        complete_document_count=sum(1 for cell in cells if cell.status == "complete"),
        partial_document_count=sum(1 for cell in cells if cell.status == "partial"),
        pending_document_count=sum(1 for cell in cells if cell.status == "pending"),
        running_document_count=sum(1 for cell in cells if cell.status == "running"),
        failed_document_count=sum(1 for cell in cells if cell.status == "failed"),
        missing_document_count=sum(1 for cell in cells if cell.status == "missing"),
        not_chunked_document_count=sum(1 for cell in cells if cell.status == "not_chunked"),
        total_chunk_count=total_chunk_count,
        embedded_chunk_count=embedded_chunk_count,
        coverage_percent=_coverage_percent(embedded_chunk_count, total_chunk_count),
    )


def _build_summary(
    documents: tuple[EmbeddingCoverageDocument, ...],
) -> EmbeddingCoverageSummary:
    profile_cells: dict[str, list[EmbeddingCoverageProfileCell]] = {}
    for document in documents:
        for cell in document.profiles:
            profile_cells.setdefault(cell.profile_name, []).append(cell)

    profile_summaries = tuple(
        _profile_summary(profile_name, cells[0].model_name, cells)
        for profile_name, cells in sorted(profile_cells.items())
        if cells
    )
    total_chunk_count = sum(document.chunk_count for document in documents)
    embedded_chunk_count = sum(
        cell.embedded_chunk_count for document in documents for cell in document.profiles
    )
    expected_embedding_count = total_chunk_count * len(profile_summaries)
    complete_cell_count = sum(
        1 for document in documents for cell in document.profiles if cell.status == "complete"
    )
    total_cell_count = sum(len(document.profiles) for document in documents)
    attention_cell_count = sum(
        1
        for document in documents
        for cell in document.profiles
        if cell.status in {"failed", "partial"}
    )
    return EmbeddingCoverageSummary(
        document_count=len(documents),
        profile_count=len(profile_summaries),
        total_chunk_count=total_chunk_count,
        expected_embedding_count=expected_embedding_count,
        embedded_chunk_count=embedded_chunk_count,
        complete_cell_count=complete_cell_count,
        incomplete_cell_count=total_cell_count - complete_cell_count,
        attention_cell_count=attention_cell_count,
        coverage_percent=_overall_coverage_percent(
            embedded_chunk_count=embedded_chunk_count,
            expected_embedding_count=expected_embedding_count,
        ),
        profile_summaries=profile_summaries,
    )


def get_embedding_coverage_matrix(
    database_url: str,
    *,
    parse_status: str | None = None,
    document_group: str | None = None,
    profile_name: str | None = None,
    limit: int = 100,
) -> EmbeddingCoverageMatrix:
    try:
        validated_limit = _validate_limit(limit, max_limit=200)
        validated_parse_status = _validate_parse_status(parse_status)
        validated_document_group = _validate_document_group(document_group)
        validated_profile_name = _validate_profile_name(profile_name)
    except ValueError as exc:
        raise InvalidEmbeddingCoverageError(str(exc)) from exc

    document_filters: list[str] = []
    profile_filters = ["is_active"]
    document_params: list[object] = []
    profile_params: list[object] = []
    if validated_parse_status is not None:
        document_filters.append("f.parse_status = %s")
        document_params.append(validated_parse_status)
    if validated_document_group is not None:
        document_filters.append("d.document_group = %s")
        document_params.append(validated_document_group)
    if validated_profile_name is not None:
        profile_filters.append("profile_name = %s")
        profile_params.append(validated_profile_name)

    where_clause = f"WHERE {' AND '.join(document_filters)}" if document_filters else ""
    profile_where_clause = f"WHERE {' AND '.join(profile_filters)}"
    vector_profile_params = [
        table.profile_name
        for table in sorted(
            EMBEDDING_VECTOR_TABLES.values(),
            key=lambda item: item.profile_name,
        )
    ]

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
                selected_profiles AS (
                    SELECT
                        profile_name,
                        model_name,
                        dimension,
                        storage_type,
                        is_active
                    FROM embedding_profiles
                    {profile_where_clause}
                    ORDER BY profile_name ASC
                ),
                chunk_stats AS (
                    SELECT
                        c.document_id,
                        count(*)::int AS chunk_count
                    FROM chunks c
                    JOIN selected_documents sd ON sd.document_id = c.document_id
                    GROUP BY c.document_id
                ),
                job_stats AS (
                    SELECT
                        c.document_id,
                        ej.profile_name,
                        count(*)::int AS job_count,
                        count(*) FILTER (WHERE ej.status = 'pending')::int AS pending_count,
                        count(*) FILTER (WHERE ej.status = 'running')::int AS running_count,
                        count(*) FILTER (WHERE ej.status = 'failed')::int AS failed_count,
                        count(*) FILTER (
                            WHERE ej.status = 'failed' AND ej.attempts < ej.max_attempts
                        )::int AS retryable_failed_count,
                        count(*) FILTER (
                            WHERE ej.status = 'failed' AND ej.attempts >= ej.max_attempts
                        )::int AS exhausted_failed_count,
                        count(*) FILTER (WHERE ej.status = 'succeeded')::int AS succeeded_job_count,
                        count(*) FILTER (WHERE ej.status = 'skipped')::int AS skipped_count,
                        max(ej.updated_at) AS latest_job_updated_at
                    FROM embedding_jobs ej
                    JOIN chunks c ON c.chunk_id = ej.chunk_id
                    JOIN selected_documents sd ON sd.document_id = c.document_id
                    GROUP BY c.document_id, ej.profile_name
                ),
                vector_stats AS (
                    {_vector_stats_union_sql()}
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
                    sp.profile_name,
                    sp.model_name,
                    sp.dimension,
                    sp.storage_type,
                    sp.is_active,
                    COALESCE(cs.chunk_count, 0)::int AS chunk_count,
                    COALESCE(js.job_count, 0)::int AS job_count,
                    COALESCE(js.pending_count, 0)::int AS pending_count,
                    COALESCE(js.running_count, 0)::int AS running_count,
                    COALESCE(js.failed_count, 0)::int AS failed_count,
                    COALESCE(js.retryable_failed_count, 0)::int AS retryable_failed_count,
                    COALESCE(js.exhausted_failed_count, 0)::int AS exhausted_failed_count,
                    COALESCE(js.succeeded_job_count, 0)::int AS succeeded_job_count,
                    COALESCE(js.skipped_count, 0)::int AS skipped_count,
                    COALESCE(vs.embedded_chunk_count, 0)::int AS embedded_chunk_count,
                    js.latest_job_updated_at,
                    vs.latest_embedding_at,
                    vs.average_embedding_elapsed_ms
                FROM selected_documents sd
                CROSS JOIN selected_profiles sp
                LEFT JOIN chunk_stats cs ON cs.document_id = sd.document_id
                LEFT JOIN job_stats js
                  ON js.document_id = sd.document_id
                 AND js.profile_name = sp.profile_name
                LEFT JOIN vector_stats vs
                  ON vs.document_id = sd.document_id
                 AND vs.profile_name = sp.profile_name
                ORDER BY sd.uploaded_at DESC, sd.document_id DESC, sp.profile_name ASC
                """,
                (
                    *document_params,
                    validated_limit,
                    *profile_params,
                    *vector_profile_params,
                ),
            )
            rows = [dict(row) for row in cursor.fetchall()]

    documents_by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        document_id = int(row["document_id"])
        cell = _row_to_profile_cell(row)
        if document_id not in documents_by_id:
            documents_by_id[document_id] = {
                "document_id": document_id,
                "file_id": int(row["file_id"]),
                "document_title": row["document_title"],
                "original_file_name": str(row["original_file_name"]),
                "file_ext": row["file_ext"],
                "document_group": str(row["document_group"]),
                "parse_status": str(row["parse_status"]),
                "access_scope": str(row["access_scope"]),
                "chunk_count": cell.chunk_count,
                "uploaded_at": row["uploaded_at"],
                "profiles": [],
            }
        documents_by_id[document_id]["profiles"].append(cell)

    documents = tuple(
        EmbeddingCoverageDocument(
            document_id=int(payload["document_id"]),
            file_id=int(payload["file_id"]),
            document_title=payload["document_title"],
            original_file_name=str(payload["original_file_name"]),
            file_ext=payload["file_ext"],
            document_group=str(payload["document_group"]),
            parse_status=str(payload["parse_status"]),
            access_scope=str(payload["access_scope"]),
            chunk_count=int(payload["chunk_count"]),
            uploaded_at=payload["uploaded_at"],
            profiles=tuple(payload["profiles"]),
        )
        for payload in documents_by_id.values()
    )
    return EmbeddingCoverageMatrix(
        summary=_build_summary(documents),
        documents=documents,
    )
