"""Core dashboard metrics aggregated from operational tables."""

from dataclasses import dataclass
from decimal import Decimal

from app.core.database import connect


@dataclass(frozen=True)
class DashboardFileTypeSummary:
    file_type: str
    file_count: int
    document_count: int
    total_file_size_bytes: int


@dataclass(frozen=True)
class DashboardDocumentGroupSummary:
    document_group: str
    file_count: int
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class DashboardChunkPolicySummary:
    chunk_policy_name: str
    chunk_count: int
    average_token_count: float | None


@dataclass(frozen=True)
class DashboardCoreMetrics:
    file_count: int
    document_count: int
    chunk_count: int
    embedding_job_count: int
    search_log_count: int
    total_file_size_bytes: int
    average_file_size_bytes: float | None
    duplicate_checksum_count: int
    average_chunk_token_count: float | None
    file_types: tuple[DashboardFileTypeSummary, ...]
    document_groups: tuple[DashboardDocumentGroupSummary, ...]
    chunk_policies: tuple[DashboardChunkPolicySummary, ...]


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def get_dashboard_core_metrics(database_url: str) -> DashboardCoreMetrics:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH duplicate_checksums AS (
                    SELECT sha256_checksum
                    FROM files
                    WHERE sha256_checksum IS NOT NULL
                    GROUP BY sha256_checksum
                    HAVING COUNT(*) > 1
                )
                SELECT
                    (SELECT COUNT(*)::int FROM files) AS file_count,
                    (SELECT COUNT(*)::int FROM documents) AS document_count,
                    (SELECT COUNT(*)::int FROM chunks) AS chunk_count,
                    (SELECT COUNT(*)::int FROM embedding_jobs) AS embedding_job_count,
                    (SELECT COUNT(*)::int FROM search_logs) AS search_log_count,
                    COALESCE((SELECT SUM(file_size_bytes) FROM files), 0)::bigint
                        AS total_file_size_bytes,
                    (SELECT AVG(file_size_bytes) FROM files WHERE file_size_bytes IS NOT NULL)
                        AS average_file_size_bytes,
                    (SELECT COUNT(*)::int FROM duplicate_checksums)
                        AS duplicate_checksum_count,
                    (SELECT AVG(token_count) FROM chunks WHERE token_count IS NOT NULL)
                        AS average_chunk_token_count
                """
            )
            totals = dict(cursor.fetchone() or {})

            cursor.execute(
                """
                WITH file_type_files AS (
                    SELECT
                        COALESCE(NULLIF(btrim(file_ext), ''), 'unknown') AS file_type,
                        COUNT(*)::int AS file_count,
                        COALESCE(SUM(file_size_bytes), 0)::bigint AS total_file_size_bytes
                    FROM files
                    GROUP BY COALESCE(NULLIF(btrim(file_ext), ''), 'unknown')
                ),
                file_type_documents AS (
                    SELECT
                        COALESCE(NULLIF(btrim(f.file_ext), ''), 'unknown') AS file_type,
                        COUNT(d.document_id)::int AS document_count
                    FROM files f
                    LEFT JOIN documents d ON d.file_id = f.file_id
                    GROUP BY COALESCE(NULLIF(btrim(f.file_ext), ''), 'unknown')
                )
                SELECT
                    file_type_files.file_type,
                    file_type_files.file_count,
                    COALESCE(file_type_documents.document_count, 0)::int AS document_count,
                    file_type_files.total_file_size_bytes
                FROM file_type_files
                LEFT JOIN file_type_documents
                    ON file_type_documents.file_type = file_type_files.file_type
                ORDER BY document_count DESC, file_count DESC, file_type ASC
                LIMIT 8
                """
            )
            file_type_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT
                    d.document_group,
                    COUNT(DISTINCT d.file_id)::int AS file_count,
                    COUNT(DISTINCT d.document_id)::int AS document_count,
                    COUNT(c.chunk_id)::int AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_group
                ORDER BY document_count DESC, chunk_count DESC, d.document_group ASC
                LIMIT 8
                """
            )
            document_group_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT
                    chunk_policy_name,
                    COUNT(*)::int AS chunk_count,
                    AVG(token_count) FILTER (WHERE token_count IS NOT NULL)
                        AS average_token_count
                FROM chunks
                GROUP BY chunk_policy_name
                ORDER BY chunk_count DESC, chunk_policy_name ASC
                LIMIT 8
                """
            )
            chunk_policy_rows = cursor.fetchall()

    return DashboardCoreMetrics(
        file_count=int(totals["file_count"]),
        document_count=int(totals["document_count"]),
        chunk_count=int(totals["chunk_count"]),
        embedding_job_count=int(totals["embedding_job_count"]),
        search_log_count=int(totals["search_log_count"]),
        total_file_size_bytes=int(totals["total_file_size_bytes"]),
        average_file_size_bytes=_float_or_none(totals["average_file_size_bytes"]),
        duplicate_checksum_count=int(totals["duplicate_checksum_count"]),
        average_chunk_token_count=_float_or_none(totals["average_chunk_token_count"]),
        file_types=tuple(
            DashboardFileTypeSummary(
                file_type=str(row["file_type"]),
                file_count=int(row["file_count"]),
                document_count=int(row["document_count"]),
                total_file_size_bytes=int(row["total_file_size_bytes"]),
            )
            for row in file_type_rows
        ),
        document_groups=tuple(
            DashboardDocumentGroupSummary(
                document_group=str(row["document_group"]),
                file_count=int(row["file_count"]),
                document_count=int(row["document_count"]),
                chunk_count=int(row["chunk_count"]),
            )
            for row in document_group_rows
        ),
        chunk_policies=tuple(
            DashboardChunkPolicySummary(
                chunk_policy_name=str(row["chunk_policy_name"]),
                chunk_count=int(row["chunk_count"]),
                average_token_count=_float_or_none(row["average_token_count"]),
            )
            for row in chunk_policy_rows
        ),
    )
