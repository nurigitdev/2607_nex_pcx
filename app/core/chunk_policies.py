"""Chunk policy read-model helpers."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.core.chunks import DEFAULT_CHUNK_POLICY_NAME
from app.core.database import connect


@dataclass(frozen=True)
class ChunkPolicySummaryRecord:
    chunk_policy_name: str
    target_token_size: int
    overlap_token_size: int
    split_strategy: str
    preserve_table: bool
    preserve_code_block: bool
    description: str | None
    is_default: bool
    chunk_count: int
    document_count: int
    total_token_count: int | None
    total_char_count: int
    average_token_count: Decimal | None
    average_char_count: Decimal | None
    created_at: datetime


class InvalidChunkPolicyManagementError(ValueError):
    """Raised when chunk policy management query input is invalid."""


def _validate_policy_name(chunk_policy_name: str) -> str:
    normalized = chunk_policy_name.strip()
    if not normalized:
        raise InvalidChunkPolicyManagementError("chunk_policy_name must not be blank")
    return normalized


def _row_to_chunk_policy_summary(row: dict[str, Any]) -> ChunkPolicySummaryRecord:
    return ChunkPolicySummaryRecord(
        chunk_policy_name=str(row["chunk_policy_name"]),
        target_token_size=int(row["target_token_size"]),
        overlap_token_size=int(row["overlap_token_size"]),
        split_strategy=str(row["split_strategy"]),
        preserve_table=bool(row["preserve_table"]),
        preserve_code_block=bool(row["preserve_code_block"]),
        description=row["description"],
        is_default=bool(row["is_default"]),
        chunk_count=int(row["chunk_count"]),
        document_count=int(row["document_count"]),
        total_token_count=(
            int(row["total_token_count"]) if row.get("total_token_count") is not None else None
        ),
        total_char_count=int(row["total_char_count"]),
        average_token_count=row["average_token_count"],
        average_char_count=row["average_char_count"],
        created_at=row["created_at"],
    )


def _chunk_policy_summary_select(where_clause: str = "") -> str:
    return f"""
        SELECT
            cp.chunk_policy_name,
            cp.target_token_size,
            cp.overlap_token_size,
            cp.split_strategy,
            cp.preserve_table,
            cp.preserve_code_block,
            cp.description,
            (cp.chunk_policy_name = %s) AS is_default,
            COALESCE(stats.chunk_count, 0) AS chunk_count,
            COALESCE(stats.document_count, 0) AS document_count,
            stats.total_token_count,
            COALESCE(stats.total_char_count, 0) AS total_char_count,
            stats.average_token_count,
            stats.average_char_count,
            cp.created_at
        FROM chunk_policies cp
        LEFT JOIN LATERAL (
            SELECT
                count(*) AS chunk_count,
                count(DISTINCT c.document_id) AS document_count,
                sum(c.token_count) AS total_token_count,
                sum(c.char_count) AS total_char_count,
                avg(c.token_count) AS average_token_count,
                avg(c.char_count) AS average_char_count
            FROM chunks c
            WHERE c.chunk_policy_name = cp.chunk_policy_name
        ) stats ON true
        {where_clause}
    """


def list_chunk_policy_summaries(database_url: str) -> list[ChunkPolicySummaryRecord]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                {_chunk_policy_summary_select()}
                ORDER BY
                    is_default DESC,
                    cp.target_token_size ASC,
                    cp.overlap_token_size ASC,
                    cp.chunk_policy_name ASC
                """,
                (DEFAULT_CHUNK_POLICY_NAME,),
            )
            rows = cursor.fetchall()
    return [_row_to_chunk_policy_summary(dict(row)) for row in rows]


def get_chunk_policy_summary(
    database_url: str,
    chunk_policy_name: str,
) -> ChunkPolicySummaryRecord | None:
    validated_policy_name = _validate_policy_name(chunk_policy_name)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                _chunk_policy_summary_select("WHERE cp.chunk_policy_name = %s"),
                (DEFAULT_CHUNK_POLICY_NAME, validated_policy_name),
            )
            row = cursor.fetchone()
    return _row_to_chunk_policy_summary(dict(row)) if row else None
