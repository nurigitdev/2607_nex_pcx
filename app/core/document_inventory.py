"""Document inventory read-model helpers."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.core.database import connect

DOCUMENT_PARSE_STATUSES = {"pending", "running", "succeeded", "failed"}


@dataclass(frozen=True)
class DocumentInventoryItem:
    document_id: int
    file_id: int
    document_title: str | None
    original_file_name: str
    file_ext: str | None
    mime_type: str | None
    file_size_bytes: int | None
    document_group: str
    security_level: str
    document_status: str
    parse_status: str
    owner_user_id: int | None
    owner_login_id: str | None
    owner_display_name: str | None
    owner_org_unit_id: int | None
    owner_org_unit_name: str | None
    access_scope: str
    uploaded_by: str | None
    uploaded_by_user_id: int | None
    uploaded_by_login_id: str | None
    uploaded_by_display_name: str | None
    chunk_count: int
    total_token_count: int | None
    total_char_count: int
    latest_pipeline_job_id: int | None
    latest_pipeline_status: str | None
    latest_pipeline_stage: str | None
    latest_pipeline_progress_percent: Decimal | None
    uploaded_at: datetime
    updated_at: datetime


class InvalidDocumentInventoryError(ValueError):
    """Raised when document inventory query input is invalid."""


def _validate_limit(limit: int, *, max_limit: int = 500) -> int:
    if limit <= 0:
        raise InvalidDocumentInventoryError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidDocumentInventoryError(f"limit must be less than or equal to {max_limit}")
    return limit


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is None or value <= 0:
        raise InvalidDocumentInventoryError(f"{field_name} must be greater than 0")


def _validate_parse_status(parse_status: str | None) -> str | None:
    if parse_status is None:
        return None
    normalized = parse_status.strip()
    if normalized not in DOCUMENT_PARSE_STATUSES:
        raise InvalidDocumentInventoryError(f"Unsupported parse_status: {parse_status}")
    return normalized


def _validate_document_group(document_group: str | None) -> str | None:
    if document_group is None:
        return None
    normalized = document_group.strip()
    if not normalized:
        raise InvalidDocumentInventoryError("document_group must not be blank")
    return normalized


def _row_to_document_inventory_item(row: dict[str, Any]) -> DocumentInventoryItem:
    return DocumentInventoryItem(
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
        document_title=row["document_title"],
        original_file_name=str(row["original_file_name"]),
        file_ext=row["file_ext"],
        mime_type=row["mime_type"],
        file_size_bytes=(
            int(row["file_size_bytes"]) if row.get("file_size_bytes") is not None else None
        ),
        document_group=str(row["document_group"]),
        security_level=str(row["security_level"]),
        document_status=str(row["document_status"]),
        parse_status=str(row["parse_status"]),
        owner_user_id=int(row["owner_user_id"]) if row.get("owner_user_id") is not None else None,
        owner_login_id=row["owner_login_id"],
        owner_display_name=row["owner_display_name"],
        owner_org_unit_id=(
            int(row["owner_org_unit_id"]) if row.get("owner_org_unit_id") is not None else None
        ),
        owner_org_unit_name=row["owner_org_unit_name"],
        access_scope=str(row["access_scope"]),
        uploaded_by=row["uploaded_by"],
        uploaded_by_user_id=(
            int(row["uploaded_by_user_id"]) if row.get("uploaded_by_user_id") is not None else None
        ),
        uploaded_by_login_id=row["uploaded_by_login_id"],
        uploaded_by_display_name=row["uploaded_by_display_name"],
        chunk_count=int(row["chunk_count"]),
        total_token_count=(
            int(row["total_token_count"]) if row.get("total_token_count") is not None else None
        ),
        total_char_count=int(row["total_char_count"]),
        latest_pipeline_job_id=(
            int(row["latest_pipeline_job_id"])
            if row.get("latest_pipeline_job_id") is not None
            else None
        ),
        latest_pipeline_status=row["latest_pipeline_status"],
        latest_pipeline_stage=row["latest_pipeline_stage"],
        latest_pipeline_progress_percent=row["latest_pipeline_progress_percent"],
        uploaded_at=row["uploaded_at"],
        updated_at=row["updated_at"],
    )


def list_document_inventory(
    database_url: str,
    *,
    parse_status: str | None = None,
    document_group: str | None = None,
    limit: int = 100,
) -> list[DocumentInventoryItem]:
    validated_limit = _validate_limit(limit)
    validated_parse_status = _validate_parse_status(parse_status)
    validated_document_group = _validate_document_group(document_group)

    filters: list[str] = []
    params: list[object] = []
    if validated_parse_status is not None:
        filters.append("f.parse_status = %s")
        params.append(validated_parse_status)
    if validated_document_group is not None:
        filters.append("d.document_group = %s")
        params.append(validated_document_group)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    d.document_id,
                    d.file_id,
                    d.document_title,
                    d.document_group,
                    d.security_level,
                    d.document_status,
                    d.owner_user_id,
                    owner.login_id AS owner_login_id,
                    owner.display_name AS owner_display_name,
                    d.owner_org_unit_id,
                    org.org_unit_name AS owner_org_unit_name,
                    d.access_scope,
                    f.original_file_name,
                    f.file_ext,
                    f.mime_type,
                    f.file_size_bytes,
                    f.parse_status,
                    f.uploaded_by,
                    f.uploaded_by_user_id,
                    uploader.login_id AS uploaded_by_login_id,
                    uploader.display_name AS uploaded_by_display_name,
                    COALESCE(chunk_stats.chunk_count, 0) AS chunk_count,
                    chunk_stats.total_token_count,
                    COALESCE(chunk_stats.total_char_count, 0) AS total_char_count,
                    latest_job.job_id AS latest_pipeline_job_id,
                    latest_job.status AS latest_pipeline_status,
                    latest_job.stage AS latest_pipeline_stage,
                    latest_job.progress_percent AS latest_pipeline_progress_percent,
                    f.uploaded_at,
                    GREATEST(f.updated_at, d.updated_at) AS updated_at
                FROM documents d
                JOIN files f ON f.file_id = d.file_id
                LEFT JOIN app_users owner ON owner.user_id = d.owner_user_id
                LEFT JOIN app_users uploader ON uploader.user_id = f.uploaded_by_user_id
                LEFT JOIN org_units org ON org.org_unit_id = d.owner_org_unit_id
                LEFT JOIN LATERAL (
                    SELECT
                        count(*) AS chunk_count,
                        sum(token_count) AS total_token_count,
                        sum(char_count) AS total_char_count
                    FROM chunks c
                    WHERE c.document_id = d.document_id
                ) chunk_stats ON true
                LEFT JOIN LATERAL (
                    SELECT
                        pj.job_id,
                        pj.status,
                        pj.stage,
                        pj.progress_percent
                    FROM pipeline_jobs pj
                    WHERE pj.document_id = d.document_id
                       OR pj.file_id = f.file_id
                    ORDER BY pj.queued_at DESC, pj.job_id DESC
                    LIMIT 1
                ) latest_job ON true
                {where_clause}
                ORDER BY f.uploaded_at DESC, d.document_id DESC
                LIMIT %s
                """,
                [*params, validated_limit],
            )
            rows = cursor.fetchall()
    return [_row_to_document_inventory_item(dict(row)) for row in rows]


def get_document_inventory_item(
    database_url: str,
    document_id: int,
) -> DocumentInventoryItem | None:
    _require_positive_id(document_id, "document_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.document_id,
                    d.file_id,
                    d.document_title,
                    d.document_group,
                    d.security_level,
                    d.document_status,
                    d.owner_user_id,
                    owner.login_id AS owner_login_id,
                    owner.display_name AS owner_display_name,
                    d.owner_org_unit_id,
                    org.org_unit_name AS owner_org_unit_name,
                    d.access_scope,
                    f.original_file_name,
                    f.file_ext,
                    f.mime_type,
                    f.file_size_bytes,
                    f.parse_status,
                    f.uploaded_by,
                    f.uploaded_by_user_id,
                    uploader.login_id AS uploaded_by_login_id,
                    uploader.display_name AS uploaded_by_display_name,
                    COALESCE(chunk_stats.chunk_count, 0) AS chunk_count,
                    chunk_stats.total_token_count,
                    COALESCE(chunk_stats.total_char_count, 0) AS total_char_count,
                    latest_job.job_id AS latest_pipeline_job_id,
                    latest_job.status AS latest_pipeline_status,
                    latest_job.stage AS latest_pipeline_stage,
                    latest_job.progress_percent AS latest_pipeline_progress_percent,
                    f.uploaded_at,
                    GREATEST(f.updated_at, d.updated_at) AS updated_at
                FROM documents d
                JOIN files f ON f.file_id = d.file_id
                LEFT JOIN app_users owner ON owner.user_id = d.owner_user_id
                LEFT JOIN app_users uploader ON uploader.user_id = f.uploaded_by_user_id
                LEFT JOIN org_units org ON org.org_unit_id = d.owner_org_unit_id
                LEFT JOIN LATERAL (
                    SELECT
                        count(*) AS chunk_count,
                        sum(token_count) AS total_token_count,
                        sum(char_count) AS total_char_count
                    FROM chunks c
                    WHERE c.document_id = d.document_id
                ) chunk_stats ON true
                LEFT JOIN LATERAL (
                    SELECT
                        pj.job_id,
                        pj.status,
                        pj.stage,
                        pj.progress_percent
                    FROM pipeline_jobs pj
                    WHERE pj.document_id = d.document_id
                       OR pj.file_id = f.file_id
                    ORDER BY pj.queued_at DESC, pj.job_id DESC
                    LIMIT 1
                ) latest_job ON true
                WHERE d.document_id = %s
                """,
                (document_id,),
            )
            row = cursor.fetchone()
    return _row_to_document_inventory_item(dict(row)) if row else None
