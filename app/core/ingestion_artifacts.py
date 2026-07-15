"""Repository helpers for ingestion extraction artifacts and source blocks."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect

EXTRACTION_PROVIDER_MODES = {"local", "remote"}
EXTRACTION_RUN_STATUSES = {"pending", "running", "succeeded", "failed", "skipped"}
EXTRACTION_ARTIFACT_TYPES = {
    "normalized_markdown",
    "plain_text",
    "parser_metadata",
    "warning_report",
    "source_snapshot",
}
DOCUMENT_BLOCK_TYPES = {
    "document",
    "heading",
    "paragraph",
    "table",
    "image",
    "figure",
    "list",
    "code",
    "page",
    "slide",
    "sheet",
}
EXTRACTION_QUALITY_SNAPSHOT_STATUSES = {"passed", "warning", "failed"}


@dataclass(frozen=True)
class ExtractionProfileInput:
    extraction_profile_name: str
    extractor_name: str
    extractor_version: str
    supported_file_types: tuple[str, ...]
    provider_mode: str = "local"
    default_options: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass(frozen=True)
class ExtractionProfileRecord:
    extraction_profile_name: str
    extractor_name: str
    extractor_version: str
    provider_mode: str
    supported_file_types: tuple[str, ...]
    default_options: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ExtractionRunInput:
    file_id: int
    document_id: int | None = None
    extraction_profile_name: str | None = None
    status: str = "pending"
    provider_mode: str = "local"
    extractor_name: str | None = None
    extractor_version: str | None = None
    elapsed_ms: int | None = None
    warning_count: int = 0
    error_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionRunRecord:
    extraction_run_id: int
    file_id: int
    document_id: int | None
    extraction_profile_name: str | None
    status: str
    provider_mode: str
    extractor_name: str | None
    extractor_version: str | None
    started_at: datetime | None
    finished_at: datetime | None
    elapsed_ms: int | None
    warning_count: int
    error_count: int
    error_code: str | None
    error_message: str | None
    runtime_metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ExtractionArtifactInput:
    file_id: int
    artifact_type: str
    extraction_run_id: int | None = None
    document_id: int | None = None
    content_text: str | None = None
    storage_path: str | None = None
    content_hash: str | None = None
    size_bytes: int | None = None
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionArtifactRecord:
    artifact_id: int
    extraction_run_id: int | None
    file_id: int
    document_id: int | None
    artifact_type: str
    content_text: str | None
    storage_path: str | None
    content_hash: str | None
    size_bytes: int | None
    language: str | None
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class DocumentBlockInput:
    artifact_id: int
    document_id: int
    block_seq: int
    block_type: str
    parent_block_id: int | None = None
    content_text: str | None = None
    content_markdown: str | None = None
    heading_path: tuple[str, ...] = ()
    source_anchor: dict[str, Any] = field(default_factory=dict)
    page_no: int | None = None
    slide_no: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentBlockRecord:
    block_id: int
    artifact_id: int
    document_id: int
    parent_block_id: int | None
    block_seq: int
    block_type: str
    content_text: str | None
    content_markdown: str | None
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
class TableArtifactInput:
    block_id: int
    content_markdown: str | None = None
    content_json: dict[str, Any] | None = None
    storage_path: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    source_anchor: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableArtifactRecord:
    table_artifact_id: int
    block_id: int
    content_markdown: str | None
    content_json: dict[str, Any] | None
    storage_path: str | None
    row_count: int | None
    column_count: int | None
    source_anchor: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ImageArtifactInput:
    block_id: int
    storage_path: str
    mime_type: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    ocr_text: str | None = None
    caption_text: str | None = None
    surrounding_text: str | None = None
    source_anchor: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageArtifactRecord:
    image_artifact_id: int
    block_id: int
    storage_path: str
    mime_type: str | None
    width_px: int | None
    height_px: int | None
    ocr_text: str | None
    caption_text: str | None
    surrounding_text: str | None
    source_anchor: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ExtractionQualitySnapshotInput:
    document_id: int
    file_id: int
    artifact_id: int
    artifact_type: str
    status: str
    block_count: int
    source_anchor_count: int
    issue_count: int
    warning_count: int
    failed_count: int
    block_summary: dict[str, Any]
    quality_payload: dict[str, Any]
    extraction_run_id: int | None = None
    extraction_profile_name: str | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    content_length: int | None = None
    content_lines: int | None = None
    source_anchor_coverage_percent: float | Decimal | None = None
    created_by: str | None = None
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class ExtractionQualitySnapshotRecord:
    snapshot_id: int
    document_id: int
    file_id: int
    artifact_id: int
    extraction_run_id: int | None
    artifact_type: str
    extraction_profile_name: str | None
    extractor_name: str | None
    extractor_version: str | None
    status: str
    content_length: int | None
    content_lines: int | None
    block_count: int
    source_anchor_count: int
    source_anchor_coverage_percent: float | None
    issue_count: int
    warning_count: int
    failed_count: int
    block_summary: dict[str, Any]
    quality_payload: dict[str, Any]
    created_by: str | None
    created_by_user_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class ExtractionQualitySnapshotSummary:
    document_id: int
    artifact_id: int | None
    snapshot_count: int
    passed_count: int
    warning_count: int
    failed_count: int
    latest_snapshot: ExtractionQualitySnapshotRecord | None


class InvalidIngestionArtifactError(ValueError):
    """Raised when ingestion artifact metadata is invalid before reaching the DB."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidIngestionArtifactError(f"{field_name} must be greater than 0")


def _validate_optional_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidIngestionArtifactError(f"{field_name} must be greater than 0")


def _require_non_blank(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise InvalidIngestionArtifactError(f"{field_name} is required")
    return stripped


def _validate_non_negative(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise InvalidIngestionArtifactError(f"{field_name} must be greater than or equal to 0")


def _validate_positive(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidIngestionArtifactError(f"{field_name} must be greater than 0")


def _validate_provider_mode(provider_mode: str) -> None:
    if provider_mode not in EXTRACTION_PROVIDER_MODES:
        raise InvalidIngestionArtifactError(f"Unsupported provider_mode: {provider_mode}")


def _validate_run_status(status: str) -> None:
    if status not in EXTRACTION_RUN_STATUSES:
        raise InvalidIngestionArtifactError(f"Unsupported extraction run status: {status}")


def _validate_artifact_type(artifact_type: str) -> None:
    if artifact_type not in EXTRACTION_ARTIFACT_TYPES:
        raise InvalidIngestionArtifactError(f"Unsupported artifact_type: {artifact_type}")


def _validate_block_type(block_type: str) -> None:
    if block_type not in DOCUMENT_BLOCK_TYPES:
        raise InvalidIngestionArtifactError(f"Unsupported block_type: {block_type}")


def _validate_quality_snapshot_status(status: str) -> None:
    if status not in EXTRACTION_QUALITY_SNAPSHOT_STATUSES:
        raise InvalidIngestionArtifactError(f"Unsupported extraction quality status: {status}")


def _validate_char_range(start: int | None, end: int | None) -> None:
    _validate_non_negative(start, "char_start")
    _validate_non_negative(end, "char_end")
    if start is not None and end is not None and end < start:
        raise InvalidIngestionArtifactError("char_end must be greater than or equal to char_start")


def _validate_percent(value: float | Decimal | None, field_name: str) -> None:
    if value is None:
        return
    if value < 0 or value > 100:
        raise InvalidIngestionArtifactError(f"{field_name} must be between 0 and 100")


def _validate_json_object(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise InvalidIngestionArtifactError(f"{field_name} must be an object")


def _validate_optional_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_blank(value, field_name)


def validate_extraction_profile_input(profile_input: ExtractionProfileInput) -> None:
    _require_non_blank(profile_input.extraction_profile_name, "extraction_profile_name")
    _require_non_blank(profile_input.extractor_name, "extractor_name")
    _require_non_blank(profile_input.extractor_version, "extractor_version")
    _validate_provider_mode(profile_input.provider_mode)
    if not profile_input.supported_file_types:
        raise InvalidIngestionArtifactError("supported_file_types is required")
    for file_type in profile_input.supported_file_types:
        _require_non_blank(file_type, "supported_file_types")


def validate_extraction_run_input(run_input: ExtractionRunInput) -> None:
    _require_positive_id(run_input.file_id, "file_id")
    _require_positive_id(run_input.document_id, "document_id")
    if run_input.extraction_profile_name is not None:
        _require_non_blank(run_input.extraction_profile_name, "extraction_profile_name")
    if run_input.extractor_name is not None:
        _require_non_blank(run_input.extractor_name, "extractor_name")
    if run_input.extractor_version is not None:
        _require_non_blank(run_input.extractor_version, "extractor_version")
    _validate_run_status(run_input.status)
    _validate_provider_mode(run_input.provider_mode)
    _validate_non_negative(run_input.elapsed_ms, "elapsed_ms")
    _validate_non_negative(run_input.warning_count, "warning_count")
    _validate_non_negative(run_input.error_count, "error_count")
    if run_input.error_code is not None:
        _require_non_blank(run_input.error_code, "error_code")
    if run_input.error_message is not None:
        _require_non_blank(run_input.error_message, "error_message")


def validate_extraction_artifact_input(artifact_input: ExtractionArtifactInput) -> None:
    _require_positive_id(artifact_input.file_id, "file_id")
    _require_positive_id(artifact_input.extraction_run_id, "extraction_run_id")
    _require_positive_id(artifact_input.document_id, "document_id")
    _validate_artifact_type(artifact_input.artifact_type)
    has_content = bool(artifact_input.content_text) or bool(artifact_input.storage_path)
    if not has_content:
        raise InvalidIngestionArtifactError("content_text or storage_path is required")
    if artifact_input.storage_path is not None:
        _require_non_blank(artifact_input.storage_path, "storage_path")
    if artifact_input.content_hash is not None:
        _require_non_blank(artifact_input.content_hash, "content_hash")
    _validate_non_negative(artifact_input.size_bytes, "size_bytes")


def validate_document_block_input(block_input: DocumentBlockInput) -> None:
    _require_positive_id(block_input.artifact_id, "artifact_id")
    _require_positive_id(block_input.document_id, "document_id")
    _require_positive_id(block_input.parent_block_id, "parent_block_id")
    _validate_non_negative(block_input.block_seq, "block_seq")
    _validate_block_type(block_input.block_type)
    _validate_positive(block_input.page_no, "page_no")
    _validate_positive(block_input.slide_no, "slide_no")
    _validate_non_negative(block_input.token_count, "token_count")
    _validate_char_range(block_input.char_start, block_input.char_end)


def validate_table_artifact_input(table_input: TableArtifactInput) -> None:
    _require_positive_id(table_input.block_id, "block_id")
    if not (table_input.content_markdown or table_input.content_json or table_input.storage_path):
        raise InvalidIngestionArtifactError(
            "content_markdown, content_json, or storage_path is required"
        )
    if table_input.storage_path is not None:
        _require_non_blank(table_input.storage_path, "storage_path")
    _validate_non_negative(table_input.row_count, "row_count")
    _validate_non_negative(table_input.column_count, "column_count")


def validate_image_artifact_input(image_input: ImageArtifactInput) -> None:
    _require_positive_id(image_input.block_id, "block_id")
    _require_non_blank(image_input.storage_path, "storage_path")
    _validate_positive(image_input.width_px, "width_px")
    _validate_positive(image_input.height_px, "height_px")


def validate_extraction_quality_snapshot_input(
    snapshot_input: ExtractionQualitySnapshotInput,
) -> None:
    _require_positive_id(snapshot_input.document_id, "document_id")
    _require_positive_id(snapshot_input.file_id, "file_id")
    _require_positive_id(snapshot_input.artifact_id, "artifact_id")
    _require_positive_id(snapshot_input.extraction_run_id, "extraction_run_id")
    _validate_optional_positive_id(snapshot_input.created_by_user_id, "created_by_user_id")
    _require_non_blank(snapshot_input.artifact_type, "artifact_type")
    _validate_quality_snapshot_status(snapshot_input.status)
    _validate_non_negative(snapshot_input.content_length, "content_length")
    _validate_non_negative(snapshot_input.content_lines, "content_lines")
    _validate_non_negative(snapshot_input.block_count, "block_count")
    _validate_non_negative(snapshot_input.source_anchor_count, "source_anchor_count")
    _validate_percent(
        snapshot_input.source_anchor_coverage_percent,
        "source_anchor_coverage_percent",
    )
    _validate_non_negative(snapshot_input.issue_count, "issue_count")
    _validate_non_negative(snapshot_input.warning_count, "warning_count")
    _validate_non_negative(snapshot_input.failed_count, "failed_count")
    _validate_json_object(snapshot_input.block_summary, "block_summary")
    _validate_json_object(snapshot_input.quality_payload, "quality_payload")
    _validate_optional_nonblank(
        snapshot_input.extraction_profile_name,
        "extraction_profile_name",
    )
    _validate_optional_nonblank(snapshot_input.extractor_name, "extractor_name")
    _validate_optional_nonblank(snapshot_input.extractor_version, "extractor_version")
    _validate_optional_nonblank(snapshot_input.created_by, "created_by")


def _row_to_extraction_profile_record(row: dict[str, Any]) -> ExtractionProfileRecord:
    return ExtractionProfileRecord(
        extraction_profile_name=str(row["extraction_profile_name"]),
        extractor_name=str(row["extractor_name"]),
        extractor_version=str(row["extractor_version"]),
        provider_mode=str(row["provider_mode"]),
        supported_file_types=tuple(row["supported_file_types"] or ()),
        default_options=dict(row["default_options"] or {}),
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_extraction_run_record(row: dict[str, Any]) -> ExtractionRunRecord:
    return ExtractionRunRecord(
        extraction_run_id=int(row["extraction_run_id"]),
        file_id=int(row["file_id"]),
        document_id=int(row["document_id"]) if row.get("document_id") is not None else None,
        extraction_profile_name=row["extraction_profile_name"],
        status=str(row["status"]),
        provider_mode=str(row["provider_mode"]),
        extractor_name=row["extractor_name"],
        extractor_version=row["extractor_version"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        elapsed_ms=int(row["elapsed_ms"]) if row.get("elapsed_ms") is not None else None,
        warning_count=int(row["warning_count"]),
        error_count=int(row["error_count"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_extraction_artifact_record(row: dict[str, Any]) -> ExtractionArtifactRecord:
    return ExtractionArtifactRecord(
        artifact_id=int(row["artifact_id"]),
        extraction_run_id=(
            int(row["extraction_run_id"]) if row.get("extraction_run_id") is not None else None
        ),
        file_id=int(row["file_id"]),
        document_id=int(row["document_id"]) if row.get("document_id") is not None else None,
        artifact_type=str(row["artifact_type"]),
        content_text=row["content_text"],
        storage_path=row["storage_path"],
        content_hash=row["content_hash"],
        size_bytes=int(row["size_bytes"]) if row.get("size_bytes") is not None else None,
        language=row["language"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def _row_to_document_block_record(row: dict[str, Any]) -> DocumentBlockRecord:
    return DocumentBlockRecord(
        block_id=int(row["block_id"]),
        artifact_id=int(row["artifact_id"]),
        document_id=int(row["document_id"]),
        parent_block_id=(
            int(row["parent_block_id"]) if row.get("parent_block_id") is not None else None
        ),
        block_seq=int(row["block_seq"]),
        block_type=str(row["block_type"]),
        content_text=row["content_text"],
        content_markdown=row["content_markdown"],
        heading_path=tuple(row["heading_path"] or ()),
        source_anchor=dict(row["source_anchor"] or {}),
        page_no=int(row["page_no"]) if row.get("page_no") is not None else None,
        slide_no=int(row["slide_no"]) if row.get("slide_no") is not None else None,
        sheet_name=row["sheet_name"],
        cell_range=row["cell_range"],
        char_start=int(row["char_start"]) if row.get("char_start") is not None else None,
        char_end=int(row["char_end"]) if row.get("char_end") is not None else None,
        token_count=int(row["token_count"]) if row.get("token_count") is not None else None,
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def _row_to_table_artifact_record(row: dict[str, Any]) -> TableArtifactRecord:
    return TableArtifactRecord(
        table_artifact_id=int(row["table_artifact_id"]),
        block_id=int(row["block_id"]),
        content_markdown=row["content_markdown"],
        content_json=dict(row["content_json"]) if row.get("content_json") is not None else None,
        storage_path=row["storage_path"],
        row_count=int(row["row_count"]) if row.get("row_count") is not None else None,
        column_count=(int(row["column_count"]) if row.get("column_count") is not None else None),
        source_anchor=dict(row["source_anchor"] or {}),
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def _row_to_image_artifact_record(row: dict[str, Any]) -> ImageArtifactRecord:
    return ImageArtifactRecord(
        image_artifact_id=int(row["image_artifact_id"]),
        block_id=int(row["block_id"]),
        storage_path=str(row["storage_path"]),
        mime_type=row["mime_type"],
        width_px=int(row["width_px"]) if row.get("width_px") is not None else None,
        height_px=int(row["height_px"]) if row.get("height_px") is not None else None,
        ocr_text=row["ocr_text"],
        caption_text=row["caption_text"],
        surrounding_text=row["surrounding_text"],
        source_anchor=dict(row["source_anchor"] or {}),
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def _row_to_extraction_quality_snapshot_record(
    row: dict[str, Any],
) -> ExtractionQualitySnapshotRecord:
    coverage = row.get("source_anchor_coverage_percent")
    return ExtractionQualitySnapshotRecord(
        snapshot_id=int(row["snapshot_id"]),
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
        artifact_id=int(row["artifact_id"]),
        extraction_run_id=(
            int(row["extraction_run_id"]) if row.get("extraction_run_id") is not None else None
        ),
        artifact_type=str(row["artifact_type"]),
        extraction_profile_name=row["extraction_profile_name"],
        extractor_name=row["extractor_name"],
        extractor_version=row["extractor_version"],
        status=str(row["status"]),
        content_length=(
            int(row["content_length"]) if row.get("content_length") is not None else None
        ),
        content_lines=int(row["content_lines"]) if row.get("content_lines") is not None else None,
        block_count=int(row["block_count"]),
        source_anchor_count=int(row["source_anchor_count"]),
        source_anchor_coverage_percent=float(coverage) if coverage is not None else None,
        issue_count=int(row["issue_count"]),
        warning_count=int(row["warning_count"]),
        failed_count=int(row["failed_count"]),
        block_summary=dict(row["block_summary"] or {}),
        quality_payload=dict(row["quality_payload"] or {}),
        created_by=row["created_by"],
        created_by_user_id=(
            int(row["created_by_user_id"]) if row.get("created_by_user_id") is not None else None
        ),
        created_at=row["created_at"],
    )


def _select_extraction_profile_columns(alias: str = "extraction_profiles") -> str:
    return f"""
        {alias}.extraction_profile_name,
        {alias}.extractor_name,
        {alias}.extractor_version,
        {alias}.provider_mode,
        {alias}.supported_file_types,
        {alias}.default_options,
        {alias}.is_active,
        {alias}.created_at,
        {alias}.updated_at
    """


def _select_extraction_run_columns(alias: str = "extraction_runs") -> str:
    return f"""
        {alias}.extraction_run_id,
        {alias}.file_id,
        {alias}.document_id,
        {alias}.extraction_profile_name,
        {alias}.status,
        {alias}.provider_mode,
        {alias}.extractor_name,
        {alias}.extractor_version,
        {alias}.started_at,
        {alias}.finished_at,
        {alias}.elapsed_ms,
        {alias}.warning_count,
        {alias}.error_count,
        {alias}.error_code,
        {alias}.error_message,
        {alias}.runtime_metadata,
        {alias}.created_at,
        {alias}.updated_at
    """


def _select_extraction_artifact_columns(alias: str = "extraction_artifacts") -> str:
    return f"""
        {alias}.artifact_id,
        {alias}.extraction_run_id,
        {alias}.file_id,
        {alias}.document_id,
        {alias}.artifact_type,
        {alias}.content_text,
        {alias}.storage_path,
        {alias}.content_hash,
        {alias}.size_bytes,
        {alias}.language,
        {alias}.metadata,
        {alias}.created_at
    """


def _select_document_block_columns(alias: str = "document_blocks") -> str:
    return f"""
        {alias}.block_id,
        {alias}.artifact_id,
        {alias}.document_id,
        {alias}.parent_block_id,
        {alias}.block_seq,
        {alias}.block_type,
        {alias}.content_text,
        {alias}.content_markdown,
        {alias}.heading_path,
        {alias}.source_anchor,
        {alias}.page_no,
        {alias}.slide_no,
        {alias}.sheet_name,
        {alias}.cell_range,
        {alias}.char_start,
        {alias}.char_end,
        {alias}.token_count,
        {alias}.metadata,
        {alias}.created_at
    """


def _select_table_artifact_columns(alias: str = "table_artifacts") -> str:
    return f"""
        {alias}.table_artifact_id,
        {alias}.block_id,
        {alias}.content_markdown,
        {alias}.content_json,
        {alias}.storage_path,
        {alias}.row_count,
        {alias}.column_count,
        {alias}.source_anchor,
        {alias}.metadata,
        {alias}.created_at
    """


def _select_image_artifact_columns(alias: str = "image_artifacts") -> str:
    return f"""
        {alias}.image_artifact_id,
        {alias}.block_id,
        {alias}.storage_path,
        {alias}.mime_type,
        {alias}.width_px,
        {alias}.height_px,
        {alias}.ocr_text,
        {alias}.caption_text,
        {alias}.surrounding_text,
        {alias}.source_anchor,
        {alias}.metadata,
        {alias}.created_at
    """


def _select_extraction_quality_snapshot_columns(
    alias: str = "extraction_quality_snapshots",
) -> str:
    return f"""
        {alias}.snapshot_id,
        {alias}.document_id,
        {alias}.file_id,
        {alias}.artifact_id,
        {alias}.extraction_run_id,
        {alias}.artifact_type,
        {alias}.extraction_profile_name,
        {alias}.extractor_name,
        {alias}.extractor_version,
        {alias}.status,
        {alias}.content_length,
        {alias}.content_lines,
        {alias}.block_count,
        {alias}.source_anchor_count,
        {alias}.source_anchor_coverage_percent,
        {alias}.issue_count,
        {alias}.warning_count,
        {alias}.failed_count,
        {alias}.block_summary,
        {alias}.quality_payload,
        {alias}.created_by,
        {alias}.created_by_user_id,
        {alias}.created_at
    """


def upsert_extraction_profile_in_connection(
    connection: Connection,
    profile_input: ExtractionProfileInput,
) -> ExtractionProfileRecord:
    validate_extraction_profile_input(profile_input)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO extraction_profiles (
                extraction_profile_name,
                extractor_name,
                extractor_version,
                provider_mode,
                supported_file_types,
                default_options,
                is_active
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (extraction_profile_name) DO UPDATE
            SET extractor_name = EXCLUDED.extractor_name,
                extractor_version = EXCLUDED.extractor_version,
                provider_mode = EXCLUDED.provider_mode,
                supported_file_types = EXCLUDED.supported_file_types,
                default_options = EXCLUDED.default_options,
                is_active = EXCLUDED.is_active,
                updated_at = now()
            RETURNING {_select_extraction_profile_columns()}
            """,
            (
                profile_input.extraction_profile_name.strip(),
                profile_input.extractor_name.strip(),
                profile_input.extractor_version.strip(),
                profile_input.provider_mode,
                list(profile_input.supported_file_types),
                Json(profile_input.default_options),
                profile_input.is_active,
            ),
        )
        return _row_to_extraction_profile_record(dict(cursor.fetchone()))


def upsert_extraction_profile(
    database_url: str,
    profile_input: ExtractionProfileInput,
) -> ExtractionProfileRecord:
    with connect(database_url) as connection:
        return upsert_extraction_profile_in_connection(connection, profile_input)


def get_extraction_profile_in_connection(
    connection: Connection,
    extraction_profile_name: str,
) -> ExtractionProfileRecord | None:
    profile_name = _require_non_blank(extraction_profile_name, "extraction_profile_name")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_select_extraction_profile_columns()}
            FROM extraction_profiles
            WHERE extraction_profile_name = %s
            """,
            (profile_name,),
        )
        row = cursor.fetchone()
    return _row_to_extraction_profile_record(dict(row)) if row else None


def get_extraction_profile(
    database_url: str,
    extraction_profile_name: str,
) -> ExtractionProfileRecord | None:
    with connect(database_url) as connection:
        return get_extraction_profile_in_connection(connection, extraction_profile_name)


def list_extraction_profiles(
    database_url: str,
    *,
    active_only: bool = False,
) -> list[ExtractionProfileRecord]:
    where_clause = "WHERE is_active" if active_only else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_extraction_profile_columns()}
                FROM extraction_profiles
                {where_clause}
                ORDER BY extraction_profile_name ASC
                """,
            )
            rows = cursor.fetchall()
    return [_row_to_extraction_profile_record(dict(row)) for row in rows]


def create_extraction_run_in_connection(
    connection: Connection,
    run_input: ExtractionRunInput,
) -> ExtractionRunRecord:
    validate_extraction_run_input(run_input)
    started_expression = "now()" if run_input.status == "running" else "NULL"
    finished_expression = (
        "now()" if run_input.status in {"succeeded", "failed", "skipped"} else "NULL"
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO extraction_runs (
                file_id,
                document_id,
                extraction_profile_name,
                status,
                provider_mode,
                extractor_name,
                extractor_version,
                started_at,
                finished_at,
                elapsed_ms,
                warning_count,
                error_count,
                error_code,
                error_message,
                runtime_metadata
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                {started_expression},
                {finished_expression},
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            RETURNING {_select_extraction_run_columns()}
            """,
            (
                run_input.file_id,
                run_input.document_id,
                (
                    run_input.extraction_profile_name.strip()
                    if run_input.extraction_profile_name is not None
                    else None
                ),
                run_input.status,
                run_input.provider_mode,
                run_input.extractor_name.strip() if run_input.extractor_name is not None else None,
                (
                    run_input.extractor_version.strip()
                    if run_input.extractor_version is not None
                    else None
                ),
                run_input.elapsed_ms,
                run_input.warning_count,
                run_input.error_count,
                run_input.error_code.strip() if run_input.error_code is not None else None,
                run_input.error_message.strip() if run_input.error_message is not None else None,
                Json(run_input.runtime_metadata),
            ),
        )
        return _row_to_extraction_run_record(dict(cursor.fetchone()))


def create_extraction_run(
    database_url: str,
    run_input: ExtractionRunInput,
) -> ExtractionRunRecord:
    with connect(database_url) as connection:
        return create_extraction_run_in_connection(connection, run_input)


def get_extraction_run_in_connection(
    connection: Connection,
    extraction_run_id: int,
) -> ExtractionRunRecord | None:
    _require_positive_id(extraction_run_id, "extraction_run_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_select_extraction_run_columns()}
            FROM extraction_runs
            WHERE extraction_run_id = %s
            """,
            (extraction_run_id,),
        )
        row = cursor.fetchone()
    return _row_to_extraction_run_record(dict(row)) if row else None


def get_extraction_run(database_url: str, extraction_run_id: int) -> ExtractionRunRecord | None:
    with connect(database_url) as connection:
        return get_extraction_run_in_connection(connection, extraction_run_id)


def list_document_extraction_runs(
    database_url: str,
    document_id: int,
) -> list[ExtractionRunRecord]:
    _require_positive_id(document_id, "document_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_extraction_run_columns()}
                FROM extraction_runs
                WHERE document_id = %s
                ORDER BY created_at DESC, extraction_run_id DESC
                """,
                (document_id,),
            )
            rows = cursor.fetchall()
    return [_row_to_extraction_run_record(dict(row)) for row in rows]


def create_extraction_artifact_in_connection(
    connection: Connection,
    artifact_input: ExtractionArtifactInput,
) -> ExtractionArtifactRecord:
    validate_extraction_artifact_input(artifact_input)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO extraction_artifacts (
                extraction_run_id,
                file_id,
                document_id,
                artifact_type,
                content_text,
                storage_path,
                content_hash,
                size_bytes,
                language,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_select_extraction_artifact_columns()}
            """,
            (
                artifact_input.extraction_run_id,
                artifact_input.file_id,
                artifact_input.document_id,
                artifact_input.artifact_type,
                artifact_input.content_text,
                (
                    artifact_input.storage_path.strip()
                    if artifact_input.storage_path is not None
                    else None
                ),
                (
                    artifact_input.content_hash.strip()
                    if artifact_input.content_hash is not None
                    else None
                ),
                artifact_input.size_bytes,
                artifact_input.language,
                Json(artifact_input.metadata),
            ),
        )
        return _row_to_extraction_artifact_record(dict(cursor.fetchone()))


def create_extraction_artifact(
    database_url: str,
    artifact_input: ExtractionArtifactInput,
) -> ExtractionArtifactRecord:
    with connect(database_url) as connection:
        return create_extraction_artifact_in_connection(connection, artifact_input)


def get_extraction_artifact_in_connection(
    connection: Connection,
    artifact_id: int,
) -> ExtractionArtifactRecord | None:
    _require_positive_id(artifact_id, "artifact_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_select_extraction_artifact_columns()}
            FROM extraction_artifacts
            WHERE artifact_id = %s
            """,
            (artifact_id,),
        )
        row = cursor.fetchone()
    return _row_to_extraction_artifact_record(dict(row)) if row else None


def get_extraction_artifact(
    database_url: str,
    artifact_id: int,
) -> ExtractionArtifactRecord | None:
    with connect(database_url) as connection:
        return get_extraction_artifact_in_connection(connection, artifact_id)


def list_document_extraction_artifacts(
    database_url: str,
    document_id: int,
    *,
    artifact_type: str | None = None,
) -> list[ExtractionArtifactRecord]:
    _require_positive_id(document_id, "document_id")
    where_type = ""
    params: tuple[object, ...] = (document_id,)
    if artifact_type is not None:
        _validate_artifact_type(artifact_type)
        where_type = "AND artifact_type = %s"
        params = (document_id, artifact_type)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_extraction_artifact_columns()}
                FROM extraction_artifacts
                WHERE document_id = %s
                  {where_type}
                ORDER BY created_at DESC, artifact_id DESC
                """,
                params,
            )
            rows = cursor.fetchall()
    return [_row_to_extraction_artifact_record(dict(row)) for row in rows]


def create_document_block_in_connection(
    connection: Connection,
    block_input: DocumentBlockInput,
) -> DocumentBlockRecord:
    validate_document_block_input(block_input)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO document_blocks (
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
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_select_document_block_columns()}
            """,
            (
                block_input.artifact_id,
                block_input.document_id,
                block_input.parent_block_id,
                block_input.block_seq,
                block_input.block_type,
                block_input.content_text,
                block_input.content_markdown,
                list(block_input.heading_path) or None,
                Json(block_input.source_anchor),
                block_input.page_no,
                block_input.slide_no,
                block_input.sheet_name,
                block_input.cell_range,
                block_input.char_start,
                block_input.char_end,
                block_input.token_count,
                Json(block_input.metadata),
            ),
        )
        return _row_to_document_block_record(dict(cursor.fetchone()))


def create_document_block(
    database_url: str,
    block_input: DocumentBlockInput,
) -> DocumentBlockRecord:
    with connect(database_url) as connection:
        return create_document_block_in_connection(connection, block_input)


def list_document_blocks(
    database_url: str,
    document_id: int,
    *,
    artifact_id: int | None = None,
) -> list[DocumentBlockRecord]:
    _require_positive_id(document_id, "document_id")
    _require_positive_id(artifact_id, "artifact_id")
    where_artifact = ""
    params: tuple[object, ...] = (document_id,)
    if artifact_id is not None:
        where_artifact = "AND artifact_id = %s"
        params = (document_id, artifact_id)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_document_block_columns()}
                FROM document_blocks
                WHERE document_id = %s
                  {where_artifact}
                ORDER BY block_seq ASC, block_id ASC
                """,
                params,
            )
            rows = cursor.fetchall()
    return [_row_to_document_block_record(dict(row)) for row in rows]


def create_table_artifact_in_connection(
    connection: Connection,
    table_input: TableArtifactInput,
) -> TableArtifactRecord:
    validate_table_artifact_input(table_input)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO table_artifacts (
                block_id,
                content_markdown,
                content_json,
                storage_path,
                row_count,
                column_count,
                source_anchor,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_select_table_artifact_columns()}
            """,
            (
                table_input.block_id,
                table_input.content_markdown,
                Json(table_input.content_json) if table_input.content_json is not None else None,
                table_input.storage_path.strip() if table_input.storage_path is not None else None,
                table_input.row_count,
                table_input.column_count,
                Json(table_input.source_anchor),
                Json(table_input.metadata),
            ),
        )
        return _row_to_table_artifact_record(dict(cursor.fetchone()))


def create_table_artifact(
    database_url: str,
    table_input: TableArtifactInput,
) -> TableArtifactRecord:
    with connect(database_url) as connection:
        return create_table_artifact_in_connection(connection, table_input)


def create_image_artifact_in_connection(
    connection: Connection,
    image_input: ImageArtifactInput,
) -> ImageArtifactRecord:
    validate_image_artifact_input(image_input)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO image_artifacts (
                block_id,
                storage_path,
                mime_type,
                width_px,
                height_px,
                ocr_text,
                caption_text,
                surrounding_text,
                source_anchor,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_select_image_artifact_columns()}
            """,
            (
                image_input.block_id,
                image_input.storage_path.strip(),
                image_input.mime_type,
                image_input.width_px,
                image_input.height_px,
                image_input.ocr_text,
                image_input.caption_text,
                image_input.surrounding_text,
                Json(image_input.source_anchor),
                Json(image_input.metadata),
            ),
        )
        return _row_to_image_artifact_record(dict(cursor.fetchone()))


def create_image_artifact(
    database_url: str,
    image_input: ImageArtifactInput,
) -> ImageArtifactRecord:
    with connect(database_url) as connection:
        return create_image_artifact_in_connection(connection, image_input)


def create_extraction_quality_snapshot_in_connection(
    connection: Connection,
    snapshot_input: ExtractionQualitySnapshotInput,
) -> ExtractionQualitySnapshotRecord:
    validate_extraction_quality_snapshot_input(snapshot_input)
    created_by = _validate_optional_nonblank(snapshot_input.created_by, "created_by")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO extraction_quality_snapshots (
                document_id,
                file_id,
                artifact_id,
                extraction_run_id,
                artifact_type,
                extraction_profile_name,
                extractor_name,
                extractor_version,
                status,
                content_length,
                content_lines,
                block_count,
                source_anchor_count,
                source_anchor_coverage_percent,
                issue_count,
                warning_count,
                failed_count,
                block_summary,
                quality_payload,
                created_by,
                created_by_user_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING {_select_extraction_quality_snapshot_columns()}
            """,
            (
                snapshot_input.document_id,
                snapshot_input.file_id,
                snapshot_input.artifact_id,
                snapshot_input.extraction_run_id,
                snapshot_input.artifact_type.strip(),
                (
                    snapshot_input.extraction_profile_name.strip()
                    if snapshot_input.extraction_profile_name is not None
                    else None
                ),
                (
                    snapshot_input.extractor_name.strip()
                    if snapshot_input.extractor_name is not None
                    else None
                ),
                (
                    snapshot_input.extractor_version.strip()
                    if snapshot_input.extractor_version is not None
                    else None
                ),
                snapshot_input.status,
                snapshot_input.content_length,
                snapshot_input.content_lines,
                snapshot_input.block_count,
                snapshot_input.source_anchor_count,
                snapshot_input.source_anchor_coverage_percent,
                snapshot_input.issue_count,
                snapshot_input.warning_count,
                snapshot_input.failed_count,
                Json(snapshot_input.block_summary),
                Json(snapshot_input.quality_payload),
                created_by,
                snapshot_input.created_by_user_id,
            ),
        )
        return _row_to_extraction_quality_snapshot_record(dict(cursor.fetchone()))


def create_extraction_quality_snapshot(
    database_url: str,
    snapshot_input: ExtractionQualitySnapshotInput,
) -> ExtractionQualitySnapshotRecord:
    with connect(database_url) as connection:
        return create_extraction_quality_snapshot_in_connection(connection, snapshot_input)


def list_extraction_quality_snapshots(
    database_url: str,
    document_id: int,
    *,
    artifact_id: int | None = None,
    limit: int = 20,
) -> list[ExtractionQualitySnapshotRecord]:
    _require_positive_id(document_id, "document_id")
    _require_positive_id(artifact_id, "artifact_id")
    if limit <= 0:
        raise InvalidIngestionArtifactError("limit must be greater than 0")
    if limit > 100:
        raise InvalidIngestionArtifactError("limit must be less than or equal to 100")

    where_artifact = ""
    params: tuple[object, ...] = (document_id, limit)
    if artifact_id is not None:
        where_artifact = "AND artifact_id = %s"
        params = (document_id, artifact_id, limit)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_extraction_quality_snapshot_columns()}
                FROM extraction_quality_snapshots
                WHERE document_id = %s
                  {where_artifact}
                ORDER BY created_at DESC, snapshot_id DESC
                LIMIT %s
                """,
                params,
            )
            rows = cursor.fetchall()
    return [_row_to_extraction_quality_snapshot_record(dict(row)) for row in rows]


def get_extraction_quality_snapshot_summary(
    database_url: str,
    document_id: int,
    *,
    artifact_id: int | None = None,
) -> ExtractionQualitySnapshotSummary:
    _require_positive_id(document_id, "document_id")
    _require_positive_id(artifact_id, "artifact_id")
    where_artifact = ""
    params: tuple[object, ...] = (document_id,)
    if artifact_id is not None:
        where_artifact = "AND artifact_id = %s"
        params = (document_id, artifact_id)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    count(*) AS snapshot_count,
                    count(*) FILTER (WHERE status = 'passed') AS passed_count,
                    count(*) FILTER (WHERE status = 'warning') AS warning_count,
                    count(*) FILTER (WHERE status = 'failed') AS failed_count
                FROM extraction_quality_snapshots
                WHERE document_id = %s
                  {where_artifact}
                """,
                params,
            )
            summary_row = dict(cursor.fetchone())
    latest_snapshots = list_extraction_quality_snapshots(
        database_url,
        document_id,
        artifact_id=artifact_id,
        limit=1,
    )
    return ExtractionQualitySnapshotSummary(
        document_id=document_id,
        artifact_id=artifact_id,
        snapshot_count=int(summary_row["snapshot_count"]),
        passed_count=int(summary_row["passed_count"]),
        warning_count=int(summary_row["warning_count"]),
        failed_count=int(summary_row["failed_count"]),
        latest_snapshot=latest_snapshots[0] if latest_snapshots else None,
    )
