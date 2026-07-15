"""Runtime contract shared by local and future remote extraction providers."""

from dataclasses import dataclass, field
from typing import Any

from app.core.ingestion_artifacts import (
    DOCUMENT_BLOCK_TYPES,
    EXTRACTION_ARTIFACT_TYPES,
    EXTRACTION_RUN_STATUSES,
    InvalidIngestionArtifactError,
)

TERMINAL_EXTRACTION_STATUSES = {"succeeded", "failed", "skipped"}


@dataclass(frozen=True)
class ExtractionRuntimeRequest:
    file_id: int
    storage_path: str
    extraction_profile_name: str
    document_id: int | None = None
    mime_type: str | None = None
    detected_file_type: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None


@dataclass(frozen=True)
class ExtractionRuntimeArtifact:
    artifact_type: str
    content_text: str | None = None
    storage_path: str | None = None
    content_hash: str | None = None
    size_bytes: int | None = None
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionRuntimeBlock:
    block_seq: int
    block_type: str
    parent_block_seq: int | None = None
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
class ExtractionRuntimeResult:
    status: str
    artifacts: tuple[ExtractionRuntimeArtifact, ...] = ()
    blocks: tuple[ExtractionRuntimeBlock, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    elapsed_ms: int | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidIngestionArtifactError(f"{field_name} must be greater than 0")


def _require_non_blank(value: str | None, field_name: str) -> str:
    stripped = (value or "").strip()
    if not stripped:
        raise InvalidIngestionArtifactError(f"{field_name} is required")
    return stripped


def _validate_non_negative(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise InvalidIngestionArtifactError(f"{field_name} must be greater than or equal to 0")


def _validate_positive(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidIngestionArtifactError(f"{field_name} must be greater than 0")


def _validate_char_range(start: int | None, end: int | None) -> None:
    _validate_non_negative(start, "char_start")
    _validate_non_negative(end, "char_end")
    if start is not None and end is not None and end < start:
        raise InvalidIngestionArtifactError("char_end must be greater than or equal to char_start")


def validate_runtime_request(request: ExtractionRuntimeRequest) -> None:
    _require_positive_id(request.file_id, "file_id")
    _require_positive_id(request.document_id, "document_id")
    _require_non_blank(request.storage_path, "storage_path")
    _require_non_blank(request.extraction_profile_name, "extraction_profile_name")
    if request.detected_file_type is not None:
        _require_non_blank(request.detected_file_type, "detected_file_type")
    if request.mime_type is not None:
        _require_non_blank(request.mime_type, "mime_type")
    if request.trace_id is not None:
        _require_non_blank(request.trace_id, "trace_id")


def validate_runtime_artifact(artifact: ExtractionRuntimeArtifact) -> None:
    if artifact.artifact_type not in EXTRACTION_ARTIFACT_TYPES:
        raise InvalidIngestionArtifactError(f"Unsupported artifact_type: {artifact.artifact_type}")
    if not artifact.content_text and not artifact.storage_path:
        raise InvalidIngestionArtifactError("content_text or storage_path is required")
    if artifact.storage_path is not None:
        _require_non_blank(artifact.storage_path, "storage_path")
    if artifact.content_hash is not None:
        _require_non_blank(artifact.content_hash, "content_hash")
    _validate_non_negative(artifact.size_bytes, "size_bytes")


def validate_runtime_block(block: ExtractionRuntimeBlock) -> None:
    _validate_non_negative(block.block_seq, "block_seq")
    _validate_non_negative(block.parent_block_seq, "parent_block_seq")
    if block.parent_block_seq == block.block_seq:
        raise InvalidIngestionArtifactError("parent_block_seq must not equal block_seq")
    if block.block_type not in DOCUMENT_BLOCK_TYPES:
        raise InvalidIngestionArtifactError(f"Unsupported block_type: {block.block_type}")
    _validate_positive(block.page_no, "page_no")
    _validate_positive(block.slide_no, "slide_no")
    _validate_non_negative(block.token_count, "token_count")
    _validate_char_range(block.char_start, block.char_end)


def validate_runtime_result(result: ExtractionRuntimeResult) -> None:
    if result.status not in EXTRACTION_RUN_STATUSES:
        raise InvalidIngestionArtifactError(
            f"Unsupported extraction result status: {result.status}"
        )
    if result.status not in TERMINAL_EXTRACTION_STATUSES:
        raise InvalidIngestionArtifactError("runtime result status must be terminal")
    _validate_non_negative(result.elapsed_ms, "elapsed_ms")
    for artifact in result.artifacts:
        validate_runtime_artifact(artifact)
    for block in result.blocks:
        validate_runtime_block(block)
    if result.status == "succeeded" and not result.artifacts:
        raise InvalidIngestionArtifactError("succeeded extraction result requires artifacts")
    if result.status == "failed" and not result.errors:
        raise InvalidIngestionArtifactError("failed extraction result requires errors")
