"""Local extraction runtime registry for local document sources."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from psycopg import Connection

from app.core.chunking import count_chunk_tokens
from app.core.database import connect
from app.core.document_parsers import (
    PARSER_NAME_MARKDOWN,
    PARSER_VERSION_MARKDOWN,
    MarkdownParser,
    ParsedBlock,
)
from app.core.extraction_runtime import (
    ExtractionRuntimeArtifact,
    ExtractionRuntimeBlock,
    ExtractionRuntimeRequest,
    ExtractionRuntimeResult,
    validate_runtime_request,
    validate_runtime_result,
)
from app.core.ingestion_artifacts import (
    DocumentBlockInput,
    DocumentBlockRecord,
    ExtractionArtifactInput,
    ExtractionArtifactRecord,
    ExtractionRunInput,
    ExtractionRunRecord,
    InvalidIngestionArtifactError,
    create_document_block_in_connection,
    create_extraction_artifact_in_connection,
    create_extraction_run_in_connection,
    get_extraction_profile_in_connection,
)

LOCAL_MARKDOWN_PROFILE_NAME = "local_markdown_default"
LOCAL_PLAIN_TEXT_PROFILE_NAME = "local_plain_text_default"
LOCAL_PLAIN_TEXT_EXTRACTOR_NAME = "local_plain_text"
LOCAL_PLAIN_TEXT_EXTRACTOR_VERSION = "0.1.0"

ERROR_CODE_LOCAL_SOURCE_NOT_FOUND = "LOCAL_SOURCE_NOT_FOUND"
ERROR_CODE_LOCAL_SOURCE_EMPTY = "LOCAL_SOURCE_EMPTY"
ERROR_CODE_LOCAL_UNSUPPORTED_PROFILE = "LOCAL_UNSUPPORTED_PROFILE"
ERROR_CODE_LOCAL_UNSUPPORTED_FILE_TYPE = "LOCAL_UNSUPPORTED_FILE_TYPE"
ERROR_CODE_LOCAL_TEXT_DECODE_FAILED = "LOCAL_TEXT_DECODE_FAILED"


LocalExtractionCallable = Callable[
    [ExtractionRuntimeRequest, Path, float],
    ExtractionRuntimeResult,
]


@dataclass(frozen=True)
class LocalExtractionHandler:
    profile_name: str
    parser_name: str
    parser_version: str
    extractor_name: str
    extractor_version: str
    supported_file_types: tuple[str, ...]
    extract: LocalExtractionCallable

    def supports_file_type(self, file_type: str | None) -> bool:
        normalized = normalize_file_type(file_type)
        return bool(normalized) and normalized in self.supported_file_types


@dataclass(frozen=True)
class PersistedExtractionRuntimeResult:
    run: ExtractionRunRecord
    artifacts: tuple[ExtractionArtifactRecord, ...] = ()
    blocks: tuple[DocumentBlockRecord, ...] = ()


def run_local_extraction(request: ExtractionRuntimeRequest) -> ExtractionRuntimeResult:
    """Run the registered local extraction profile and return a terminal runtime result."""

    validate_runtime_request(request)
    started = perf_counter()
    source_path = Path(request.storage_path)
    profile_name = request.extraction_profile_name.strip()
    file_type = normalize_file_type(request.detected_file_type) or normalize_file_type(
        source_path.suffix
    )

    if not source_path.is_file():
        return _failed_result(
            started,
            ERROR_CODE_LOCAL_SOURCE_NOT_FOUND,
            f"Source file was not found: {source_path}",
        )

    handler = get_local_extraction_handler(profile_name)
    if handler is None:
        return _failed_result(
            started,
            ERROR_CODE_LOCAL_UNSUPPORTED_PROFILE,
            f"Unsupported local extraction profile: {profile_name}",
        )

    if not handler.supports_file_type(file_type):
        return _failed_result(
            started,
            ERROR_CODE_LOCAL_UNSUPPORTED_FILE_TYPE,
            f"{profile_name} cannot process {source_path.suffix or '(none)'} files",
        )

    result = handler.extract(request, source_path, started)
    validate_runtime_result(result)
    return result


def normalize_file_type(file_type: str | None) -> str:
    return (file_type or "").strip().lower().removeprefix(".")


def list_local_extraction_handlers() -> tuple[LocalExtractionHandler, ...]:
    return LOCAL_EXTRACTION_HANDLERS


def get_local_extraction_handler(profile_name: str | None) -> LocalExtractionHandler | None:
    normalized = (profile_name or "").strip()
    return LOCAL_EXTRACTION_HANDLER_BY_PROFILE.get(normalized)


def select_local_extraction_handler(
    file_type: str | None,
) -> LocalExtractionHandler | None:
    normalized = normalize_file_type(file_type)
    for handler in LOCAL_EXTRACTION_HANDLERS:
        if handler.supports_file_type(normalized):
            return handler
    return None


def select_local_extraction_profile_name(file_type: str | None) -> str | None:
    handler = select_local_extraction_handler(file_type)
    return handler.profile_name if handler else None


def _extract_markdown_source(
    request: ExtractionRuntimeRequest,
    source_path: Path,
    started: float,
) -> ExtractionRuntimeResult:
    source_text, failed_result = _read_utf8_text_source(source_path, started)
    if failed_result is not None:
        return failed_result
    return _extract_markdown(request, source_text, source_path, started)


def _extract_plain_text_source(
    request: ExtractionRuntimeRequest,
    source_path: Path,
    started: float,
) -> ExtractionRuntimeResult:
    source_text, failed_result = _read_utf8_text_source(source_path, started)
    if failed_result is not None:
        return failed_result
    return _extract_plain_text(request, source_text, source_path, started)


def _read_utf8_text_source(
    source_path: Path,
    started: float,
) -> tuple[str, ExtractionRuntimeResult | None]:
    try:
        source_text = _normalize_line_endings(source_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        return "", _failed_result(
            started,
            ERROR_CODE_LOCAL_TEXT_DECODE_FAILED,
            f"Source file could not be decoded as UTF-8: {exc}",
        )

    if not source_text.strip():
        return "", _failed_result(
            started,
            ERROR_CODE_LOCAL_SOURCE_EMPTY,
            "Source file does not contain extractable text",
        )

    return source_text, None


LOCAL_EXTRACTION_HANDLERS = (
    LocalExtractionHandler(
        profile_name=LOCAL_MARKDOWN_PROFILE_NAME,
        parser_name=PARSER_NAME_MARKDOWN,
        parser_version=PARSER_VERSION_MARKDOWN,
        extractor_name="local_markdown",
        extractor_version=PARSER_VERSION_MARKDOWN,
        supported_file_types=("md",),
        extract=_extract_markdown_source,
    ),
    LocalExtractionHandler(
        profile_name=LOCAL_PLAIN_TEXT_PROFILE_NAME,
        parser_name=LOCAL_PLAIN_TEXT_EXTRACTOR_NAME,
        parser_version=LOCAL_PLAIN_TEXT_EXTRACTOR_VERSION,
        extractor_name=LOCAL_PLAIN_TEXT_EXTRACTOR_NAME,
        extractor_version=LOCAL_PLAIN_TEXT_EXTRACTOR_VERSION,
        supported_file_types=("txt", "text"),
        extract=_extract_plain_text_source,
    ),
)

LOCAL_EXTRACTION_HANDLER_BY_PROFILE = {
    handler.profile_name: handler for handler in LOCAL_EXTRACTION_HANDLERS
}

SUPPORTED_LOCAL_PROFILE_SUFFIXES = {
    handler.profile_name: {f".{file_type}" for file_type in handler.supported_file_types}
    for handler in LOCAL_EXTRACTION_HANDLERS
}


def persist_extraction_runtime_result(
    database_url: str,
    request: ExtractionRuntimeRequest,
    result: ExtractionRuntimeResult,
) -> PersistedExtractionRuntimeResult:
    validate_runtime_request(request)
    validate_runtime_result(result)
    with connect(database_url) as connection:
        return persist_extraction_runtime_result_in_connection(connection, request, result)


def persist_extraction_runtime_result_in_connection(
    connection: Connection,
    request: ExtractionRuntimeRequest,
    result: ExtractionRuntimeResult,
) -> PersistedExtractionRuntimeResult:
    validate_runtime_request(request)
    validate_runtime_result(result)
    profile = get_extraction_profile_in_connection(
        connection,
        request.extraction_profile_name,
    )
    if profile is None:
        raise InvalidIngestionArtifactError(
            f"Extraction profile was not found: {request.extraction_profile_name}"
        )
    if result.blocks and request.document_id is None:
        raise InvalidIngestionArtifactError("document_id is required when persisting blocks")

    run = create_extraction_run_in_connection(
        connection,
        ExtractionRunInput(
            file_id=request.file_id,
            document_id=request.document_id,
            extraction_profile_name=profile.extraction_profile_name,
            status=result.status,
            provider_mode=profile.provider_mode,
            extractor_name=profile.extractor_name,
            extractor_version=profile.extractor_version,
            elapsed_ms=result.elapsed_ms,
            warning_count=len(result.warnings),
            error_count=len(result.errors),
            error_code=(
                str(result.runtime_metadata.get("error_code"))
                if result.runtime_metadata.get("error_code")
                else None
            ),
            error_message=result.errors[0] if result.errors else None,
            runtime_metadata=result.runtime_metadata,
        ),
    )

    artifacts = tuple(
        create_extraction_artifact_in_connection(
            connection,
            ExtractionArtifactInput(
                extraction_run_id=run.extraction_run_id,
                file_id=request.file_id,
                document_id=request.document_id,
                artifact_type=artifact.artifact_type,
                content_text=artifact.content_text,
                storage_path=artifact.storage_path,
                content_hash=artifact.content_hash,
                size_bytes=artifact.size_bytes,
                language=artifact.language,
                metadata=artifact.metadata,
            ),
        )
        for artifact in result.artifacts
    )
    if result.blocks and not artifacts:
        raise InvalidIngestionArtifactError("runtime blocks require a persisted artifact")

    block_records = _persist_blocks(
        connection,
        request.document_id,
        artifacts[0],
        result.blocks,
    )
    return PersistedExtractionRuntimeResult(
        run=run,
        artifacts=artifacts,
        blocks=block_records,
    )


def _extract_markdown(
    request: ExtractionRuntimeRequest,
    source_text: str,
    source_path: Path,
    started: float,
) -> ExtractionRuntimeResult:
    parser = MarkdownParser()
    parsed_document = parser.parse_text(source_text, source_path=str(source_path))
    line_offsets = _line_offsets(source_text)
    blocks = _markdown_blocks_to_runtime_blocks(parsed_document.blocks, line_offsets)
    artifact = ExtractionRuntimeArtifact(
        artifact_type="normalized_markdown",
        content_text=source_text,
        content_hash=_hash_text(source_text),
        size_bytes=len(source_text.encode("utf-8")),
        metadata={
            "source_path": str(source_path),
            "source_file_name": source_path.name,
            "parser_name": parsed_document.parser_name,
            "parser_version": parsed_document.parser_version,
            "line_count": parsed_document.line_count,
            "block_count": len(blocks),
            "options": request.options,
        },
    )
    return ExtractionRuntimeResult(
        status="succeeded",
        artifacts=(artifact,),
        blocks=blocks,
        elapsed_ms=_elapsed_ms(started),
        runtime_metadata={
            "profile_name": LOCAL_MARKDOWN_PROFILE_NAME,
            "extractor_name": PARSER_NAME_MARKDOWN,
            "extractor_version": PARSER_VERSION_MARKDOWN,
            "source_path": str(source_path),
            "line_count": parsed_document.line_count,
            "block_count": len(blocks),
        },
    )


def _extract_plain_text(
    request: ExtractionRuntimeRequest,
    source_text: str,
    source_path: Path,
    started: float,
) -> ExtractionRuntimeResult:
    blocks = tuple(_plain_text_blocks(source_text))
    artifact = ExtractionRuntimeArtifact(
        artifact_type="plain_text",
        content_text=source_text,
        content_hash=_hash_text(source_text),
        size_bytes=len(source_text.encode("utf-8")),
        metadata={
            "source_path": str(source_path),
            "source_file_name": source_path.name,
            "line_count": len(source_text.splitlines()),
            "block_count": len(blocks),
            "options": request.options,
        },
    )
    return ExtractionRuntimeResult(
        status="succeeded",
        artifacts=(artifact,),
        blocks=blocks,
        elapsed_ms=_elapsed_ms(started),
        runtime_metadata={
            "profile_name": LOCAL_PLAIN_TEXT_PROFILE_NAME,
            "extractor_name": LOCAL_PLAIN_TEXT_EXTRACTOR_NAME,
            "extractor_version": LOCAL_PLAIN_TEXT_EXTRACTOR_VERSION,
            "source_path": str(source_path),
            "line_count": len(source_text.splitlines()),
            "block_count": len(blocks),
        },
    )


def _markdown_blocks_to_runtime_blocks(
    parsed_blocks: tuple[ParsedBlock, ...],
    line_offsets: tuple[int, ...],
) -> tuple[ExtractionRuntimeBlock, ...]:
    runtime_blocks: list[ExtractionRuntimeBlock] = []
    latest_heading_seq_by_level: dict[int, int] = {}

    for block_seq, block in enumerate(parsed_blocks):
        heading_level = int(block.metadata.get("level", len(block.heading_path) or 1))
        parent_block_seq: int | None = None
        if block.block_type == "heading":
            parent_block_seq = latest_heading_seq_by_level.get(heading_level - 1)
            latest_heading_seq_by_level = {
                level: seq
                for level, seq in latest_heading_seq_by_level.items()
                if level < heading_level
            }
            latest_heading_seq_by_level[heading_level] = block_seq
        elif latest_heading_seq_by_level:
            parent_block_seq = latest_heading_seq_by_level[max(latest_heading_seq_by_level)]

        runtime_blocks.append(
            _markdown_block_to_runtime_block(
                block,
                block_seq,
                parent_block_seq,
                line_offsets,
            )
        )
    return tuple(runtime_blocks)


def _markdown_block_to_runtime_block(
    block: ParsedBlock,
    block_seq: int,
    parent_block_seq: int | None,
    line_offsets: tuple[int, ...],
) -> ExtractionRuntimeBlock:
    block_type = "code" if block.block_type == "code_block" else block.block_type
    content_markdown = _format_markdown_block(block)
    char_start, char_end = _char_range_for_lines(line_offsets, block.start_line, block.end_line)
    source_anchor: dict[str, Any] = {
        "start_line": block.start_line,
        "end_line": block.end_line,
    }
    if block.block_type == "table":
        source_anchor["table_index"] = block_seq
    return ExtractionRuntimeBlock(
        block_seq=block_seq,
        block_type=block_type,
        parent_block_seq=parent_block_seq,
        content_text=block.text,
        content_markdown=content_markdown,
        heading_path=block.heading_path,
        source_anchor=source_anchor,
        char_start=char_start,
        char_end=char_end,
        token_count=count_chunk_tokens(content_markdown),
        metadata=block.metadata,
    )


def _plain_text_blocks(source_text: str) -> list[ExtractionRuntimeBlock]:
    lines = source_text.splitlines(keepends=True)
    blocks: list[ExtractionRuntimeBlock] = []
    paragraph_lines: list[str] = []
    paragraph_start_line: int | None = None
    paragraph_start_char: int | None = None
    char_offset = 0

    for index, raw_line in enumerate(lines, start=1):
        line_text = raw_line.rstrip("\n")
        if line_text.strip():
            if paragraph_start_line is None:
                paragraph_start_line = index
                paragraph_start_char = char_offset
            paragraph_lines.append(line_text.strip())
        elif paragraph_lines:
            blocks.append(
                _build_plain_text_block(
                    blocks,
                    paragraph_lines,
                    paragraph_start_line,
                    index - 1,
                    paragraph_start_char,
                    char_offset,
                )
            )
            paragraph_lines = []
            paragraph_start_line = None
            paragraph_start_char = None
        char_offset += len(raw_line)

    if paragraph_lines:
        blocks.append(
            _build_plain_text_block(
                blocks,
                paragraph_lines,
                paragraph_start_line,
                len(lines),
                paragraph_start_char,
                len(source_text),
            )
        )
    return blocks


def _build_plain_text_block(
    blocks: list[ExtractionRuntimeBlock],
    paragraph_lines: list[str],
    start_line: int | None,
    end_line: int,
    start_char: int | None,
    end_char: int,
) -> ExtractionRuntimeBlock:
    content_text = " ".join(paragraph_lines)
    return ExtractionRuntimeBlock(
        block_seq=len(blocks),
        block_type="paragraph",
        content_text=content_text,
        content_markdown=content_text,
        source_anchor={
            "start_line": start_line or 1,
            "end_line": end_line,
        },
        char_start=start_char if start_char is not None else 0,
        char_end=end_char,
        token_count=count_chunk_tokens(content_text),
    )


def _persist_blocks(
    connection: Connection,
    document_id: int | None,
    artifact: ExtractionArtifactRecord,
    runtime_blocks: tuple[ExtractionRuntimeBlock, ...],
) -> tuple[DocumentBlockRecord, ...]:
    if document_id is None:
        return ()
    block_id_by_seq: dict[int, int] = {}
    records: list[DocumentBlockRecord] = []
    for block in runtime_blocks:
        parent_block_id = (
            block_id_by_seq.get(block.parent_block_seq)
            if block.parent_block_seq is not None
            else None
        )
        record = create_document_block_in_connection(
            connection,
            DocumentBlockInput(
                artifact_id=artifact.artifact_id,
                document_id=document_id,
                parent_block_id=parent_block_id,
                block_seq=block.block_seq,
                block_type=block.block_type,
                content_text=block.content_text,
                content_markdown=block.content_markdown,
                heading_path=block.heading_path,
                source_anchor=block.source_anchor,
                page_no=block.page_no,
                slide_no=block.slide_no,
                sheet_name=block.sheet_name,
                cell_range=block.cell_range,
                char_start=block.char_start,
                char_end=block.char_end,
                token_count=block.token_count,
                metadata=block.metadata,
            ),
        )
        block_id_by_seq[block.block_seq] = record.block_id
        records.append(record)
    return tuple(records)


def _format_markdown_block(block: ParsedBlock) -> str:
    if block.block_type == "heading":
        level = int(block.metadata.get("level", len(block.heading_path) or 1))
        return f"{'#' * level} {block.text}"
    if block.block_type == "code_block":
        language = block.metadata.get("language")
        fence = f"```{language}" if language else "```"
        return f"{fence}\n{block.text}\n```"
    return block.text


def _line_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    position = 0
    for line in text.splitlines(keepends=True):
        position += len(line)
        offsets.append(position)
    if not text.endswith("\n"):
        offsets[-1] = len(text)
    return tuple(offsets)


def _char_range_for_lines(
    line_offsets: tuple[int, ...],
    start_line: int,
    end_line: int,
) -> tuple[int, int]:
    start_index = max(start_line - 1, 0)
    end_index = min(end_line, len(line_offsets) - 1)
    return line_offsets[start_index], line_offsets[end_index]


def _failed_result(
    started: float,
    error_code: str,
    error_message: str,
) -> ExtractionRuntimeResult:
    result = ExtractionRuntimeResult(
        status="failed",
        errors=(error_message,),
        elapsed_ms=_elapsed_ms(started),
        runtime_metadata={"error_code": error_code},
    )
    validate_runtime_result(result)
    return result


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _elapsed_ms(started: float) -> int:
    return max(int((perf_counter() - started) * 1000), 0)
