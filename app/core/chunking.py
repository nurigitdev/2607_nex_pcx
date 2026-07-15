"""Heading-aware chunking for parsed documents."""

import re
from dataclasses import dataclass

from app.core.chunks import (
    DEFAULT_CHUNK_POLICY_NAME,
    ChunkInput,
    calculate_chunk_content_hash,
)
from app.core.document_parsers import ParsedBlock, ParsedDocument
from app.core.ingestion_artifacts import DocumentBlockRecord

TOKEN_RE = re.compile(r"\S+")
SPLIT_STRATEGY_HEADING_AWARE = "heading-aware"


@dataclass(frozen=True)
class ChunkPolicy:
    chunk_policy_name: str
    target_token_size: int
    overlap_token_size: int
    split_strategy: str = SPLIT_STRATEGY_HEADING_AWARE
    preserve_table: bool = True
    preserve_code_block: bool = True


@dataclass(frozen=True)
class _ChunkUnit:
    block: ParsedBlock
    text: str
    token_count: int


@dataclass(frozen=True)
class _DocumentBlockChunkUnit:
    block: DocumentBlockRecord
    text: str
    token_count: int


class InvalidChunkPolicyError(ValueError):
    """Raised when chunk policy settings are invalid."""


DEFAULT_CHUNK_POLICIES: dict[str, ChunkPolicy] = {
    "heading_512_64": ChunkPolicy(
        chunk_policy_name="heading_512_64",
        target_token_size=512,
        overlap_token_size=64,
    ),
    "heading_1000_200": ChunkPolicy(
        chunk_policy_name="heading_1000_200",
        target_token_size=1000,
        overlap_token_size=200,
    ),
    "heading_1500_200": ChunkPolicy(
        chunk_policy_name="heading_1500_200",
        target_token_size=1500,
        overlap_token_size=200,
    ),
}


def get_chunk_policy(chunk_policy_name: str = DEFAULT_CHUNK_POLICY_NAME) -> ChunkPolicy:
    try:
        return DEFAULT_CHUNK_POLICIES[chunk_policy_name]
    except KeyError as exc:
        raise InvalidChunkPolicyError(f"Unsupported chunk policy: {chunk_policy_name}") from exc


def validate_chunk_policy(policy: ChunkPolicy) -> None:
    if not policy.chunk_policy_name.strip():
        raise InvalidChunkPolicyError("chunk_policy_name is required")
    if policy.target_token_size <= 0:
        raise InvalidChunkPolicyError("target_token_size must be greater than 0")
    if policy.overlap_token_size < 0:
        raise InvalidChunkPolicyError("overlap_token_size must be greater than or equal to 0")
    if policy.overlap_token_size >= policy.target_token_size:
        raise InvalidChunkPolicyError("overlap_token_size must be smaller than target_token_size")
    if policy.split_strategy != SPLIT_STRATEGY_HEADING_AWARE:
        raise InvalidChunkPolicyError(f"Unsupported split strategy: {policy.split_strategy}")


def count_chunk_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def chunk_parsed_document(
    document: ParsedDocument,
    *,
    document_id: int,
    policy: ChunkPolicy | None = None,
) -> list[ChunkInput]:
    if document_id <= 0:
        raise ValueError("document_id must be greater than 0")

    effective_policy = policy or get_chunk_policy()
    validate_chunk_policy(effective_policy)
    units = [
        _block_to_unit(block) for block in document.blocks if _format_block_text(block).strip()
    ]
    chunks: list[list[_ChunkUnit]] = []
    current_units: list[_ChunkUnit] = []

    for unit in units:
        if unit.block.block_type == "heading" and current_units:
            chunks.append(current_units)
            current_units = []

        would_exceed_target = (
            current_units
            and _unit_token_count(current_units) + unit.token_count
            > effective_policy.target_token_size
        )
        if would_exceed_target:
            chunks.append(current_units)
            current_units = _overlap_tail(current_units, effective_policy.overlap_token_size)
            if (
                _unit_token_count(current_units) + unit.token_count
                > effective_policy.target_token_size
            ):
                current_units = []

        current_units.append(unit)

    if current_units:
        chunks.append(current_units)

    return [
        _units_to_chunk_input(
            document,
            document_id=document_id,
            chunk_seq=chunk_seq,
            units=chunk_units,
            policy=effective_policy,
        )
        for chunk_seq, chunk_units in enumerate(chunks)
    ]


def chunk_document_blocks(
    blocks: list[DocumentBlockRecord],
    *,
    document_id: int,
    policy: ChunkPolicy | None = None,
    parser_name: str | None = None,
    parser_version: str | None = None,
) -> list[ChunkInput]:
    if document_id <= 0:
        raise ValueError("document_id must be greater than 0")

    effective_policy = policy or get_chunk_policy()
    validate_chunk_policy(effective_policy)
    ordered_blocks = sorted(blocks, key=lambda block: (block.block_seq, block.block_id))
    for block in ordered_blocks:
        if block.document_id != document_id:
            raise ValueError("all document blocks must belong to the target document_id")

    units = [
        _document_block_to_unit(block)
        for block in ordered_blocks
        if _format_document_block_text(block).strip()
    ]
    chunks: list[list[_DocumentBlockChunkUnit]] = []
    current_units: list[_DocumentBlockChunkUnit] = []

    for unit in units:
        if unit.block.block_type == "heading" and current_units:
            chunks.append(current_units)
            current_units = []

        would_exceed_target = (
            current_units
            and _document_unit_token_count(current_units) + unit.token_count
            > effective_policy.target_token_size
        )
        if would_exceed_target:
            chunks.append(current_units)
            current_units = _document_overlap_tail(
                current_units,
                effective_policy.overlap_token_size,
            )
            if (
                _document_unit_token_count(current_units) + unit.token_count
                > effective_policy.target_token_size
            ):
                current_units = []

        current_units.append(unit)

    if current_units:
        chunks.append(current_units)

    return [
        _document_units_to_chunk_input(
            document_id=document_id,
            chunk_seq=chunk_seq,
            units=chunk_units,
            policy=effective_policy,
            parser_name=parser_name,
            parser_version=parser_version,
        )
        for chunk_seq, chunk_units in enumerate(chunks)
    ]


def _block_to_unit(block: ParsedBlock) -> _ChunkUnit:
    text = _format_block_text(block)
    return _ChunkUnit(block=block, text=text, token_count=count_chunk_tokens(text))


def _format_block_text(block: ParsedBlock) -> str:
    if block.block_type == "heading":
        level = int(block.metadata.get("level", len(block.heading_path) or 1))
        return f"{'#' * level} {block.text}"
    if block.block_type == "code_block":
        language = block.metadata.get("language")
        fence = f"```{language}" if language else "```"
        return f"{fence}\n{block.text}\n```"
    return block.text


def _unit_token_count(units: list[_ChunkUnit]) -> int:
    return sum(unit.token_count for unit in units)


def _document_block_to_unit(block: DocumentBlockRecord) -> _DocumentBlockChunkUnit:
    text = _format_document_block_text(block)
    return _DocumentBlockChunkUnit(
        block=block,
        text=text,
        token_count=count_chunk_tokens(text),
    )


def _format_document_block_text(block: DocumentBlockRecord) -> str:
    if block.content_markdown:
        return block.content_markdown
    return block.content_text or ""


def _document_unit_token_count(units: list[_DocumentBlockChunkUnit]) -> int:
    return sum(unit.token_count for unit in units)


def _overlap_tail(units: list[_ChunkUnit], overlap_token_size: int) -> list[_ChunkUnit]:
    if overlap_token_size == 0:
        return []

    selected: list[_ChunkUnit] = []
    selected_tokens = 0
    for unit in reversed(units):
        if selected and selected_tokens + unit.token_count > overlap_token_size:
            break
        selected.append(unit)
        selected_tokens += unit.token_count
        if selected_tokens >= overlap_token_size:
            break
    return list(reversed(selected))


def _document_overlap_tail(
    units: list[_DocumentBlockChunkUnit],
    overlap_token_size: int,
) -> list[_DocumentBlockChunkUnit]:
    if overlap_token_size == 0:
        return []

    selected: list[_DocumentBlockChunkUnit] = []
    selected_tokens = 0
    for unit in reversed(units):
        if selected and selected_tokens + unit.token_count > overlap_token_size:
            break
        selected.append(unit)
        selected_tokens += unit.token_count
        if selected_tokens >= overlap_token_size:
            break
    return list(reversed(selected))


def _units_to_chunk_input(
    document: ParsedDocument,
    *,
    document_id: int,
    chunk_seq: int,
    units: list[_ChunkUnit],
    policy: ChunkPolicy,
) -> ChunkInput:
    chunk_text = "\n\n".join(unit.text for unit in units)
    token_count = count_chunk_tokens(chunk_text)
    metadata = {
        "block_types": [unit.block.block_type for unit in units],
        "source_blocks": [
            {
                "block_type": unit.block.block_type,
                "start_line": unit.block.start_line,
                "end_line": unit.block.end_line,
            }
            for unit in units
        ],
        "start_line": min(unit.block.start_line for unit in units),
        "end_line": max(unit.block.end_line for unit in units),
        "source_path": document.source_path,
        "split_strategy": policy.split_strategy,
        "target_token_size": policy.target_token_size,
        "overlap_token_size": policy.overlap_token_size,
    }
    return ChunkInput(
        document_id=document_id,
        chunk_seq=chunk_seq,
        chunk_text=chunk_text,
        chunk_policy_name=policy.chunk_policy_name,
        parser_name=document.parser_name,
        parser_version=document.parser_version,
        heading_path=_common_heading_path([unit.block.heading_path for unit in units]),
        token_count=token_count,
        content_hash=calculate_chunk_content_hash(chunk_text),
        metadata=metadata,
    )


def _document_units_to_chunk_input(
    *,
    document_id: int,
    chunk_seq: int,
    units: list[_DocumentBlockChunkUnit],
    policy: ChunkPolicy,
    parser_name: str | None,
    parser_version: str | None,
) -> ChunkInput:
    chunk_text = "\n\n".join(unit.text for unit in units)
    token_count = count_chunk_tokens(chunk_text)
    block_types = [unit.block.block_type for unit in units]
    source_blocks = [
        {
            "block_id": unit.block.block_id,
            "block_seq": unit.block.block_seq,
            "block_type": unit.block.block_type,
            "source_anchor": unit.block.source_anchor,
        }
        for unit in units
    ]
    metadata = {
        "block_ids": [unit.block.block_id for unit in units],
        "artifact_ids": sorted({unit.block.artifact_id for unit in units}),
        "block_types": block_types,
        "source_blocks": source_blocks,
        "split_strategy": policy.split_strategy,
        "target_token_size": policy.target_token_size,
        "overlap_token_size": policy.overlap_token_size,
    }
    source_anchor = _merged_source_anchor(units, source_blocks)
    return ChunkInput(
        document_id=document_id,
        artifact_id=_common_artifact_id(units),
        block_id=units[0].block.block_id,
        chunk_seq=chunk_seq,
        chunk_type=_chunk_type_for_document_units(units),
        chunk_text=chunk_text,
        content_markdown=chunk_text,
        chunk_policy_name=policy.chunk_policy_name,
        parser_name=parser_name,
        parser_version=parser_version,
        heading_path=_common_heading_path([unit.block.heading_path for unit in units]),
        source_anchor=source_anchor,
        page_no=_common_optional_value([unit.block.page_no for unit in units]),
        slide_no=_common_optional_value([unit.block.slide_no for unit in units]),
        sheet_name=_common_optional_value([unit.block.sheet_name for unit in units]),
        cell_range=_common_optional_value([unit.block.cell_range for unit in units]),
        source_char_start=_minimum_optional_value([unit.block.char_start for unit in units]),
        source_char_end=_maximum_optional_value([unit.block.char_end for unit in units]),
        token_count=token_count,
        content_hash=calculate_chunk_content_hash(chunk_text),
        metadata=metadata,
    )


def _merged_source_anchor(
    units: list[_DocumentBlockChunkUnit],
    source_blocks: list[dict[str, object]],
) -> dict[str, object]:
    source_anchor: dict[str, object] = {
        "source_blocks": source_blocks,
        "start_block_seq": min(unit.block.block_seq for unit in units),
        "end_block_seq": max(unit.block.block_seq for unit in units),
    }
    start_lines = [
        unit.block.source_anchor.get("start_line")
        for unit in units
        if isinstance(unit.block.source_anchor.get("start_line"), int)
    ]
    end_lines = [
        unit.block.source_anchor.get("end_line")
        for unit in units
        if isinstance(unit.block.source_anchor.get("end_line"), int)
    ]
    if start_lines:
        source_anchor["start_line"] = min(start_lines)
    if end_lines:
        source_anchor["end_line"] = max(end_lines)
    return source_anchor


def _common_artifact_id(units: list[_DocumentBlockChunkUnit]) -> int | None:
    artifact_ids = {unit.block.artifact_id for unit in units}
    return artifact_ids.pop() if len(artifact_ids) == 1 else None


def _chunk_type_for_document_units(units: list[_DocumentBlockChunkUnit]) -> str:
    if len(units) != 1:
        return "text"
    block_type = units[0].block.block_type
    if block_type in {"table", "image", "figure", "code", "list", "heading"}:
        return block_type
    return "text"


def _common_optional_value[T](values: list[T | None]) -> T | None:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    first_value = present_values[0]
    return first_value if all(value == first_value for value in present_values) else None


def _minimum_optional_value(values: list[int | None]) -> int | None:
    present_values = [value for value in values if value is not None]
    return min(present_values) if present_values else None


def _maximum_optional_value(values: list[int | None]) -> int | None:
    present_values = [value for value in values if value is not None]
    return max(present_values) if present_values else None


def _common_heading_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    common: list[str] = []
    for candidates in zip(*paths, strict=False):
        if len(set(candidates)) != 1:
            break
        common.append(candidates[0])
    return tuple(common)
