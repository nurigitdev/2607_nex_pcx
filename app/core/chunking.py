"""Heading-aware chunking for parsed documents."""

import re
from dataclasses import dataclass

from app.core.chunks import (
    DEFAULT_CHUNK_POLICY_NAME,
    ChunkInput,
    calculate_chunk_content_hash,
)
from app.core.document_parsers import ParsedBlock, ParsedDocument

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


def _common_heading_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    common: list[str] = []
    for candidates in zip(*paths, strict=False):
        if len(set(candidates)) != 1:
            break
        common.append(candidates[0])
    return tuple(common)
