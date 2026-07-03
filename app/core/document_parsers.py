"""Document parser interfaces and Markdown parser foundation."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PARSER_NAME_MARKDOWN = "markdown"
PARSER_VERSION_MARKDOWN = "0.1.0"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


class UnsupportedParserFileError(ValueError):
    """Raised when a parser cannot handle the supplied file."""


@dataclass(frozen=True)
class ParsedBlock:
    block_type: str
    text: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_type": self.block_type,
            "text": self.text,
            "heading_path": list(self.heading_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ParsedDocument:
    parser_name: str
    parser_version: str
    source_path: str | None
    line_count: int
    blocks: tuple[ParsedBlock, ...]

    @property
    def extracted_text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)

    @property
    def extracted_text_size(self) -> int:
        return len(self.extracted_text)

    def to_dict(self, *, include_source_path: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "line_count": self.line_count,
            "blocks": [block.to_dict() for block in self.blocks],
        }
        if include_source_path:
            payload["source_path"] = self.source_path
        return payload


class MarkdownParser:
    parser_name = PARSER_NAME_MARKDOWN
    parser_version = PARSER_VERSION_MARKDOWN

    def parse_path(self, path: Path) -> ParsedDocument:
        if path.suffix.lower() != ".md":
            raise UnsupportedParserFileError(f"Unsupported Markdown file extension: {path.suffix}")
        return self.parse_text(path.read_text(encoding="utf-8"), source_path=str(path))

    def parse_text(self, text: str, *, source_path: str | None = None) -> ParsedDocument:
        lines = text.splitlines()
        heading_stack: list[str] = []
        blocks: list[ParsedBlock] = []
        index = 0

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            heading_match = HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
                blocks.append(
                    ParsedBlock(
                        block_type="heading",
                        text=title,
                        heading_path=tuple(heading_stack),
                        start_line=index + 1,
                        end_line=index + 1,
                        metadata={"level": level},
                    )
                )
                index += 1
                continue

            fence_match = FENCE_RE.match(line)
            if fence_match:
                block, index = self._parse_code_block(lines, index, tuple(heading_stack))
                blocks.append(block)
                continue

            if _starts_table(lines, index):
                block, index = self._parse_table(lines, index, tuple(heading_stack))
                blocks.append(block)
                continue

            block, index = self._parse_paragraph(lines, index, tuple(heading_stack))
            blocks.append(block)

        return ParsedDocument(
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            source_path=source_path,
            line_count=len(lines),
            blocks=tuple(blocks),
        )

    def _parse_code_block(
        self,
        lines: list[str],
        start_index: int,
        heading_path: tuple[str, ...],
    ) -> tuple[ParsedBlock, int]:
        fence_match = FENCE_RE.match(lines[start_index])
        if fence_match is None:
            msg = "Code block parser called without a fence"
            raise RuntimeError(msg)

        fence = fence_match.group(1)
        info_string = fence_match.group(2).strip()
        language = info_string.split(maxsplit=1)[0] if info_string else None
        content_lines: list[str] = []
        index = start_index + 1
        end_line = len(lines)

        while index < len(lines):
            if _is_closing_fence(lines[index], fence):
                end_line = index + 1
                index += 1
                break
            content_lines.append(lines[index])
            index += 1

        metadata = {"language": language} if language else {}
        return (
            ParsedBlock(
                block_type="code_block",
                text="\n".join(content_lines),
                heading_path=heading_path,
                start_line=start_index + 1,
                end_line=end_line,
                metadata=metadata,
            ),
            index,
        )

    def _parse_table(
        self,
        lines: list[str],
        start_index: int,
        heading_path: tuple[str, ...],
    ) -> tuple[ParsedBlock, int]:
        table_lines = [lines[start_index], lines[start_index + 1]]
        index = start_index + 2

        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            table_lines.append(lines[index])
            index += 1

        columns = _split_table_row(table_lines[0])
        metadata = {
            "columns": columns,
            "row_count": max(len(table_lines) - 2, 0),
        }
        return (
            ParsedBlock(
                block_type="table",
                text="\n".join(table_lines),
                heading_path=heading_path,
                start_line=start_index + 1,
                end_line=start_index + len(table_lines),
                metadata=metadata,
            ),
            index,
        )

    def _parse_paragraph(
        self,
        lines: list[str],
        start_index: int,
        heading_path: tuple[str, ...],
    ) -> tuple[ParsedBlock, int]:
        paragraph_lines: list[str] = []
        index = start_index

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                break
            starts_new_structural_block = (
                HEADING_RE.match(line) or FENCE_RE.match(line) or _starts_table(lines, index)
            )
            if paragraph_lines and starts_new_structural_block:
                break
            paragraph_lines.append(line.strip())
            index += 1

        return (
            ParsedBlock(
                block_type="paragraph",
                text=" ".join(paragraph_lines),
                heading_path=heading_path,
                start_line=start_index + 1,
                end_line=start_index + len(paragraph_lines),
                metadata={},
            ),
            index,
        )


def _is_closing_fence(line: str, opening_fence: str) -> bool:
    stripped = line.strip()
    fence_char = opening_fence[0]
    return stripped.startswith(fence_char * len(opening_fence)) and set(stripped) == {fence_char}


def _starts_table(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and _is_table_separator(lines[index + 1])


def _is_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(TABLE_SEPARATOR_CELL_RE.match(cell) for cell in cells)


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|") if cell.strip()]
