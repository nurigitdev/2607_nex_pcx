"""DOCX export helpers for generation Markdown output."""

from __future__ import annotations

import io
import re

from docx import Document
from docx.document import Document as DocxDocument

GENERATION_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$")
_MARKDOWN_UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
_MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def markdown_to_docx_bytes(markdown: str, *, title: str | None = None) -> bytes:
    """Convert a practical subset of Markdown to a DOCX byte payload."""

    document = Document()
    if title:
        document.core_properties.title = title
    _add_markdown(document, markdown)

    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _add_markdown(document: DocxDocument, markdown: str) -> None:
    lines = markdown.splitlines()
    index = 0
    in_code_block = False
    code_lines: list[str] = []
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_block:
                _add_code_block(document, code_lines)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            index += 1
            continue
        if in_code_block:
            code_lines.append(line)
            index += 1
            continue

        if not stripped:
            index += 1
            continue

        if _is_markdown_table_start(lines, index):
            table_lines: list[str] = []
            while index < len(lines) and _is_markdown_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1
            _add_table(document, table_lines)
            continue

        heading_match = _MARKDOWN_HEADING_PATTERN.match(stripped)
        if heading_match:
            level = min(len(heading_match.group(1)), 4)
            document.add_heading(_plain_text(heading_match.group(2)), level=level)
            index += 1
            continue

        unordered_match = _MARKDOWN_UNORDERED_LIST_PATTERN.match(line)
        if unordered_match:
            document.add_paragraph(_plain_text(unordered_match.group(1)), style="List Bullet")
            index += 1
            continue

        ordered_match = _MARKDOWN_ORDERED_LIST_PATTERN.match(line)
        if ordered_match:
            document.add_paragraph(_plain_text(ordered_match.group(1)), style="List Number")
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines) and _is_paragraph_continuation(lines[index]):
            paragraph_lines.append(lines[index].strip())
            index += 1
        document.add_paragraph(_plain_text(" ".join(paragraph_lines)))

    if code_lines:
        _add_code_block(document, code_lines)


def _is_paragraph_continuation(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped
        and not stripped.startswith("```")
        and not _MARKDOWN_HEADING_PATTERN.match(stripped)
        and not _MARKDOWN_UNORDERED_LIST_PATTERN.match(line)
        and not _MARKDOWN_ORDERED_LIST_PATTERN.match(line)
        and not _is_markdown_table_line(line)
    )


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and _is_markdown_table_line(lines[index])
        and _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(lines[index + 1].strip()) is not None
    )


def _is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _add_table(document: DocxDocument, table_lines: list[str]) -> None:
    rows = [
        _split_table_row(line)
        for line in table_lines
        if _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(line.strip()) is None
    ]
    rows = [row for row in rows if row]
    if not rows:
        return
    column_count = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for column_index in range(column_count):
            cell = table.cell(row_index, column_index)
            cell.text = _plain_text(row[column_index]) if column_index < len(row) else ""


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _add_code_block(document: DocxDocument, code_lines: list[str]) -> None:
    if not code_lines:
        return
    paragraph = document.add_paragraph()
    run = paragraph.add_run("\n".join(code_lines))
    run.font.name = "Courier New"


def _plain_text(value: str) -> str:
    text = value.replace("**", "").replace("__", "").replace("*", "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()
