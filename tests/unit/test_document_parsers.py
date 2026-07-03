from pathlib import Path

import pytest

from app.core.document_parsers import (
    MarkdownParser,
    ParsedBlock,
    UnsupportedParserFileError,
)


def test_markdown_parser_extracts_heading_hierarchy() -> None:
    document = MarkdownParser().parse_text("# Root\n\n## Child\n\nBody")

    assert [block.heading_path for block in document.blocks] == [
        ("Root",),
        ("Root", "Child"),
        ("Root", "Child"),
    ]
    assert document.blocks[-1].text == "Body"


def test_markdown_parser_preserves_fenced_code_block() -> None:
    document = MarkdownParser().parse_text("# Root\n\n```sql\nselect 1;\n```")
    code_block = document.blocks[1]

    assert code_block == ParsedBlock(
        block_type="code_block",
        text="select 1;",
        heading_path=("Root",),
        start_line=3,
        end_line=5,
        metadata={"language": "sql"},
    )


def test_markdown_parser_preserves_table_as_single_block() -> None:
    document = MarkdownParser().parse_text(
        "# Root\n\n| Key | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
    )
    table_block = document.blocks[1]

    assert table_block.block_type == "table"
    assert table_block.start_line == 3
    assert table_block.end_line == 6
    assert table_block.metadata == {"columns": ["Key", "Value"], "row_count": 2}


def test_markdown_parser_reads_markdown_path(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    source_path.write_text("# Title\n\nText", encoding="utf-8")

    document = MarkdownParser().parse_path(source_path)

    assert document.source_path == str(source_path)
    assert document.parser_name == "markdown"
    assert document.parser_version == "0.1.0"
    assert document.extracted_text == "Title\n\nText"
    assert document.extracted_text_size == len("Title\n\nText")


def test_markdown_parser_rejects_non_markdown_path(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("plain text", encoding="utf-8")

    with pytest.raises(UnsupportedParserFileError, match="Unsupported Markdown"):
        MarkdownParser().parse_path(source_path)


def test_markdown_parser_handles_unclosed_code_fence() -> None:
    document = MarkdownParser().parse_text("# Root\n\n```python\nprint('open')")
    code_block = document.blocks[1]

    assert code_block.block_type == "code_block"
    assert code_block.end_line == 4
    assert code_block.text == "print('open')"
