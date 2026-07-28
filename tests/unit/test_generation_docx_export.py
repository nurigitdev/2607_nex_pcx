from io import BytesIO

from docx import Document

from app.core.generation_docx_export import markdown_to_docx_bytes


def test_markdown_to_docx_bytes_converts_headings_lists_tables_and_code() -> None:
    docx_bytes = markdown_to_docx_bytes(
        "\n".join(
            (
                "# 보고서 초안",
                "",
                "## 요약",
                "첫 문단입니다.",
                "",
                "- 항목 A",
                "1. 항목 B",
                "",
                "| 구분 | 값 |",
                "| --- | --- |",
                "| 상태 | 정상 |",
                "",
                "```json",
                '{"ok": true}',
                "```",
            )
        ),
        title="Pytest DOCX",
    )

    document = Document(BytesIO(docx_bytes))
    paragraph_text = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    table_text = [
        [cell.text for cell in row.cells] for table in document.tables for row in table.rows
    ]

    assert document.core_properties.title == "Pytest DOCX"
    assert paragraph_text[:5] == ["보고서 초안", "요약", "첫 문단입니다.", "항목 A", "항목 B"]
    assert '{"ok": true}' in paragraph_text[-1]
    assert table_text == [["구분", "값"], ["상태", "정상"]]


def test_markdown_to_docx_bytes_ignores_empty_tables_and_unclosed_code_blocks() -> None:
    docx_bytes = markdown_to_docx_bytes(
        "\n".join(
            (
                "plain paragraph",
                "",
                "| --- | --- |",
                "",
                "```",
                "unclosed code",
            )
        )
    )

    document = Document(BytesIO(docx_bytes))
    paragraph_text = [paragraph.text for paragraph in document.paragraphs if paragraph.text]

    assert "plain paragraph" in paragraph_text
    assert "unclosed code" in paragraph_text
    assert document.tables == []
