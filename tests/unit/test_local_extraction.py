from pathlib import Path

from app.core.extraction_runtime import ExtractionRuntimeRequest
from app.core.local_extraction import (
    ERROR_CODE_LOCAL_SOURCE_NOT_FOUND,
    ERROR_CODE_LOCAL_UNSUPPORTED_FILE_TYPE,
    LOCAL_MARKDOWN_PROFILE_NAME,
    LOCAL_PLAIN_TEXT_PROFILE_NAME,
    run_local_extraction,
)


def test_run_local_markdown_extraction_returns_artifact_and_blocks(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.md"
    source_path.write_text(
        "# Title\r\n\r\n"
        "Intro paragraph.\r\n\r\n"
        "## Data\r\n\r\n"
        "| A | B |\r\n"
        "| --- | --- |\r\n"
        "| 1 | 2 |\r\n\r\n"
        "```python\r\n"
        "print(1)\r\n"
        "```\r\n",
        encoding="utf-8",
    )

    result = run_local_extraction(
        ExtractionRuntimeRequest(
            file_id=1,
            document_id=2,
            storage_path=str(source_path),
            extraction_profile_name=LOCAL_MARKDOWN_PROFILE_NAME,
            mime_type="text/markdown",
            detected_file_type="md",
        )
    )

    assert result.status == "succeeded"
    assert result.errors == ()
    assert result.artifacts[0].artifact_type == "normalized_markdown"
    assert "\r" not in result.artifacts[0].content_text
    assert result.artifacts[0].metadata["parser_name"] == "markdown"
    assert [block.block_type for block in result.blocks] == [
        "heading",
        "paragraph",
        "heading",
        "table",
        "code",
    ]
    assert result.blocks[1].parent_block_seq == 0
    assert result.blocks[2].parent_block_seq == 0
    assert result.blocks[3].parent_block_seq == 2
    assert result.blocks[3].source_anchor == {
        "start_line": 7,
        "end_line": 9,
        "table_index": 3,
    }
    assert result.blocks[4].metadata == {"language": "python"}
    assert result.runtime_metadata["block_count"] == 5


def test_run_local_plain_text_extraction_splits_paragraphs(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text(
        "First paragraph\r\ncontinues.\r\n\r\nSecond paragraph.",
        encoding="utf-8",
    )

    result = run_local_extraction(
        ExtractionRuntimeRequest(
            file_id=1,
            document_id=2,
            storage_path=str(source_path),
            extraction_profile_name=LOCAL_PLAIN_TEXT_PROFILE_NAME,
            detected_file_type="txt",
        )
    )

    assert result.status == "succeeded"
    assert result.artifacts[0].artifact_type == "plain_text"
    assert result.blocks[0].content_text == "First paragraph continues."
    assert result.blocks[0].source_anchor == {"start_line": 1, "end_line": 2}
    assert result.blocks[1].content_text == "Second paragraph."
    assert result.runtime_metadata["extractor_name"] == "local_plain_text"


def test_run_local_extraction_returns_failed_for_missing_source(tmp_path: Path) -> None:
    result = run_local_extraction(
        ExtractionRuntimeRequest(
            file_id=1,
            storage_path=str(tmp_path / "missing.md"),
            extraction_profile_name=LOCAL_MARKDOWN_PROFILE_NAME,
        )
    )

    assert result.status == "failed"
    assert result.runtime_metadata["error_code"] == ERROR_CODE_LOCAL_SOURCE_NOT_FOUND
    assert "not found" in result.errors[0]


def test_run_local_extraction_returns_failed_for_profile_suffix_mismatch(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_text("# Not really PDF", encoding="utf-8")

    result = run_local_extraction(
        ExtractionRuntimeRequest(
            file_id=1,
            storage_path=str(source_path),
            extraction_profile_name=LOCAL_MARKDOWN_PROFILE_NAME,
        )
    )

    assert result.status == "failed"
    assert result.runtime_metadata["error_code"] == ERROR_CODE_LOCAL_UNSUPPORTED_FILE_TYPE
    assert "cannot process" in result.errors[0]
