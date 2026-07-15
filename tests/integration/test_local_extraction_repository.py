from pathlib import Path
from uuid import uuid4

import pytest

from app.core.chunking import chunk_document_blocks
from app.core.chunks import list_document_chunks, replace_document_chunks
from app.core.database import connect
from app.core.extraction_runtime import ExtractionRuntimeRequest
from app.core.ingestion_artifacts import list_document_blocks
from app.core.local_extraction import (
    LOCAL_MARKDOWN_PROFILE_NAME,
    persist_extraction_runtime_result,
    run_local_extraction,
)

pytestmark = pytest.mark.integration


def _create_document(database_url: str, source_path: Path) -> tuple[int, int]:
    checksum = f"local-extraction-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    mime_type,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    detected_file_type
                )
                VALUES (%s, %s, '.md', 'text/markdown', %s, %s, %s, 'md')
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    source_path.stat().st_size,
                    checksum,
                    str(source_path),
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Local extraction fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_local_markdown_extraction_result_can_be_persisted(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "slice-217.md"
    source_path.write_text("# Local Extraction\n\nBody text.", encoding="utf-8")
    file_id, document_id = _create_document(migrated_database_url, source_path)

    try:
        request = ExtractionRuntimeRequest(
            file_id=file_id,
            document_id=document_id,
            storage_path=str(source_path),
            extraction_profile_name=LOCAL_MARKDOWN_PROFILE_NAME,
            mime_type="text/markdown",
            detected_file_type="md",
        )
        runtime_result = run_local_extraction(request)
        persisted = persist_extraction_runtime_result(
            migrated_database_url,
            request,
            runtime_result,
        )
        blocks = list_document_blocks(
            migrated_database_url,
            document_id,
            artifact_id=persisted.artifacts[0].artifact_id,
        )

        assert persisted.run.status == "succeeded"
        assert persisted.run.extraction_profile_name == LOCAL_MARKDOWN_PROFILE_NAME
        assert persisted.run.extractor_name == "local_markdown"
        assert persisted.run.error_count == 0
        assert persisted.artifacts[0].artifact_type == "normalized_markdown"
        assert persisted.artifacts[0].content_hash
        assert persisted.artifacts[0].metadata["parser_name"] == "markdown"
        assert [block.block_type for block in persisted.blocks] == ["heading", "paragraph"]
        assert blocks == list(persisted.blocks)
        assert blocks[1].parent_block_id == blocks[0].block_id
        assert blocks[0].source_anchor == {"start_line": 1, "end_line": 1}
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_persisted_document_blocks_can_be_chunked_with_lineage(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "slice-218.md"
    source_path.write_text("# Chunk Lineage\n\nTrace this paragraph.", encoding="utf-8")
    file_id, document_id = _create_document(migrated_database_url, source_path)

    try:
        request = ExtractionRuntimeRequest(
            file_id=file_id,
            document_id=document_id,
            storage_path=str(source_path),
            extraction_profile_name=LOCAL_MARKDOWN_PROFILE_NAME,
            mime_type="text/markdown",
            detected_file_type="md",
        )
        runtime_result = run_local_extraction(request)
        persisted = persist_extraction_runtime_result(
            migrated_database_url,
            request,
            runtime_result,
        )
        block_inputs = list_document_blocks(
            migrated_database_url,
            document_id,
            artifact_id=persisted.artifacts[0].artifact_id,
        )
        chunk_inputs = chunk_document_blocks(
            block_inputs,
            document_id=document_id,
            parser_name="markdown",
            parser_version="0.1.0",
        )
        chunks = replace_document_chunks(
            migrated_database_url,
            document_id,
            chunk_inputs,
        )
        stored_chunks = list_document_chunks(migrated_database_url, document_id)

        assert len(chunks) == 1
        assert chunks == stored_chunks
        assert chunks[0].artifact_id == persisted.artifacts[0].artifact_id
        assert chunks[0].block_id == persisted.blocks[0].block_id
        assert chunks[0].chunk_text == "# Chunk Lineage\n\nTrace this paragraph."
        assert chunks[0].content_markdown == chunks[0].chunk_text
        assert chunks[0].source_anchor["start_line"] == 1
        assert chunks[0].source_anchor["end_line"] == 3
        assert chunks[0].source_char_start == 0
        assert chunks[0].source_char_end == len(source_path.read_text(encoding="utf-8"))
        assert chunks[0].metadata["block_ids"] == [
            persisted.blocks[0].block_id,
            persisted.blocks[1].block_id,
        ]
    finally:
        _cleanup_file(migrated_database_url, file_id)
