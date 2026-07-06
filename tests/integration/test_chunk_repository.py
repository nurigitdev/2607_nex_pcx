from uuid import uuid4

import pytest

from app.core.chunks import (
    ChunkInput,
    InvalidChunkError,
    calculate_chunk_content_hash,
    create_chunk,
    list_document_chunks,
    replace_document_chunks,
    replace_document_chunks_in_connection,
)
from app.core.database import connect

pytestmark = pytest.mark.integration


def _create_document(database_url: str) -> tuple[int, int]:
    checksum = f"chunk-repository-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path
                )
                VALUES (%s, %s, '.md', 1, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Chunk repository fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_create_and_list_document_chunks(migrated_database_url: str) -> None:
    file_id, document_id = _create_document(migrated_database_url)
    try:
        chunk = create_chunk(
            migrated_database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=0,
                chunk_text="First markdown chunk",
                parser_name="markdown",
                parser_version="0.1.0",
                heading_path=("Overview",),
                token_count=3,
                metadata={"block_type": "paragraph"},
            ),
        )
        chunks = list_document_chunks(migrated_database_url, document_id)

        assert chunk.chunk_id == chunks[0].chunk_id
        assert chunk.content_hash == calculate_chunk_content_hash("First markdown chunk")
        assert chunk.char_count == len("First markdown chunk")
        assert chunk.heading_path == ("Overview",)
        assert chunk.metadata == {"block_type": "paragraph"}
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_replace_document_chunks_links_neighbors_and_replaces_policy_set(
    migrated_database_url: str,
) -> None:
    file_id, document_id = _create_document(migrated_database_url)
    try:
        initial = replace_document_chunks(
            migrated_database_url,
            document_id,
            [
                ChunkInput(document_id=document_id, chunk_seq=0, chunk_text="Alpha"),
                ChunkInput(document_id=document_id, chunk_seq=1, chunk_text="Beta"),
                ChunkInput(document_id=document_id, chunk_seq=2, chunk_text="Gamma"),
            ],
        )
        replacement = replace_document_chunks(
            migrated_database_url,
            document_id,
            [
                ChunkInput(document_id=document_id, chunk_seq=0, chunk_text="Delta"),
                ChunkInput(document_id=document_id, chunk_seq=1, chunk_text="Epsilon"),
            ],
        )
        stored = list_document_chunks(migrated_database_url, document_id)

        assert [chunk.chunk_text for chunk in initial] == ["Alpha", "Beta", "Gamma"]
        assert [chunk.chunk_text for chunk in replacement] == ["Delta", "Epsilon"]
        assert [chunk.chunk_text for chunk in stored] == ["Delta", "Epsilon"]
        assert stored[0].prev_chunk_id is None
        assert stored[0].next_chunk_id == stored[1].chunk_id
        assert stored[1].prev_chunk_id == stored[0].chunk_id
        assert stored[1].next_chunk_id is None
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_replace_document_chunks_validates_policy_and_sequence_before_db() -> None:
    with pytest.raises(InvalidChunkError, match="contiguous"):
        replace_document_chunks_in_connection(
            None,  # type: ignore[arg-type]
            1,
            [
                ChunkInput(document_id=1, chunk_seq=1, chunk_text="Out of order"),
            ],
        )

    with pytest.raises(InvalidChunkError, match="target chunk_policy_name"):
        replace_document_chunks_in_connection(
            None,  # type: ignore[arg-type]
            1,
            [
                ChunkInput(
                    document_id=1,
                    chunk_seq=0,
                    chunk_text="Wrong policy",
                    chunk_policy_name="heading_1000_200",
                ),
            ],
        )
