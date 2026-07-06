from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    generate_mock_embedding,
    get_chunk_embedding,
    store_chunk_embedding,
)

pytestmark = pytest.mark.integration


def _create_chunk(database_url: str, chunk_text: str) -> tuple[int, int]:
    checksum = f"embedding-vector-{uuid4()}"
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
                (file_id, f"Embedding vector fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    char_count
                )
                VALUES (%s, 0, %s, %s, 'heading_512_64', %s)
                RETURNING chunk_id
                """,
                (document_id, chunk_text, f"chunk-{checksum}", len(chunk_text)),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_store_and_get_vector_embedding(migrated_database_url: str) -> None:
    chunk_text = "Vector repository test"
    file_id, chunk_id = _create_chunk(migrated_database_url, chunk_text)
    try:
        embedding = generate_mock_embedding(
            chunk_text,
            profile_name="kure_v1_1024",
            dimension=1024,
        )

        stored = store_chunk_embedding(
            migrated_database_url,
            EmbeddingVectorInput(
                chunk_id=chunk_id,
                profile_name="kure_v1_1024",
                embedding=embedding,
                elapsed_ms=11,
            ),
        )
        replacement = store_chunk_embedding(
            migrated_database_url,
            EmbeddingVectorInput(
                chunk_id=chunk_id,
                profile_name="kure_v1_1024",
                embedding=embedding,
                elapsed_ms=12,
            ),
        )
        loaded = get_chunk_embedding(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_id=chunk_id,
        )

        assert stored.chunk_id == chunk_id
        assert stored.dimension == 1024
        assert stored.storage_type == "vector"
        assert stored.embedding_text.startswith("[")
        assert replacement.elapsed_ms == 12
        assert loaded == replacement
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_store_and_get_halfvec_embedding(migrated_database_url: str) -> None:
    chunk_text = "Halfvec repository test"
    file_id, chunk_id = _create_chunk(migrated_database_url, chunk_text)
    try:
        embedding = generate_mock_embedding(
            chunk_text,
            profile_name="qwen3_4b_2560",
            dimension=2560,
        )

        stored = store_chunk_embedding(
            migrated_database_url,
            EmbeddingVectorInput(
                chunk_id=chunk_id,
                profile_name="qwen3_4b_2560",
                embedding=embedding,
                elapsed_ms=21,
            ),
        )
        loaded = get_chunk_embedding(
            migrated_database_url,
            profile_name="qwen3_4b_2560",
            chunk_id=chunk_id,
        )

        assert stored.dimension == 2560
        assert stored.storage_type == "halfvec"
        assert stored.elapsed_ms == 21
        assert loaded == stored
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_get_chunk_embedding_returns_none_when_missing(migrated_database_url: str) -> None:
    file_id, chunk_id = _create_chunk(migrated_database_url, "Missing embedding")
    try:
        assert (
            get_chunk_embedding(
                migrated_database_url,
                profile_name="kure_v1_1024",
                chunk_id=chunk_id,
            )
            is None
        )
    finally:
        _cleanup_file(migrated_database_url, file_id)
