from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    generate_mock_embedding,
    store_chunk_embedding,
)
from app.core.vector_search import VectorSearchInput, search_similar_chunks

pytestmark = pytest.mark.integration


def _create_search_chunks(
    database_url: str,
    chunk_texts: list[str],
    *,
    document_status: str = "active",
    file_ext: str = ".md",
) -> tuple[int, list[int]]:
    checksum = f"vector-search-{uuid4()}"
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
                    storage_path,
                    document_group
                )
                VALUES (%s, %s, %s, 1, %s, %s, 'slice-024')
                RETURNING file_id
                """,
                (
                    f"{checksum}{file_ext}",
                    f"{checksum}.stored{file_ext}",
                    file_ext,
                    checksum,
                    f"/tmp/{checksum}{file_ext}",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    document_status,
                    access_scope
                )
                VALUES (%s, %s, 'slice-024', %s, 'company')
                RETURNING document_id
                """,
                (file_id, f"Vector search fixture {checksum}", document_status),
            )
            document_id = cursor.fetchone()["document_id"]
            chunk_ids = []
            for index, chunk_text in enumerate(chunk_texts):
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_seq,
                        chunk_text,
                        content_hash,
                        chunk_policy_name,
                        parser_name,
                        parser_version,
                        heading_path,
                        page_no,
                        token_count,
                        char_count
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        'heading_512_64',
                        'markdown',
                        '1.0',
                        %s,
                        1,
                        %s,
                        %s
                    )
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        index,
                        chunk_text,
                        f"chunk-{checksum}-{index}",
                        ["Section", f"Item {index}"],
                        len(chunk_text.split()),
                        len(chunk_text),
                    ),
                )
                chunk_ids.append(cursor.fetchone()["chunk_id"])

    return file_id, chunk_ids


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _store_mock_embeddings(
    database_url: str,
    *,
    profile_name: str,
    chunk_ids: list[int],
    chunk_texts: list[str],
) -> None:
    dimension = 2560 if profile_name == "qwen3_4b_2560" else 1024
    for chunk_id, chunk_text in zip(chunk_ids, chunk_texts, strict=True):
        store_chunk_embedding(
            database_url,
            EmbeddingVectorInput(
                chunk_id=chunk_id,
                profile_name=profile_name,
                embedding=generate_mock_embedding(
                    chunk_text,
                    profile_name=profile_name,
                    dimension=dimension,
                ),
                elapsed_ms=7,
            ),
        )


def test_search_similar_chunks_returns_ranked_metadata(
    migrated_database_url: str,
) -> None:
    chunk_texts = [
        "Alpha onboarding policy",
        "Beta reimbursement workflow",
        "Gamma security handbook",
    ]
    file_id, chunk_ids = _create_search_chunks(migrated_database_url, chunk_texts)
    try:
        _store_mock_embeddings(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
        )

        results = search_similar_chunks(
            migrated_database_url,
            VectorSearchInput(
                query_text="Beta reimbursement workflow",
                profile_name="kure_v1_1024",
                top_k=2,
                chunk_policy_name="heading_512_64",
                document_group="slice-024",
                file_type=".md",
            ),
        )

        assert len(results) == 2
        assert results[0].rank == 1
        assert results[0].chunk_id == chunk_ids[1]
        assert results[0].chunk_text == "Beta reimbursement workflow"
        assert results[0].chunk_preview == "Beta reimbursement workflow"
        assert results[0].distance == pytest.approx(0.0, abs=1e-6)
        assert results[0].score == pytest.approx(1.0, abs=1e-6)
        assert results[0].profile_name == "kure_v1_1024"
        assert results[0].document_group == "slice-024"
        assert results[0].original_file_name.endswith(".md")
        assert results[0].heading_path == ("Section", "Item 1")
        assert results[0].embedding_elapsed_ms == 7
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_search_similar_chunks_supports_halfvec_profile(
    migrated_database_url: str,
) -> None:
    chunk_texts = ["Halfvec search anchor", "Halfvec alternate chunk"]
    file_id, chunk_ids = _create_search_chunks(migrated_database_url, chunk_texts)
    try:
        _store_mock_embeddings(
            migrated_database_url,
            profile_name="qwen3_4b_2560",
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
        )

        results = search_similar_chunks(
            migrated_database_url,
            VectorSearchInput(
                query_text="Halfvec search anchor",
                profile_name="qwen3_4b_2560",
                top_k=1,
            ),
        )

        assert [result.chunk_id for result in results] == [chunk_ids[0]]
        assert results[0].distance == pytest.approx(0.0, abs=1e-3)
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_search_similar_chunks_excludes_inactive_documents(
    migrated_database_url: str,
) -> None:
    active_texts = ["Visible search anchor"]
    archived_texts = ["Visible search anchor"]
    active_file_id, active_chunk_ids = _create_search_chunks(migrated_database_url, active_texts)
    archived_file_id, archived_chunk_ids = _create_search_chunks(
        migrated_database_url,
        archived_texts,
        document_status="archived",
    )
    try:
        _store_mock_embeddings(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_ids=active_chunk_ids,
            chunk_texts=active_texts,
        )
        _store_mock_embeddings(
            migrated_database_url,
            profile_name="kure_v1_1024",
            chunk_ids=archived_chunk_ids,
            chunk_texts=archived_texts,
        )

        results = search_similar_chunks(
            migrated_database_url,
            VectorSearchInput(
                query_text="Visible search anchor",
                profile_name="kure_v1_1024",
                top_k=5,
            ),
        )

        assert active_chunk_ids[0] in {result.chunk_id for result in results}
        assert archived_chunk_ids[0] not in {result.chunk_id for result in results}
    finally:
        _cleanup_file(migrated_database_url, active_file_id)
        _cleanup_file(migrated_database_url, archived_file_id)
