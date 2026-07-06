from uuid import uuid4

import pytest

from app.core.config import get_settings
from app.core.database import connect
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    generate_mock_embedding,
    store_chunk_embedding,
)
from app.core.migrations import upgrade
from app.core.vector_search import VectorSearchInput, search_similar_chunks

pytestmark = pytest.mark.integration


@pytest.fixture()
def migrated_database_url() -> str:
    database_url = get_settings().test_database_url
    if not database_url:
        pytest.skip("NEX_PCX_TEST_DATABASE_URL is not set")
    upgrade("head", database_url)
    return database_url


def _create_corpus(database_url: str, chunk_texts: list[str]) -> tuple[int, list[int]]:
    checksum = f"vector-regression-{uuid4()}"
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
                VALUES (%s, %s, '.md', 1, %s, %s, 'regression')
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
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    access_scope
                )
                VALUES (%s, %s, 'regression', 'company')
                RETURNING document_id
                """,
                (file_id, f"Vector regression fixture {checksum}"),
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
                        char_count
                    )
                    VALUES (%s, %s, %s, %s, 'heading_512_64', %s)
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        index,
                        chunk_text,
                        f"chunk-{checksum}-{index}",
                        len(chunk_text),
                    ),
                )
                chunk_ids.append(cursor.fetchone()["chunk_id"])

    return file_id, chunk_ids


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _pg_literal_rounded(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(format(value, ".8g")) for value in values)


def _cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return 1 - sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def test_vector_search_ranking_matches_mock_embedding_regression(
    migrated_database_url: str,
) -> None:
    profile_name = "kure_v1_1024"
    dimension = 1024
    query_text = "NeX PCX reimbursement policy anchor"
    corpus = [
        "NeX PCX onboarding checklist",
        "NeX PCX reimbursement policy anchor",
        "NeX PCX security classification guide",
        "NeX PCX vacation approval workflow",
    ]
    file_id, chunk_ids = _create_corpus(migrated_database_url, corpus)
    try:
        stored_embeddings = []
        for chunk_id, chunk_text in zip(chunk_ids, corpus, strict=True):
            embedding = generate_mock_embedding(
                chunk_text,
                profile_name=profile_name,
                dimension=dimension,
            )
            stored_embeddings.append(_pg_literal_rounded(embedding))
            store_chunk_embedding(
                migrated_database_url,
                EmbeddingVectorInput(
                    chunk_id=chunk_id,
                    profile_name=profile_name,
                    embedding=embedding,
                    elapsed_ms=5,
                ),
            )

        query_embedding = _pg_literal_rounded(
            generate_mock_embedding(
                query_text,
                profile_name=profile_name,
                dimension=dimension,
            )
        )
        expected_chunk_ids = [
            chunk_id
            for chunk_id, _distance in sorted(
                (
                    (
                        chunk_id,
                        _cosine_distance(query_embedding, stored_embedding),
                    )
                    for chunk_id, stored_embedding in zip(chunk_ids, stored_embeddings, strict=True)
                ),
                key=lambda item: (item[1], item[0]),
            )
        ]

        results = search_similar_chunks(
            migrated_database_url,
            VectorSearchInput(
                query_text=query_text,
                profile_name=profile_name,
                top_k=len(corpus),
                document_group="regression",
            ),
        )

        assert [result.chunk_id for result in results] == expected_chunk_ids
        assert results[0].chunk_text == query_text
        assert [result.rank for result in results] == [1, 2, 3, 4]
    finally:
        _cleanup_file(migrated_database_url, file_id)
