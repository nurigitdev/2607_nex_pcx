from uuid import uuid4

import pytest

from app.core.bm25_keyword_index import (
    DEFAULT_BM25_TOKENIZER_NAME,
    list_chunk_keyword_statistics,
    list_chunk_keyword_terms,
    refresh_chunk_policy_keyword_index,
    replace_chunk_keyword_terms,
)
from app.core.chunks import ChunkInput, create_chunk
from app.core.database import connect

pytestmark = pytest.mark.integration


def _create_policy(database_url: str, policy_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chunk_policies (
                    chunk_policy_name,
                    target_token_size,
                    overlap_token_size,
                    split_strategy,
                    description
                )
                VALUES (%s, 256, 32, 'bm25-fixture', 'BM25 keyword index fixture')
                """,
                (policy_name,),
            )


def _create_document(database_url: str) -> tuple[int, int]:
    checksum = f"bm25-index-{uuid4()}"
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
                (file_id, f"BM25 index fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_fixture(database_url: str, file_id: int, policy_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunk_keyword_statistics WHERE chunk_policy_name = %s",
                (policy_name,),
            )
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            cursor.execute(
                "DELETE FROM chunk_policies WHERE chunk_policy_name = %s",
                (policy_name,),
            )


def test_replace_chunk_keyword_terms_is_idempotent(
    migrated_database_url: str,
) -> None:
    policy_name = f"bm25_policy_{uuid4().hex}"
    _create_policy(migrated_database_url, policy_name)
    file_id, document_id = _create_document(migrated_database_url)
    try:
        chunk = create_chunk(
            migrated_database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=0,
                chunk_text="Alpha alpha beta",
                chunk_policy_name=policy_name,
            ),
        )

        initial_terms = replace_chunk_keyword_terms(
            migrated_database_url,
            chunk_id=chunk.chunk_id,
            chunk_policy_name=policy_name,
            chunk_text=chunk.chunk_text,
        )
        replacement_terms = replace_chunk_keyword_terms(
            migrated_database_url,
            chunk_id=chunk.chunk_id,
            chunk_policy_name=policy_name,
            chunk_text="Gamma",
        )
        stored_terms = list_chunk_keyword_terms(
            migrated_database_url,
            chunk_id=chunk.chunk_id,
        )

        assert [(term.term, term.term_frequency) for term in initial_terms] == [
            ("alpha", 2),
            ("beta", 1),
        ]
        assert replacement_terms == stored_terms
        assert [(term.term, term.term_frequency) for term in stored_terms] == [("gamma", 1)]
    finally:
        _cleanup_fixture(migrated_database_url, file_id, policy_name)


def test_refresh_chunk_policy_keyword_index_rebuilds_terms_and_statistics(
    migrated_database_url: str,
) -> None:
    policy_name = f"bm25_policy_{uuid4().hex}"
    _create_policy(migrated_database_url, policy_name)
    file_id, document_id = _create_document(migrated_database_url)
    try:
        chunks = [
            create_chunk(
                migrated_database_url,
                ChunkInput(
                    document_id=document_id,
                    chunk_seq=0,
                    chunk_text="Alpha alpha beta 한국어",
                    chunk_policy_name=policy_name,
                ),
            ),
            create_chunk(
                migrated_database_url,
                ChunkInput(
                    document_id=document_id,
                    chunk_seq=1,
                    chunk_text="Beta gamma 한국어",
                    chunk_policy_name=policy_name,
                ),
            ),
            create_chunk(
                migrated_database_url,
                ChunkInput(
                    document_id=document_id,
                    chunk_seq=2,
                    chunk_text="... !!!",
                    chunk_policy_name=policy_name,
                ),
            ),
        ]
        replace_chunk_keyword_terms(
            migrated_database_url,
            chunk_id=chunks[0].chunk_id,
            chunk_policy_name=policy_name,
            chunk_text="stale stale",
        )

        result = refresh_chunk_policy_keyword_index(
            migrated_database_url,
            chunk_policy_name=policy_name,
        )
        repeated = refresh_chunk_policy_keyword_index(
            migrated_database_url,
            chunk_policy_name=policy_name,
        )
        first_chunk_terms = list_chunk_keyword_terms(
            migrated_database_url,
            chunk_id=chunks[0].chunk_id,
        )
        statistics = {
            record.term: record
            for record in list_chunk_keyword_statistics(
                migrated_database_url,
                chunk_policy_name=policy_name,
            )
        }

        assert result.chunk_policy_name == policy_name
        assert result.tokenizer_name == DEFAULT_BM25_TOKENIZER_NAME
        assert result.chunk_count == 3
        assert result.term_row_count == 6
        assert result.statistics_row_count == 4
        assert repeated == result
        assert [(term.term, term.term_frequency) for term in first_chunk_terms] == [
            ("alpha", 2),
            ("beta", 1),
            ("한국어", 1),
        ]
        assert statistics["alpha"].document_frequency == 1
        assert statistics["beta"].document_frequency == 2
        assert statistics["한국어"].document_frequency == 2
        assert statistics["gamma"].corpus_chunk_count == 3
        assert float(statistics["gamma"].average_document_length) == pytest.approx(2.3333)
    finally:
        _cleanup_fixture(migrated_database_url, file_id, policy_name)


def test_refresh_chunk_policy_keyword_index_handles_empty_policy(
    migrated_database_url: str,
) -> None:
    policy_name = f"bm25_policy_{uuid4().hex}"
    _create_policy(migrated_database_url, policy_name)
    file_id, document_id = _create_document(migrated_database_url)
    try:
        result = refresh_chunk_policy_keyword_index(
            migrated_database_url,
            chunk_policy_name=policy_name,
        )
        statistics = list_chunk_keyword_statistics(
            migrated_database_url,
            chunk_policy_name=policy_name,
        )

        assert document_id > 0
        assert result.chunk_count == 0
        assert result.term_row_count == 0
        assert result.statistics_row_count == 0
        assert result.average_document_length == 0
        assert statistics == []
    finally:
        _cleanup_fixture(migrated_database_url, file_id, policy_name)
