from uuid import uuid4

import pytest

from app.core.bm25_keyword_index import refresh_chunk_policy_keyword_index
from app.core.bm25_search import BM25SearchInput, search_bm25_chunks
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
                VALUES (%s, 256, 32, 'bm25-search-fixture', 'BM25 search fixture')
                """,
                (policy_name,),
            )


def _create_document(
    database_url: str,
    *,
    document_group: str,
    document_status: str = "active",
    file_ext: str = ".md",
) -> tuple[int, int]:
    checksum = f"bm25-search-{uuid4()}"
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
                VALUES (%s, %s, %s, 1, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}{file_ext}",
                    f"{checksum}.stored{file_ext}",
                    file_ext,
                    checksum,
                    f"/tmp/{checksum}{file_ext}",
                    document_group,
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
                VALUES (%s, %s, %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, f"BM25 search fixture {checksum}", document_group, document_status),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_fixture(database_url: str, file_ids: list[int], policy_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunk_keyword_statistics WHERE chunk_policy_name = %s",
                (policy_name,),
            )
            for file_id in file_ids:
                cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            cursor.execute(
                "DELETE FROM chunk_policies WHERE chunk_policy_name = %s",
                (policy_name,),
            )


def _create_chunks(
    database_url: str,
    document_id: int,
    policy_name: str,
    texts: list[str],
) -> list[int]:
    return [
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=index,
                chunk_text=text,
                chunk_policy_name=policy_name,
                parser_name="markdown",
                parser_version="1.0",
                heading_path=("BM25", f"Item {index}"),
                page_no=1,
                token_count=len(text.split()),
            ),
        ).chunk_id
        for index, text in enumerate(texts)
    ]


def test_search_bm25_chunks_returns_ranked_metadata_and_score_components(
    migrated_database_url: str,
) -> None:
    policy_name = f"bm25_policy_{uuid4().hex}"
    document_group = f"slice-307-{uuid4().hex}"
    _create_policy(migrated_database_url, policy_name)
    file_id, document_id = _create_document(
        migrated_database_url,
        document_group=document_group,
    )
    try:
        chunk_ids = _create_chunks(
            migrated_database_url,
            document_id,
            policy_name,
            [
                "Beta beta reimbursement workflow",
                "Alpha beta general policy",
                "Gamma security handbook",
            ],
        )
        refresh_chunk_policy_keyword_index(
            migrated_database_url,
            chunk_policy_name=policy_name,
        )

        results = search_bm25_chunks(
            migrated_database_url,
            BM25SearchInput(
                query_text="beta reimbursement",
                top_k=2,
                chunk_policy_name=policy_name,
                document_group=document_group,
                file_type=".md",
            ),
        )

        assert len(results) == 2
        assert [result.rank for result in results] == [1, 2]
        assert results[0].chunk_id == chunk_ids[0]
        assert results[0].score > results[1].score
        assert results[0].chunk_text == "Beta beta reimbursement workflow"
        assert results[0].chunk_preview == "Beta beta reimbursement workflow"
        assert results[0].matched_term_count == 2
        assert results[0].document_length == pytest.approx(4.0)
        assert results[0].score_components["query_terms"] == ["beta", "reimbursement"]
        assert results[0].score_components["k1"] == pytest.approx(1.2)
        assert results[0].document_group == document_group
        assert results[0].original_file_name.endswith(".md")
        assert results[0].heading_path == ("BM25", "Item 0")
    finally:
        _cleanup_fixture(migrated_database_url, [file_id], policy_name)


def test_search_bm25_chunks_filters_inactive_documents(
    migrated_database_url: str,
) -> None:
    policy_name = f"bm25_policy_{uuid4().hex}"
    document_group = f"slice-307-{uuid4().hex}"
    _create_policy(migrated_database_url, policy_name)
    active_file_id, active_document_id = _create_document(
        migrated_database_url,
        document_group=document_group,
    )
    archived_file_id, archived_document_id = _create_document(
        migrated_database_url,
        document_group=document_group,
        document_status="archived",
    )
    try:
        active_chunk_ids = _create_chunks(
            migrated_database_url,
            active_document_id,
            policy_name,
            ["Visible active anchor"],
        )
        archived_chunk_ids = _create_chunks(
            migrated_database_url,
            archived_document_id,
            policy_name,
            ["Visible active anchor"],
        )
        refresh_chunk_policy_keyword_index(
            migrated_database_url,
            chunk_policy_name=policy_name,
        )

        results = search_bm25_chunks(
            migrated_database_url,
            BM25SearchInput(
                query_text="visible active",
                top_k=5,
                chunk_policy_name=policy_name,
                document_group=document_group,
            ),
        )

        assert active_chunk_ids[0] in {result.chunk_id for result in results}
        assert archived_chunk_ids[0] not in {result.chunk_id for result in results}
    finally:
        _cleanup_fixture(
            migrated_database_url,
            [active_file_id, archived_file_id],
            policy_name,
        )


def test_search_bm25_chunks_returns_empty_for_punctuation_query(
    migrated_database_url: str,
) -> None:
    assert (
        search_bm25_chunks(
            migrated_database_url,
            BM25SearchInput(query_text="... !!!"),
        )
        == []
    )
