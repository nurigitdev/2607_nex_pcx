from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.bm25_index_coverage import get_bm25_index_coverage_matrix
from app.core.bm25_keyword_index import refresh_chunk_policy_keyword_index
from app.core.chunks import ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.file_metadata import (
    FileMetadataInput,
    create_file_metadata,
    mark_file_parse_succeeded,
)
from app.main import create_app

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
                VALUES (%s, 384, 64, 'bm25-coverage-fixture', 'BM25 coverage fixture')
                """,
                (policy_name,),
            )


def _create_fixture(database_url: str) -> dict[str, object]:
    suffix = uuid4().hex
    policy_name = f"bm25_coverage_{suffix}"
    checksum = f"bm25-coverage-{suffix}"
    document_group = f"bm25-coverage-{suffix}"
    _create_policy(database_url, policy_name)
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"bm25-coverage-{suffix}.md",
            stored_file_name=f"bm25-coverage-{suffix}.stored.md",
            file_size_bytes=256,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/bm25-coverage-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group=document_group,
            security_level="internal",
            uploaded_by="integration-test",
            document_title=f"BM25 Coverage Document {suffix}",
        ),
    )
    document_id = created.file.document_id
    assert document_id is not None
    chunks = [
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=0,
                chunk_text="BM25 coverage alpha beta",
                chunk_policy_name=policy_name,
                token_count=4,
            ),
        ),
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=1,
                chunk_text="BM25 coverage beta gamma",
                chunk_policy_name=policy_name,
                token_count=4,
            ),
        ),
    ]
    mark_file_parse_succeeded(
        database_url,
        created.file.file_id,
        parser_name="markdown",
        parser_version="slice-311",
        extracted_text_size=64,
    )
    refresh_chunk_policy_keyword_index(database_url, chunk_policy_name=policy_name)
    return {
        "checksum": checksum,
        "document_group": document_group,
        "document_id": document_id,
        "file_id": created.file.file_id,
        "policy_name": policy_name,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }


def _cleanup_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunk_keyword_statistics WHERE chunk_policy_name = %s",
                (fixture["policy_name"],),
            )
            cursor.execute("DELETE FROM files WHERE file_id = %s", (fixture["file_id"],))
            cursor.execute(
                "DELETE FROM chunk_policies WHERE chunk_policy_name = %s",
                (fixture["policy_name"],),
            )


def test_bm25_index_coverage_repository_api_and_page(
    migrated_database_url: str,
) -> None:
    fixture = _create_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        matrix = get_bm25_index_coverage_matrix(
            migrated_database_url,
            document_group=str(fixture["document_group"]),
            chunk_policy_name=str(fixture["policy_name"]),
        )
        create_chunk(
            migrated_database_url,
            ChunkInput(
                document_id=int(fixture["document_id"]),
                chunk_seq=2,
                chunk_text="BM25 coverage chunk added after refresh",
                chunk_policy_name=str(fixture["policy_name"]),
                token_count=6,
            ),
        )
        stale_matrix = get_bm25_index_coverage_matrix(
            migrated_database_url,
            document_group=str(fixture["document_group"]),
            chunk_policy_name=str(fixture["policy_name"]),
        )

        with TestClient(app) as client:
            api_response = client.get(
                "/api/admin/bm25-index-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "chunk_policy_name": fixture["policy_name"],
                },
            )
            filtered_response = client.get(
                "/api/admin/bm25-index-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "chunk_policy_name": fixture["policy_name"],
                    "parse_status": "succeeded",
                },
            )
            bad_response = client.get(
                "/api/admin/bm25-index-coverage",
                params={"tokenizer_name": "unknown"},
            )
            page_response = client.get(
                "/admin/bm25-index-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "chunk_policy_name": fixture["policy_name"],
                },
            )
            english_page_response = client.get(
                "/admin/bm25-index-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "chunk_policy_name": fixture["policy_name"],
                    "lang": "en",
                },
            )

        row = matrix.rows[0]
        stale_row = stale_matrix.rows[0]
        payload = api_response.json()
        payload_row = payload["rows"][0]

        assert row.status == "complete"
        assert row.chunk_count == 2
        assert row.indexed_chunk_count == 2
        assert row.statistics_corpus_chunk_count == 2
        assert matrix.summary.coverage_percent == 100
        assert stale_row.status == "stale"
        assert stale_row.chunk_count == 3
        assert stale_row.indexed_chunk_count == 2
        assert stale_row.policy_chunk_count == 3
        assert stale_matrix.summary.stale_row_count == 1
        assert stale_matrix.summary.attention_row_count == 1
        assert api_response.status_code == 200
        assert payload["summary"]["document_count"] == 1
        assert payload["summary"]["stale_row_count"] == 1
        assert payload_row["status"] == "stale"
        assert payload_row["coverage_label"] == "66.67%"
        assert payload_row["statistics_corpus_chunk_count"] == 2
        assert filtered_response.status_code == 200
        assert filtered_response.json()["summary"]["document_policy_count"] == 1
        assert bad_response.status_code == 400
        assert page_response.status_code == 200
        assert "data-bm25-index-coverage-page" in page_response.text
        assert "BM25 Index Coverage" in page_response.text
        assert str(fixture["policy_name"]) in page_response.text
        assert "2 / 3" in page_response.text
        assert "Stale" in page_response.text
        assert english_page_response.status_code == 200
        assert "Document x Policy BM25 Matrix" in english_page_response.text
    finally:
        _cleanup_fixture(migrated_database_url, fixture)
