from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.chunks import ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_coverage import get_embedding_coverage_matrix
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    create_embedding_job,
    mark_embedding_job_failed,
    mark_embedding_job_succeeded,
)
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    get_embedding_vector_table,
    store_chunk_embedding,
)
from app.core.file_metadata import (
    FileMetadataInput,
    create_file_metadata,
    mark_file_parse_succeeded,
)
from app.main import create_app

pytestmark = pytest.mark.integration


COMPLETE_PROFILE = "bge_m3_1024"
FAILED_PROFILE = "kure_v1_1024"


def _embedding_values(profile_name: str) -> tuple[float, ...]:
    table = get_embedding_vector_table(profile_name)
    return tuple(0.001 for _ in range(table.dimension))


def _create_coverage_fixture(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-164-{suffix}"
    group = f"slice-164-{suffix}"
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"coverage-{suffix}.md",
            stored_file_name=f"coverage-{suffix}.stored.md",
            file_size_bytes=128,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/coverage-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group=group,
            security_level="internal",
            uploaded_by="integration-test",
            document_title=f"Coverage Document {suffix}",
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
                chunk_text="Coverage first chunk",
                token_count=3,
            ),
        ),
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=1,
                chunk_text="Coverage second chunk",
                token_count=4,
            ),
        ),
    ]
    mark_file_parse_succeeded(
        database_url,
        created.file.file_id,
        parser_name="markdown",
        parser_version="slice-164",
        extracted_text_size=42,
    )

    for chunk in chunks:
        job = create_embedding_job(
            database_url,
            EmbeddingJobInput(chunk_id=chunk.chunk_id, profile_name=COMPLETE_PROFILE),
        ).job
        store_chunk_embedding(
            database_url,
            EmbeddingVectorInput(
                chunk_id=chunk.chunk_id,
                profile_name=COMPLETE_PROFILE,
                embedding=_embedding_values(COMPLETE_PROFILE),
                elapsed_ms=12,
            ),
        )
        mark_embedding_job_succeeded(database_url, job.job_id)

    failed_job = create_embedding_job(
        database_url,
        EmbeddingJobInput(chunk_id=chunks[0].chunk_id, profile_name=FAILED_PROFILE),
    ).job
    mark_embedding_job_failed(
        database_url,
        failed_job.job_id,
        error_code="TEST_PROVIDER_ERROR",
        error_message="coverage fixture failure",
    )

    return {
        "checksum": checksum,
        "document_group": group,
        "document_id": document_id,
    }


def _cleanup_coverage_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM files WHERE sha256_checksum = %s",
                (fixture["checksum"],),
            )


def test_embedding_coverage_matrix_repository_api_and_page(
    migrated_database_url: str,
) -> None:
    fixture = _create_coverage_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        matrix = get_embedding_coverage_matrix(
            migrated_database_url,
            document_group=str(fixture["document_group"]),
        )

        with TestClient(app) as client:
            api_response = client.get(
                "/api/admin/embedding-coverage",
                params={"document_group": fixture["document_group"]},
            )
            filtered_api_response = client.get(
                "/api/admin/embedding-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "profile_name": COMPLETE_PROFILE,
                },
            )
            bad_response = client.get("/api/admin/embedding-coverage", params={"limit": 0})
            page_response = client.get(
                "/admin/embedding-coverage",
                params={"document_group": fixture["document_group"]},
            )
            english_page_response = client.get(
                "/admin/embedding-coverage",
                params={"document_group": fixture["document_group"], "lang": "en"},
            )

        document = matrix.documents[0]
        cells = {cell.profile_name: cell for cell in document.profiles}
        payload = api_response.json()
        payload_document = payload["documents"][0]
        payload_cells = {cell["profile_name"]: cell for cell in payload_document["profiles"]}

        assert matrix.summary.document_count == 1
        assert matrix.summary.profile_count >= 2
        assert matrix.summary.total_chunk_count == 2
        assert cells[COMPLETE_PROFILE].status == "complete"
        assert cells[COMPLETE_PROFILE].embedded_chunk_count == 2
        assert cells[COMPLETE_PROFILE].coverage_percent == 100
        assert cells[FAILED_PROFILE].status == "failed"
        assert cells[FAILED_PROFILE].failed_count == 1
        assert api_response.status_code == 200
        assert payload["summary"]["document_count"] == 1
        assert payload_cells[COMPLETE_PROFILE]["status"] == "complete"
        assert payload_cells[COMPLETE_PROFILE]["coverage_label"] == "100.00%"
        assert payload_cells[FAILED_PROFILE]["status"] == "failed"
        assert filtered_api_response.status_code == 200
        assert filtered_api_response.json()["summary"]["profile_count"] == 1
        assert bad_response.status_code == 400
        assert page_response.status_code == 200
        assert "data-embedding-coverage-page" in page_response.text
        assert "임베딩 Coverage Matrix" in page_response.text
        assert COMPLETE_PROFILE in page_response.text
        assert "2 / 2" in page_response.text
        assert english_page_response.status_code == 200
        assert "Embedding Coverage Matrix" in english_page_response.text
    finally:
        _cleanup_coverage_fixture(migrated_database_url, fixture)
