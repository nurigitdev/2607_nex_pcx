from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.chunk_policies import get_chunk_policy_summary, list_chunk_policy_summaries
from app.core.chunks import DEFAULT_CHUNK_POLICY_NAME, ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.file_metadata import FileMetadataInput, create_file_metadata
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_chunk_policy_fixture(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-052-{suffix}"
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"chunk-policy-{suffix}.md",
            stored_file_name=f"chunk-policy-{suffix}.stored.md",
            file_size_bytes=128,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/chunk-policy-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group=f"slice-052-{suffix}",
            security_level="internal",
            uploaded_by="integration-test",
            document_title=f"Chunk Policy Document {suffix}",
        ),
    )
    document_id = created.file.document_id
    assert document_id is not None

    create_chunk(
        database_url,
        ChunkInput(
            document_id=document_id,
            chunk_seq=0,
            chunk_text="Default chunk policy fixture",
            token_count=4,
        ),
    )
    create_chunk(
        database_url,
        ChunkInput(
            document_id=document_id,
            chunk_seq=1,
            chunk_text="Longer chunk policy fixture",
            chunk_policy_name="heading_1000_200",
            token_count=5,
        ),
    )
    return {
        "checksum": checksum,
        "document_id": document_id,
    }


def _cleanup_chunk_policy_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM files WHERE sha256_checksum = %s",
                (fixture["checksum"],),
            )


def test_chunk_policy_repository_api_and_page(migrated_database_url: str) -> None:
    fixture = _create_chunk_policy_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        summaries = list_chunk_policy_summaries(migrated_database_url)
        default_policy = get_chunk_policy_summary(
            migrated_database_url,
            DEFAULT_CHUNK_POLICY_NAME,
        )
        large_policy = get_chunk_policy_summary(migrated_database_url, "heading_1000_200")
        missing_policy = get_chunk_policy_summary(migrated_database_url, "missing_policy")

        with TestClient(app) as client:
            api_response = client.get("/api/chunk-policies")
            detail_response = client.get("/api/chunk-policies/heading_1000_200")
            missing_response = client.get("/api/chunk-policies/missing_policy")
            page_response = client.get("/admin/chunk-policies")

        api_payload = api_response.json()
        detail_payload = detail_response.json()["chunk_policy"]

        assert summaries[0].chunk_policy_name == DEFAULT_CHUNK_POLICY_NAME
        assert default_policy is not None
        assert default_policy.is_default is True
        assert default_policy.chunk_count >= 1
        assert default_policy.document_count >= 1
        assert large_policy is not None
        assert large_policy.chunk_count >= 1
        assert large_policy.average_token_count is not None
        assert missing_policy is None
        assert api_response.status_code == 200
        assert api_payload["default_chunk_policy_name"] == DEFAULT_CHUNK_POLICY_NAME
        assert len(api_payload["chunk_policies"]) >= 3
        assert detail_response.status_code == 200
        assert detail_payload["chunk_policy_name"] == "heading_1000_200"
        assert detail_payload["chunk_count"] >= 1
        assert missing_response.status_code == 404
        assert page_response.status_code == 200
        assert "Chunk Policies" in page_response.text
        assert "heading_1000_200" in page_response.text
        assert "Default" in page_response.text
        assert "In use" in page_response.text
    finally:
        _cleanup_chunk_policy_fixture(migrated_database_url, fixture)
