from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.extraction_runtime import ExtractionRuntimeRequest
from app.core.local_extraction import (
    LOCAL_MARKDOWN_PROFILE_NAME,
    persist_extraction_runtime_result,
    run_local_extraction,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_document(database_url: str, source_path: Path) -> tuple[int, int]:
    checksum = f"artifact-api-{uuid4()}"
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
                (file_id, f"Artifact API Fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_document_ingestion_artifact_api_and_page(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "slice-220.md"
    source_path.write_text("# Artifact API\n\nInspect this block.", encoding="utf-8")
    file_id, document_id = _create_document(migrated_database_url, source_path)
    app = create_app(Settings(database_url=migrated_database_url))

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
        artifact_id = persisted.artifacts[0].artifact_id

        with TestClient(app) as client:
            api_response = client.get(f"/api/documents/{document_id}/ingestion-artifacts")
            selected_response = client.get(
                f"/api/documents/{document_id}/ingestion-artifacts",
                params={"artifact_id": artifact_id},
            )
            missing_artifact_response = client.get(
                f"/api/documents/{document_id}/ingestion-artifacts",
                params={"artifact_id": 999_999_999},
            )
            page_response = client.get(
                f"/documents/{document_id}/artifacts",
                params={"lang": "ko", "artifact_id": artifact_id},
            )

        payload = api_response.json()
        selected_payload = selected_response.json()

        assert api_response.status_code == 200
        assert payload["document"]["document_id"] == document_id
        assert payload["selected_artifact_id"] == artifact_id
        assert payload["extraction_runs"][0]["status"] == "succeeded"
        assert payload["artifacts"][0]["artifact_id"] == artifact_id
        assert payload["artifacts"][0]["content_preview"].startswith("# Artifact API")
        assert [block["block_type"] for block in payload["blocks"]] == [
            "heading",
            "paragraph",
        ]
        assert selected_payload["selected_artifact_id"] == artifact_id
        assert missing_artifact_response.status_code == 404
        assert page_response.status_code == 200
        assert "Artifact API" in page_response.text
        assert "Blocks" in page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)
