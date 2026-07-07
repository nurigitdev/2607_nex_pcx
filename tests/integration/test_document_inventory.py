from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.chunks import ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.document_inventory import list_document_inventory
from app.core.file_metadata import (
    FileMetadataInput,
    create_file_metadata,
    mark_file_parse_succeeded,
)
from app.core.pipeline_jobs import PipelineJobInput, create_pipeline_job, mark_pipeline_succeeded
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_document_fixture(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-049-{suffix}"
    group = f"slice-049-{suffix}"
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"inventory-{suffix}.md",
            stored_file_name=f"inventory-{suffix}.stored.md",
            file_size_bytes=84,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/inventory-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group=group,
            security_level="internal",
            uploaded_by="integration-test",
            document_title=f"Inventory Document {suffix}",
        ),
    )
    document_id = created.file.document_id
    assert document_id is not None

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                "SELECT org_unit_id FROM org_units WHERE org_unit_name = 'Platform Team'",
            )
            org_unit_id = cursor.fetchone()["org_unit_id"]
            cursor.execute(
                """
                UPDATE files
                SET uploaded_by_user_id = %s
                WHERE file_id = %s
                """,
                (user_id, created.file.file_id),
            )
            cursor.execute(
                """
                UPDATE documents
                SET owner_user_id = %s,
                    owner_org_unit_id = %s,
                    access_scope = 'team'
                WHERE document_id = %s
                """,
                (user_id, org_unit_id, document_id),
            )

    create_chunk(
        database_url,
        ChunkInput(
            document_id=document_id,
            chunk_seq=0,
            chunk_text="Inventory first chunk",
            token_count=3,
        ),
    )
    create_chunk(
        database_url,
        ChunkInput(
            document_id=document_id,
            chunk_seq=1,
            chunk_text="Inventory second chunk",
            token_count=4,
        ),
    )
    mark_file_parse_succeeded(
        database_url,
        created.file.file_id,
        parser_name="markdown",
        parser_version="slice-049",
        extracted_text_size=42,
    )
    job = create_pipeline_job(
        database_url,
        PipelineJobInput(
            job_type="document_ingestion",
            file_id=created.file.file_id,
            document_id=document_id,
            total_units=2,
        ),
    )
    completed_job = mark_pipeline_succeeded(database_url, job.job_id, message="Inventory ready")
    assert completed_job is not None

    return {
        "checksum": checksum,
        "document_group": group,
        "document_id": document_id,
        "file_id": created.file.file_id,
        "job_id": completed_job.job_id,
    }


def _cleanup_document_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM files WHERE sha256_checksum = %s",
                (fixture["checksum"],),
            )


def _permission_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id IN ('alice.member', 'bob.member', 'pcx.admin')
                """)
            users = {row["login_id"]: int(row["user_id"]) for row in cursor.fetchall()}
            cursor.execute("""
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name IN ('Platform Team', 'Business Team')
                """)
            orgs = {row["org_unit_name"]: int(row["org_unit_id"]) for row in cursor.fetchall()}
    return {**users, **orgs}


def test_document_inventory_repository_api_and_page(migrated_database_url: str) -> None:
    fixture = _create_document_fixture(migrated_database_url)
    ids = _permission_ids(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        documents = list_document_inventory(
            migrated_database_url,
            document_group=str(fixture["document_group"]),
        )

        with TestClient(app) as client:
            api_response = client.get(
                "/api/documents",
                params={"document_group": fixture["document_group"]},
            )
            detail_response = client.get(f"/api/documents/{fixture['document_id']}")
            detail_page_response = client.get(f"/documents/{fixture['document_id']}")
            bad_detail_response = client.get(
                f"/api/documents/{fixture['document_id']}",
                params={"chunk_policy_name": " "},
            )
            missing_detail_response = client.get("/api/documents/999999999")
            filtered_response = client.get(
                "/api/documents",
                params={"parse_status": "failed", "document_group": fixture["document_group"]},
            )
            bad_response = client.get("/api/documents", params={"limit": 0})
            page_response = client.get(
                "/documents",
                params={"document_group": fixture["document_group"]},
            )
            api_permission_update_response = client.put(
                f"/api/documents/{fixture['document_id']}/permissions",
                json={
                    "owner_user_id": ids["bob.member"],
                    "owner_org_unit_id": ids["Business Team"],
                    "access_scope": "org_tree",
                    "updated_by_user_id": ids["pcx.admin"],
                },
            )
            html_permission_update_response = client.post(
                f"/documents/{fixture['document_id']}/permissions",
                data={
                    "owner_user_id": str(ids["alice.member"]),
                    "owner_org_unit_id": str(ids["Platform Team"]),
                    "access_scope": "company",
                    "updated_by_user_id": str(ids["pcx.admin"]),
                },
            )
            invalid_permission_update_response = client.put(
                f"/api/documents/{fixture['document_id']}/permissions",
                json={"access_scope": "private"},
            )
            missing_permission_update_response = client.put(
                "/api/documents/999999999/permissions",
                json={"access_scope": "personal"},
            )

        item = documents[0]
        payload = api_response.json()["documents"][0]
        detail_payload = detail_response.json()
        api_permission_payload = api_permission_update_response.json()["document"]

        assert len(documents) == 1
        assert item.document_id == fixture["document_id"]
        assert item.file_id == fixture["file_id"]
        assert item.document_group == fixture["document_group"]
        assert item.parse_status == "succeeded"
        assert item.owner_login_id == "alice.member"
        assert item.owner_org_unit_name == "Platform Team"
        assert item.access_scope == "team"
        assert item.chunk_count == 2
        assert item.total_token_count == 7
        assert item.latest_pipeline_job_id == fixture["job_id"]
        assert item.latest_pipeline_status == "succeeded"
        assert api_response.status_code == 200
        assert payload["document_id"] == fixture["document_id"]
        assert payload["chunk_count"] == 2
        assert payload["latest_pipeline_status"] == "succeeded"
        assert detail_response.status_code == 200
        assert detail_payload["document"]["document_id"] == fixture["document_id"]
        assert [chunk["chunk_seq"] for chunk in detail_payload["chunks"]] == [0, 1]
        assert detail_payload["chunks"][0]["chunk_text"] == "Inventory first chunk"
        assert bad_detail_response.status_code == 400
        assert missing_detail_response.status_code == 404
        assert filtered_response.status_code == 200
        assert filtered_response.json()["documents"] == []
        assert bad_response.status_code == 400
        assert page_response.status_code == 200
        assert "Documents" in page_response.text
        assert "Inventory Document" in page_response.text
        assert str(fixture["job_id"]) in page_response.text
        assert f'href="/documents/{fixture["document_id"]}"' in page_response.text
        assert detail_page_response.status_code == 200
        assert "Metadata" in detail_page_response.text
        assert "Chunks" in detail_page_response.text
        assert "문서 권한 편집" in detail_page_response.text
        assert "Inventory first chunk" in detail_page_response.text
        assert api_permission_update_response.status_code == 200
        assert api_permission_payload["owner_login_id"] == "bob.member"
        assert api_permission_payload["owner_org_unit_name"] == "Business Team"
        assert api_permission_payload["access_scope"] == "org_tree"
        assert html_permission_update_response.status_code == 200
        assert "문서 권한 metadata가 저장되었습니다." in html_permission_update_response.text
        assert "company" in html_permission_update_response.text
        assert invalid_permission_update_response.status_code == 400
        assert missing_permission_update_response.status_code == 404
    finally:
        _cleanup_document_fixture(migrated_database_url, fixture)
