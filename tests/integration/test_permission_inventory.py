from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.file_metadata import FileMetadataInput, create_file_metadata
from app.core.permission_inventory import (
    get_permission_inventory,
    get_permission_readiness_summary,
    list_permission_org_units,
    list_permission_users,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _seed_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id IN ('alice.member', 'pcx.admin')
                """)
            users = {row["login_id"]: int(row["user_id"]) for row in cursor.fetchall()}
            cursor.execute("""
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name IN ('Platform Team', 'NeX Company')
                """)
            orgs = {row["org_unit_name"]: int(row["org_unit_id"]) for row in cursor.fetchall()}
    return {**users, **orgs}


def _create_permission_inventory_document(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-055-{suffix}"
    ids = _seed_ids(database_url)
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"permission-inventory-{suffix}.md",
            stored_file_name=f"permission-inventory-{suffix}.stored.md",
            file_size_bytes=55,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/permission-inventory-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group="slice-055",
            security_level="internal",
            uploaded_by="permission-inventory-test",
            document_title=f"Permission Inventory {suffix}",
        ),
    )
    assert created.file.document_id is not None
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE files
                SET uploaded_by_user_id = %s
                WHERE file_id = %s
                """,
                (ids["alice.member"], created.file.file_id),
            )
            cursor.execute(
                """
                UPDATE documents
                SET owner_user_id = %s,
                    owner_org_unit_id = %s,
                    access_scope = 'team'
                WHERE document_id = %s
                """,
                (ids["alice.member"], ids["Platform Team"], created.file.document_id),
            )
    return {
        "checksum": checksum,
        "document_id": created.file.document_id,
        "file_id": created.file.file_id,
    }


def _cleanup_permission_inventory_document(database_url: str, checksum: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE sha256_checksum = %s", (checksum,))


def _create_permission_readiness_issue_document(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-060-{suffix}"
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"permission-readiness-{suffix}.md",
            stored_file_name=f"permission-readiness-{suffix}.stored.md",
            file_size_bytes=60,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/permission-readiness-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group="slice-060",
            security_level="internal",
            uploaded_by="permission-readiness-test",
            document_title=f"Permission Readiness {suffix}",
        ),
    )
    assert created.file.document_id is not None
    return {
        "checksum": checksum,
        "document_id": created.file.document_id,
        "file_id": created.file.file_id,
    }


def test_permission_inventory_repository_api_and_page(migrated_database_url: str) -> None:
    fixture = _create_permission_inventory_document(migrated_database_url)
    issue_fixture = _create_permission_readiness_issue_document(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        inventory = get_permission_inventory(migrated_database_url)
        readiness = get_permission_readiness_summary(migrated_database_url)
        users = list_permission_users(migrated_database_url)
        org_units = list_permission_org_units(migrated_database_url)
        users_by_login = {user.login_id: user for user in users}
        orgs_by_name = {org.org_unit_name: org for org in org_units}

        with TestClient(app) as client:
            inventory_response = client.get("/api/admin/permissions")
            readiness_response = client.get("/api/admin/permissions/readiness")
            users_response = client.get("/api/admin/permissions/users", params={"limit": 2})
            orgs_response = client.get("/api/admin/permissions/org-units")
            memberships_response = client.get("/api/admin/permissions/memberships")
            bad_users_response = client.get(
                "/api/admin/permissions/users",
                params={"limit": 0},
            )
            bad_readiness_response = client.get(
                "/api/admin/permissions/readiness",
                params={"issue_limit": 0},
            )
            page_response = client.get("/admin/permissions")
            english_page_response = client.get("/admin/permissions?lang=en")

        payload = inventory_response.json()["permission_inventory"]
        readiness_payload = readiness_response.json()["permission_readiness"]
        access_counts = {
            scope["access_scope"]: scope["document_count"]
            for scope in payload["summary"]["access_scope_counts"]
        }

        assert inventory.summary.active_user_count >= 5
        assert readiness.issue_document_count >= 1
        assert readiness.missing_uploader_count >= 1
        assert readiness.personal_missing_owner_count >= 1
        assert any(issue.document_id == issue_fixture["document_id"] for issue in readiness.issues)
        assert inventory.summary.org_unit_count >= 5
        assert inventory.summary.membership_count >= 5
        assert users_by_login["pcx.admin"].managed_org_unit_count >= 5
        assert users_by_login["alice.member"].primary_role_name == "member"
        assert users_by_login["alice.member"].primary_org_unit_name == "Platform Team"
        assert users_by_login["alice.member"].managed_org_unit_count == 0
        assert users_by_login["alice.member"].owned_document_count >= 1
        assert users_by_login["alice.member"].uploaded_file_count >= 1
        assert orgs_by_name["Platform Team"].membership_count >= 2
        assert orgs_by_name["Platform Team"].owned_document_count >= 1
        assert inventory_response.status_code == 200
        assert readiness_response.status_code == 200
        assert readiness_payload["document_count"] >= 2
        assert readiness_payload["issue_document_count"] >= 1
        assert any(
            issue["document_id"] == issue_fixture["document_id"]
            and "missing_uploader" in issue["issue_codes"]
            and "personal_missing_owner" in issue["issue_codes"]
            for issue in readiness_payload["issues"]
        )
        assert payload["summary"]["active_user_count"] >= 5
        assert access_counts["team"] >= 1
        assert any(user["login_id"] == "alice.member" for user in payload["users"])
        assert users_response.status_code == 200
        assert len(users_response.json()["users"]) == 2
        assert orgs_response.status_code == 200
        assert any(
            org_unit["org_unit_name"] == "Platform Team"
            for org_unit in orgs_response.json()["org_units"]
        )
        assert memberships_response.status_code == 200
        assert any(
            membership["role_name"] == "team_lead"
            for membership in memberships_response.json()["memberships"]
        )
        assert bad_users_response.status_code == 400
        assert bad_readiness_response.status_code == 400
        assert page_response.status_code == 200
        assert "권한 시뮬레이션" in page_response.text
        assert "권한 Metadata 준비도" in page_response.text
        assert "업로더 누락" in page_response.text
        assert "alice.member" in page_response.text
        assert "Platform Team" in page_response.text
        assert english_page_response.status_code == 200
        assert "Permission Simulation" in english_page_response.text
        assert "Permission Metadata Readiness" in english_page_response.text
        assert "Managed orgs" in english_page_response.text
    finally:
        _cleanup_permission_inventory_document(migrated_database_url, str(fixture["checksum"]))
        _cleanup_permission_inventory_document(
            migrated_database_url,
            str(issue_fixture["checksum"]),
        )
