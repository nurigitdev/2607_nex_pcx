from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def test_identity_permission_tables_and_seed_rows(migrated_database_url: str) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('app_users', 'org_units', 'user_org_memberships')
        """,
    )
    active_users = fetch_one(
        migrated_database_url,
        "SELECT count(*) AS count FROM app_users WHERE is_active",
    )
    membership_roles = fetch_one(
        migrated_database_url,
        """
        SELECT count(DISTINCT role_name) AS count
        FROM user_org_memberships
        WHERE role_name IN ('member', 'team_lead', 'group_lead', 'admin')
        """,
    )
    company_org = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM org_units
        WHERE org_unit_type = 'company'
        """,
    )

    assert table_count["count"] == 3
    assert active_users["count"] >= 5
    assert membership_roles["count"] == 4
    assert company_org["count"] >= 1


def test_org_units_preserve_seed_hierarchy(migrated_database_url: str) -> None:
    platform_team = fetch_one(
        migrated_database_url,
        """
        SELECT parent.org_unit_name AS parent_name
        FROM org_units child
        JOIN org_units parent ON parent.org_unit_id = child.parent_org_unit_id
        WHERE child.org_unit_name = 'Platform Team'
        """,
    )
    group_parent = fetch_one(
        migrated_database_url,
        """
        SELECT parent.org_unit_type AS parent_type
        FROM org_units child
        JOIN org_units parent ON parent.org_unit_id = child.parent_org_unit_id
        WHERE child.org_unit_name = 'Platform Group'
        """,
    )

    assert platform_team["parent_name"] == "Platform Group"
    assert group_parent["parent_type"] == "division"


def test_documents_default_to_personal_access_scope(migrated_database_url: str) -> None:
    checksum = f"permission-default-{uuid4()}"
    with connect(migrated_database_url) as connection:
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
                VALUES (%s, 'Permission default')
                RETURNING access_scope, permission_metadata
                """,
                (file_id,),
            )
            document_row = cursor.fetchone()
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))

    assert document_row["access_scope"] == "personal"
    assert document_row["permission_metadata"] == {}


def test_files_and_documents_reference_identity_metadata(
    migrated_database_url: str,
) -> None:
    checksum = f"permission-reference-{uuid4()}"
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                "SELECT org_unit_id FROM org_units WHERE org_unit_name = 'Platform Team'"
            )
            org_unit_id = cursor.fetchone()["org_unit_id"]
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    uploaded_by_user_id
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    user_id,
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    owner_user_id,
                    owner_org_unit_id,
                    access_scope,
                    permission_metadata
                )
                VALUES (%s, 'Team document', %s, %s, 'team', '{"fixture": true}')
                RETURNING owner_user_id, owner_org_unit_id, access_scope, permission_metadata
                """,
                (file_id, user_id, org_unit_id),
            )
            document_row = cursor.fetchone()
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))

    assert document_row["owner_user_id"] == user_id
    assert document_row["owner_org_unit_id"] == org_unit_id
    assert document_row["access_scope"] == "team"
    assert document_row["permission_metadata"] == {"fixture": True}


def test_permission_check_constraints(migrated_database_url: str) -> None:
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute("""
                    INSERT INTO org_units (org_unit_name, org_unit_type)
                    VALUES ('Invalid Org', 'squad')
                    """)
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                "SELECT org_unit_id FROM org_units WHERE org_unit_name = 'Platform Team'"
            )
            org_unit_id = cursor.fetchone()["org_unit_id"]
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO user_org_memberships (user_id, org_unit_id, role_name)
                    VALUES (%s, %s, 'owner')
                    """,
                    (user_id, org_unit_id),
                )
        connection.rollback()

        checksum = f"invalid-access-scope-{uuid4()}"
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
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO documents (file_id, document_title, access_scope)
                    VALUES (%s, 'Invalid scope', 'department')
                    """,
                    (file_id,),
                )
        connection.rollback()


def test_user_delete_cascades_memberships(migrated_database_url: str) -> None:
    login_id = f"cascade-{uuid4()}"
    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO app_users (login_id, display_name)
                VALUES (%s, 'Cascade User')
                RETURNING user_id
                """,
                (login_id,),
            )
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                "SELECT org_unit_id FROM org_units WHERE org_unit_name = 'Platform Team'"
            )
            org_unit_id = cursor.fetchone()["org_unit_id"]
            cursor.execute(
                """
                INSERT INTO user_org_memberships (user_id, org_unit_id, role_name)
                VALUES (%s, %s, 'member')
                """,
                (user_id, org_unit_id),
            )
            cursor.execute("DELETE FROM app_users WHERE user_id = %s", (user_id,))
            cursor.execute(
                """
                SELECT count(*) AS count
                FROM user_org_memberships
                WHERE user_id = %s
                """,
                (user_id,),
            )
            membership_count = cursor.fetchone()

    assert membership_count["count"] == 0
