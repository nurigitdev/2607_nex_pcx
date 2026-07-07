from hashlib import sha256
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.main import create_app


def cleanup_checksum(database_url: str, checksum: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE sha256_checksum = %s", (checksum,))


def seed_permission_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id IN ('alice.member', 'chloe.teamlead')
                """)
            users = {row["login_id"]: int(row["user_id"]) for row in cursor.fetchall()}
            cursor.execute("""
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name = 'Platform Team'
                """)
            orgs = {row["org_unit_name"]: int(row["org_unit_id"]) for row in cursor.fetchall()}
    return {**users, **orgs}


def post_upload_form(
    client: TestClient,
    *,
    file_name: str,
    content: bytes,
    mime_type: str,
    data_overrides: dict[str, str] | None = None,
):
    data = {
        "document_group": "ui-slice-007",
        "security_level": "restricted",
        "uploaded_by": "ui-test",
    }
    data.update(data_overrides or {})
    return client.post(
        "/files/upload",
        data=data,
        files={"file": (file_name, content, mime_type)},
    )


def test_file_upload_ui_stores_file_and_shows_result(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    content = b"# Slice 007\n\nUpload UI test."
    checksum = sha256(content).hexdigest()
    ids = seed_permission_ids(migrated_database_url)
    app = create_app(
        Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
    )

    try:
        with TestClient(app) as client:
            response = post_upload_form(
                client,
                file_name="slice-007.md",
                content=content,
                mime_type="text/markdown",
                data_overrides={
                    "uploaded_by_user_id": str(ids["alice.member"]),
                    "owner_user_id": str(ids["chloe.teamlead"]),
                    "owner_org_unit_id": str(ids["Platform Team"]),
                    "access_scope": "team",
                },
            )

        db_row = fetch_one(
            migrated_database_url,
            """
            SELECT
                f.file_id,
                f.sha256_checksum,
                f.document_group,
                f.security_level,
                f.uploaded_by_user_id,
                d.owner_user_id,
                d.owner_org_unit_id,
                d.access_scope,
                pj.job_id AS pipeline_job_id
            FROM files f
            JOIN documents d ON d.file_id = f.file_id
            JOIN pipeline_jobs pj ON pj.file_id = f.file_id
            WHERE f.sha256_checksum = %s
            """,
            (checksum,),
        )

        assert response.status_code == 200
        assert "File uploaded and metadata stored." in response.text
        assert "Stored" in response.text
        assert "Pipeline Job ID" in response.text
        assert "Pipeline Status" in response.text
        assert "queued" in response.text
        assert str(db_row["pipeline_job_id"]) in response.text
        assert "slice-007.md" in response.text
        assert checksum in response.text
        assert "접근 범위" in response.text
        assert str(ids["alice.member"]) in response.text
        assert str(ids["chloe.teamlead"]) in response.text
        assert str(ids["Platform Team"]) in response.text
        assert db_row["document_group"] == "ui-slice-007"
        assert db_row["security_level"] == "restricted"
        assert db_row["uploaded_by_user_id"] == ids["alice.member"]
        assert db_row["owner_user_id"] == ids["chloe.teamlead"]
        assert db_row["owner_org_unit_id"] == ids["Platform Team"]
        assert db_row["access_scope"] == "team"
        assert len(list(tmp_path.iterdir())) == 1
    finally:
        cleanup_checksum(migrated_database_url, checksum)


def test_file_upload_ui_shows_duplicate_result(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    content = b"# Duplicate UI\n\nSame content."
    checksum = sha256(content).hexdigest()
    app = create_app(
        Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
    )

    try:
        with TestClient(app) as client:
            created = post_upload_form(
                client,
                file_name="first.md",
                content=content,
                mime_type="text/markdown",
            )
            duplicate = post_upload_form(
                client,
                file_name="second.md",
                content=content,
                mime_type="text/markdown",
            )

        file_count = fetch_one(
            migrated_database_url,
            "SELECT count(*) AS count FROM files WHERE sha256_checksum = %s",
            (checksum,),
        )

        assert created.status_code == 200
        assert duplicate.status_code == 200
        assert "Duplicate checksum detected. Existing file metadata was returned." in duplicate.text
        assert file_count["count"] == 1
        assert len(list(tmp_path.iterdir())) == 1
    finally:
        cleanup_checksum(migrated_database_url, checksum)


def test_file_upload_ui_shows_unsupported_extension_error(tmp_path: Path) -> None:
    app = create_app(
        Settings(database_url="postgresql://example/db", upload_storage_dir=tmp_path),
    )

    with TestClient(app) as client:
        response = post_upload_form(
            client,
            file_name="script.exe",
            content=b"not allowed",
            mime_type="application/octet-stream",
        )

    assert response.status_code == 200
    assert "Unsupported file extension: .exe" in response.text
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
