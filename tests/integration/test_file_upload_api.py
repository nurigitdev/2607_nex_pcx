from hashlib import sha256
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.file_uploads import InvalidUploadFileNameError
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


def post_file(
    client: TestClient,
    *,
    file_name: str,
    content: bytes,
    mime_type: str,
    data_overrides: dict[str, str] | None = None,
):
    data = {
        "document_group": "slice-006",
        "security_level": "internal",
        "uploaded_by": "integration-test",
    }
    data.update(data_overrides or {})
    return client.post(
        "/api/files",
        data=data,
        files={"file": (file_name, content, mime_type)},
    )


def test_upload_file_api_stores_file_and_metadata(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    content = b"# Slice 006\n\nUpload API test."
    checksum = sha256(content).hexdigest()
    ids = seed_permission_ids(migrated_database_url)
    app = create_app(
        Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
    )

    try:
        with TestClient(app) as client:
            response = post_file(
                client,
                file_name="slice-006.md",
                content=content,
                mime_type="text/markdown",
                data_overrides={
                    "uploaded_by_user_id": str(ids["alice.member"]),
                    "owner_user_id": str(ids["chloe.teamlead"]),
                    "owner_org_unit_id": str(ids["Platform Team"]),
                    "access_scope": "team",
                },
            )

        body = response.json()
        file_payload = body["file"]
        stored_path = Path(file_payload["storage_path"])
        db_row = fetch_one(
            migrated_database_url,
            """
            SELECT
                f.file_id,
                f.sha256_checksum,
                f.document_group,
                f.uploaded_by_user_id,
                d.document_id,
                d.owner_user_id,
                d.owner_org_unit_id,
                d.access_scope,
                pj.job_id AS pipeline_job_id,
                pj.requested_by_user_id,
                pj.metadata AS pipeline_metadata,
                pj.status AS pipeline_status,
                pj.stage AS pipeline_stage
            FROM files f
            JOIN documents d ON d.file_id = f.file_id
            JOIN pipeline_jobs pj ON pj.document_id = d.document_id
            WHERE f.sha256_checksum = %s
            """,
            (checksum,),
        )
        created_event = fetch_one(
            migrated_database_url,
            """
            SELECT count(*) AS count
            FROM pipeline_job_events
            WHERE job_id = %s
              AND event_type = 'created'
            """,
            (body["pipeline_job_id"],),
        )

        assert response.status_code == 201
        assert body["duplicate"] is False
        assert body["pipeline_job_id"] == db_row["pipeline_job_id"]
        assert body["pipeline_job"] == {
            "job_id": db_row["pipeline_job_id"],
            "status": "queued",
            "stage": "upload_saved",
            "progress_percent": "0.00",
        }
        assert file_payload["original_file_name"] == "slice-006.md"
        assert file_payload["file_ext"] == ".md"
        assert file_payload["file_size_bytes"] == len(content)
        assert file_payload["sha256_checksum"] == checksum
        assert file_payload["document_group"] == "slice-006"
        assert file_payload["uploaded_by_user_id"] == ids["alice.member"]
        assert file_payload["owner_user_id"] == ids["chloe.teamlead"]
        assert file_payload["owner_org_unit_id"] == ids["Platform Team"]
        assert file_payload["access_scope"] == "team"
        assert stored_path.parent == tmp_path
        assert stored_path.read_bytes() == content
        assert db_row["file_id"] == file_payload["file_id"]
        assert db_row["document_id"] == file_payload["document_id"]
        assert db_row["uploaded_by_user_id"] == ids["alice.member"]
        assert db_row["owner_user_id"] == ids["chloe.teamlead"]
        assert db_row["owner_org_unit_id"] == ids["Platform Team"]
        assert db_row["access_scope"] == "team"
        assert db_row["requested_by_user_id"] == ids["alice.member"]
        assert db_row["pipeline_metadata"]["access_scope"] == "team"
        assert db_row["pipeline_status"] == "queued"
        assert db_row["pipeline_stage"] == "upload_saved"
        assert created_event["count"] == 1
    finally:
        cleanup_checksum(migrated_database_url, checksum)


def test_upload_file_api_returns_duplicate_for_same_checksum(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    content = b"# Duplicate\n\nSame content."
    checksum = sha256(content).hexdigest()
    app = create_app(
        Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
    )

    try:
        with TestClient(app) as client:
            created = post_file(
                client,
                file_name="first.md",
                content=content,
                mime_type="text/markdown",
            )
            duplicate = post_file(
                client,
                file_name="second.md",
                content=content,
                mime_type="text/markdown",
            )

        created_body = created.json()
        duplicate_body = duplicate.json()
        file_count = fetch_one(
            migrated_database_url,
            "SELECT count(*) AS count FROM files WHERE sha256_checksum = %s",
            (checksum,),
        )
        job_count = fetch_one(
            migrated_database_url,
            """
            SELECT count(*) AS count
            FROM pipeline_jobs pj
            JOIN files f ON f.file_id = pj.file_id
            WHERE f.sha256_checksum = %s
            """,
            (checksum,),
        )

        assert created.status_code == 201
        assert duplicate.status_code == 200
        assert created_body["pipeline_job_id"] is not None
        assert created_body["pipeline_job"]["status"] == "queued"
        assert duplicate_body["duplicate"] is True
        assert duplicate_body["pipeline_job_id"] is None
        assert duplicate_body["pipeline_job"] is None
        assert duplicate_body["file"]["file_id"] == created_body["file"]["file_id"]
        assert file_count["count"] == 1
        assert job_count["count"] == 1
        assert len(list(tmp_path.iterdir())) == 1
    finally:
        cleanup_checksum(migrated_database_url, checksum)


def test_upload_file_api_rejects_invalid_access_scope(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(database_url=migrated_database_url, upload_storage_dir=tmp_path),
    )

    with TestClient(app) as client:
        response = post_file(
            client,
            file_name="invalid-scope.md",
            content=b"# Invalid scope",
            mime_type="text/markdown",
            data_overrides={"access_scope": "private"},
        )

    assert response.status_code == 400
    assert "Unsupported access_scope" in response.json()["detail"]
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_upload_file_api_rejects_unsupported_extension(tmp_path: Path) -> None:
    app = create_app(
        Settings(database_url="postgresql://example/db", upload_storage_dir=tmp_path),
    )

    with TestClient(app) as client:
        response = post_file(
            client,
            file_name="script.exe",
            content=b"not allowed",
            mime_type="application/octet-stream",
        )

    assert response.status_code == 415
    assert "Unsupported file extension" in response.json()["detail"]
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_upload_file_api_returns_bad_request_for_invalid_upload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_store_upload(**kwargs):
        raise InvalidUploadFileNameError("file name is required")

    monkeypatch.setattr("app.main.store_upload", fake_store_upload)
    app = create_app(
        Settings(database_url="postgresql://example/db", upload_storage_dir=tmp_path),
    )

    with TestClient(app) as client:
        response = post_file(
            client,
            file_name="example.md",
            content=b"invalid upload",
            mime_type="text/markdown",
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "file name is required"
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
