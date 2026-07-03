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


def post_file(client: TestClient, *, file_name: str, content: bytes, mime_type: str):
    return client.post(
        "/api/files",
        data={
            "document_group": "slice-006",
            "security_level": "internal",
            "uploaded_by": "integration-test",
        },
        files={"file": (file_name, content, mime_type)},
    )


def test_upload_file_api_stores_file_and_metadata(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    content = b"# Slice 006\n\nUpload API test."
    checksum = sha256(content).hexdigest()
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
            )

        body = response.json()
        file_payload = body["file"]
        stored_path = Path(file_payload["storage_path"])
        db_row = fetch_one(
            migrated_database_url,
            """
            SELECT f.file_id, f.sha256_checksum, f.document_group, d.document_id
            FROM files f
            JOIN documents d ON d.file_id = f.file_id
            WHERE f.sha256_checksum = %s
            """,
            (checksum,),
        )

        assert response.status_code == 201
        assert body["duplicate"] is False
        assert file_payload["original_file_name"] == "slice-006.md"
        assert file_payload["file_ext"] == ".md"
        assert file_payload["file_size_bytes"] == len(content)
        assert file_payload["sha256_checksum"] == checksum
        assert file_payload["document_group"] == "slice-006"
        assert stored_path.parent == tmp_path
        assert stored_path.read_bytes() == content
        assert db_row["file_id"] == file_payload["file_id"]
        assert db_row["document_id"] == file_payload["document_id"]
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

        assert created.status_code == 201
        assert duplicate.status_code == 200
        assert duplicate_body["duplicate"] is True
        assert duplicate_body["file"]["file_id"] == created_body["file"]["file_id"]
        assert file_count["count"] == 1
        assert len(list(tmp_path.iterdir())) == 1
    finally:
        cleanup_checksum(migrated_database_url, checksum)


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
