from uuid import uuid4

import pytest

from app.core.database import connect, fetch_one
from app.core.file_metadata import (
    FileMetadataInput,
    create_file_metadata,
    create_file_metadata_in_connection,
    find_file_by_checksum,
)

pytestmark = pytest.mark.integration


def make_metadata(checksum: str, **overrides) -> FileMetadataInput:
    values = {
        "original_file_name": f"{checksum}.md",
        "stored_file_name": f"{checksum}.stored.md",
        "file_size_bytes": 42,
        "sha256_checksum": checksum,
        "storage_path": f"/tmp/nex_pcx/{checksum}.stored.md",
        "mime_type": "text/markdown",
        "document_group": "slice-005",
        "security_level": "internal",
        "uploaded_by": "integration-test",
        "document_title": f"Document {checksum}",
    }
    values.update(overrides)
    return FileMetadataInput(**values)


def cleanup_checksum(database_url: str, checksum: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE sha256_checksum = %s", (checksum,))


def test_create_file_metadata_creates_file_and_document(migrated_database_url: str) -> None:
    checksum = f"file-metadata-{uuid4()}"
    try:
        result = create_file_metadata(migrated_database_url, make_metadata(checksum))

        file_row = fetch_one(
            migrated_database_url,
            """
            SELECT file_id, document_group, security_level, parse_status
            FROM files
            WHERE sha256_checksum = %s
            """,
            (checksum,),
        )
        document_row = fetch_one(
            migrated_database_url,
            """
            SELECT document_id, file_id, document_group, security_level, document_status
            FROM documents
            WHERE file_id = %s
            """,
            (result.file.file_id,),
        )

        assert not result.duplicate
        assert result.file.document_id == document_row["document_id"]
        assert result.file.file_ext == ".md"
        assert result.file.document_group == "slice-005"
        assert file_row["parse_status"] == "pending"
        assert document_row["document_status"] == "active"
        assert document_row["document_group"] == file_row["document_group"]
        assert document_row["security_level"] == file_row["security_level"]
    finally:
        cleanup_checksum(migrated_database_url, checksum)


def test_create_file_metadata_returns_duplicate_for_existing_checksum(
    migrated_database_url: str,
) -> None:
    checksum = f"duplicate-{uuid4()}"
    try:
        created = create_file_metadata(migrated_database_url, make_metadata(checksum))
        duplicate = create_file_metadata(
            migrated_database_url,
            make_metadata(
                checksum,
                original_file_name="same-content.md",
                stored_file_name="same-content.stored.md",
                storage_path="/tmp/nex_pcx/same-content.stored.md",
            ),
        )

        file_count = fetch_one(
            migrated_database_url,
            "SELECT count(*) AS count FROM files WHERE sha256_checksum = %s",
            (checksum,),
        )
        document_count = fetch_one(
            migrated_database_url,
            "SELECT count(*) AS count FROM documents WHERE file_id = %s",
            (created.file.file_id,),
        )

        assert not created.duplicate
        assert duplicate.duplicate
        assert duplicate.file.file_id == created.file.file_id
        assert duplicate.file.document_id == created.file.document_id
        assert file_count["count"] == 1
        assert document_count["count"] == 1
    finally:
        cleanup_checksum(migrated_database_url, checksum)


def test_find_file_by_checksum_returns_none_when_missing(migrated_database_url: str) -> None:
    with connect(migrated_database_url) as connection:
        assert find_file_by_checksum(connection, f"missing-{uuid4()}") is None


def test_create_file_metadata_in_connection_uses_transaction(
    migrated_database_url: str,
) -> None:
    checksum = f"transaction-{uuid4()}"
    try:
        with connect(migrated_database_url) as connection:
            result = create_file_metadata_in_connection(connection, make_metadata(checksum))

        stored = fetch_one(
            migrated_database_url,
            """
            SELECT f.file_id, d.document_id
            FROM files f
            JOIN documents d ON d.file_id = f.file_id
            WHERE f.sha256_checksum = %s
            """,
            (checksum,),
        )

        assert stored["file_id"] == result.file.file_id
        assert stored["document_id"] == result.file.document_id
    finally:
        cleanup_checksum(migrated_database_url, checksum)
