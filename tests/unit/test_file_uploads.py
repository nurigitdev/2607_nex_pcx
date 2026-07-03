from hashlib import sha256
from io import BytesIO

import pytest

from app.core.file_metadata import (
    CreateFileMetadataResult,
    FileMetadataRecord,
    UnsupportedFileExtensionError,
)
from app.core.file_uploads import (
    InvalidUploadFileNameError,
    build_stored_file_name,
    sanitize_upload_file_name,
    store_upload,
    write_stream_with_checksum,
)


def make_record(**overrides) -> FileMetadataRecord:
    values = {
        "file_id": 1,
        "document_id": 10,
        "original_file_name": "example.md",
        "stored_file_name": "stored.md",
        "file_ext": ".md",
        "mime_type": "text/markdown",
        "file_size_bytes": 12,
        "sha256_checksum": "abc123",
        "storage_path": "/tmp/stored.md",
        "document_group": "default",
        "security_level": "internal",
        "parse_status": "pending",
    }
    values.update(overrides)
    return FileMetadataRecord(**values)


def test_sanitize_upload_file_name_uses_basename() -> None:
    assert sanitize_upload_file_name("../reports/example.md") == "example.md"


def test_sanitize_upload_file_name_rejects_empty_name() -> None:
    with pytest.raises(InvalidUploadFileNameError, match="file name"):
        sanitize_upload_file_name(" ")


def test_sanitize_upload_file_name_rejects_missing_name() -> None:
    with pytest.raises(InvalidUploadFileNameError, match="file name"):
        sanitize_upload_file_name(None)


def test_build_stored_file_name_keeps_supported_extension() -> None:
    stored_file_name = build_stored_file_name("Report.PDF")

    assert stored_file_name.endswith(".pdf")


def test_build_stored_file_name_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileExtensionError, match="Unsupported file extension"):
        build_stored_file_name("script.exe")


def test_write_stream_with_checksum_writes_file(tmp_path) -> None:
    target_path = tmp_path / "stored.md"
    content = b"hello upload"

    file_size, checksum = write_stream_with_checksum(BytesIO(content), target_path)

    assert target_path.read_bytes() == content
    assert file_size == len(content)
    assert checksum == sha256(content).hexdigest()


def test_store_upload_creates_metadata_and_file(monkeypatch, tmp_path) -> None:
    captured_metadata = None

    def fake_create_file_metadata(database_url, metadata):
        nonlocal captured_metadata
        captured_metadata = metadata
        record = make_record(
            original_file_name=metadata.original_file_name,
            stored_file_name=metadata.stored_file_name,
            file_size_bytes=metadata.file_size_bytes,
            sha256_checksum=metadata.sha256_checksum,
            storage_path=metadata.storage_path,
            document_group=metadata.document_group,
            security_level=metadata.security_level,
        )
        return CreateFileMetadataResult(file=record, duplicate=False)

    monkeypatch.setattr("app.core.file_uploads.create_file_metadata", fake_create_file_metadata)

    result = store_upload(
        database_url="postgresql://example/db",
        upload_stream=BytesIO(b"hello upload"),
        original_file_name="example.md",
        storage_dir=tmp_path,
        mime_type="text/markdown",
        document_group="docs",
        security_level="restricted",
        uploaded_by="tester",
    )

    assert not result.duplicate
    assert captured_metadata is not None
    assert captured_metadata.document_title == "example"
    assert captured_metadata.uploaded_by == "tester"
    assert (tmp_path / captured_metadata.stored_file_name).read_bytes() == b"hello upload"


def test_store_upload_removes_new_file_when_duplicate(monkeypatch, tmp_path) -> None:
    def fake_create_file_metadata(database_url, metadata):
        record = make_record(
            original_file_name="existing.md",
            stored_file_name="existing.md",
            storage_path="/tmp/existing.md",
            sha256_checksum=metadata.sha256_checksum,
        )
        return CreateFileMetadataResult(file=record, duplicate=True)

    monkeypatch.setattr("app.core.file_uploads.create_file_metadata", fake_create_file_metadata)

    result = store_upload(
        database_url="postgresql://example/db",
        upload_stream=BytesIO(b"same content"),
        original_file_name="duplicate.md",
        storage_dir=tmp_path,
    )

    assert result.duplicate
    assert list(tmp_path.iterdir()) == []


def test_store_upload_removes_new_file_when_metadata_write_fails(monkeypatch, tmp_path) -> None:
    def fake_create_file_metadata(database_url, metadata):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("app.core.file_uploads.create_file_metadata", fake_create_file_metadata)

    with pytest.raises(RuntimeError, match="database unavailable"):
        store_upload(
            database_url="postgresql://example/db",
            upload_stream=BytesIO(b"content"),
            original_file_name="example.md",
            storage_dir=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []
