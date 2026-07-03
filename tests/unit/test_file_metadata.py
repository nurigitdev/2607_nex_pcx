from pathlib import Path

import pytest

from app.core.file_metadata import (
    FileMetadataInput,
    InvalidFileMetadataError,
    UnsupportedFileExtensionError,
    calculate_sha256,
    normalize_file_ext,
    validate_file_metadata,
)


def make_metadata(**overrides) -> FileMetadataInput:
    values = {
        "original_file_name": "example.md",
        "stored_file_name": "example-stored.md",
        "file_size_bytes": 12,
        "sha256_checksum": "abc123",
        "storage_path": "/tmp/example-stored.md",
        "mime_type": "text/markdown",
    }
    values.update(overrides)
    return FileMetadataInput(**values)


def test_calculate_sha256_reads_file_in_chunks(tmp_path: Path) -> None:
    source = tmp_path / "example.md"
    source.write_bytes(b"hello nex pcx")
    expected = "924c812cdb689ceadc8e0aab86bbfcaf0763ac5239e28b82522057f11ed88cb5"

    assert calculate_sha256(source) == expected


def test_normalize_file_ext_lowercases_suffix() -> None:
    assert normalize_file_ext("Report.PDF") == ".pdf"


def test_validate_file_metadata_accepts_supported_extension() -> None:
    assert validate_file_metadata(make_metadata(original_file_name="report.DOCX")) == ".docx"


def test_validate_file_metadata_rejects_missing_name() -> None:
    with pytest.raises(InvalidFileMetadataError, match="original_file_name"):
        validate_file_metadata(make_metadata(original_file_name=" "))


def test_validate_file_metadata_rejects_missing_stored_name() -> None:
    with pytest.raises(InvalidFileMetadataError, match="stored_file_name"):
        validate_file_metadata(make_metadata(stored_file_name=" "))


def test_validate_file_metadata_rejects_negative_size() -> None:
    with pytest.raises(InvalidFileMetadataError, match="file_size_bytes"):
        validate_file_metadata(make_metadata(file_size_bytes=-1))


def test_validate_file_metadata_rejects_missing_checksum() -> None:
    with pytest.raises(InvalidFileMetadataError, match="sha256_checksum"):
        validate_file_metadata(make_metadata(sha256_checksum=" "))


def test_validate_file_metadata_rejects_missing_storage_path() -> None:
    with pytest.raises(InvalidFileMetadataError, match="storage_path"):
        validate_file_metadata(make_metadata(storage_path=" "))


def test_validate_file_metadata_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileExtensionError, match="Unsupported file extension"):
        validate_file_metadata(make_metadata(original_file_name="script.exe"))
