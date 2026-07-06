from pathlib import Path

import pytest

from app.core.file_metadata import (
    FileMetadataInput,
    InvalidFileMetadataError,
    UnsupportedFileExtensionError,
    calculate_sha256,
    create_file_metadata_in_connection,
    get_file_metadata_in_connection,
    mark_file_parse_failed_in_connection,
    mark_file_parse_running_in_connection,
    mark_file_parse_succeeded_in_connection,
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


def test_get_file_metadata_rejects_non_positive_id_before_db() -> None:
    with pytest.raises(InvalidFileMetadataError, match="file_id"):
        get_file_metadata_in_connection(None, 0)  # type: ignore[arg-type]


def test_mark_file_parse_running_validates_inputs_before_db() -> None:
    with pytest.raises(InvalidFileMetadataError, match="parser_name"):
        mark_file_parse_running_in_connection(
            None,  # type: ignore[arg-type]
            1,
            parser_name=" ",
            parser_version="0.1.0",
        )

    with pytest.raises(InvalidFileMetadataError, match="parser_version"):
        mark_file_parse_running_in_connection(
            None,  # type: ignore[arg-type]
            1,
            parser_name="markdown",
            parser_version=" ",
        )


def test_mark_file_parse_succeeded_validates_inputs_before_db() -> None:
    with pytest.raises(InvalidFileMetadataError, match="parser_name"):
        mark_file_parse_succeeded_in_connection(
            None,  # type: ignore[arg-type]
            1,
            parser_name=" ",
            parser_version="0.1.0",
            extracted_text_size=1,
        )

    with pytest.raises(InvalidFileMetadataError, match="parser_version"):
        mark_file_parse_succeeded_in_connection(
            None,  # type: ignore[arg-type]
            1,
            parser_name="markdown",
            parser_version=" ",
            extracted_text_size=1,
        )

    with pytest.raises(InvalidFileMetadataError, match="extracted_text_size"):
        mark_file_parse_succeeded_in_connection(
            None,  # type: ignore[arg-type]
            1,
            parser_name="markdown",
            parser_version="0.1.0",
            extracted_text_size=-1,
        )


def test_mark_file_parse_failed_validates_inputs_before_db() -> None:
    with pytest.raises(InvalidFileMetadataError, match="error_message"):
        mark_file_parse_failed_in_connection(
            None,  # type: ignore[arg-type]
            1,
            error_message=" ",
        )

    with pytest.raises(InvalidFileMetadataError, match="parser_name"):
        mark_file_parse_failed_in_connection(
            None,  # type: ignore[arg-type]
            1,
            error_message="failed",
            parser_name=" ",
        )

    with pytest.raises(InvalidFileMetadataError, match="parser_version"):
        mark_file_parse_failed_in_connection(
            None,  # type: ignore[arg-type]
            1,
            error_message="failed",
            parser_version=" ",
        )


class _RowCountZeroCursor:
    rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, *args, **kwargs) -> None:
        return None


class _RowCountZeroConnection:
    def cursor(self):
        return _RowCountZeroCursor()


def test_parse_status_updates_return_none_when_file_is_missing() -> None:
    connection = _RowCountZeroConnection()

    assert (
        mark_file_parse_running_in_connection(
            connection,  # type: ignore[arg-type]
            1,
            parser_name="markdown",
            parser_version="0.1.0",
        )
        is None
    )
    assert (
        mark_file_parse_succeeded_in_connection(
            connection,  # type: ignore[arg-type]
            1,
            parser_name="markdown",
            parser_version="0.1.0",
            extracted_text_size=42,
        )
        is None
    )
    assert (
        mark_file_parse_failed_in_connection(
            connection,  # type: ignore[arg-type]
            1,
            error_message="failed",
        )
        is None
    )


class _InsertConflictCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, *args, **kwargs) -> None:
        return None

    def fetchone(self):
        return None


class _InsertConflictConnection:
    def cursor(self):
        return _InsertConflictCursor()


def test_create_file_metadata_raises_when_conflict_row_cannot_be_found(monkeypatch) -> None:
    monkeypatch.setattr("app.core.file_metadata.find_file_by_checksum", lambda *args: None)

    with pytest.raises(RuntimeError, match="Duplicate checksum"):
        create_file_metadata_in_connection(
            _InsertConflictConnection(),  # type: ignore[arg-type]
            make_metadata(),
        )
