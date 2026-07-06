"""File metadata persistence and duplicate detection."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from psycopg import Connection

from app.core.database import connect

SUPPORTED_FILE_EXTENSIONS = {".pdf", ".docx", ".hwpx", ".pptx", ".xlsx", ".md"}


@dataclass(frozen=True)
class FileMetadataInput:
    original_file_name: str
    stored_file_name: str
    file_size_bytes: int
    sha256_checksum: str
    storage_path: str
    mime_type: str | None = None
    document_group: str = "default"
    security_level: str = "internal"
    uploaded_by: str | None = None
    document_title: str | None = None


@dataclass(frozen=True)
class FileMetadataRecord:
    file_id: int
    document_id: int | None
    original_file_name: str
    stored_file_name: str
    file_ext: str
    mime_type: str | None
    file_size_bytes: int
    sha256_checksum: str
    storage_path: str
    document_group: str
    security_level: str
    parse_status: str


@dataclass(frozen=True)
class CreateFileMetadataResult:
    file: FileMetadataRecord
    duplicate: bool


class UnsupportedFileExtensionError(ValueError):
    """Raised when a file extension is outside the MVP upload scope."""


class InvalidFileMetadataError(ValueError):
    """Raised when file metadata is incomplete or invalid."""


def _require_positive_id(value: int, field_name: str) -> None:
    if value <= 0:
        raise InvalidFileMetadataError(f"{field_name} must be greater than 0")


def calculate_sha256(file_path: Path) -> str:
    digest = sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_file_ext(file_name: str) -> str:
    return Path(file_name).suffix.lower()


def validate_file_metadata(metadata: FileMetadataInput) -> str:
    if not metadata.original_file_name.strip():
        raise InvalidFileMetadataError("original_file_name is required")
    if not metadata.stored_file_name.strip():
        raise InvalidFileMetadataError("stored_file_name is required")
    if metadata.file_size_bytes < 0:
        raise InvalidFileMetadataError("file_size_bytes must be greater than or equal to 0")
    if not metadata.sha256_checksum.strip():
        raise InvalidFileMetadataError("sha256_checksum is required")
    if not metadata.storage_path.strip():
        raise InvalidFileMetadataError("storage_path is required")

    file_ext = normalize_file_ext(metadata.original_file_name)
    if file_ext not in SUPPORTED_FILE_EXTENSIONS:
        raise UnsupportedFileExtensionError(f"Unsupported file extension: {file_ext or '(none)'}")
    return file_ext


def row_to_file_metadata_record(row: dict[str, Any]) -> FileMetadataRecord:
    return FileMetadataRecord(
        file_id=int(row["file_id"]),
        document_id=int(row["document_id"]) if row.get("document_id") is not None else None,
        original_file_name=str(row["original_file_name"]),
        stored_file_name=str(row["stored_file_name"]),
        file_ext=str(row["file_ext"]),
        mime_type=row["mime_type"],
        file_size_bytes=int(row["file_size_bytes"]),
        sha256_checksum=str(row["sha256_checksum"]),
        storage_path=str(row["storage_path"]),
        document_group=str(row["document_group"]),
        security_level=str(row["security_level"]),
        parse_status=str(row["parse_status"]),
    )


def find_file_by_checksum(connection: Connection, checksum: str) -> FileMetadataRecord | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                f.file_id,
                d.document_id,
                f.original_file_name,
                f.stored_file_name,
                f.file_ext,
                f.mime_type,
                f.file_size_bytes,
                f.sha256_checksum,
                f.storage_path,
                f.document_group,
                f.security_level,
                f.parse_status
            FROM files f
            LEFT JOIN documents d ON d.file_id = f.file_id
            WHERE f.sha256_checksum = %s
            ORDER BY d.document_id
            LIMIT 1
            """,
            (checksum,),
        )
        row = cursor.fetchone()
        return row_to_file_metadata_record(dict(row)) if row else None


def get_file_metadata_in_connection(
    connection: Connection,
    file_id: int,
) -> FileMetadataRecord | None:
    _require_positive_id(file_id, "file_id")
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                f.file_id,
                d.document_id,
                f.original_file_name,
                f.stored_file_name,
                f.file_ext,
                f.mime_type,
                f.file_size_bytes,
                f.sha256_checksum,
                f.storage_path,
                f.document_group,
                f.security_level,
                f.parse_status
            FROM files f
            LEFT JOIN documents d ON d.file_id = f.file_id
            WHERE f.file_id = %s
            ORDER BY d.document_id
            LIMIT 1
            """,
            (file_id,),
        )
        row = cursor.fetchone()
    return row_to_file_metadata_record(dict(row)) if row else None


def get_file_metadata(database_url: str, file_id: int) -> FileMetadataRecord | None:
    with connect(database_url) as connection:
        return get_file_metadata_in_connection(connection, file_id)


def mark_file_parse_running_in_connection(
    connection: Connection,
    file_id: int,
    *,
    parser_name: str,
    parser_version: str,
) -> FileMetadataRecord | None:
    _require_positive_id(file_id, "file_id")
    if not parser_name.strip():
        raise InvalidFileMetadataError("parser_name is required")
    if not parser_version.strip():
        raise InvalidFileMetadataError("parser_version is required")

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE files
            SET parser_name = %s,
                parser_version = %s,
                parse_status = 'running',
                parse_error_message = NULL,
                updated_at = now()
            WHERE file_id = %s
            """,
            (parser_name.strip(), parser_version.strip(), file_id),
        )
        if cursor.rowcount == 0:
            return None
    return get_file_metadata_in_connection(connection, file_id)


def mark_file_parse_running(
    database_url: str,
    file_id: int,
    *,
    parser_name: str,
    parser_version: str,
) -> FileMetadataRecord | None:
    with connect(database_url) as connection:
        return mark_file_parse_running_in_connection(
            connection,
            file_id,
            parser_name=parser_name,
            parser_version=parser_version,
        )


def mark_file_parse_succeeded_in_connection(
    connection: Connection,
    file_id: int,
    *,
    parser_name: str,
    parser_version: str,
    extracted_text_size: int,
) -> FileMetadataRecord | None:
    _require_positive_id(file_id, "file_id")
    if not parser_name.strip():
        raise InvalidFileMetadataError("parser_name is required")
    if not parser_version.strip():
        raise InvalidFileMetadataError("parser_version is required")
    if extracted_text_size < 0:
        raise InvalidFileMetadataError("extracted_text_size must be greater than or equal to 0")

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE files
            SET parser_name = %s,
                parser_version = %s,
                parse_status = 'succeeded',
                parse_error_message = NULL,
                extracted_text_size = %s,
                updated_at = now()
            WHERE file_id = %s
            """,
            (
                parser_name.strip(),
                parser_version.strip(),
                extracted_text_size,
                file_id,
            ),
        )
        if cursor.rowcount == 0:
            return None
    return get_file_metadata_in_connection(connection, file_id)


def mark_file_parse_succeeded(
    database_url: str,
    file_id: int,
    *,
    parser_name: str,
    parser_version: str,
    extracted_text_size: int,
) -> FileMetadataRecord | None:
    with connect(database_url) as connection:
        return mark_file_parse_succeeded_in_connection(
            connection,
            file_id,
            parser_name=parser_name,
            parser_version=parser_version,
            extracted_text_size=extracted_text_size,
        )


def mark_file_parse_failed_in_connection(
    connection: Connection,
    file_id: int,
    *,
    error_message: str,
    parser_name: str | None = None,
    parser_version: str | None = None,
) -> FileMetadataRecord | None:
    _require_positive_id(file_id, "file_id")
    if not error_message.strip():
        raise InvalidFileMetadataError("error_message is required")
    if parser_name is not None and not parser_name.strip():
        raise InvalidFileMetadataError("parser_name must not be blank")
    if parser_version is not None and not parser_version.strip():
        raise InvalidFileMetadataError("parser_version must not be blank")

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE files
            SET parser_name = COALESCE(%s, parser_name),
                parser_version = COALESCE(%s, parser_version),
                parse_status = 'failed',
                parse_error_message = %s,
                updated_at = now()
            WHERE file_id = %s
            """,
            (
                parser_name.strip() if parser_name is not None else None,
                parser_version.strip() if parser_version is not None else None,
                error_message.strip(),
                file_id,
            ),
        )
        if cursor.rowcount == 0:
            return None
    return get_file_metadata_in_connection(connection, file_id)


def mark_file_parse_failed(
    database_url: str,
    file_id: int,
    *,
    error_message: str,
    parser_name: str | None = None,
    parser_version: str | None = None,
) -> FileMetadataRecord | None:
    with connect(database_url) as connection:
        return mark_file_parse_failed_in_connection(
            connection,
            file_id,
            error_message=error_message,
            parser_name=parser_name,
            parser_version=parser_version,
        )


def create_file_metadata_in_connection(
    connection: Connection,
    metadata: FileMetadataInput,
) -> CreateFileMetadataResult:
    file_ext = validate_file_metadata(metadata)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO files (
                original_file_name,
                stored_file_name,
                file_ext,
                mime_type,
                file_size_bytes,
                sha256_checksum,
                storage_path,
                document_group,
                security_level,
                uploaded_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sha256_checksum) DO NOTHING
            RETURNING
                file_id,
                original_file_name,
                stored_file_name,
                file_ext,
                mime_type,
                file_size_bytes,
                sha256_checksum,
                storage_path,
                document_group,
                security_level,
                parse_status
            """,
            (
                metadata.original_file_name,
                metadata.stored_file_name,
                file_ext,
                metadata.mime_type,
                metadata.file_size_bytes,
                metadata.sha256_checksum,
                metadata.storage_path,
                metadata.document_group,
                metadata.security_level,
                metadata.uploaded_by,
            ),
        )
        inserted_file_row = cursor.fetchone()

    if inserted_file_row is None:
        duplicate = find_file_by_checksum(connection, metadata.sha256_checksum)
        if duplicate is None:
            msg = "Duplicate checksum detected, but existing file metadata was not found"
            raise RuntimeError(msg)
        return CreateFileMetadataResult(file=duplicate, duplicate=True)

    file_row = dict(inserted_file_row)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (
                file_id,
                document_title,
                document_group,
                security_level
            )
            VALUES (%s, %s, %s, %s)
            RETURNING document_id
            """,
            (
                file_row["file_id"],
                metadata.document_title or metadata.original_file_name,
                metadata.document_group,
                metadata.security_level,
            ),
        )
        document_row = cursor.fetchone()

    file_row["document_id"] = document_row["document_id"]
    return CreateFileMetadataResult(
        file=row_to_file_metadata_record(file_row),
        duplicate=False,
    )


def create_file_metadata(
    database_url: str,
    metadata: FileMetadataInput,
) -> CreateFileMetadataResult:
    with connect(database_url) as connection:
        return create_file_metadata_in_connection(connection, metadata)
