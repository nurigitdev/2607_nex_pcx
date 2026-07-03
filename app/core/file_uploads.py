"""Local file upload storage pipeline."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from app.core.file_metadata import (
    SUPPORTED_FILE_EXTENSIONS,
    FileMetadataInput,
    FileMetadataRecord,
    UnsupportedFileExtensionError,
    create_file_metadata,
    normalize_file_ext,
)

UPLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class FileUploadResult:
    file: FileMetadataRecord
    duplicate: bool


class InvalidUploadFileNameError(ValueError):
    """Raised when an upload does not include a usable file name."""


def sanitize_upload_file_name(file_name: str | None) -> str:
    if not file_name:
        raise InvalidUploadFileNameError("file name is required")

    sanitized = Path(file_name).name.strip()
    if not sanitized or sanitized in {".", ".."}:
        raise InvalidUploadFileNameError("file name is required")
    return sanitized


def build_stored_file_name(original_file_name: str) -> str:
    file_ext = normalize_file_ext(original_file_name)
    if file_ext not in SUPPORTED_FILE_EXTENSIONS:
        raise UnsupportedFileExtensionError(f"Unsupported file extension: {file_ext or '(none)'}")
    return f"{uuid4().hex}{file_ext}"


def write_stream_with_checksum(upload_stream: BinaryIO, target_path: Path) -> tuple[int, str]:
    digest = sha256()
    file_size_bytes = 0
    with target_path.open("wb") as target_file:
        while chunk := upload_stream.read(UPLOAD_CHUNK_SIZE):
            target_file.write(chunk)
            digest.update(chunk)
            file_size_bytes += len(chunk)
    return file_size_bytes, digest.hexdigest()


def store_upload(
    *,
    database_url: str,
    upload_stream: BinaryIO,
    original_file_name: str | None,
    storage_dir: Path,
    mime_type: str | None = None,
    document_group: str = "default",
    security_level: str = "internal",
    uploaded_by: str | None = None,
) -> FileUploadResult:
    safe_file_name = sanitize_upload_file_name(original_file_name)
    stored_file_name = build_stored_file_name(safe_file_name)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / stored_file_name

    try:
        file_size_bytes, checksum = write_stream_with_checksum(upload_stream, storage_path)
        metadata = FileMetadataInput(
            original_file_name=safe_file_name,
            stored_file_name=stored_file_name,
            file_size_bytes=file_size_bytes,
            sha256_checksum=checksum,
            storage_path=str(storage_path),
            mime_type=mime_type,
            document_group=document_group,
            security_level=security_level,
            uploaded_by=uploaded_by,
            document_title=Path(safe_file_name).stem,
        )
        metadata_result = create_file_metadata(database_url, metadata)
        if metadata_result.duplicate:
            storage_path.unlink(missing_ok=True)
        return FileUploadResult(file=metadata_result.file, duplicate=metadata_result.duplicate)
    except Exception:
        storage_path.unlink(missing_ok=True)
        raise
