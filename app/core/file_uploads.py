"""Local file upload storage pipeline."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from app.core.database import connect
from app.core.file_metadata import (
    SUPPORTED_FILE_EXTENSIONS,
    CreateFileMetadataResult,
    FileMetadataInput,
    FileMetadataRecord,
    UnsupportedFileExtensionError,
    create_file_metadata_in_connection,
    normalize_file_ext,
)
from app.core.pipeline_jobs import (
    PipelineJobInput,
    PipelineJobRecord,
    create_pipeline_job_in_connection,
)

UPLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class FileUploadResult:
    file: FileMetadataRecord
    duplicate: bool
    pipeline_job: PipelineJobRecord | None = None


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


def create_upload_metadata_and_pipeline_job(
    database_url: str,
    metadata: FileMetadataInput,
) -> FileUploadResult:
    with connect(database_url) as connection:
        metadata_result: CreateFileMetadataResult = create_file_metadata_in_connection(
            connection,
            metadata,
        )
        if metadata_result.duplicate:
            return FileUploadResult(file=metadata_result.file, duplicate=True)

        if metadata_result.file.document_id is None:
            msg = "Pipeline job cannot be queued without a document_id"
            raise RuntimeError(msg)

        pipeline_job = create_pipeline_job_in_connection(
            connection,
            PipelineJobInput(
                job_type="document_ingestion",
                file_id=metadata_result.file.file_id,
                document_id=metadata_result.file.document_id,
                requested_by_user_id=metadata.uploaded_by_user_id,
                metadata={
                    "original_file_name": metadata_result.file.original_file_name,
                    "document_group": metadata_result.file.document_group,
                    "security_level": metadata_result.file.security_level,
                    "sha256_checksum": metadata_result.file.sha256_checksum,
                    "uploaded_by": metadata.uploaded_by,
                    "uploaded_by_user_id": metadata.uploaded_by_user_id,
                    "owner_user_id": metadata.owner_user_id,
                    "owner_org_unit_id": metadata.owner_org_unit_id,
                    "access_scope": metadata.access_scope,
                },
            ),
        )
        return FileUploadResult(
            file=metadata_result.file,
            duplicate=False,
            pipeline_job=pipeline_job,
        )


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
    uploaded_by_user_id: int | None = None,
    owner_user_id: int | None = None,
    owner_org_unit_id: int | None = None,
    access_scope: str = "personal",
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
            uploaded_by_user_id=uploaded_by_user_id,
            owner_user_id=owner_user_id if owner_user_id is not None else uploaded_by_user_id,
            owner_org_unit_id=owner_org_unit_id,
            access_scope=access_scope,
            permission_metadata={
                "source": "upload",
                "uploaded_by_user_id": uploaded_by_user_id,
                "owner_user_id": (
                    owner_user_id if owner_user_id is not None else uploaded_by_user_id
                ),
                "owner_org_unit_id": owner_org_unit_id,
                "access_scope": access_scope,
            },
        )
        upload_result = create_upload_metadata_and_pipeline_job(database_url, metadata)
        if upload_result.duplicate:
            storage_path.unlink(missing_ok=True)
        return upload_result
    except Exception:
        storage_path.unlink(missing_ok=True)
        raise
